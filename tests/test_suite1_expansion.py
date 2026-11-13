"""Comprehensive tests for IVIMBench v1.6.0 Suite I 3-Axis Architectural Expansion."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ivimbench
from ivimbench.audit import (
    AuditAttestation,
    AuditChain,
    build_suite1_audit_chain,
    run_audit,
    verify_suite_audit_lock,
    write_suite_audit_lock,
)
from ivimbench.bridges.avery_stack import (
    ConformalIVIMEstimator,
    detect_voxel_ood,
    evaluate_deployment_shift,
    get_canonical_fitter,
)
from ivimbench.cli import main as cli_main
from ivimbench.cli import suite_audit_main
from ivimbench.closed_loop.sentinel_stopping import evaluate_regret_vs_coverage_stopping
from ivimbench.decision.voi import compute_decision_calibration_gap
from ivimbench.io.clinical_bridge import export_slicer_rt_bundle
from ivimbench.io.volumes import read_nifti
from ivimbench.metrics.coverage import (
    MondrianConformalQuantile,
    SequentialConformalCalibrator,
    SplitConformalQuantile,
    SplitConformalResidual,
    calibrate_conformal_intervals,
)
from ivimbench.phantoms.dro import generate_dro
from ivimbench.phantoms.synthetic import generate_cohort
from ivimbench.registry import run_estimator


# --------------------------------------------------------------------------- #
# Axis 1: Canonical Cross-Package Delegation Contract & Standalone Fallbacks
# --------------------------------------------------------------------------- #
def test_canonical_fitter_delegation_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    cohort = generate_cohort(protocol="osipi", organ="liver", n_voxels=20, snr=35.0, seed=42)

    fitter_sextant = get_canonical_fitter(prefer_sextant=True)
    fitter_builtin = get_canonical_fitter(prefer_sextant=False)
    assert fitter_builtin.backend == "ivimbench_builtin"
    if importlib.util.find_spec("sextant_core") is not None:
        assert fitter_sextant.backend == "sextant_core"

    for fitter in (fitter_sextant, fitter_builtin):
        seg = fitter.fit_segmented(cohort.signals, cohort.bvalues)
        assert seg["point"].shape == (20, 3)
        assert seg["railed"].shape == (20, 3)

        nlls = fitter.fit_nlls(cohort.signals[:8], cohort.bvalues)
        assert nlls["point"].shape == (8, 3)

        crlb = fitter.compute_crlb(cohort.bvalues, cohort.params[:6], snr=35.0)
        assert crlb["std"].shape == (6, 3)
        assert np.all(crlb["std"] >= 0.0)

        railed = fitter.classify_railing(nlls["point"][:, 1], "Dstar")
        assert railed["railed_mask"].shape == (8,)
        assert railed["railed_mask"].dtype == np.bool_

        # 2D (b, TE) T2-IVIM fit check
        b_grid = np.array([0.0, 20.0, 80.0, 200.0, 600.0, 0.0, 20.0, 80.0, 200.0, 600.0])
        te_grid = np.array([45.0, 45.0, 45.0, 45.0, 45.0, 90.0, 90.0, 90.0, 90.0, 90.0])
        d0, ds0, f0, t2p0, t2t0 = 1.1e-3, 25.0e-3, 0.22, 110.0, 55.0
        w_p = f0 * np.exp(-te_grid / t2p0)
        w_t = (1.0 - f0) * np.exp(-te_grid / t2t0)
        s_2d = (w_p * np.exp(-b_grid * ds0) + w_t * np.exp(-b_grid * d0)) / (
            f0 * np.exp(-45.0 / t2p0) + (1.0 - f0) * np.exp(-45.0 / t2t0)
        )
        res_2d = fitter.fit_t2_ivim(np.vstack([s_2d, s_2d]), b_grid, te_grid)
        assert np.asarray(res_2d["D"]).shape == (2,)
        assert np.all(np.isfinite(res_2d["D"]))

    # Simulate environment without sextant_core installed
    real_find_spec = importlib.util.find_spec

    def _no_sextant(name: str, package: str | None = None) -> Any:
        if name == "sextant_core" or name.startswith("sextant_core."):
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", _no_sextant)
    fitter_fallback = get_canonical_fitter(prefer_sextant=True)
    assert fitter_fallback.backend == "ivimbench_builtin"
    audit_fallback = run_audit(seed=42, continuity_voxels=40, roundtrip_voxels=8, rt_fractions=3)
    assert audit_fallback.metrics["fitter_backend"] == "ivimbench_builtin"


def test_conformal_uq_and_shift_ood_delegation_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = generate_cohort(protocol="osipi", organ="pancreas", n_voxels=160, snr=30.0, seed=42)
    y_cal = cohort.params[:100]
    y_test = cohort.params[100:]
    rng = np.random.Generator(np.random.PCG64(42))
    pred_cal = y_cal + rng.normal(0.0, [0.1e-3, 4e-3, 0.025], size=y_cal.shape)
    pred_test = y_test + rng.normal(0.0, [0.1e-3, 4e-3, 0.025], size=y_test.shape)
    scale_cal = np.broadcast_to(np.array([0.1e-3, 4e-3, 0.025]), y_cal.shape)
    scale_test = np.broadcast_to(np.array([0.1e-3, 4e-3, 0.025]), y_test.shape)

    split_cal = SplitConformalResidual(alpha=0.10)
    lo_s, hi_s = split_cal.calibrate_apply(pred_cal, y_cal, pred_test)
    assert lo_s.shape == y_test.shape and np.all(hi_s >= lo_s)

    levels = np.array([0.05, 0.50, 0.95], dtype=np.float64)
    q_cal = np.stack([pred_cal - scale_cal, pred_cal, pred_cal + scale_cal], axis=-1)
    q_test = np.stack([pred_test - scale_test, pred_test, pred_test + scale_test], axis=-1)

    cqr_cal = SplitConformalQuantile(q_levels=levels)
    q_c = cqr_cal.calibrate_apply(q_cal, y_cal, q_test)
    assert q_c.shape == (60, 3, 3) and np.all(q_c[:, :, 2] >= q_c[:, :, 0])

    groups_cal = (cohort.signals[:100, 0] > np.median(cohort.signals[:100, 0])).astype(int)
    groups_test = (cohort.signals[100:, 0] > np.median(cohort.signals[:100, 0])).astype(int)
    mond_cal = MondrianConformalQuantile(q_levels=levels)
    q_m = mond_cal.calibrate_apply(q_cal, y_cal, groups_cal, q_test, groups_test)
    assert q_m.shape == (60, 3, 3) and np.all(q_m[:, :, 2] >= q_m[:, :, 0])

    # Sequential ACI & Conformal PID calibration
    for seq_method in ("aci", "conformal_pid"):
        seq = SequentialConformalCalibrator(method=seq_method, alpha=0.10)  # type: ignore[arg-type]
        lo_q, hi_q, tr_q = seq.calibrate_stream(
            pred_cal[:, 2], y_cal[:, 2], pred_test[:, 2], y_test[:, 2]
        )
        assert lo_q.shape == (60,)
        assert tr_q.shape == (60,)
        assert np.all(hi_q >= lo_q)

        est_seq = ConformalIVIMEstimator(
            method=seq_method,  # type: ignore[arg-type]
            alpha=0.10,
            n_train=80,
            n_calibration=80,
            seed=42,
        )
        out_seq = est_seq.fit_predict(cohort.signals[:40], bvalues="osipi")
        assert out_seq["lower"].shape == (40, 3)

    # Shift monitor & VoxelOODDetector delegation (non-monotonic tail rise violates normalized IVIM decay)
    shifted_signals = cohort.signals[100:] + 0.55 * (cohort.bvalues[None, :] / 800.0)
    shift_info = evaluate_deployment_shift(
        cohort.signals[:100], shifted_signals, bvalues=cohort.bvalues, seed=42
    )
    assert isinstance(shift_info["fires"], bool)
    assert "backend" in shift_info

    ood_info = detect_voxel_ood(
        cohort.signals[:100], shifted_signals, bvalues=cohort.bvalues, alpha_ood=0.05
    )
    assert ood_info["ood_mask"].shape == (60,)
    assert float(ood_info["flagged_fraction"]) > 0.20

    # Verify clean fallback when conformal_ivim is absent
    real_find_spec = importlib.util.find_spec

    def _no_conformal(name: str, package: str | None = None) -> Any:
        if name == "conformal_ivim" or name.startswith("conformal_ivim."):
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", _no_conformal)
    lo_fb, hi_fb, backend_fb = calibrate_conformal_intervals(
        "cqr",
        cal_true=y_cal,
        cal_lo=pred_cal - scale_cal,
        cal_hi=pred_cal + scale_cal,
        test_lo=pred_test - scale_test,
        test_hi=pred_test + scale_test,
        alpha=0.10,
    )
    assert backend_fb == "ivimbench_builtin"
    assert lo_fb.shape == y_test.shape and np.all(hi_fb >= lo_fb)

    fb_ood = detect_voxel_ood(cohort.signals[:100], shifted_signals, use_external=False)
    assert fb_ood["backend"] == "ivimbench_builtin"
    assert float(fb_ood["flagged_fraction"]) > 0.20


# --------------------------------------------------------------------------- #
# Axis 2: Minos Gap, Sentinel Stopping & Cryptographic AuditAttestation
# --------------------------------------------------------------------------- #
def test_minos_decision_calibration_gap_and_shift_separation() -> None:
    rep = compute_decision_calibration_gap(cost_asymmetry=3.5, alpha=0.10, seed=42)
    assert rep.status == "PASS"
    assert rep.tau_stat > 0.0
    assert rep.tau_star > 0.0
    assert np.isclose(rep.gap, rep.tau_star - rep.tau_stat)
    assert rep.loss_at_tau_star <= rep.loss_at_tau_stat + 1e-12
    assert rep.value_of_calibration >= 0.0
    assert rep.value_of_trust_gate >= 0.0
    assert rep.observable_shift_auc >= 0.75
    assert 0.35 <= rep.hidden_truth_shift_auc <= 0.65


def test_sentinel_regret_vs_coverage_stopping() -> None:
    rep = evaluate_regret_vs_coverage_stopping(seed=42, n_fractions=7, n_boot=120)
    assert rep.n_fractions == 7
    assert 1 <= rep.coverage_stop_fraction <= 7
    assert 1 <= rep.regret_stop_fraction <= 7
    assert rep.stopping_gap_ci95[0] <= rep.stopping_fraction_gap <= rep.stopping_gap_ci95[1]
    assert isinstance(rep.separation_gate_passed, bool)
    assert len(rep.fraction_trajectory) == 7


def test_cryptographic_audit_chain_and_lockfile_cli(tmp_path: Path) -> None:
    chain = build_suite1_audit_chain(seed=42)
    assert isinstance(chain, AuditChain)
    assert len(chain.chain_sha256) == 64
    assert len(chain.stages) == 3

    stage3 = run_audit(seed=42)
    assert isinstance(stage3.attestation, AuditAttestation)
    assert len(stage3.attestation.input_sha256) == 64
    assert len(stage3.attestation.report_sha256) == 64
    assert "python" in stage3.attestation.platform_fingerprint

    lock_file = tmp_path / "audit-lock.json"
    rc_write = suite_audit_main(["--seed", "42", "--write-lock", str(lock_file), "--json"])
    assert rc_write == 0
    assert lock_file.is_file()

    rc_verify = suite_audit_main(["--seed", "42", "--verify-lock", str(lock_file)])
    assert rc_verify == 0
    ver_dict = verify_suite_audit_lock(chain, lock_file)
    assert ver_dict["verified"] is True

    # Tamper with chain_sha256 and ensure verification fails with non-zero exit code
    tampered = json.loads(lock_file.read_text(encoding="utf-8"))
    tampered["chain_sha256"] = "0" * 64
    tampered_path = tmp_path / "tampered-lock.json"
    write_suite_audit_lock(tampered, tampered_path)
    rc_bad = suite_audit_main(["--seed", "42", "--verify-lock", str(tampered_path)])
    assert rc_bad == 1


# --------------------------------------------------------------------------- #
# Axis 3: 3D Slicer / DICOM-RT Bundle & Interactive Volumetric HTML Viewer
# --------------------------------------------------------------------------- #
def test_export_slicer_rt_bundle_and_halt_gating(tmp_path: Path) -> None:
    dro = generate_dro(shape=(10, 10, 8), organ="pancreas", protocol="osipi", snr=30.0, seed=42)
    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 30.0],
            [0.0, -2.0, 0.0, 40.0],
            [0.0, 0.0, 3.5, -15.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    sig_fg, _ = dro.to_voxels(foreground_only=True)
    fit_res = run_estimator("conformal_ivim", sig_fg, dro.bvalues, alpha=0.10, seed=42)
    chain = build_suite1_audit_chain(seed=42)

    out_dir = tmp_path / "slicer_bundle"
    bundle = export_slicer_rt_bundle(
        output_dir=out_dir,
        subject_id="07",
        fit_result=fit_res,
        audit_chain=chain,
        volume_meta={
            "mask": dro.foreground_mask,
            "shape": dro.spatial_shape,
            "affine": affine,
            "signals": sig_fg,
            "bvalues": dro.bvalues,
        },
        decision_threshold=0.25,
        seed=42,
    )

    elig_path = bundle.paths["rt_escalation_eligible_mask"]
    halt_path = bundle.paths["ood_or_railed_halt_mask"]
    assert elig_path.name == "sub-07_desc-rt_escalation_eligible_mask.nii.gz"
    assert halt_path.name == "sub-07_desc-ood_or_railed_halt_mask.nii.gz"

    import gzip
    import struct

    with gzip.open(elig_path, "rb") as gz_e:
        assert struct.unpack("<h", gz_e.read(348)[70:72])[0] == 2  # NIFTI DT_UINT8
    with gzip.open(halt_path, "rb") as gz_h:
        assert struct.unpack("<h", gz_h.read(348)[70:72])[0] == 2  # NIFTI DT_UINT8

    elig_arr, elig_aff, _ = read_nifti(elig_path)
    halt_arr, halt_aff, _ = read_nifti(halt_path)
    np.testing.assert_allclose(elig_aff, affine, atol=1e-5)
    np.testing.assert_allclose(halt_aff, affine, atol=1e-5)
    assert elig_arr.dtype == np.uint8
    assert halt_arr.dtype == np.uint8
    assert set(np.unique(elig_arr)).issubset({0, 1})
    assert set(np.unique(halt_arr)).issubset({0, 1})
    # Eligible and halt masks must be strictly disjoint
    assert int(np.sum((elig_arr > 0) & (halt_arr > 0))) == 0

    # Verify LUTs, 3D Slicer loader, RT action plan JSON, and HTML viewer
    assert (out_dir / "ivim_railing_lut.ctbl").is_file()
    assert (out_dir / "ivim_uq_width_lut.ctbl").is_file()
    slicer_py = (out_dir / "load_in_slicer.py").read_text(encoding="utf-8")
    assert "labelOpacity=0.40" in slicer_py
    assert bundle.chain_sha256 in slicer_py

    rt_plan = json.loads((out_dir / "sub-07_rt_action_plan.json").read_text(encoding="utf-8"))
    assert rt_plan["dicom_rt_metadata"]["PatientIdentityRemoved"] == "YES"

    viewer_html = (out_dir / "volume_viewer.html").read_text(encoding="utf-8")
    assert "#121b2d" in viewer_html
    assert "#d4af37" in viewer_html
    assert "z-slice-slider" in viewer_html
    for label in (
        "D Map",
        "D* Map",
        "f Map",
        "Conformal CI Width",
        "D* Boundary-Railed Mask",
        "RT Escalation Gate Mask",
    ):
        assert label in viewer_html

    # Verify AuditChain HALT gates out 100% of RT escalation voxels
    halted_bundle = export_slicer_rt_bundle(
        output_dir=tmp_path / "slicer_halted",
        subject_id="08",
        fit_result=fit_res,
        audit_chain={"overall_status": "HALT", "chain_sha256": "f" * 64},
        volume_meta={
            "mask": dro.foreground_mask,
            "shape": dro.spatial_shape,
            "affine": affine,
            "signals": sig_fg,
            "bvalues": dro.bvalues,
        },
        decision_threshold=0.50,
        seed=42,
    )
    assert halted_bundle.n_rt_eligible_voxels == 0


def test_new_cli_subcommands_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert hasattr(ivimbench, "export_slicer_rt_bundle")
    assert hasattr(ivimbench, "compute_decision_calibration_gap")
    assert hasattr(ivimbench, "evaluate_regret_vs_coverage_stopping")

    # 1. ivimbench minos-gap --include-sentinel --json
    rc_minos = cli_main(["minos-gap", "--include-sentinel", "--json"])
    out_minos = json.loads(capsys.readouterr().out)
    assert rc_minos == 0
    assert "gap" in out_minos
    assert "sentinel_stopping" in out_minos

    # 2. ivimbench export-slicer --demo-volume --out <dir> --json
    slicer_out = tmp_path / "cli_slicer_rt"
    rc_slicer = cli_main(
        ["export-slicer", "--demo-volume", "--out", str(slicer_out), "--json"]
    )
    out_slicer = json.loads(capsys.readouterr().out)
    assert rc_slicer == 0
    assert (slicer_out / "load_in_slicer.py").is_file()
    assert (slicer_out / "volume_viewer.html").is_file()
    assert out_slicer["overall_audit_status"] in ("PASS", "WARN")

    # 3. ivimbench fit-volume --demo-volume --out <dir> emits volume_viewer.html
    vol_out = tmp_path / "cli_fit_vol"
    rc_vol = cli_main(
        ["fit-volume", "--demo-volume", "--estimator", "segmented", "--out", str(vol_out), "--json"]
    )
    out_vol = json.loads(capsys.readouterr().out)
    assert rc_vol == 0
    assert "volume_viewer" in out_vol["files"]
    assert (vol_out / "volume_viewer.html").is_file()
