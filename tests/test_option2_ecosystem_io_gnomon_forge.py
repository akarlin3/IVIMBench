"""Comprehensive deterministic tests for IVIMBench v1.4.0 expansion:
1. License-safe cross-package `Trust -> Value -> Action` bridge (`avery_stack.py`)
2. Volumetric 3D/4D NIfTI & DICOM clinical image I/O (`io/volumes.py`)
3. `Gnomon` clean-room reproduce-or-refute verification harness (`gnomon/`)
4. `Forge` 1.5T MR-Linac 7 MV FFF & Electron Return Effect (ERE) physics (`closed_loop/ere_forge.py`)
5. CLI subcommands (`pipeline`, `fit-volume`, `reproduce-gnomon`, `forge-ere-check`)
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ivimbench
from ivimbench.bridges.avery_stack import (
    ConformalIVIMEstimator,
    PipelineExecutionReport,
    SextantNLLSEstimator,
    ecosystem_availability,
    run_trust_value_action_pipeline,
)
from ivimbench.cli import main as cli_main
from ivimbench.closed_loop import (
    ParameterizedDoseGeometryTwin,
    SyntheticOrganTwin,
    SyntheticTwinLoop,
    compare_closed_loop_policies,
    compute_ere_boundary_perturbation,
    estimate_mc_simulation_floor,
    generate_unity_7mv_fff_spectrum,
    verify_photon_beam_purity,
)
from ivimbench.gnomon import GNOMON_TARGETS, run_gnomon_cleanroom_audit
from ivimbench.io import (
    IO_EXTRA_INSTALL_MSG,
    export_dro_to_nifti,
    fit_dwi_volume,
    load_dwi_volume,
    read_nifti,
    save_parameter_maps,
    write_nifti,
)
from ivimbench.phantoms.dro import generate_dro
from ivimbench.phantoms.synthetic import generate_cohort
from ivimbench.registry import list_estimators, run_estimator


# --------------------------------------------------------------------------- #
# Part 1: License-Safe Ecosystem Bridge & Unified Pipeline
# --------------------------------------------------------------------------- #
def test_pyproject_license_boundary_and_version() -> None:
    assert ivimbench.__version__ == "1.5.0"
    pyproject_text = (
        Path(__file__).resolve().parents[1] / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert 'version = "1.5.0"' in pyproject_text
    assert "nibabel>=5.0" in pyproject_text
    assert "pydicom>=2.4" in pyproject_text
    assert "conformal-ivim>=1.0.0" in pyproject_text
    # Strict MIT boundary: AGPL-3.0 sextant-core / sextant-ivim must NEVER appear in pyproject.toml
    assert "sextant-core" not in pyproject_text
    assert "sextant-ivim" not in pyproject_text
    assert "sextant_core" not in pyproject_text


def test_registered_bridge_estimators_and_methods() -> None:
    reg_names = list_estimators()
    assert "conformal_ivim" in reg_names
    assert "sextant_nlls" in reg_names

    eco = ecosystem_availability()
    assert eco["ivimbench"]["license"] == "MIT"
    assert "conformal_ivim" in eco
    assert "sextant_core" in eco

    cohort = generate_cohort(protocol="osipi", organ="pancreas", n_voxels=48, snr=30.0, seed=42)
    out_nlls = run_estimator("sextant_nlls", cohort.signals, cohort.bvalues, alpha=0.10, seed=42)
    assert out_nlls.point.shape == (48, 3)
    assert out_nlls.railed is not None and out_nlls.railed.shape == (48, 3)
    assert "dstar_railing_ci" in out_nlls.extras

    for method in ("split", "cqr", "mondrian"):
        est = ConformalIVIMEstimator(
            method=method,
            alpha=0.10,
            organ="pancreas",
            snr=30.0,
            n_train=120,
            n_calibration=100,
            monitor_n_null=60,
            seed=42,
        )
        res = est.fit_predict(cohort.signals, cohort.bvalues, alpha=0.10)
        assert res["point"].shape == (48, 3)
        assert res["lower"].shape == (48, 3)
        assert res["upper"].shape == (48, 3)
        assert np.all(res["upper"] >= res["lower"])
        assert res["conformal_method"] == method
        assert "shift_monitor" in res


def test_dynamic_external_sextant_and_conformal_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dynamic runtime dispatch when `sextant_core` or `conformal_ivim` is discoverable."""
    cohort = generate_cohort(protocol="osipi", organ="liver", n_voxels=32, snr=30.0, seed=17)

    # 1. Mock sextant_core module
    fake_sextant: Any = types.ModuleType("sextant_core")
    fake_sextant.__spec__ = types.SimpleNamespace(name="sextant_core")

    def _fake_fit_ivim(b: np.ndarray, sig: np.ndarray, **_: Any) -> Any:
        n = sig.shape[0]
        return types.SimpleNamespace(
            D=np.full(n, 1.1e-3),
            D_star=np.where(np.arange(n) % 2 == 0, 0.1995, 0.035),
            f=np.full(n, 0.18),
            S0=np.ones(n),
        )

    def _fake_classify_railing(
        ds: np.ndarray, bounds_D_star: tuple[float, float] = (3e-3, 0.20), rel_tol: float = 0.01
    ) -> dict[str, np.ndarray]:
        tol = rel_tol * (bounds_D_star[1] - bounds_D_star[0])
        m = (ds - bounds_D_star[0] <= tol) | (bounds_D_star[1] - ds <= tol)
        return {"railed_mask": m}

    def _fake_boot_ci(mask: np.ndarray, n_boot: int = 100, seed: int = 42) -> tuple[float, float, float]:
        p = 100.0 * float(np.mean(mask))
        return (p, max(p - 5.0, 0.0), min(p + 5.0, 100.0))

    fake_sextant.fit_ivim = _fake_fit_ivim
    fake_sextant.classify_railing = _fake_classify_railing
    fake_sextant.bootstrap_railing_ci = _fake_boot_ci
    monkeypatch.setitem(sys.modules, "sextant_core", fake_sextant)

    sex_est = SextantNLLSEstimator(use_external=True, seed=17)
    sex_res = sex_est.fit_predict(cohort.signals, cohort.bvalues, alpha=0.10)
    assert sex_res["backend"] == "sextant_core"
    assert abs(sex_res["dstar_railing_ci"]["estimate"] - 0.50) < 1e-6

    # 2. Mock conformal_ivim module
    fake_conf: Any = types.ModuleType("conformal_ivim")
    fake_conf.__spec__ = types.SimpleNamespace(name="conformal_ivim")

    class _FakeConformalIVIM:
        def __init__(self, method: str = "cqr", alpha: float = 0.10, **_: Any) -> None:
            self.method = method
            self.alpha = alpha

        def fit(self, x: np.ndarray, y: np.ndarray, **_: Any) -> _FakeConformalIVIM:
            return self

        def calibrate(
            self, x: np.ndarray, y: np.ndarray, alpha: float = 0.10, strata: Any = None, **_: Any
        ) -> _FakeConformalIVIM:
            self.last_strata = strata
            return self

        def predict(self, x: np.ndarray, **_: Any) -> np.ndarray:
            n = x.shape[0]
            return np.column_stack([np.full(n, 1.1e-3), np.full(n, 0.03), np.full(n, 0.18)])

        def predict_interval(
            self, x: np.ndarray, alpha: float = 0.10, strata: Any = None, **_: Any
        ) -> tuple[np.ndarray, np.ndarray]:
            pt = self.predict(x)
            return 0.85 * pt, 1.15 * pt

    class _FakeMonitor:
        def __init__(self, **_: Any) -> None:
            pass

        def fit_signals(self, x_ref: np.ndarray, **_: Any) -> _FakeMonitor:
            return self

        def evaluate_signals(self, x_test: np.ndarray, **_: Any) -> dict[str, Any]:
            return {
                "fires": False,
                "auc": 0.51,
                "maha": {"stat": 0.04, "p_value": 0.62},
                "resid": {"ratio": 1.01},
            }

    fake_conf.ConformalIVIM = _FakeConformalIVIM
    fake_conf.DeploymentValidityMonitor = _FakeMonitor
    monkeypatch.setitem(sys.modules, "conformal_ivim", fake_conf)

    c_est = ConformalIVIMEstimator(
        method="mondrian",
        use_external=True,
        n_train=60,
        n_calibration=60,
        seed=17,
    )
    c_res = c_est.fit_predict(cohort.signals, cohort.bvalues, alpha=0.10)
    assert c_res["backend"] == "conformal_ivim"
    assert c_res["shift_monitor"]["backend"] == "conformal_ivim.DeploymentValidityMonitor"


def test_run_trust_value_action_pipeline_determinism_and_schema() -> None:
    rep_a = run_trust_value_action_pipeline(
        protocol="osipi",
        organ="pancreas",
        snr=30.0,
        conformal_method="cqr",
        alpha=0.10,
        fractions=4,
        dro_shape=(16, 16),
        seed=42,
    )
    rep_b = run_trust_value_action_pipeline(
        protocol="osipi",
        organ="pancreas",
        snr=30.0,
        conformal_method="cqr",
        alpha=0.10,
        fractions=4,
        dro_shape=(16, 16),
        seed=42,
    )
    assert isinstance(rep_a, PipelineExecutionReport)
    assert rep_a.to_json() == rep_b.to_json()

    payload = json.loads(rep_a.to_json())
    for required_section in (
        "config",
        "ecosystem",
        "stage1_substrate_fit",
        "stage2_trust_shift",
        "stage3_value_of_information",
        "stage4_closed_loop_action",
    ):
        assert required_section in payload

    assert payload["stage1_substrate_fit"]["n_foreground_voxels"] > 0
    assert 0.0 <= payload["stage1_substrate_fit"]["dstar_railing_rate"] <= 1.0
    assert payload["stage2_trust_shift"]["in_distribution_shift"]["fires"] is False
    assert payload["stage2_trust_shift"]["ood_stress_shift"]["fires"] is True
    assert payload["stage3_value_of_information"]["conformal_regret"] >= 0.0
    assert len(payload["stage4_closed_loop_action"]["conformal_trajectory"]) == 4

    summary_table = rep_a.format_summary_table()
    assert "1. Substrate & NLLS Fit" in summary_table
    assert "2. Trust & Shift Gate" in summary_table
    assert "3. Value-of-Information Gate" in summary_table
    assert "4. Closed-Loop Adaptive RT" in summary_table


# --------------------------------------------------------------------------- #
# Part 2: Volumetric 3D/4D NIfTI & DICOM Clinical Image I/O
# --------------------------------------------------------------------------- #
def test_nifti_volume_roundtrip_exact_affine_and_parameter_maps(tmp_path: Path) -> None:
    dro = generate_dro(shape=(10, 12, 8), organ="pancreas", protocol="osipi", snr=30.0, seed=42)

    # Non-trivial oblique/rotated 4x4 float64 scanner affine to test exact preservation
    theta = 0.173456789123
    c, s = float(np.cos(theta)), float(np.sin(theta))
    oblique_affine = np.array(
        [
            [1.85 * c, -1.85 * s, 0.0, -94.123456789],
            [1.85 * s, 1.85 * c, 0.0, -112.987654321],
            [0.0, 0.0, 3.75, 28.333333333],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    dro_dir = tmp_path / "dro_nifti"
    exported = export_dro_to_nifti(dro, dro_dir, affine=oblique_affine, backend="numpy")
    assert exported["dwi"].is_file()
    assert exported["bval"].is_file()
    assert exported["roi_mask"].is_file()
    assert exported["gt_Dstar_map"].is_file()

    vol = load_dwi_volume(
        exported["dwi"],
        exported["bval"],
        mask_path=exported["roi_mask"],
        backend="numpy",
    )
    assert vol.spatial_shape == (10, 12, 8)
    assert vol.dwi_4d.shape == (10, 12, 8, dro.bvalues.size)
    assert vol.signals.shape == (int(np.sum(dro.foreground_mask)), dro.bvalues.size)
    np.testing.assert_allclose(vol.affine, oblique_affine, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(vol.bvalues, dro.bvalues, rtol=0.0, atol=1e-9)

    fit_out = run_estimator("sextant_nlls", vol.signals, vol.bvalues, alpha=0.10, seed=42)
    maps_dir = tmp_path / "fitted_maps"
    written = save_parameter_maps(
        maps_dir,
        fit_out,
        vol,
        outside_fill=np.nan,
        backend="numpy",
    )

    expected_keys = {
        "D_map",
        "Dstar_map",
        "f_map",
        "D_ci_width",
        "Dstar_ci_width",
        "f_ci_width",
        "Dstar_railed_mask",
        "exchangeability_report",
    }
    assert expected_keys.issubset(set(written.keys()))

    d_vol, d_aff, _ = read_nifti(written["D_map"], backend="numpy")
    assert d_vol.shape == (10, 12, 8)
    np.testing.assert_allclose(d_aff, oblique_affine, rtol=0.0, atol=1e-12)
    assert np.all(np.isnan(d_vol[~vol.mask]))
    assert np.all(np.isfinite(d_vol[vol.mask]))

    railed_vol, r_aff, _ = read_nifti(written["Dstar_railed_mask"], backend="numpy")
    assert railed_vol.dtype == np.uint8
    assert railed_vol.shape == (10, 12, 8)
    assert set(np.unique(railed_vol).tolist()).issubset({0, 1})
    assert np.all(railed_vol[~vol.mask] == 0)
    np.testing.assert_allclose(r_aff, oblique_affine, rtol=0.0, atol=1e-12)

    exch_json = json.loads(written["exchangeability_report"].read_text(encoding="utf-8"))
    assert exch_json["spatial_shape"] == [10, 12, 8]
    assert exch_json["n_masked_voxels"] == vol.n_masked
    assert "exchangeability" in exch_json


def test_io_missing_optional_dependencies_error_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dcm_dir = tmp_path / "fake_dicom_series"
    dcm_dir.mkdir()
    bval_file = tmp_path / "series.bval"
    bval_file.write_text("0 50 400 800\n", encoding="utf-8")

    monkeypatch.setattr("ivimbench.io.volumes.has_pydicom", lambda: False)
    with pytest.raises(ImportError, match=r"Install ivimbench\[io\] for NIfTI/DICOM support"):
        load_dwi_volume(dcm_dir, bval_file)

    monkeypatch.setattr("ivimbench.io.volumes.has_nibabel", lambda: False)
    dummy = np.ones((4, 4, 4), dtype=np.float32)
    with pytest.raises(ImportError, match=r"Install ivimbench\[io\] for NIfTI/DICOM support"):
        write_nifti(dummy, np.eye(4), tmp_path / "x.nii.gz", backend="nibabel")
    assert IO_EXTRA_INSTALL_MSG == "Install ivimbench[io] for NIfTI/DICOM support"


def test_fit_dwi_volume_convenience_pipeline(tmp_path: Path) -> None:
    dro_2d = generate_dro(shape=(12, 12), organ="liver", protocol="osipi", snr=30.0, seed=7)
    in_dir = tmp_path / "dro2d"
    exp = export_dro_to_nifti(dro_2d, in_dir, backend="numpy")
    vol, est_out, written = fit_dwi_volume(
        exp["dwi"],
        exp["bval"],
        tmp_path / "out2d",
        mask_path=exp["roi_mask"],
        estimator="segmented",
        outside_fill=0.0,
        backend="numpy",
    )
    assert vol.spatial_shape == (12, 12, 1)
    assert est_out.point.shape[0] == vol.n_masked
    f_vol, _, _ = read_nifti(written["f_map"], backend="numpy")
    assert np.all(f_vol[~vol.mask] == 0.0)


# --------------------------------------------------------------------------- #
# Part 3A: Gnomon Clean-Room Reproduce-or-Refute Harness
# --------------------------------------------------------------------------- #
def test_gnomon_cleanroom_audit_targets_and_sd_separation() -> None:
    assert "nlls_dstar_railing_rate" in GNOMON_TARGETS
    assert GNOMON_TARGETS["nlls_dstar_railing_rate"].reference_target == pytest.approx(0.547)
    assert GNOMON_TARGETS["nlls_dstar_railing_rate"].verified_window == (0.520, 0.564)

    verdict = run_gnomon_cleanroom_audit(n_voxels=2000, seed=42)
    verdict_repeat = run_gnomon_cleanroom_audit(n_voxels=2000, seed=42)
    assert verdict.to_json() == verdict_repeat.to_json()

    assert verdict.overall_status == "PASS"
    for check in verdict.checks.values():
        assert check.status == "PASS"
        assert check.ci_95[0] <= check.observed <= check.ci_95[1]

    # Verify C1 NLLS D* railing inside [0.520, 0.564]
    c1 = verdict.checks["nlls_dstar_railing_rate"]
    assert 0.520 <= c1.observed <= 0.564

    # Verify C2 vs C3: high-D* Gaussian shape failure vs quantile recovery
    high_laplace = verdict.tercile_coverage["Laplace-SD"]["high_tercile"]
    high_mcmc_sd = verdict.tercile_coverage["MCMC-SD"]["high_tercile"]
    high_mcmc_q = verdict.tercile_coverage["MCMC-quantile"]["high_tercile"]
    high_conf_q = verdict.tercile_coverage["conformal-quantile"]["high_tercile"]
    assert max(high_laplace, high_mcmc_sd) < 0.82
    assert min(high_mcmc_q, high_conf_q) >= 0.855

    # Verify C4 vs C5: explicit separation between honest_crlb and floored_railed
    marg_honest = verdict.marginal_coverage["honest_crlb"]
    marg_floored = verdict.marginal_coverage["floored_railed"]
    assert 0.79 <= marg_honest <= 0.915
    assert 0.28 <= marg_floored <= 0.68
    assert (marg_honest - marg_floored) > 0.25

    dispositions = {entry.claim_id: entry.action for entry in verdict.claims_ledger}
    assert dispositions["C1_NLLS_DSTAR_RAILING"] == "KEEP"
    assert dispositions["C4_HONEST_CRLB_MARGINAL_SEPARATION"] == "REFRAME"
    assert dispositions["C5_FLOORED_RAILED_ARTIFACT_SEPARATION"] == "DROP"


# --------------------------------------------------------------------------- #
# Part 3B: Forge 1.5T MR-Linac 7 MV FFF & Electron Return Effect (ERE) Physics
# --------------------------------------------------------------------------- #
def test_forge_7mv_fff_spectrum_and_ere_boundary_physics() -> None:
    spec = generate_unity_7mv_fff_spectrum(n_bins=64)
    assert spec.n_bins == 64
    assert spec.particle == "gamma"
    assert spec.zero_proton_contamination is True
    assert spec.proton_fraction == 0.0
    assert np.isclose(np.sum(spec.weights), 1.0)
    assert 1.85 <= spec.mean_energy_mev <= 2.35

    purity_bad = verify_photon_beam_purity(["gamma", "proton", "gamma"])
    assert purity_bad["zero_proton_contamination"] is False
    assert purity_bad["n_proton"] == 1

    # 3D water -> air -> water phantom irradiated along +x (beam_axis=0)
    dose_3d = np.full((20, 12, 12), 2.0, dtype=np.float64)
    density_3d = np.ones((20, 12, 12), dtype=np.float64)
    density_3d[8:13, :, :] = 0.0012  # Air cavity: exit at i=7, re-entry at i=13

    ere_15t = compute_ere_boundary_perturbation(
        dose_3d, density_3d, b_field_tesla=1.5, beam_axis=0
    )
    assert ere_15t.within_reference_band is True
    assert 12.4 <= ere_15t.peak_exit_enhancement_pct <= 20.9
    # Positive dose hotspot on upstream water->air exit face (i=7)
    assert float(np.mean(ere_15t.delta_dose[7, :, :])) > 0.20
    # Electronic disequilibrium dose dip on downstream air->water re-entry face (i=13)
    assert float(np.mean(ere_15t.delta_dose[13, :, :])) < -0.05

    # Strictly zero ERE perturbation at B = 0.0 T
    ere_0t = compute_ere_boundary_perturbation(
        dose_3d, density_3d, b_field_tesla=0.0, beam_axis=0
    )
    assert float(np.max(np.abs(ere_0t.delta_dose))) == 0.0
    np.testing.assert_array_equal(ere_0t.perturbed_dose, dose_3d)

    floor_rep = estimate_mc_simulation_floor(
        n_cases=10000, voxels_per_case=(64, 64, 64), histories_per_case=1e7
    )
    assert floor_rep.photon_purity_verified is True
    assert floor_rep.total_core_hours > 1000.0
    assert floor_rep.ere_gate_status == "ANALYTICAL_IN_BAND"


def test_closed_loop_twins_with_mr_linac_ere_toggle() -> None:
    dose_grid = np.full((8, 8, 8), 2.0, dtype=np.float64)
    organ_mask = np.ones((8, 8, 8), dtype=bool)
    density_map = np.ones((8, 8, 8), dtype=np.float64)
    density_map[4:6, :, :] = 0.0012  # Internal air pocket adjacent to target

    twin_no_ere = ParameterizedDoseGeometryTwin(
        dose_grid=dose_grid,
        organ_mask=organ_mask,
        density_map=density_map,
        apply_mr_linac_ere=False,
        seed=42,
    )
    twin_ere = ParameterizedDoseGeometryTwin(
        dose_grid=dose_grid,
        organ_mask=organ_mask,
        density_map=density_map,
        apply_mr_linac_ere=True,
        b_field_tesla=1.5,
        seed=42,
    )

    rep_off = SyntheticTwinLoop(
        substrate=twin_no_ere, n_fractions=3, snr=30.0, interval_mode="conformal", seed=42
    ).run()
    rep_on = SyntheticTwinLoop(
        substrate=twin_ere,
        n_fractions=3,
        snr=30.0,
        interval_mode="conformal",
        apply_mr_linac_ere=True,
        seed=42,
    ).run()

    assert rep_off.summary["apply_mr_linac_ere"] is False
    assert rep_on.summary["apply_mr_linac_ere"] is True
    assert rep_off.summary["mean_ere_peak_exit_pct"] == 0.0
    assert rep_on.summary["mean_ere_peak_exit_pct"] > 5.0

    cmp_organ_ere = compare_closed_loop_policies(
        substrate=SyntheticOrganTwin(shape=(12, 12), apply_mr_linac_ere=True, seed=42),
        n_fractions=3,
        snr=30.0,
        apply_mr_linac_ere=True,
        seed=42,
    )
    assert cmp_organ_ere.runs["conformal_trust_gated"].summary["mean_ere_peak_exit_pct"] > 0.0


# --------------------------------------------------------------------------- #
# Part 4: CLI Subcommands (`pipeline`, `fit-volume`, `reproduce-gnomon`, `forge-ere-check`)
# --------------------------------------------------------------------------- #
def test_v140_cli_subcommands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pipe_dir = tmp_path / "cli_pipe"
    rc1 = cli_main(
        [
            "pipeline",
            "--organ",
            "pancreas",
            "--protocol",
            "osipi",
            "--conformal",
            "cqr",
            "--fractions",
            "2",
            "--out",
            str(pipe_dir),
            "--json",
        ]
    )
    assert rc1 == 0
    out1 = json.loads(capsys.readouterr().out)
    assert out1["config"]["conformal_method"] == "cqr"
    assert (pipe_dir / "pipeline_report.json").is_file()

    vol_dir = tmp_path / "cli_vol"
    rc2 = cli_main(
        [
            "fit-volume",
            "--demo-volume",
            "--estimator",
            "segmented",
            "--out",
            str(vol_dir),
            "--json",
        ]
    )
    assert rc2 == 0
    out2 = json.loads(capsys.readouterr().out)
    assert out2["spatial_shape"] == [10, 10, 8]
    assert (vol_dir / "Dstar_railed_mask.nii.gz").is_file()

    rc3 = cli_main(["reproduce-gnomon", "--n-voxels", "2000", "--seed", "42", "--json"])
    assert rc3 == 0
    out3 = json.loads(capsys.readouterr().out)
    assert out3["overall_status"] == "PASS"

    rc4 = cli_main(["forge-ere-check", "--json"])
    assert rc4 == 0
    out4 = json.loads(capsys.readouterr().out)
    assert out4["qa_status"] == "PASS"
    assert out4["ere_zero_field_max_abs_gy"] == 0.0
