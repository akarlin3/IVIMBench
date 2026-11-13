"""Tests for domain-agnostic ivimbench.transport (levy-wall) and ivimbench.decision (minos-gate) entrypoints."""

from __future__ import annotations

import importlib.metadata
import json

import numpy as np
import pytest

import ivimbench
from ivimbench.audit import AuditAttestation, AuditReport
from ivimbench.decision import (
    MinosCalibrationGapReport,
    SentinelStoppingReport,
    compute_decision_calibration_gap,
    evaluate_regret_vs_coverage_stopping,
    run_decision_audit,
)
from ivimbench.decision.cli import main as minos_gate_main
from ivimbench.transport import (
    DEFAULT_Q_GRID,
    FractionalTransportCRLB,
    MisspecificationAliasingReport,
    TransportWallReport,
    compute_fractional_transport_crlb,
    discrete_multiexponential_decay,
    map_anomalous_identifiability_wall,
    probe_misspecification_conditional_aliasing,
    run_transport_audit,
    stretched_exponential_decay,
)
from ivimbench.transport.cli import main as levy_wall_main


def test_transport_domain_agnostic_physics_and_procrustes() -> None:
    q = np.asarray(DEFAULT_Q_GRID, dtype=np.float64)

    # Continuity limit: alpha = 1.0 equals exact mono-exponential
    s_stretch = stretched_exponential_decay(q, K=1.25e-3, alpha=1.0, S0=1.5)
    s_mono = 1.5 * np.exp(-q * 1.25e-3)
    np.testing.assert_allclose(s_stretch, s_mono, rtol=0.0, atol=1e-14)

    # Discrete multi-exponential mixture normalization and monotonicity
    s_mix = discrete_multiexponential_decay(
        q, rates=[1.0e-3, 25.0e-3], weights=[0.80, 0.20], S0=1.0
    )
    assert s_mix.shape == q.shape
    assert np.isclose(s_mix[0], 1.0)
    assert np.all(np.diff(s_mix) <= 0.0)

    # Fisher CRLB decreases monotonically with SNR
    crlb_lo = compute_fractional_transport_crlb(q, K=1.1e-3, alpha=0.75, snr=15.0)
    crlb_hi = compute_fractional_transport_crlb(q, K=1.1e-3, alpha=0.75, snr=60.0)
    assert isinstance(crlb_lo, FractionalTransportCRLB)
    assert crlb_lo.cv_alpha > crlb_hi.cv_alpha > 0.0
    assert crlb_lo["sigma_K"] > crlb_hi["sigma_K"] > 0.0

    # Joint (K, alpha) identifiability wall
    wall = map_anomalous_identifiability_wall(
        q_schedules={"compact_q": q},
        snr_grid=[15.0, 50.0],
        alpha_grid=[0.65, 1.0],
        n_replicates=12,
        n_boot=60,
        seed=42,
    )
    assert isinstance(wall, TransportWallReport)
    assert len(wall.dataframe) == 4
    assert "crlb_cv_alpha" in wall.dataframe.columns

    # Procrustes conditional coverage aliasing under continuous gamma dispersion
    proc = probe_misspecification_conditional_aliasing(
        q, family="dispersion_gamma", snr=40.0, seed=42, n_samples=300
    )
    assert isinstance(proc, MisspecificationAliasingReport)
    assert abs(proc.global_marginal_coverage - 0.90) <= 0.08
    assert proc.conditional_coverage_aliasing_gap >= 0.05
    assert proc.aliasing_detected is True

    # Deterministic transport audit with cryptographic attestation
    rep1 = run_transport_audit(seed=42)
    rep2 = run_transport_audit(seed=42)
    assert isinstance(rep1, AuditReport)
    assert rep1.status == "PASS"
    assert isinstance(rep1.attestation, AuditAttestation)
    assert rep1.attestation.report_sha256 == rep2.attestation.report_sha256


def test_levy_wall_cli_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    rc_aud = levy_wall_main(["audit", "--json", "--seed", "42"])
    out_aud = json.loads(capsys.readouterr().out)
    assert rc_aud == 0
    assert out_aud["package"] == "levy-wall"
    assert out_aud["status"] == "PASS"
    assert len(out_aud["attestation"]["report_sha256"]) == 64

    rc_wall = levy_wall_main(
        [
            "crlb-wall",
            "--snr-min",
            "10",
            "--snr-max",
            "80",
            "--n-snr",
            "2",
            "--alphas",
            "0.7,1.0",
            "--n-replicates",
            "8",
            "--n-boot",
            "40",
            "--json",
        ]
    )
    out_wall = json.loads(capsys.readouterr().out)
    assert rc_wall == 0
    assert len(out_wall["records"]) == 4

    rc_proc = levy_wall_main(
        ["procrustes-check", "--family", "dispersion_gamma", "--n-samples", "240", "--json"]
    )
    out_proc = json.loads(capsys.readouterr().out)
    assert rc_proc == 0
    assert out_proc["aliasing_detected"] is True


def test_decision_domain_agnostic_api_and_minos_gate_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rng = np.random.Generator(np.random.PCG64(42))
    truth = rng.normal(0.50, 0.18, size=300)
    pred = truth + rng.normal(0.0, 0.08, size=300)
    half_w = np.full(300, 0.09, dtype=np.float64)

    gap_rep = compute_decision_calibration_gap(
        (pred - half_w, pred + half_w),
        truth,
        cost_asymmetry=3.0,
        threshold_grid=np.linspace(0.25, 3.0, 111),
        seed=42,
    )
    assert isinstance(gap_rep, MinosCalibrationGapReport)
    assert np.isclose(gap_rep.gap, gap_rep.tau_star - gap_rep.tau_stat)
    assert gap_rep.loss_at_tau_star <= gap_rep.loss_at_tau_stat + 1e-12

    # Domain-agnostic (T, N) trajectory dict support in Sentinel stopping
    cov_traj = np.tile(np.array([0.89, 0.90, 0.90, 0.90, 0.90, 0.90])[:, None], (1, 120))
    reg_traj = np.tile(np.array([0.48, 0.39, 0.30, 0.21, 0.14, 0.14])[:, None], (1, 120))
    sent_rep = evaluate_regret_vs_coverage_stopping(
        {"coverage": cov_traj, "regret": reg_traj},
        seed=42,
        n_boot=80,
    )
    assert isinstance(sent_rep, SentinelStoppingReport)
    assert sent_rep.coverage_stop_fraction == 2
    assert sent_rep.regret_stop_fraction == 6
    assert sent_rep.separation_gate_passed is True

    # Deterministic decision audit
    aud1 = run_decision_audit(seed=42)
    aud2 = run_decision_audit(seed=42)
    assert aud1.status == "PASS"
    assert aud1.attestation is not None and aud2.attestation is not None
    assert aud1.attestation.report_sha256 == aud2.attestation.report_sha256

    # minos-gate CLI subcommands
    rc_aud = minos_gate_main(["audit", "--json", "--seed", "42"])
    out_aud = json.loads(capsys.readouterr().out)
    assert rc_aud == 0
    assert out_aud["package"] == "minos-gate"
    assert out_aud["status"] == "PASS"

    rc_gap = minos_gate_main(["gap-demo", "--cost-asymmetry", "3.0", "--json"])
    out_gap = json.loads(capsys.readouterr().out)
    assert rc_gap == 0
    assert "gap" in out_gap and "value_of_calibration" in out_gap

    rc_sent = minos_gate_main(["sentinel-stopping", "--n-boot", "80", "--json"])
    out_sent = json.loads(capsys.readouterr().out)
    assert rc_sent == 0
    assert "stopping_fraction_gap" in out_sent


def test_four_console_script_entrypoints_registered() -> None:
    assert hasattr(ivimbench, "transport")
    assert hasattr(ivimbench, "run_transport_audit")
    assert hasattr(ivimbench, "run_decision_audit")

    eps = {
        ep.name: ep.value
        for ep in importlib.metadata.entry_points(group="console_scripts")
        if ep.dist is not None and ep.dist.name == "ivimbench"
    }
    assert eps["ivimbench"] == "ivimbench.cli:main"
    assert eps["averycorp-suite"] == "ivimbench.cli:suite_main"
    assert eps["levy-wall"] == "ivimbench.transport.cli:main"
    assert eps["minos-gate"] == "ivimbench.decision.cli:main"
