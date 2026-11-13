"""Standardized clean-room audit for domain-agnostic statistical decision theory & sequential stopping (``minos-gate``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._version import __version__
from ..closed_loop.sentinel_stopping import evaluate_regret_vs_coverage_stopping
from .voi import compute_decision_calibration_gap

if TYPE_CHECKING:
    from ..audit import AuditReport

__all__ = ["run_decision_audit"]


def run_decision_audit(seed: int = 42) -> AuditReport:
    """Run deterministic *Minos* calibration-gap & *Sentinel* sequential-stopping audit."""
    from ..audit import (
        SUITE_II_NAME,
        AuditAttestation,
        AuditReport,
        StatusLiteral,
        compute_input_sha256,
        compute_report_sha256,
        get_platform_fingerprint,
    )

    seed_int = int(seed)
    rng = np.random.Generator(np.random.PCG64(seed_int))
    truth = rng.normal(0.16, 0.055, size=360)
    pred = truth + rng.normal(0.0, 0.028, size=360)
    half_w = np.full(360, 0.032, dtype=np.float64)
    input_tensor = np.column_stack([pred - half_w, pred + half_w, truth])

    gap_rep = compute_decision_calibration_gap(
        (pred - half_w, pred + half_w),
        truth,
        cost_asymmetry=3.0,
        alpha=0.10,
        seed=seed_int,
    )
    sent_rep = evaluate_regret_vs_coverage_stopping(
        None,
        seed=seed_int,
        cost_asymmetry=3.0,
        n_boot=150,
    )

    auc_sep = float(gap_rep.observable_shift_auc - gap_rep.hidden_truth_shift_auc)
    loss_optimal = bool(gap_rep.loss_at_tau_star <= gap_rep.loss_at_tau_stat + 1.0e-12)
    shift_honest = bool(
        gap_rep.observable_shift_auc >= 0.75
        and 0.35 <= gap_rep.hidden_truth_shift_auc <= 0.65
    )

    if loss_optimal and shift_honest and sent_rep.separation_gate_passed:
        status: StatusLiteral = "PASS"
        verdict = (
            f"MINOS_SENTINEL_VERIFIED: G={gap_rep.gap:+.3f} (tau_stat={gap_rep.tau_stat:.3f}, "
            f"tau*={gap_rep.tau_star:.3f}), VoC={gap_rep.value_of_calibration:.4f}, "
            f"VoTG={gap_rep.value_of_trust_gate:.4f}, AUC sep={auc_sep:+.3f}; "
            f"Sentinel F{sent_rep.coverage_stop_fraction} vs F{sent_rep.regret_stop_fraction}."
        )
    elif loss_optimal and shift_honest:
        status = "WARN"
        verdict = "MINOS_VERIFIED_SENTINEL_EQUIVALENT: Minos gap & shift separation passed; Sentinel rules coincided."
    else:
        status = "HALT"
        verdict = "MINOS_FAILURE: Decision utility optimality or shift-observability separation violated."

    stage_role = "Domain-Agnostic Decision-Calibration Gap (Minos) & Sequential Stopping Gate (Sentinel)"
    metrics: dict[str, float | int | str | bool] = {
        "tau_stat": round(gap_rep.tau_stat, 6),
        "tau_star": round(gap_rep.tau_star, 6),
        "calibration_gap": round(gap_rep.gap, 6),
        "value_of_calibration": round(gap_rep.value_of_calibration, 6),
        "value_of_trust_gate": round(gap_rep.value_of_trust_gate, 6),
        "observable_shift_auc": round(gap_rep.observable_shift_auc, 6),
        "hidden_truth_shift_auc": round(gap_rep.hidden_truth_shift_auc, 6),
        "shift_auc_separation": round(auc_sep, 6),
        "sentinel_coverage_stop": int(sent_rep.coverage_stop_fraction),
        "sentinel_regret_stop": int(sent_rep.regret_stop_fraction),
        "sentinel_excess_regret": round(sent_rep.excess_regret_of_coverage_stop, 6),
        "sentinel_separation_passed": bool(sent_rep.separation_gate_passed),
    }
    thresholds: dict[str, float | int | str] = {
        "min_observable_shift_auc": 0.75,
        "hidden_shift_auc_band": "[0.35, 0.65]",
        "min_stopping_fraction_gap": 1,
    }
    att = AuditAttestation(
        input_sha256=compute_input_sha256(input_tensor),
        report_sha256=compute_report_sha256(
            package="minos-gate",
            version=__version__,
            suite=SUITE_II_NAME,
            stage_role=stage_role,
            status=status,
            verdict=verdict,
            seed=seed_int,
            metrics=metrics,
            thresholds=thresholds,
        ),
        platform_fingerprint=get_platform_fingerprint(),
    )
    return AuditReport(
        package="minos-gate",
        version=__version__,
        suite=SUITE_II_NAME,
        stage_role=stage_role,
        status=status,
        verdict=verdict,
        metrics=metrics,
        thresholds=thresholds,
        seed=seed_int,
        attestation=att,
    )
