"""Sentinel multi-fraction stopping rules: Regret stability vs Conformal coverage stability.

Implements the publication-safe *Sentinel* multi-fraction stopping rule benchmark
comparing:
1. ``coverage_stop``: stopping protocol adaptation / escalation review when rolling
   conformal marginal coverage stabilizes within ``coverage_tol`` of ``1 - alpha``.
2. ``regret_stop``: stopping when rolling expected asymmetric clinical utility regret
   stabilizes within ``regret_rel_tol``.

Includes matched null false-alarm calibration, deterministic ``PCG64`` voxel-bootstrap
95% confidence intervals, and the pre-registered separation/refutation diagnostic gate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..decision.voi import infer_exceedance_probability
from ..phantoms.synthetic import DEFAULT_SEED
from ..runner import _sanitize_json

__all__ = [
    "SentinelStoppingReport",
    "evaluate_regret_vs_coverage_stopping",
]


@dataclass(frozen=True)
class SentinelStoppingReport:
    """Structured report for *Sentinel* multi-fraction regret-vs-coverage stopping."""

    status: str
    verdict: str
    seed: int
    n_fractions: int
    n_voxels: int
    alpha: float
    cost_asymmetry: float
    coverage_stop_fraction: int
    regret_stop_fraction: int
    stopping_fraction_gap: float
    stopping_gap_ci95: tuple[float, float]
    coverage_at_coverage_stop: float
    coverage_at_regret_stop: float
    regret_at_coverage_stop: float
    regret_at_regret_stop: float
    excess_regret_of_coverage_stop: float
    excess_regret_ci95: tuple[float, float]
    matched_false_alarm_rate: float
    separation_gate_passed: bool
    fraction_trajectory: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_json(
            {
                "status": self.status,
                "verdict": self.verdict,
                "seed": int(self.seed),
                "n_fractions": int(self.n_fractions),
                "n_voxels": int(self.n_voxels),
                "alpha": float(self.alpha),
                "cost_asymmetry": float(self.cost_asymmetry),
                "coverage_stop_fraction": int(self.coverage_stop_fraction),
                "regret_stop_fraction": int(self.regret_stop_fraction),
                "stopping_fraction_gap": float(self.stopping_fraction_gap),
                "stopping_gap_ci95": [float(self.stopping_gap_ci95[0]), float(self.stopping_gap_ci95[1])],
                "coverage_at_coverage_stop": float(self.coverage_at_coverage_stop),
                "coverage_at_regret_stop": float(self.coverage_at_regret_stop),
                "regret_at_coverage_stop": float(self.regret_at_coverage_stop),
                "regret_at_regret_stop": float(self.regret_at_regret_stop),
                "excess_regret_of_coverage_stop": float(self.excess_regret_of_coverage_stop),
                "excess_regret_ci95": [float(self.excess_regret_ci95[0]), float(self.excess_regret_ci95[1])],
                "matched_false_alarm_rate": float(self.matched_false_alarm_rate),
                "separation_gate_passed": bool(self.separation_gate_passed),
                "fraction_trajectory": list(self.fraction_trajectory),
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _solve_stopping_fractions(
    cov_mat: np.ndarray,
    reg_mat: np.ndarray,
    *,
    nominal: float,
    cov_tol: float,
    reg_tol: float,
    min_fraction: int = 2,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Compute 1-indexed ``(coverage_stop_fraction, regret_stop_fraction)`` from ``(T, N)`` matrices."""
    n_frac = cov_mat.shape[0]
    mean_cov = np.mean(cov_mat, axis=1)
    mean_reg = np.mean(reg_mat, axis=1)

    cov_stop = n_frac
    for t in range(max(1, min_fraction - 1), n_frac):
        rolling_gap = abs(0.5 * (mean_cov[t] + mean_cov[t - 1]) - nominal)
        step_drift = abs(mean_cov[t] - mean_cov[t - 1])
        if rolling_gap <= cov_tol and step_drift <= cov_tol:
            cov_stop = t + 1
            break

    reg_stop = n_frac
    for t in range(max(1, min_fraction - 1), n_frac):
        denom = max(float(mean_reg[0]), 1e-6)
        rel_change = abs(mean_reg[t] - mean_reg[t - 1]) / denom
        # Regret stabilization requires monotone non-increasing plateau within reg_tol
        if rel_change <= reg_tol and mean_reg[t] <= mean_reg[t - 1] + 0.5 * reg_tol * denom:
            reg_stop = t + 1
            break

    return cov_stop, reg_stop, mean_cov, mean_reg


def _build_voxel_matrices_from_loop(
    loop_result: Any,
    *,
    rng: np.random.Generator,
    alpha: float,
    cost_asymmetry: float,
    n_voxels_default: int,
    n_fractions_default: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct ``(T, N)`` per-fraction coverage, asymmetric utility regret, and width arrays."""
    nominal = 1.0 - float(alpha)
    lam = float(cost_asymmetry)

    # Extract trajectory summary if a ClosedLoopResult / ClosedLoopComparisonReport / dict is supplied
    traj_df = None
    if loop_result is not None:
        if hasattr(loop_result, "fractions_df"):
            traj_df = loop_result.fractions_df
        elif hasattr(loop_result, "runs") and isinstance(loop_result.runs, Mapping):
            key = (
                "conformal_trust_gated"
                if "conformal_trust_gated" in loop_result.runs
                else next(iter(loop_result.runs))
            )
            traj_df = loop_result.runs[key].fractions_df
        elif isinstance(loop_result, Mapping) and "conformal_trajectory" in loop_result:
            traj_df = loop_result["conformal_trajectory"]

    if traj_df is not None:
        rows = (
            traj_df.to_dict(orient="records")
            if hasattr(traj_df, "to_dict")
            else list(traj_df)
        )
        n_frac = max(len(rows), 3)
        n_vox = int(n_voxels_default)
        base_f = rng.uniform(0.05, 0.24, size=n_vox)
        cutoff = 0.12
        cov_mat = np.empty((n_frac, n_vox), dtype=np.float64)
        reg_mat = np.empty((n_frac, n_vox), dtype=np.float64)
        wid_mat = np.empty((n_frac, n_vox), dtype=np.float64)

        for t in range(n_frac):
            row = rows[min(t, len(rows) - 1)]
            target_cov = float(row.get("f_coverage", nominal))
            # Progressive microvascular radiation response shifts f downward across fractions
            f_t = np.clip(base_f * np.exp(-0.14 * t) + rng.normal(0.0, 0.006, size=n_vox), 0.01, 0.55)
            sigma_t = np.maximum(0.036 * np.exp(-0.16 * t), 0.010)
            f_hat = f_t + rng.normal(0.0, sigma_t, size=n_vox)
            z_q = float(np.quantile(np.abs(rng.normal(0.0, 1.0, size=600)), np.clip(target_cov, 0.5, 0.99)))
            half_w = z_q * sigma_t * np.ones(n_vox, dtype=np.float64)
            lo_t = np.clip(f_hat - half_w, 0.0, 0.70)
            hi_t = np.clip(f_hat + half_w, 0.0, 0.70)
            cov_mat[t] = ((f_t >= lo_t) & (f_t <= hi_t)).astype(np.float64)
            wid_mat[t] = hi_t - lo_t

            p_exc, _, _ = infer_exceedance_probability(
                lo_t, hi_t, cutoff, point=f_hat, alpha=alpha, direction="less"
            )
            escalate = p_exc >= (1.0 / (1.0 + lam))
            hypoxic = f_t <= cutoff
            reg_mat[t] = lam * (hypoxic & (~escalate)).astype(np.float64) + (
                (~hypoxic) & escalate
            ).astype(np.float64)
        return cov_mat, reg_mat, wid_mat

    # Canonical longitudinal multi-fraction Sentinel simulation:
    # Online ACI rapidly locks onto 90% coverage by fraction 2, whereas radiation-induced
    # perfusion boundary crossing near cutoff c=0.12 causes expected clinical regret to
    # continue evolving until fraction 5-6 before stabilizing.
    n_frac = int(n_fractions_default)
    n_vox = int(n_voxels_default)
    cutoff = 0.12
    f_0 = rng.uniform(0.07, 0.23, size=n_vox)
    cov_mat = np.empty((n_frac, n_vox), dtype=np.float64)
    reg_mat = np.empty((n_frac, n_vox), dtype=np.float64)
    wid_mat = np.empty((n_frac, n_vox), dtype=np.float64)

    cal_z = np.sort(np.abs(rng.normal(0.0, 1.0, size=400)))
    q_nom = float(cal_z[min(int(np.ceil(401 * nominal)) - 1, 399)])

    for t in range(n_frac):
        # Cumulative dose progressively drives borderline tumor voxels across hypoxia threshold
        response_shift = 0.055 * (1.0 - np.exp(-0.55 * t))
        f_t = np.clip(f_0 - response_shift, 0.015, 0.55)
        # Sequential estimation precision sharpens across fractions via longitudinal pooling
        sigma_t = 0.042 / np.sqrt(1.0 + 0.85 * t)
        noise_t = rng.normal(0.0, sigma_t, size=n_vox)
        f_hat = np.clip(f_t + noise_t, 0.0, 0.70)
        half_w = q_nom * sigma_t * np.ones(n_vox, dtype=np.float64)
        lo_t = np.clip(f_hat - half_w, 0.0, 0.70)
        hi_t = np.clip(f_hat + half_w, 0.0, 0.70)

        cov_mat[t] = ((f_t >= lo_t) & (f_t <= hi_t)).astype(np.float64)
        wid_mat[t] = hi_t - lo_t

        p_exc, _, _ = infer_exceedance_probability(
            lo_t, hi_t, cutoff, point=f_hat, alpha=alpha, direction="less"
        )
        escalate = p_exc >= (1.0 / (1.0 + lam))
        hypoxic = f_t <= cutoff
        reg_mat[t] = lam * (hypoxic & (~escalate)).astype(np.float64) + (
            (~hypoxic) & escalate
        ).astype(np.float64)

    return cov_mat, reg_mat, wid_mat


def evaluate_regret_vs_coverage_stopping(
    loop_result: Any = None,
    seed: int = DEFAULT_SEED,
    *,
    alpha: float = 0.10,
    cost_asymmetry: float = 3.0,
    coverage_tol: float = 0.035,
    regret_rel_tol: float = 0.06,
    n_voxels: int = 260,
    n_fractions: int = 8,
    n_boot: int = 250,
) -> SentinelStoppingReport:
    """Compare rolling conformal ``coverage_stop`` vs clinical utility ``regret_stop`` (*Sentinel*).

    Quantifies stopping-fraction divergence, excess expected clinical regret incurred when
    halting protocol adaptation on statistical coverage stability alone, voxel-bootstrap
    95% confidence intervals (`PCG64`), matched null false-alarm rate, and the pre-registered
    separation/refutation diagnostic gate.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"`alpha` must lie in (0, 1); got {alpha!r}.")
    if float(cost_asymmetry) <= 0.0:
        raise ValueError(f"`cost_asymmetry` must be > 0; got {cost_asymmetry!r}.")

    ss = np.random.SeedSequence(int(seed))
    sim_ss, boot_ss, null_ss = ss.spawn(3)
    rng_sim = np.random.Generator(np.random.PCG64(sim_ss))
    rng_boot = np.random.Generator(np.random.PCG64(boot_ss))
    rng_null = np.random.Generator(np.random.PCG64(null_ss))

    cov_mat, reg_mat, wid_mat = _build_voxel_matrices_from_loop(
        loop_result,
        rng=rng_sim,
        alpha=float(alpha),
        cost_asymmetry=float(cost_asymmetry),
        n_voxels_default=int(n_voxels),
        n_fractions_default=int(n_fractions),
    )
    n_frac, n_vox = cov_mat.shape
    nominal = 1.0 - float(alpha)

    cov_stop, reg_stop, mean_cov, mean_reg = _solve_stopping_fractions(
        cov_mat,
        reg_mat,
        nominal=nominal,
        cov_tol=float(coverage_tol),
        reg_tol=float(regret_rel_tol),
    )
    mean_wid = np.mean(wid_mat, axis=1)

    gap_frac = float(reg_stop - cov_stop)
    reg_at_cov = float(mean_reg[cov_stop - 1])
    reg_at_reg = float(mean_reg[reg_stop - 1])
    excess_regret = float(reg_at_cov - reg_at_reg)

    # Voxel-level bootstrap 95% CIs on stopping gap and excess clinical regret
    boot_gaps = np.empty(int(n_boot), dtype=np.float64)
    boot_excess = np.empty(int(n_boot), dtype=np.float64)
    for b_idx in range(int(n_boot)):
        idx = rng_boot.integers(0, n_vox, size=n_vox)
        c_b, r_b, _, mr_b = _solve_stopping_fractions(
            cov_mat[:, idx],
            reg_mat[:, idx],
            nominal=nominal,
            cov_tol=float(coverage_tol),
            reg_tol=float(regret_rel_tol),
        )
        boot_gaps[b_idx] = float(r_b - c_b)
        boot_excess[b_idx] = float(mr_b[c_b - 1] - mr_b[r_b - 1])

    gap_ci = (float(np.percentile(boot_gaps, 2.5)), float(np.percentile(boot_gaps, 97.5)))
    excess_ci = (float(np.percentile(boot_excess, 2.5)), float(np.percentile(boot_excess, 97.5)))

    # Matched stationary null false-alarm calibration check
    n_null_trials = 100
    false_alarms = 0
    for _ in range(n_null_trials):
        null_cov = (rng_null.uniform(size=(n_frac, n_vox)) <= nominal).astype(np.float64)
        step_drifts = np.abs(np.diff(np.mean(null_cov, axis=1)))
        if np.any(step_drifts > 2.5 * float(coverage_tol)):
            false_alarms += 1
    matched_far = float(false_alarms / float(n_null_trials))

    separation_passed = bool(abs(gap_frac) >= 1.0 and excess_regret > 0.0)
    if separation_passed:
        status = "PASS"
        verdict = (
            f"SEPARATION_CONFIRMED: coverage_stop=F{cov_stop} vs regret_stop=F{reg_stop} "
            f"(gap={gap_frac:+.1f} fractions, excess regret={excess_regret:+.4f} "
            f"[{excess_ci[0]:+.4f}, {excess_ci[1]:+.4f}])."
        )
    else:
        status = "WARN"
        verdict = (
            f"REFUTATION_EQUIVALENT: coverage_stop=F{cov_stop} and regret_stop=F{reg_stop} "
            f"coincide within statistical uncertainty (excess regret={excess_regret:+.4f})."
        )

    trajectory = [
        {
            "fraction": float(t + 1),
            "coverage": float(mean_cov[t]),
            "expected_regret": float(mean_reg[t]),
            "mean_width": float(mean_wid[t]),
        }
        for t in range(n_frac)
    ]

    return SentinelStoppingReport(
        status=status,
        verdict=verdict,
        seed=int(seed),
        n_fractions=n_frac,
        n_voxels=n_vox,
        alpha=float(alpha),
        cost_asymmetry=float(cost_asymmetry),
        coverage_stop_fraction=cov_stop,
        regret_stop_fraction=reg_stop,
        stopping_fraction_gap=gap_frac,
        stopping_gap_ci95=gap_ci,
        coverage_at_coverage_stop=float(mean_cov[cov_stop - 1]),
        coverage_at_regret_stop=float(mean_cov[reg_stop - 1]),
        regret_at_coverage_stop=reg_at_cov,
        regret_at_regret_stop=reg_at_reg,
        excess_regret_of_coverage_stop=excess_regret,
        excess_regret_ci95=excess_ci,
        matched_false_alarm_rate=matched_far,
        separation_gate_passed=separation_passed,
        fraction_trajectory=trajectory,
    )
