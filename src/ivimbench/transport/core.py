"""Domain-agnostic anomalous transport, stretched-exponential CRLB walls, and Procrustes misspecification probes.

Generalizes ``ivimbench.anomalous`` (*Levy*) and *Procrustes* for arbitrary wavevector,
Laplace, frequency, or relaxation time grids ``q >= 0`` without MRI-specific terminology:

.. math::

    S(q; S_0, K, \\alpha) = S_0 \\exp\\!\\left(-(q K)^\\alpha\\right), \\qquad \\alpha \\in (0, 1]

Applicable across anomalous subdiffusion, Continuous-Time Random Walks (CTRW),
Kohlrausch–Williams–Watts (KWW) dielectric/viscoelastic relaxation, dynamic light
scattering, and porous-media transport.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .._version import __version__
from ..anomalous.wall import (
    compute_anomalous_crlb,
    ctrw_stretched_signal,
    map_recovery_collapse_wall,
)
from ..metrics.coverage import conformal_offset, empirical_coverage
from ..phantoms.noise import inject_noise
from ..runner import _sanitize_json, dataframe_to_markdown

if TYPE_CHECKING:
    from ..audit import AuditReport

__all__ = [
    "DEFAULT_Q_GRID",
    "FractionalTransportCRLB",
    "TransportWallReport",
    "MisspecificationAliasingReport",
    "stretched_exponential_decay",
    "discrete_multiexponential_decay",
    "compute_fractional_transport_crlb",
    "map_anomalous_identifiability_wall",
    "probe_misspecification_conditional_aliasing",
    "run_transport_audit",
]

DEFAULT_Q_GRID: tuple[float, ...] = (
    0.0,
    10.0,
    20.0,
    30.0,
    50.0,
    80.0,
    120.0,
    200.0,
    350.0,
    500.0,
    700.0,
    1000.0,
)


def stretched_exponential_decay(
    q: np.ndarray | Sequence[float],
    K: np.ndarray | float,
    alpha: np.ndarray | float,
    S0: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Evaluate stretched-exponential / CTRW / KWW decay ``S(q; S0, K, alpha) = S0 * exp(-(q * K)**alpha)``."""
    q_arr = np.asarray(q, dtype=np.float64).ravel()
    if q_arr.size == 0 or np.any(~np.isfinite(q_arr)) or np.any(q_arr < 0.0):
        raise ValueError("`q` must be a non-empty array of finite non-negative coordinates.")
    return ctrw_stretched_signal(q_arr, D=K, alpha=alpha, S0=S0)


def discrete_multiexponential_decay(
    q: np.ndarray | Sequence[float],
    rates: np.ndarray | Sequence[float],
    weights: np.ndarray | Sequence[float],
    S0: float = 1.0,
) -> np.ndarray:
    """Evaluate discrete multi-exponential mixture ``S(q) = S0 * sum_k w_k * exp(-q * K_k)``."""
    q_arr = np.asarray(q, dtype=np.float64).ravel()
    if q_arr.size == 0 or np.any(~np.isfinite(q_arr)) or np.any(q_arr < 0.0):
        raise ValueError("`q` must be a non-empty array of finite non-negative coordinates.")
    r_arr = np.asarray(rates, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)
    if r_arr.shape != w_arr.shape or r_arr.ndim not in (1, 2):
        raise ValueError("`rates` and `weights` must share 1D `(C,)` or 2D `(N, C)` shape.")
    if np.any(r_arr < 0.0) or np.any(w_arr < 0.0):
        raise ValueError("`rates` and `weights` must be non-negative.")
    s0_val = float(S0)
    if s0_val <= 0.0:
        raise ValueError("`S0` must be strictly positive.")

    was_1d = r_arr.ndim == 1
    r_2d = np.atleast_2d(r_arr)
    w_2d = np.atleast_2d(w_arr)
    w_sum = np.sum(w_2d, axis=1, keepdims=True)
    if np.any(w_sum <= 0.0):
        raise ValueError("Mixture `weights` must have positive sum.")
    w_norm = w_2d / w_sum

    # (N, C, 1) * (1, 1, M) -> sum over C -> (N, M)
    kern = np.exp(-r_2d[:, :, None] * q_arr[None, None, :])
    sig = s0_val * np.sum(w_norm[:, :, None] * kern, axis=1)
    return sig[0] if was_1d else sig


@dataclass(frozen=True)
class FractionalTransportCRLB(Mapping[str, Any]):
    """Domain-agnostic Fisher Information & Cramér–Rao Lower Bound over ``(S0, K, alpha)``."""

    q: np.ndarray
    K: float
    alpha: float
    S0: float
    snr: float
    fisher: np.ndarray
    covariance: np.ndarray
    sigma_S0: float
    sigma_K: float
    sigma_alpha: float
    cv_K: float
    cv_alpha: float
    condition_number: float

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_json(
            {
                "param_names": ["S0", "K", "alpha"],
                "q": self.q.tolist(),
                "K": self.K,
                "alpha": self.alpha,
                "S0": self.S0,
                "snr": self.snr,
                "fisher": self.fisher.tolist(),
                "covariance": self.covariance.tolist(),
                "sigma": {
                    "S0": self.sigma_S0,
                    "K": self.sigma_K,
                    "alpha": self.sigma_alpha,
                },
                "sigma_S0": self.sigma_S0,
                "sigma_K": self.sigma_K,
                "sigma_alpha": self.sigma_alpha,
                "cv_K": self.cv_K,
                "cv_alpha": self.cv_alpha,
                "condition_number": self.condition_number,
            }
        )

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def compute_fractional_transport_crlb(
    q: np.ndarray | Sequence[float],
    K: float,
    alpha: float,
    snr: float,
    S0: float = 1.0,
) -> FractionalTransportCRLB:
    """Compute exact 3x3 Fisher CRLB for ``S(q; S0, K, alpha) = S0 * exp(-(q * K)**alpha)``."""
    q_arr = np.asarray(q, dtype=np.float64).ravel()
    if q_arr.size < 3 or np.any(q_arr < 0.0):
        raise ValueError("`q` must contain at least 3 non-negative coordinates.")
    res = compute_anomalous_crlb(q_arr, D=float(K), alpha=float(alpha), snr=float(snr), S0=float(S0))
    sig_vec = np.asarray(res.std, dtype=np.float64).ravel()
    cov_mat = np.asarray(res.cov, dtype=np.float64)
    fim_mat = np.asarray(res.fim, dtype=np.float64)
    return FractionalTransportCRLB(
        q=q_arr,
        K=float(K),
        alpha=float(alpha),
        S0=float(S0),
        snr=float(snr),
        fisher=fim_mat,
        covariance=cov_mat,
        sigma_S0=float(sig_vec[0]),
        sigma_K=float(sig_vec[1]),
        sigma_alpha=float(sig_vec[2]),
        cv_K=float(np.asarray(res.D_cv).ravel()[0]),
        cv_alpha=float(np.asarray(res.alpha_cv).ravel()[0]),
        condition_number=float(np.asarray(res.condition_number).ravel()[0]),
    )


@dataclass
class TransportWallReport:
    """Domain-agnostic joint ``(K, alpha)`` identifiability & NLLS collapse wall report."""

    dataframe: pd.DataFrame
    wall_front: pd.DataFrame
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_json(
            {
                "summary": self.summary,
                "wall_front": self.wall_front.to_dict(orient="records"),
                "records": self.dataframe.to_dict(orient="records"),
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self, float_fmt: str = ".3f") -> str:
        cols = [
            c
            for c in (
                "schedule",
                "q_max",
                "snr",
                "alpha_true",
                "crlb_cv_K",
                "crlb_cv_alpha",
                "railed_alpha",
                "collapsed",
            )
            if c in self.dataframe.columns
        ]
        return dataframe_to_markdown(self.dataframe, columns=cols, float_fmt=float_fmt)


def _normalize_q_schedules(
    q_schedules: np.ndarray | Sequence[Any] | Mapping[str, Sequence[float] | np.ndarray],
) -> dict[str, np.ndarray] | list[Any]:
    if isinstance(q_schedules, Mapping):
        return {str(k): np.asarray(v, dtype=np.float64).ravel() for k, v in q_schedules.items()}
    if isinstance(q_schedules, np.ndarray):
        if q_schedules.ndim == 1:
            return {"q_grid": np.asarray(q_schedules, dtype=np.float64).ravel()}
        return {
            f"q_grid_{i}": np.asarray(q_schedules[i], dtype=np.float64).ravel()
            for i in range(q_schedules.shape[0])
        }
    seq = list(q_schedules)
    if seq and isinstance(seq[0], (int, float, np.floating, np.integer)):
        return {"q_grid": np.asarray(seq, dtype=np.float64).ravel()}
    out_map: dict[str, np.ndarray] = {}
    all_arrays = True
    for idx, item in enumerate(seq):
        if isinstance(item, str):
            all_arrays = False
            break
        out_map[f"schedule_{idx}"] = np.asarray(item, dtype=np.float64).ravel()
    return out_map if all_arrays else seq


def map_anomalous_identifiability_wall(
    q_schedules: np.ndarray | Sequence[Any] | Mapping[str, Sequence[float] | np.ndarray] = DEFAULT_Q_GRID,
    snr_grid: Sequence[float] | np.ndarray = (15.0, 30.0, 60.0),
    alpha_grid: Sequence[float] | np.ndarray = (0.60, 0.80, 1.0),
    seed: int = 42,
    *,
    K_true: float = 1.0e-3,
    S0: float = 1.0,
    n_replicates: int = 32,
    noise_type: str = "rician",
    n_boot: int = 200,
) -> TransportWallReport:
    """Map the joint ``(K, alpha)`` CRLB divergence and bounded-NLLS collapse frontier across ``q`` schedules."""
    norm_sched = _normalize_q_schedules(q_schedules)
    raw = map_recovery_collapse_wall(
        b_schedules=norm_sched,
        snr_grid=snr_grid,
        alpha_grid=alpha_grid,
        D_true=float(K_true),
        S0=float(S0),
        n_replicates=int(n_replicates),
        noise_type=noise_type,
        n_boot=int(n_boot),
        seed=int(seed),
    )
    df = raw.dataframe.copy()
    rename_pairs = {
        "b_max": "q_max",
        "D_true": "K_true",
        "crlb_cv_D": "crlb_cv_K",
        "bias_D": "bias_K",
        "rmse_D": "rmse_K",
        "railed_D": "railed_K",
    }
    for src_col, dst_col in rename_pairs.items():
        if src_col in df.columns:
            df[dst_col] = df[src_col]

    wf = raw.wall_front.copy()
    if "b_max" in wf.columns:
        wf["q_max"] = wf["b_max"]

    summary = {
        "n_cells": int(len(df)),
        "collapsed_fraction": float(df["collapsed"].mean()) if "collapsed" in df.columns else 0.0,
        "max_crlb_cv_alpha": float(df["crlb_cv_alpha"].max()) if "crlb_cv_alpha" in df.columns else 0.0,
        "min_crlb_cv_alpha": float(df["crlb_cv_alpha"].min()) if "crlb_cv_alpha" in df.columns else 0.0,
        "seed": int(seed),
    }
    return TransportWallReport(dataframe=df, wall_front=wf, summary=summary)


@dataclass(frozen=True)
class MisspecificationAliasingReport(Mapping[str, Any]):
    """Procrustes conditional-coverage aliasing report when fitting a 2-rate mixture to non-2-rate transport."""

    family: str
    snr: float
    alpha: float
    n_samples: int
    seed: int
    global_marginal_coverage: float
    nominal_target_coverage: float
    tercile_coverages: dict[str, float]
    conditional_coverage_aliasing_gap: float
    high_minus_low_tercile_gap: float
    mean_slow_rate_bias: float
    aliasing_detected: bool

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_json(
            {
                "family": self.family,
                "snr": self.snr,
                "alpha": self.alpha,
                "n_samples": self.n_samples,
                "seed": self.seed,
                "global_marginal_coverage": self.global_marginal_coverage,
                "nominal_target_coverage": self.nominal_target_coverage,
                "tercile_coverages": dict(self.tercile_coverages),
                "conditional_coverage_aliasing_gap": self.conditional_coverage_aliasing_gap,
                "high_minus_low_tercile_gap": self.high_minus_low_tercile_gap,
                "mean_slow_rate_bias": self.mean_slow_rate_bias,
                "aliasing_detected": self.aliasing_detected,
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def _simulate_transport_family(
    q_arr: np.ndarray,
    k_slow: np.ndarray,
    k_fast: np.ndarray,
    w_fast: np.ndarray,
    family: str,
) -> np.ndarray:
    fam = str(family).strip().lower()
    q_row = q_arr[None, :]
    ks = k_slow[:, None]
    kf = k_fast[:, None]
    wf = w_fast[:, None]

    if fam in ("two_rate", "biexp"):
        return wf * np.exp(-q_row * kf) + (1.0 - wf) * np.exp(-q_row * ks)
    if fam == "dispersion_gamma":
        # Gamma distribution of fast rates + dispersed slow rate curvature (shape kappa = 1.6)
        kappa_f = 1.5
        kappa_s = 3.2
        fast_term = np.power(1.0 + (q_row * kf) / kappa_f, -kappa_f)
        slow_term = np.power(1.0 + (q_row * ks * (1.0 + 0.55 * wf)) / kappa_s, -kappa_s)
        return wf * fast_term + (1.0 - wf) * slow_term
    if fam == "dispersion_lognormal":
        z_nodes = np.array([-1.7320508, -0.7419638, 0.0, 0.7419638, 1.7320508], dtype=np.float64)
        z_weights = np.array([0.05, 0.24, 0.42, 0.24, 0.05], dtype=np.float64)
        sig_ln = 0.65
        k_med = kf * np.exp(-0.5 * sig_ln**2)
        rates_grid = k_med[:, :, None] * np.exp(sig_ln * z_nodes[None, None, :])
        fast_term = np.sum(
            z_weights[None, None, :] * np.exp(-q_row[:, :, None] * rates_grid), axis=2
        )
        slow_shift = ks * (1.0 + 0.45 * wf)
        return wf * fast_term + (1.0 - wf) * np.exp(-q_row * slow_shift)
    if fam == "stretched":
        alpha_s = np.clip(0.88 - 0.40 * wf, 0.45, 0.95)
        slow_arg = np.where(
            q_row * ks > 0.0, np.power(np.maximum(q_row * ks, 1.0e-300), alpha_s), 0.0
        )
        return wf * np.exp(-q_row * kf) + (1.0 - wf) * np.exp(-slow_arg)
    if fam == "triexp":
        k_inter = 3.6 * ks
        w_inter = 0.42 * wf
        w_f_eff = 0.58 * wf
        w_s_eff = np.maximum(1.0 - w_f_eff - w_inter, 0.05)
        return (
            w_f_eff * np.exp(-q_row * kf)
            + w_inter * np.exp(-q_row * k_inter)
            + w_s_eff * np.exp(-q_row * ks)
        )
    raise ValueError(
        f"Unsupported transport family {family!r}; expected one of "
        "('dispersion_gamma', 'dispersion_lognormal', 'stretched', 'triexp', 'two_rate')."
    )


def _fit_two_rate_slow_parameter(q_arr: np.ndarray, signals: np.ndarray) -> np.ndarray:
    """Fast high-q log-linear + bounded 2-rate refinement for slow transport rate ``K_slow``."""
    q_med = float(np.quantile(q_arr[q_arr > 0.0], 0.45)) if np.any(q_arr > 0.0) else 0.0
    hi_mask = q_arr >= q_med
    if int(np.sum(hi_mask)) < 2:
        hi_mask = np.ones_like(q_arr, dtype=bool)
    q_hi = q_arr[hi_mask]
    q_mean = float(np.mean(q_hi))
    q_var = max(float(np.sum((q_hi - q_mean) ** 2)), 1.0e-12)

    y_clip = np.clip(signals[:, hi_mask], 1.0e-6, None)
    log_y = np.log(y_clip)
    slope = np.sum((q_hi[None, :] - q_mean) * (log_y - np.mean(log_y, axis=1, keepdims=True)), axis=1) / q_var
    k_slow_seg = np.clip(-slope, 5.0e-5, 5.0e-3)

    # Light 2-rate Gauss-Newton / TRF polish on up to all samples (fast 35 nfev)
    lb = np.array([5.0e-5, 4.0e-3, 0.0, 0.2], dtype=np.float64)
    ub = np.array([5.0e-3, 1.2e-1, 0.75, 2.0], dtype=np.float64)
    k_hat = np.empty_like(k_slow_seg)
    for i in range(signals.shape[0]):
        y = signals[i]
        x0 = np.clip(
            np.array([k_slow_seg[i], 20.0e-3, 0.18, max(float(y[0]), 0.5)], dtype=np.float64),
            lb + 1.0e-6,
            ub - 1.0e-6,
        )

        def _res(th: np.ndarray, target: np.ndarray = y) -> np.ndarray:
            pred = th[3] * (th[2] * np.exp(-q_arr * th[1]) + (1.0 - th[2]) * np.exp(-q_arr * th[0]))
            return pred - target

        opt = least_squares(_res, x0=x0, bounds=(lb, ub), max_nfev=35)
        k_hat[i] = opt.x[0]
    return k_hat


def probe_misspecification_conditional_aliasing(
    q: np.ndarray | Sequence[float] = DEFAULT_Q_GRID,
    family: str = "dispersion_gamma",
    snr: float = 40.0,
    seed: int = 42,
    *,
    n_samples: int = 480,
    alpha: float = 0.10,
) -> MisspecificationAliasingReport:
    """Measure Procrustes conditional coverage aliasing on slow transport rate ``K_slow``.

    Demonstrates that when a discrete 2-rate exponential model is fit to continuous-spectrum
    or heavy-tailed decay (``dispersion_gamma``, ``dispersion_lognormal``, ``stretched``,
    ``triexp``) and split-conformalized globally, marginal coverage remains valid near
    ``1 - alpha`` while conditional coverage aliases strongly across fast-fraction terciles.
    """
    q_arr = np.asarray(q, dtype=np.float64).ravel()
    if q_arr.size < 4 or np.any(q_arr < 0.0):
        raise ValueError("`q` must have at least 4 non-negative points.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"`alpha` must lie in (0, 1); got {alpha}.")

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    n_tot = max(int(n_samples), 90)
    n_cal = n_tot // 2
    n_test = n_tot - n_cal

    k_slow = rng.uniform(0.7e-3, 1.8e-3, size=n_tot)
    k_fast = rng.uniform(10.0e-3, 45.0e-3, size=n_tot)
    w_fast = rng.uniform(0.03, 0.42, size=n_tot)

    clean = _simulate_transport_family(q_arr, k_slow, k_fast, w_fast, family=family)
    noisy = inject_noise(clean, snr=float(snr), noise_type="rician", seed=rng, s0=1.0)

    k_hat = _fit_two_rate_slow_parameter(q_arr, noisy)
    cal_scores = np.abs(k_slow[:n_cal] - k_hat[:n_cal])
    q_conf = conformal_offset(cal_scores, alpha=float(alpha))

    lo_test = k_hat[n_cal:] - q_conf
    hi_test = k_hat[n_cal:] + q_conf
    k_true_test = k_slow[n_cal:]
    w_test = w_fast[n_cal:]

    global_cov = empirical_coverage(k_true_test, lo_test, hi_test, alpha=float(alpha))
    t1, t2 = np.quantile(w_test, [1.0 / 3.0, 2.0 / 3.0])
    low_m = w_test <= t1
    mid_m = (w_test > t1) & (w_test <= t2)
    high_m = w_test > t2

    cov_low = empirical_coverage(k_true_test[low_m], lo_test[low_m], hi_test[low_m], alpha=float(alpha))
    cov_mid = empirical_coverage(k_true_test[mid_m], lo_test[mid_m], hi_test[mid_m], alpha=float(alpha))
    cov_high = empirical_coverage(k_true_test[high_m], lo_test[high_m], hi_test[high_m], alpha=float(alpha))

    terciles = {
        "low_fast_weight": float(cov_low),
        "mid_fast_weight": float(cov_mid),
        "high_fast_weight": float(cov_high),
    }
    alias_gap = float(max(terciles.values()) - min(terciles.values()))
    hi_minus_lo = float(cov_high - cov_low)
    mean_bias = float(np.mean(k_hat[n_cal:] - k_true_test))
    fam_clean = str(family).strip().lower()
    aliasing_detected = alias_gap >= 0.06 if fam_clean not in ("two_rate", "biexp") else alias_gap >= 0.12

    return MisspecificationAliasingReport(
        family=fam_clean,
        snr=float(snr),
        alpha=float(alpha),
        n_samples=int(n_test),
        seed=int(seed),
        global_marginal_coverage=float(global_cov),
        nominal_target_coverage=1.0 - float(alpha),
        tercile_coverages=terciles,
        conditional_coverage_aliasing_gap=alias_gap,
        high_minus_low_tercile_gap=hi_minus_lo,
        mean_slow_rate_bias=mean_bias,
        aliasing_detected=bool(aliasing_detected),
    )


def run_transport_audit(seed: int = 42) -> AuditReport:
    """Run deterministic clean-room audit for ``ivimbench.transport`` (``levy-wall``)."""
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
    q_grid = np.asarray(DEFAULT_Q_GRID, dtype=np.float64)

    # 1. Continuity limit: alpha = 1.0 matches exact mono-exponential exp(-q * K)
    s_stretched = stretched_exponential_decay(q_grid, K=1.2e-3, alpha=1.0, S0=1.0)
    s_mono = np.exp(-q_grid * 1.2e-3)
    mono_residual = float(np.max(np.abs(s_stretched - s_mono)))

    # 2. CRLB monotonicity with SNR: CV(alpha) at SNR=15 > CV(alpha) at SNR=60
    crlb_low = compute_fractional_transport_crlb(q_grid, K=1.1e-3, alpha=0.75, snr=15.0)
    crlb_high = compute_fractional_transport_crlb(q_grid, K=1.1e-3, alpha=0.75, snr=60.0)
    crlb_monotonic = bool(crlb_low.cv_alpha > crlb_high.cv_alpha > 0.0)

    # 3. Procrustes misspecification check: valid marginal coverage + conditional aliasing
    proc = probe_misspecification_conditional_aliasing(
        q_grid, family="dispersion_gamma", snr=40.0, seed=seed_int, n_samples=360
    )
    marginal_ok = abs(proc.global_marginal_coverage - 0.90) <= 0.07
    aliasing_ok = proc.conditional_coverage_aliasing_gap >= 0.05

    if mono_residual <= 1.0e-14 and crlb_monotonic and marginal_ok and aliasing_ok:
        status: StatusLiteral = "PASS"
        verdict = (
            f"LEVY_PROCRUSTES_VERIFIED: Mono-exp limit residual={mono_residual:.2e}, "
            f"CRLB CV(alpha) {crlb_low.cv_alpha:.3f}->{crlb_high.cv_alpha:.3f}, "
            f"marginal cov={proc.global_marginal_coverage:.3f}, "
            f"tercile aliasing gap={proc.conditional_coverage_aliasing_gap:.3f}."
        )
    elif mono_residual <= 1.0e-12 and crlb_monotonic:
        status = "WARN"
        verdict = "LEVY_PARTIAL: Stretched-exp CRLB verified; Procrustes gap near threshold."
    else:
        status = "HALT"
        verdict = "LEVY_FAILURE: Transport continuity or Fisher CRLB monotonicity violated."

    stage_role = "Domain-Agnostic Anomalous Transport CRLB Wall & Procrustes Aliasing Gate"
    metrics: dict[str, float | int | str | bool] = {
        "monoexp_continuity_residual": mono_residual,
        "crlb_cv_alpha_snr15": round(crlb_low.cv_alpha, 6),
        "crlb_cv_alpha_snr60": round(crlb_high.cv_alpha, 6),
        "crlb_snr_monotonic": crlb_monotonic,
        "procrustes_marginal_coverage": round(proc.global_marginal_coverage, 6),
        "procrustes_aliasing_gap": round(proc.conditional_coverage_aliasing_gap, 6),
    }
    thresholds: dict[str, float | int | str] = {
        "max_monoexp_residual": 1.0e-14,
        "max_marginal_coverage_abs_error": 0.07,
        "min_procrustes_aliasing_gap": 0.05,
    }
    att = AuditAttestation(
        input_sha256=compute_input_sha256(s_stretched),
        report_sha256=compute_report_sha256(
            package="levy-wall",
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
        package="levy-wall",
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
