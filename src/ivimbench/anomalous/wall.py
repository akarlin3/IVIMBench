"""Anomalous fractional-order diffusion forward models, Fisher CRLBs, and recovery collapse walls.

Implements:
1. Continuous-Time Random Walk (CTRW) / fractional Bloch–Torrey stretched-exponential
   forward physics ``S(b; S0, D, alpha) = S0 * exp(-(b * D)**alpha)`` for
   ``alpha in (0, 1]`` and its two-compartment fractional IVIM extension, paired
   with exact Rician and multi-coil non-central chi magnitude noise (:mod:`ivimbench.phantoms.noise`).
2. Analytical ``3 x 3`` Fisher Information Matrix and Cramér–Rao Lower Bounds
   (:func:`compute_anomalous_crlb`) over ``(S0, D, alpha)`` with closed-form
   partial derivatives ``dS/dalpha = -S(b) * (b * D)**alpha * ln(b * D)``.
3. Joint ``(D, alpha)`` identifiability & recovery collapse wall mapper
   (:func:`map_recovery_collapse_wall`) across acquisition ``b``-schedules,
   ``b_max``, and SNR regimes, reporting relative CRLB CVs, bounded-NLLS bias and
   boundary-railing rates, and ``PCG64``-seeded 95% bootstrap confidence intervals.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..acquisition.crlb import resolve_schedule_and_nex
from ..metrics.railing import DEFAULT_RAIL_TOL, bootstrap_railing_ci, classify_railing
from ..phantoms.noise import inject_noise
from ..protocols import AcquisitionProtocol, resolve_bvalues
from ..runner import _sanitize_json, dataframe_to_latex, dataframe_to_markdown

__all__ = [
    "ANOMALOUS_PARAM_NAMES",
    "DEFAULT_ANOMALOUS_BOUNDS",
    "AnomalousCRLBResult",
    "AnomalousNLLSFitResult",
    "RecoveryWallReport",
    "ctrw_stretched_signal",
    "anomalous_stretched_signal",
    "fractional_ivim_signal",
    "simulate_anomalous_signals",
    "anomalous_signal_jacobian",
    "anomalous_fisher_matrix",
    "compute_anomalous_crlb",
    "fit_anomalous_nlls",
    "map_recovery_collapse_wall",
]

ANOMALOUS_PARAM_NAMES: tuple[str, str, str] = ("S0", "D", "alpha")

DEFAULT_ANOMALOUS_BOUNDS: dict[str, tuple[float, float]] = {
    "D": (1.0e-4, 4.0e-3),
    "alpha": (0.10, 1.0),
}


def _validate_alpha_domain(alpha_arr: np.ndarray, name: str = "alpha") -> None:
    if np.any(~np.isfinite(alpha_arr)) or np.any((alpha_arr <= 0.0) | (alpha_arr > 1.0)):
        raise ValueError(f"`{name}` must lie in (0, 1]; got range [{np.min(alpha_arr)}, {np.max(alpha_arr)}].")


def _validate_positive(arr: np.ndarray, name: str) -> None:
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"`{name}` must be strictly positive; got min={np.min(arr)}.")


def ctrw_stretched_signal(
    bvalues: np.ndarray | Sequence[float] | str | AcquisitionProtocol,
    D: np.ndarray | float,
    alpha: np.ndarray | float,
    S0: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Evaluate the CTRW / fractional Bloch–Torrey stretched-exponential forward model.

    .. math::

        S(b; S_0, D, \\alpha) = S_0 \\exp\\!\\left(-(b D)^\\alpha\\right),
        \\qquad \\alpha \\in (0, 1]

    At ``alpha = 1.0``, reduces identically to mono-exponential Gaussian diffusion
    ``S0 * exp(-b * D)``.

    Parameters
    ----------
    bvalues : 1D array-like, protocol key, or AcquisitionProtocol
        Diffusion weighting ``b`` in ``s/mm^2`` (must be ``>= 0``).
    D : float or array-like
        Effective anomalous diffusion coefficient in ``mm^2/s`` (``> 0``).
    alpha : float or array-like
        Fractional-order anomalous stretching exponent in ``(0, 1]``.
    S0 : float or array-like, default=1.0
        Baseline ``b = 0`` signal intensity (``> 0``).

    Returns
    -------
    np.ndarray
        Clean magnitude signal of shape ``(n_b,)`` for scalar inputs or
        ``(..., n_b)`` for broadcast array inputs.
    """
    b_arr, _ = resolve_bvalues(bvalues) if isinstance(bvalues, (str, AcquisitionProtocol)) else (
        np.asarray(bvalues, dtype=np.float64).ravel(),
        "custom",
    )
    if np.any(b_arr < 0.0):
        raise ValueError("`bvalues` must be non-negative.")

    d_raw = np.asarray(D, dtype=np.float64)
    a_raw = np.asarray(alpha, dtype=np.float64)
    s0_raw = np.asarray(S0, dtype=np.float64)
    _validate_positive(d_raw, "D")
    _validate_alpha_domain(a_raw, "alpha")
    _validate_positive(s0_raw, "S0")

    d_b, a_b, s0_b = np.broadcast_arrays(d_raw, a_raw, s0_raw)
    scalar = d_b.ndim == 0
    lead_shape = d_b.shape

    d_col = np.atleast_1d(d_b).ravel()[:, None]
    a_col = np.atleast_1d(a_b).ravel()[:, None]
    s0_col = np.atleast_1d(s0_b).ravel()[:, None]

    bd = d_col * b_arr[None, :]
    exponent = np.where(
        bd > 0.0,
        np.where(a_col == 1.0, bd, np.power(np.maximum(bd, 1.0e-300), a_col)),
        0.0,
    )
    sig = s0_col * np.exp(-exponent)
    return sig[0] if scalar else sig.reshape(lead_shape + (b_arr.size,))


anomalous_stretched_signal = ctrw_stretched_signal


def fractional_ivim_signal(
    bvalues: np.ndarray | Sequence[float] | str | AcquisitionProtocol,
    D: np.ndarray | float,
    Dstar: np.ndarray | float,
    f: np.ndarray | float,
    alpha: np.ndarray | float,
    *,
    beta: np.ndarray | float = 1.0,
    S0: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Two-compartment fractional IVIM forward model for ``alpha, beta in (0, 1]``.

    .. math::

        S(b) = S_0 \\left[ f \\exp\\!\\left(-(b D^*)^\\beta\\right)
        + (1 - f) \\exp\\!\\left(-(b D)^\\alpha\\right) \\right]

    Reduces continuously to canonical bi-exponential IVIM at ``alpha = beta = 1.0``
    and to single-compartment CTRW stretched-exponential decay at ``f = 0.0``.
    """
    b_arr, _ = resolve_bvalues(bvalues) if isinstance(bvalues, (str, AcquisitionProtocol)) else (
        np.asarray(bvalues, dtype=np.float64).ravel(),
        "custom",
    )
    if np.any(b_arr < 0.0):
        raise ValueError("`bvalues` must be non-negative.")

    d_raw = np.asarray(D, dtype=np.float64)
    ds_raw = np.asarray(Dstar, dtype=np.float64)
    f_raw = np.asarray(f, dtype=np.float64)
    a_raw = np.asarray(alpha, dtype=np.float64)
    be_raw = np.asarray(beta, dtype=np.float64)
    s0_raw = np.asarray(S0, dtype=np.float64)

    _validate_positive(d_raw, "D")
    _validate_positive(ds_raw, "Dstar")
    _validate_alpha_domain(a_raw, "alpha")
    _validate_alpha_domain(be_raw, "beta")
    _validate_positive(s0_raw, "S0")
    if np.any((f_raw < 0.0) | (f_raw > 1.0)):
        raise ValueError("`f` must lie in [0, 1].")

    d_b, ds_b, f_b, a_b, be_b, s0_b = np.broadcast_arrays(
        d_raw, ds_raw, f_raw, a_raw, be_raw, s0_raw
    )
    scalar = d_b.ndim == 0
    lead_shape = d_b.shape

    d_col = np.atleast_1d(d_b).ravel()[:, None]
    ds_col = np.atleast_1d(ds_b).ravel()[:, None]
    f_col = np.atleast_1d(f_b).ravel()[:, None]
    a_col = np.atleast_1d(a_b).ravel()[:, None]
    be_col = np.atleast_1d(be_b).ravel()[:, None]
    s0_col = np.atleast_1d(s0_b).ravel()[:, None]

    bd = d_col * b_arr[None, :]
    bds = ds_col * b_arr[None, :]

    tissue_arg = np.where(
        bd > 0.0,
        np.where(a_col == 1.0, bd, np.power(np.maximum(bd, 1.0e-300), a_col)),
        0.0,
    )
    perf_arg = np.where(
        bds > 0.0,
        np.where(be_col == 1.0, bds, np.power(np.maximum(bds, 1.0e-300), be_col)),
        0.0,
    )
    sig = s0_col * (f_col * np.exp(-perf_arg) + (1.0 - f_col) * np.exp(-tissue_arg))
    return sig[0] if scalar else sig.reshape(lead_shape + (b_arr.size,))


def simulate_anomalous_signals(
    bvalues: np.ndarray | Sequence[float] | str | AcquisitionProtocol,
    D: np.ndarray | float,
    alpha: np.ndarray | float,
    *,
    snr: float | np.ndarray | None = 30.0,
    S0: np.ndarray | float = 1.0,
    Dstar: np.ndarray | float | None = None,
    f: np.ndarray | float | None = None,
    beta: np.ndarray | float = 1.0,
    noise_type: str = "rician",
    n_coils: int = 4,
    seed: int | np.random.Generator = 42,
) -> np.ndarray:
    """Simulate CTRW or two-compartment fractional IVIM signals under exact Rician or nc-chi noise.

    Uses ``numpy.random.PCG64`` when ``seed`` is an integer to guarantee deterministic,
    100% synthetic magnitude acquisitions.
    """
    if (Dstar is None) != (f is None):
        raise ValueError("Provide both `Dstar` and `f` for two-compartment fractional IVIM, or neither.")
    if Dstar is not None and f is not None:
        clean = fractional_ivim_signal(
            bvalues, D=D, Dstar=Dstar, f=f, alpha=alpha, beta=beta, S0=S0
        )
    else:
        clean = ctrw_stretched_signal(bvalues, D=D, alpha=alpha, S0=S0)

    rng = seed if isinstance(seed, np.random.Generator) else np.random.Generator(np.random.PCG64(int(seed)))
    return inject_noise(
        clean,
        snr=snr,
        noise_type=noise_type,
        n_coils=int(n_coils),
        seed=rng,
        s0=S0,
    )


def anomalous_signal_jacobian(
    bvalues: np.ndarray | Sequence[float] | str | AcquisitionProtocol,
    D: np.ndarray | float,
    alpha: np.ndarray | float,
    S0: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Analytical Jacobian ``dS / d(S0, D, alpha)`` of ``S(b) = S0 * exp(-(b * D)**alpha)``.

    Closed-form partial derivatives:
    - ``dS / dS0    = exp(-(b * D)**alpha)``
    - ``dS / dD     = -S(b) * alpha * (b * D)**alpha / D``  (``0`` at ``b = 0``)
    - ``dS / dalpha = -S(b) * (b * D)**alpha * ln(b * D)``  (``0`` at ``b = 0``)

    Returns
    -------
    np.ndarray
        Jacobian array of shape ``(n_b, 3)`` for scalar parameters or ``(..., n_b, 3)``
        for vectorized inputs, ordered ``("S0", "D", "alpha")``.
    """
    b_arr, _ = resolve_bvalues(bvalues) if isinstance(bvalues, (str, AcquisitionProtocol)) else (
        np.asarray(bvalues, dtype=np.float64).ravel(),
        "custom",
    )
    if np.any(b_arr < 0.0):
        raise ValueError("`bvalues` must be non-negative.")

    d_raw = np.asarray(D, dtype=np.float64)
    a_raw = np.asarray(alpha, dtype=np.float64)
    s0_raw = np.asarray(S0, dtype=np.float64)
    _validate_positive(d_raw, "D")
    _validate_alpha_domain(a_raw, "alpha")
    _validate_positive(s0_raw, "S0")

    d_b, a_b, s0_b = np.broadcast_arrays(d_raw, a_raw, s0_raw)
    scalar = d_b.ndim == 0
    lead_shape = d_b.shape

    d_exp = np.atleast_1d(d_b).ravel()[:, None]
    a_exp = np.atleast_1d(a_b).ravel()[:, None]
    s0_exp = np.atleast_1d(s0_b).ravel()[:, None]
    bd = d_exp * b_arr[None, :]
    pos = bd > 0.0
    safe_bd = np.where(pos, bd, 1.0)
    bd_pow = np.where(pos, np.where(a_exp == 1.0, safe_bd, np.power(safe_bd, a_exp)), 0.0)
    exp_term = np.exp(-bd_pow)
    sig = s0_exp * exp_term

    j_s0 = exp_term
    j_d = np.where(pos, -sig * a_exp * bd_pow / d_exp, 0.0)
    j_alpha = np.where(pos, -sig * bd_pow * np.log(safe_bd), 0.0)

    jac = np.stack([j_s0, j_d, j_alpha], axis=-1)
    return jac[0] if scalar else jac.reshape(lead_shape + (b_arr.size, 3))


def anomalous_fisher_matrix(
    bvalues: np.ndarray | Sequence[float] | str | AcquisitionProtocol,
    D: np.ndarray | float,
    alpha: np.ndarray | float,
    snr: float | np.ndarray,
    S0: np.ndarray | float = 1.0,
    *,
    nex: Sequence[float] | np.ndarray | float | None = None,
) -> np.ndarray:
    """Analytical ``3 x 3`` Fisher Information Matrix over ``theta = (S0, D, alpha)``.

    .. math::

        I(\\theta) = \\frac{1}{\\sigma_0^2} \\sum_{k=1}^{N_b} n_k \\,
        \\nabla_\\theta S(b_k; \\theta) \\nabla_\\theta S(b_k; \\theta)^\\top,
        \\qquad \\sigma_0 = \\frac{S_0}{\\mathrm{SNR}}
    """
    b_arr, w_nex, _, _ = resolve_schedule_and_nex(bvalues, nex=nex)
    d_raw = np.asarray(D, dtype=np.float64)
    a_raw = np.asarray(alpha, dtype=np.float64)
    s0_raw = np.asarray(S0, dtype=np.float64)
    snr_raw = np.asarray(snr, dtype=np.float64)
    _validate_positive(snr_raw, "snr")

    d_b, a_b, s0_b, snr_b = np.broadcast_arrays(d_raw, a_raw, s0_raw, snr_raw)
    scalar = d_b.ndim == 0

    d_1d = np.atleast_1d(d_b).ravel()
    a_1d = np.atleast_1d(a_b).ravel()
    s0_1d = np.atleast_1d(s0_b).ravel()
    snr_1d = np.atleast_1d(snr_b).ravel()

    jac = anomalous_signal_jacobian(b_arr, d_1d, a_1d, S0=s0_1d)
    sigma0 = s0_1d / snr_1d
    jac_w = jac * w_nex[None, :, None]
    fim = np.einsum("nbi,nbj->nij", jac_w, jac) / (sigma0**2)[:, None, None]
    fim = 0.5 * (fim + np.swapaxes(fim, -1, -2))
    if scalar:
        return fim[0]
    return fim.reshape(d_b.shape + (3, 3))


@dataclass(frozen=True)
class AnomalousCRLBResult:
    """Analytical ``3 x 3`` Fisher Information and CRLB container over ``(S0, D, alpha)``."""

    fim: np.ndarray
    cov: np.ndarray
    std: np.ndarray
    cv: np.ndarray
    d_alpha_corr: np.ndarray | float
    condition_number: np.ndarray | float
    identifiable_mask: np.ndarray | bool
    jacobian: np.ndarray
    bvalues: np.ndarray
    nex: np.ndarray
    param_order: tuple[str, str, str] = ANOMALOUS_PARAM_NAMES

    @property
    def S0_std(self) -> np.ndarray | float:
        return float(self.std[0]) if self.std.ndim == 1 else self.std[..., 0]

    @property
    def D_std(self) -> np.ndarray | float:
        return float(self.std[1]) if self.std.ndim == 1 else self.std[..., 1]

    @property
    def alpha_std(self) -> np.ndarray | float:
        return float(self.std[2]) if self.std.ndim == 1 else self.std[..., 2]

    @property
    def S0_cv(self) -> np.ndarray | float:
        return float(self.cv[0]) if self.cv.ndim == 1 else self.cv[..., 0]

    @property
    def D_cv(self) -> np.ndarray | float:
        return float(self.cv[1]) if self.cv.ndim == 1 else self.cv[..., 1]

    @property
    def alpha_cv(self) -> np.ndarray | float:
        return float(self.cv[2]) if self.cv.ndim == 1 else self.cv[..., 2]

    def to_dict(self) -> dict[str, Any]:
        std_2d = np.atleast_2d(self.std)
        cv_2d = np.atleast_2d(self.cv)
        corr_1d = np.atleast_1d(np.asarray(self.d_alpha_corr, dtype=np.float64)).ravel()
        id_1d = np.atleast_1d(np.asarray(self.identifiable_mask, dtype=bool)).ravel()
        out: dict[str, Any] = {
            "param_order": list(self.param_order),
            "b_max": float(np.max(self.bvalues)),
            "n_bvalues": int(self.bvalues.size),
            "identifiable_fraction": float(np.mean(id_1d)),
            "median_d_alpha_corr": float(np.nanmedian(corr_1d)),
            "parameters": {},
        }
        for j, name in enumerate(self.param_order):
            fin = np.isfinite(std_2d[:, j])
            out["parameters"][name] = {
                "median_std": float(np.median(std_2d[fin, j])) if np.any(fin) else float("inf"),
                "median_cv": float(np.median(cv_2d[fin, j])) if np.any(fin) else float("inf"),
            }
        return out


def compute_anomalous_crlb(
    bvalues: np.ndarray | Sequence[float] | str | AcquisitionProtocol,
    D: np.ndarray | float,
    alpha: np.ndarray | float,
    snr: float | np.ndarray,
    S0: np.ndarray | float = 1.0,
    *,
    nex: Sequence[float] | np.ndarray | float | None = None,
    max_cond: float = 1.0e14,
) -> AnomalousCRLBResult:
    """Derive and evaluate the analytical ``3 x 3`` FIM and CRLBs over ``(S0, D, alpha)``.

    Parameters
    ----------
    bvalues : sequence of float, protocol key, or AcquisitionProtocol
        Diffusion ``b``-value schedule in ``s/mm^2`` (requires at least 3 distinct
        ``b``-values for joint ``(S0, D, alpha)`` identifiability).
    D : float or array-like
        Anomalous diffusion coefficient ``D`` in ``mm^2/s`` (``> 0``).
    alpha : float or array-like
        Fractional-order stretching exponent ``alpha`` in ``(0, 1]``.
    snr : float or array-like
        Nominal ``b = 0`` signal-to-noise ratio (``> 0``).
    S0 : float or array-like, default=1.0
        Baseline ``b = 0`` signal amplitude (``> 0``).
    nex : float, sequence of float, or None, optional
        Optional per-``b`` signal averages (NEX).
    max_cond : float, default=1e14
        Maximum condition number of the dimensionless parameter-scaled FIM.
    """
    b_arr, w_nex, _, _ = resolve_schedule_and_nex(bvalues, nex=nex)
    d_raw = np.asarray(D, dtype=np.float64)
    a_raw = np.asarray(alpha, dtype=np.float64)
    s0_raw = np.asarray(S0, dtype=np.float64)
    snr_raw = np.asarray(snr, dtype=np.float64)
    _validate_positive(d_raw, "D")
    _validate_alpha_domain(a_raw, "alpha")
    _validate_positive(s0_raw, "S0")
    _validate_positive(snr_raw, "snr")

    d_b, a_b, s0_b, snr_b = np.broadcast_arrays(d_raw, a_raw, s0_raw, snr_raw)
    scalar = d_b.ndim == 0
    lead_shape = d_b.shape

    d_1d = np.atleast_1d(d_b).ravel()
    a_1d = np.atleast_1d(a_b).ravel()
    s0_1d = np.atleast_1d(s0_b).ravel()
    snr_1d = np.atleast_1d(snr_b).ravel()
    n = d_1d.size

    jac_3d = anomalous_signal_jacobian(b_arr, d_1d, a_1d, S0=s0_1d)
    fim_3d = anomalous_fisher_matrix(b_arr, d_1d, a_1d, snr=snr_1d, S0=s0_1d, nex=w_nex)

    cov_3d = np.full((n, 3, 3), np.inf, dtype=np.float64)
    std_2d = np.full((n, 3), np.inf, dtype=np.float64)
    corr_1d = np.full(n, np.nan, dtype=np.float64)
    cond_1d = np.full(n, np.inf, dtype=np.float64)
    ident_1d = np.zeros(n, dtype=bool)

    theta_2d = np.column_stack([s0_1d, d_1d, a_1d])
    scales = np.maximum(np.abs(theta_2d), 1.0e-6)

    for i in range(n):
        mat = fim_3d[i]
        if not np.all(np.isfinite(mat)):
            continue
        sc = scales[i]
        mat_scaled = (mat * sc[:, None]) * sc[None, :]
        cond_val = float(np.linalg.cond(mat_scaled))
        cond_1d[i] = cond_val
        if not np.isfinite(cond_val) or cond_val > float(max_cond):
            continue
        try:
            inv_scaled = np.linalg.inv(mat_scaled)
            inv_mat = (inv_scaled * sc[:, None]) * sc[None, :]
            diag3 = np.diag(inv_mat)
            if np.all(np.isfinite(diag3)) and np.all(diag3 > 0.0):
                cov_3d[i] = inv_mat
                sd = np.sqrt(diag3)
                std_2d[i] = sd
                corr_1d[i] = float(np.clip(inv_mat[1, 2] / (sd[1] * sd[2]), -1.0, 1.0))
                ident_1d[i] = True
        except np.linalg.LinAlgError:
            continue

    cv_2d = std_2d / np.maximum(np.abs(theta_2d), 1.0e-12)

    if scalar:
        return AnomalousCRLBResult(
            fim=fim_3d[0],
            cov=cov_3d[0],
            std=std_2d[0],
            cv=cv_2d[0],
            d_alpha_corr=float(corr_1d[0]),
            condition_number=float(cond_1d[0]),
            identifiable_mask=bool(ident_1d[0]),
            jacobian=jac_3d[0],
            bvalues=b_arr,
            nex=w_nex,
        )

    return AnomalousCRLBResult(
        fim=fim_3d.reshape(lead_shape + (3, 3)),
        cov=cov_3d.reshape(lead_shape + (3, 3)),
        std=std_2d.reshape(lead_shape + (3,)),
        cv=cv_2d.reshape(lead_shape + (3,)),
        d_alpha_corr=corr_1d.reshape(lead_shape),
        condition_number=cond_1d.reshape(lead_shape),
        identifiable_mask=ident_1d.reshape(lead_shape),
        jacobian=jac_3d.reshape(lead_shape + (b_arr.size, 3)),
        bvalues=b_arr,
        nex=w_nex,
    )


def _ctrw_residuals(theta: np.ndarray, b: np.ndarray, y: np.ndarray) -> np.ndarray:
    s0, d_val, a_val = theta
    bd = b * d_val
    pos = bd > 0.0
    safe_bd = np.where(pos, bd, 1.0)
    bd_pow = np.where(pos, np.power(safe_bd, a_val), 0.0)
    return s0 * np.exp(-bd_pow) - y


def _ctrw_jacobian(theta: np.ndarray, b: np.ndarray, y: np.ndarray) -> np.ndarray:
    _ = y
    s0, d_val, a_val = theta
    bd = b * d_val
    pos = bd > 0.0
    safe_bd = np.where(pos, bd, 1.0)
    bd_pow = np.where(pos, np.power(safe_bd, a_val), 0.0)
    exp_term = np.exp(-bd_pow)
    sig = s0 * exp_term
    j_s0 = exp_term
    j_d = np.where(pos, -sig * a_val * bd_pow / d_val, 0.0)
    j_a = np.where(pos, -sig * bd_pow * np.log(safe_bd), 0.0)
    return np.column_stack([j_s0, j_d, j_a])


@dataclass(frozen=True)
class AnomalousNLLSFitResult:
    """Bounded NLLS fit container for ``S(b; S0, D, alpha) = S0 * exp(-(b * D)**alpha)``."""

    S0: np.ndarray
    D: np.ndarray
    alpha: np.ndarray
    sigma: np.ndarray
    railed_D: np.ndarray
    railed_alpha: np.ndarray
    success: np.ndarray

    @property
    def params(self) -> np.ndarray:
        """Fitted ``(n, 3)`` matrix ordered ``(S0, D, alpha)``."""
        return np.column_stack([self.S0, self.D, self.alpha])


def fit_anomalous_nlls(
    signals: np.ndarray,
    bvalues: np.ndarray | Sequence[float] | str | AcquisitionProtocol,
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    s0_bounds: tuple[float, float] = (0.1, 2.5),
    init: tuple[float, float] = (1.1e-3, 0.80),
    rail_tol: float = DEFAULT_RAIL_TOL,
    max_nfev: int = 200,
) -> AnomalousNLLSFitResult:
    """Fit bounded CTRW stretched-exponential parameters ``(S0, D, alpha)`` via TRF NLLS."""
    b_arr, _ = resolve_bvalues(bvalues) if isinstance(bvalues, (str, AcquisitionProtocol)) else (
        np.asarray(bvalues, dtype=np.float64).ravel(),
        "custom",
    )
    sig = np.atleast_2d(np.asarray(signals, dtype=np.float64))
    if sig.shape[1] != b_arr.size:
        raise ValueError(f"`signals` shape {sig.shape} incompatible with bvalues length {b_arr.size}.")

    active_bounds = dict(DEFAULT_ANOMALOUS_BOUNDS)
    if bounds is not None:
        active_bounds.update(bounds)
    d_lo, d_hi = active_bounds["D"]
    a_lo, a_hi = active_bounds["alpha"]

    n = sig.shape[0]
    b0_idx = int(np.argmin(b_arr))
    b_hi_idx = int(np.argmax(b_arr))
    b_span = max(float(b_arr[b_hi_idx] - b_arr[b0_idx]), 1.0)

    s0_out = np.empty(n, dtype=np.float64)
    d_out = np.empty(n, dtype=np.float64)
    a_out = np.empty(n, dtype=np.float64)
    sigma_out = np.empty((n, 3), dtype=np.float64)
    success_out = np.empty(n, dtype=bool)
    x_scale = np.array([1.0, 1.0e-3, 0.25], dtype=np.float64)

    for i in range(n):
        y = sig[i]
        s0_obs = max(float(y[b0_idx]), 1.0e-4)
        ratio_hi = float(np.clip(y[b_hi_idx] / s0_obs, 1.0e-4, 0.98))
        d_guess = float(np.clip(-np.log(ratio_hi) / b_span, d_lo * 1.05, d_hi * 0.95))
        lo3 = np.array([s0_bounds[0] * s0_obs, d_lo, a_lo], dtype=np.float64)
        hi3 = np.array([s0_bounds[1] * s0_obs, d_hi, a_hi], dtype=np.float64)
        span3 = hi3 - lo3
        x0 = np.clip(
            np.array([s0_obs, 0.5 * (d_guess + init[0]), init[1]], dtype=np.float64),
            lo3 + 1.0e-8,
            hi3 - 1.0e-8,
        )
        res = least_squares(
            _ctrw_residuals,
            x0,
            jac=_ctrw_jacobian,
            bounds=(lo3, hi3),
            args=(b_arr, y),
            method="trf",
            x_scale=x_scale,
            max_nfev=int(max_nfev),
        )
        s0_out[i] = float(res.x[0])
        d_out[i] = float(res.x[1])
        a_out[i] = float(res.x[2])
        success_out[i] = bool(res.success)

        dof = max(b_arr.size - 3, 1)
        resid_var = float(np.sum(res.fun**2)) / dof
        cov = resid_var * np.linalg.pinv(res.jac.T @ res.jac)
        var3 = np.clip(np.diag(cov), 0.0, span3**2)
        sigma_out[i] = np.sqrt(var3)

    railed_d = classify_railing(d_out, bounds=(d_lo, d_hi), rel_tol=rail_tol)["railed_mask"]
    railed_a = classify_railing(a_out, bounds=(a_lo, a_hi), rel_tol=rail_tol)["railed_mask"]
    return AnomalousNLLSFitResult(
        S0=s0_out,
        D=d_out,
        alpha=a_out,
        sigma=sigma_out,
        railed_D=railed_d,
        railed_alpha=railed_a,
        success=success_out,
    )


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = values.size
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = values[idx].mean(axis=1)
    tail = 0.5 * (1.0 - float(confidence))
    lo, hi = np.percentile(means, [100.0 * tail, 100.0 * (1.0 - tail)])
    return float(lo), float(hi)


def _bootstrap_rmse_ci(
    sq_errors: np.ndarray,
    *,
    n_boot: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = sq_errors.size
    idx = rng.integers(0, n, size=(int(n_boot), n))
    rmses = np.sqrt(sq_errors[idx].mean(axis=1))
    tail = 0.5 * (1.0 - float(confidence))
    lo, hi = np.percentile(rmses, [100.0 * tail, 100.0 * (1.0 - tail)])
    return float(lo), float(hi)


def _resolve_schedule_list(
    b_schedules: (
        str
        | AcquisitionProtocol
        | Mapping[str, Sequence[float] | np.ndarray | str | AcquisitionProtocol]
        | Sequence[str | AcquisitionProtocol | Sequence[float] | np.ndarray]
    ),
) -> list[tuple[str, np.ndarray]]:
    if isinstance(b_schedules, (str, AcquisitionProtocol)):
        b_arr, name = resolve_bvalues(b_schedules)
        return [(name, b_arr)]
    if isinstance(b_schedules, Mapping):
        resolved: list[tuple[str, np.ndarray]] = []
        for key, spec in b_schedules.items():
            b_arr, _ = resolve_bvalues(spec)
            resolved.append((str(key), b_arr))
        return resolved
    # Check if 1D numeric b-value array was passed directly
    arr_try = None
    if isinstance(b_schedules, np.ndarray) and b_schedules.ndim == 1 and np.issubdtype(b_schedules.dtype, np.number):
        arr_try = b_schedules
    elif (
        isinstance(b_schedules, Sequence)
        and len(b_schedules) > 0
        and isinstance(b_schedules[0], (int, float, np.number))
    ):
        arr_try = np.asarray(b_schedules, dtype=np.float64)
    if arr_try is not None:
        b_arr, name = resolve_bvalues(arr_try)
        return [(name, b_arr)]

    out: list[tuple[str, np.ndarray]] = []
    for idx, item in enumerate(b_schedules):
        b_arr, name = resolve_bvalues(item)
        label = f"schedule_{idx + 1}_bmax{int(round(float(np.max(b_arr))))}" if name == "custom" else name
        out.append((label, b_arr))
    if not out:
        raise ValueError("`b_schedules` must contain at least one b-value schedule.")
    return out


@dataclass
class RecoveryWallReport:
    """Joint ``(D, alpha)`` anomalous diffusion identifiability & collapse wall report."""

    dataframe: pd.DataFrame
    wall_front: pd.DataFrame
    summary: dict[str, Any]
    metadata: dict[str, Any]

    TABLE_COLS: tuple[str, ...] = (
        "schedule",
        "b_max",
        "snr",
        "alpha",
        "crlb_cv_D",
        "crlb_cv_alpha",
        "alpha_bias",
        "alpha_railed_rate",
        "alpha_railed_ci_lo",
        "alpha_railed_ci_hi",
        "is_collapsed",
    )

    def format_table(self) -> str:
        present = [c for c in self.TABLE_COLS if c in self.dataframe.columns]
        return self.dataframe[present].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )

    def to_markdown(self, float_fmt: str = ".4f") -> str:
        return dataframe_to_markdown(
            self.dataframe,
            columns=self.TABLE_COLS,
            float_fmt=float_fmt,
        )

    def to_latex(
        self,
        *,
        caption: str = "Anomalous fractional-order diffusion (D, alpha) CRLB and NLLS recovery wall.",
        label: str = "tab:anomalous_wall",
        float_fmt: str = ".4f",
    ) -> str:
        return dataframe_to_latex(
            self.dataframe,
            columns=self.TABLE_COLS,
            caption=caption,
            label=label,
            float_fmt=float_fmt,
        )

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_json(
            {
                "summary": self.summary,
                "metadata": self.metadata,
                "wall_front": self.wall_front.to_dict(orient="records"),
                "records": self.dataframe.to_dict(orient="records"),
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def map_recovery_collapse_wall(
    b_schedules: (
        str
        | AcquisitionProtocol
        | Mapping[str, Sequence[float] | np.ndarray | str | AcquisitionProtocol]
        | Sequence[str | AcquisitionProtocol | Sequence[float] | np.ndarray]
    ),
    snr_grid: Sequence[float] | np.ndarray,
    alpha_grid: Sequence[float] | np.ndarray,
    seed: int = 42,
    *,
    D_true: float = 1.0e-3,
    S0: float = 1.0,
    n_replicates: int = 64,
    noise_type: str = "rician",
    n_coils: int = 4,
    n_boot: int = 400,
    confidence: float = 0.95,
    cv_collapse_threshold: float = 0.25,
    railing_collapse_threshold: float = 0.20,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> RecoveryWallReport:
    """Quantify the joint ``(D, alpha)`` identifiability & recovery collapse wall.

    Evaluates analytical ``3 x 3`` CRLB coefficients of variation alongside
    ``PCG64``-seeded Monte Carlo bounded-NLLS recovery bias, RMSE, boundary-railing
    rates, and 95% percentile bootstrap confidence intervals across ``b_schedules``,
    ``snr_grid``, and ``alpha_grid``, identifying the exact ``SNR x b_max`` regime
    where fractional order ``alpha`` collapses.
    """
    schedules = _resolve_schedule_list(b_schedules)
    snrs = np.sort(np.atleast_1d(np.asarray(snr_grid, dtype=np.float64)).ravel())
    alphas = np.atleast_1d(np.asarray(alpha_grid, dtype=np.float64)).ravel()
    if snrs.size == 0 or alphas.size == 0:
        raise ValueError("`snr_grid` and `alpha_grid` must be non-empty.")
    _validate_positive(snrs, "snr_grid")
    _validate_alpha_domain(alphas, "alpha_grid")
    if int(n_replicates) < 4:
        raise ValueError("`n_replicates` must be >= 4.")

    seed_seq = np.random.SeedSequence(int(seed))
    total_cells = len(schedules) * int(snrs.size) * int(alphas.size)
    cell_streams = seed_seq.spawn(total_cells)

    rows: list[dict[str, Any]] = []
    cell_idx = 0

    for sched_name, b_arr in schedules:
        b_max = float(np.max(b_arr))
        for snr_val in snrs:
            for alpha_val in alphas:
                child_ss = cell_streams[cell_idx]
                cell_idx += 1
                sim_ss, boot_ss = child_ss.spawn(2)
                rng_sim = np.random.Generator(np.random.PCG64(sim_ss))
                rng_boot = np.random.Generator(np.random.PCG64(boot_ss))

                crlb = compute_anomalous_crlb(
                    b_arr,
                    D=float(D_true),
                    alpha=float(alpha_val),
                    snr=float(snr_val),
                    S0=float(S0),
                )

                clean_1d = ctrw_stretched_signal(
                    b_arr, D=float(D_true), alpha=float(alpha_val), S0=float(S0)
                )
                clean_rep = np.broadcast_to(clean_1d[None, :], (int(n_replicates), b_arr.size))
                noisy_rep = inject_noise(
                    clean_rep,
                    snr=float(snr_val),
                    noise_type=noise_type,
                    n_coils=int(n_coils),
                    seed=rng_sim,
                    s0=float(S0),
                )
                fit = fit_anomalous_nlls(noisy_rep, b_arr, bounds=bounds)

                a_err = fit.alpha - float(alpha_val)
                d_err = fit.D - float(D_true)
                alpha_bias = float(np.mean(a_err))
                alpha_rel_bias = float(alpha_bias / float(alpha_val))
                d_rel_bias = float(np.mean(d_err) / float(D_true))
                alpha_rmse = float(np.sqrt(np.mean(a_err**2)))
                d_rmse = float(np.sqrt(np.mean(d_err**2)))
                emp_cv_alpha = float(np.std(fit.alpha, ddof=1) / float(alpha_val))
                emp_cv_d = float(np.std(fit.D, ddof=1) / float(D_true))

                any_railed_mask = fit.railed_alpha | fit.railed_D
                rail_seed = int(rng_boot.integers(0, 2**31 - 1))
                rail_ci = bootstrap_railing_ci(
                    fit.railed_alpha,
                    n_boot=int(n_boot),
                    confidence=float(confidence),
                    seed=rail_seed,
                )
                bias_lo, bias_hi = _bootstrap_mean_ci(
                    a_err, n_boot=int(n_boot), confidence=float(confidence), rng=rng_boot
                )
                rmse_lo, rmse_hi = _bootstrap_rmse_ci(
                    a_err**2, n_boot=int(n_boot), confidence=float(confidence), rng=rng_boot
                )

                crlb_cv_alpha = float(crlb.alpha_cv)
                crlb_cv_d = float(crlb.D_cv)
                is_collapsed = bool(
                    (not bool(crlb.identifiable_mask))
                    or (crlb_cv_alpha >= float(cv_collapse_threshold))
                    or (rail_ci.point >= float(railing_collapse_threshold))
                )

                rows.append(
                    {
                        "schedule": sched_name,
                        "n_b": int(b_arr.size),
                        "b_max": b_max,
                        "snr": float(snr_val),
                        "alpha": float(alpha_val),
                        "D_true": float(D_true),
                        "crlb_cv_S0": float(crlb.S0_cv),
                        "crlb_cv_D": crlb_cv_d,
                        "crlb_cv_alpha": crlb_cv_alpha,
                        "crlb_corr_D_alpha": float(crlb.d_alpha_corr),
                        "condition_number": float(crlb.condition_number),
                        "emp_cv_D": emp_cv_d,
                        "emp_cv_alpha": emp_cv_alpha,
                        "D_rel_bias": d_rel_bias,
                        "D_rmse": d_rmse,
                        "alpha_bias": alpha_bias,
                        "alpha_rel_bias": alpha_rel_bias,
                        "alpha_bias_ci_lo": bias_lo,
                        "alpha_bias_ci_hi": bias_hi,
                        "alpha_rmse": alpha_rmse,
                        "alpha_rmse_ci_lo": rmse_lo,
                        "alpha_rmse_ci_hi": rmse_hi,
                        "alpha_railed_rate": float(rail_ci.point),
                        "alpha_railed_ci_lo": float(rail_ci.lo),
                        "alpha_railed_ci_hi": float(rail_ci.hi),
                        "D_railed_rate": float(np.mean(fit.railed_D)),
                        "any_railed_rate": float(np.mean(any_railed_mask)),
                        "is_collapsed": is_collapsed,
                    }
                )

    df = pd.DataFrame(rows)

    # Identify exact SNR x b_max collapse wall frontier per (schedule, b_max, alpha)
    front_rows: list[dict[str, Any]] = []
    for (sched_name, b_max_val, alpha_val), grp in df.groupby(
        ["schedule", "b_max", "alpha"], sort=False
    ):
        grp_sorted = grp.sort_values("snr")
        non_collapsed = grp_sorted[~grp_sorted["is_collapsed"]]
        collapsed = grp_sorted[grp_sorted["is_collapsed"]]
        critical_snr = (
            float(non_collapsed["snr"].iloc[0]) if len(non_collapsed) > 0 else float("inf")
        )
        max_collapsed_snr = (
            float(collapsed["snr"].iloc[-1]) if len(collapsed) > 0 else 0.0
        )
        front_rows.append(
            {
                "schedule": str(sched_name),
                "b_max": float(b_max_val),
                "alpha": float(alpha_val),
                "critical_snr": critical_snr,
                "max_collapsed_snr": max_collapsed_snr,
                "collapsed_fraction": float(np.mean(grp_sorted["is_collapsed"])),
                "min_crlb_cv_alpha": float(grp_sorted["crlb_cv_alpha"].min()),
                "max_alpha_railed_rate": float(grp_sorted["alpha_railed_rate"].max()),
            }
        )
    wall_front = pd.DataFrame(front_rows)

    finite_crit = wall_front["critical_snr"].to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(finite_crit)
    summary: dict[str, Any] = {
        "n_cells": int(len(df)),
        "collapsed_cell_fraction": float(df["is_collapsed"].mean()),
        "median_crlb_cv_alpha": float(df["crlb_cv_alpha"].median()),
        "median_crlb_cv_D": float(df["crlb_cv_D"].median()),
        "mean_alpha_railed_rate": float(df["alpha_railed_rate"].mean()),
        "median_critical_snr": (
            float(np.median(finite_crit[finite_mask])) if np.any(finite_mask) else float("inf")
        ),
        "cv_collapse_threshold": float(cv_collapse_threshold),
        "railing_collapse_threshold": float(railing_collapse_threshold),
    }
    metadata: dict[str, Any] = {
        "seed": int(seed),
        "D_true": float(D_true),
        "S0": float(S0),
        "n_replicates": int(n_replicates),
        "noise_type": noise_type,
        "n_coils": int(n_coils),
        "n_boot": int(n_boot),
        "confidence": float(confidence),
        "schedules": [s[0] for s in schedules],
        "snr_grid": [float(x) for x in snrs],
        "alpha_grid": [float(x) for x in alphas],
    }
    return RecoveryWallReport(
        dataframe=df,
        wall_front=wall_front,
        summary=summary,
        metadata=metadata,
    )
