"""Prediction interval sharpness, normalized width, and Cramér-Rao Lower Bound (CRLB) efficiency.

Ported and unified from Caliper (``caliper.metrics``), Vernier (``vernier.crlb``),
Gauge (``gauge.forward``), and Sextant (``sextant_core.calibration``):
- **Absolute width** (:func:`absolute_width`): Mean prediction interval width
  ``E[upper - lower]`` in physical parameter units.
- **Normalized width** (:func:`normalized_width`): Interval width scaled by the
  parameter's physiological domain span ``upper_bound - lower_bound``, enabling
  dimensionless comparison across ``(D, D*, f)``.
- **Relative width** (:func:`relative_width`): Interval width scaled per-voxel by
  ground-truth magnitude ``|theta_true|``.
- **CRLB & Efficiency Ratio** (:func:`crlb_std`, :func:`crlb_interval_width`,
  :func:`crlb_efficiency_ratio`): Exact Fisher information matrix and Cramér-Rao
  lower bound standard errors for the bi-exponential IVIM signal model.

CRLB Efficiency Definition
--------------------------
At ground-truth ``theta_i = (D_i, D*_i, f_i)`` and noise scale
``sigma_i = S0 / SNR_i``, the Fisher information matrix is
``I(theta_i) = (1 / sigma_i^2) J(theta_i)^T W J(theta_i)``. The minimum
asymptotic ``(1 - alpha)`` Gaussian interval width for any locally unbiased
estimator of parameter ``j`` is:

.. math::

    W_{\\text{CRLB}, i, j}(1 - \\alpha) = 2 \\, z_{1 - \\alpha/2} \\sqrt{\\left[I(\\theta_i)^{-1}\\right]_{jj}}

Comparing predicted interval width ``W_hat_{i,j} = upper_{i,j} - lower_{i,j}``
against ``W_CRLB`` yields two reciprocal summary ratios over identifiable voxels:

1. **Sharpness-to-CRLB Ratio** (``sharpness_to_crlb_ratio``):
   ``median_i(W_hat_{i,j} / W_CRLB_{i,j})``. Equals ``1.0`` at the Cramér-Rao
   bound; values ``> 1.0`` measure interval width inflation relative to the
   information-theoretic floor; values ``< 1.0`` indicate either informative
   prior shrinkage or uncalibrated over-confidence.
2. **CRLB Efficiency** (``crlb_efficiency``):
   ``median_i(W_CRLB_{i,j} / W_hat_{i,j})``, the classical statistical efficiency
   in ``(0, 1]`` for calibrated unbiased intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "CRLBEfficiencyResult",
    "interval_widths",
    "absolute_width",
    "sharpness",
    "normalized_width",
    "relative_width",
    "ivim_jacobian",
    "fisher_information",
    "crlb_std",
    "crlb_interval_width",
    "crlb_efficiency_ratio",
    "crlb_dstar_wall_ratio",
]


@dataclass(frozen=True)
class CRLBEfficiencyResult:
    """Summary of prediction-interval sharpness relative to the Cramér-Rao lower bound."""

    sharpness_to_crlb_ratio: np.ndarray
    mean_sharpness_to_crlb_ratio: np.ndarray
    crlb_efficiency: np.ndarray
    mean_predicted_width: np.ndarray
    mean_crlb_width: np.ndarray
    identifiable_fraction: np.ndarray
    alpha: float

    def to_dict(self, param_names: Sequence[str] = ("D", "Dstar", "f")) -> dict[str, Any]:
        """Serialize per-parameter CRLB efficiency metrics into a nested dictionary."""
        out: dict[str, Any] = {"alpha": self.alpha, "parameters": {}}
        for j, name in enumerate(param_names):
            out["parameters"][name] = {
                "sharpness_to_crlb_ratio": float(self.sharpness_to_crlb_ratio[j]),
                "mean_sharpness_to_crlb_ratio": float(self.mean_sharpness_to_crlb_ratio[j]),
                "crlb_efficiency": float(self.crlb_efficiency[j]),
                "mean_predicted_width": float(self.mean_predicted_width[j]),
                "mean_crlb_width": float(self.mean_crlb_width[j]),
                "identifiable_fraction": float(self.identifiable_fraction[j]),
            }
        return out


def interval_widths(
    lower: np.ndarray | Sequence[float],
    upper: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Elementwise prediction interval width ``upper - lower``."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError(f"`lower` ({lo.shape}) and `upper` ({hi.shape}) shapes must match.")
    widths = hi - lo
    if np.any(widths < -1e-12):
        raise ValueError("`upper` must be >= `lower` elementwise.")
    return np.maximum(widths, 0.0)


def absolute_width(
    lower: np.ndarray | Sequence[float],
    upper: np.ndarray | Sequence[float],
    axis: int | None = 0,
) -> float | np.ndarray:
    """Mean absolute interval width along ``axis`` (scalar if 1D)."""
    w = interval_widths(lower, upper)
    if w.ndim == 1 or axis is None:
        return float(np.mean(w))
    return np.mean(w, axis=axis)


sharpness = absolute_width


def normalized_width(
    lower: np.ndarray | Sequence[float],
    upper: np.ndarray | Sequence[float],
    bounds: tuple[float, float] | Mapping[str, tuple[float, float]] | np.ndarray,
    param_names: Sequence[str] = ("D", "Dstar", "f"),
) -> float | np.ndarray:
    """Mean interval width normalized by parameter domain span ``bound_hi - bound_lo``.

    Parameters
    ----------
    lower, upper : array-like
        1D ``(n,)`` or 2D ``(n, n_params)`` interval endpoints.
    bounds : (lo, hi) tuple, dict of {name: (lo, hi)}, or (n_params, 2) array
        Physiological or optimization support bounds.
    param_names : sequence of str, default=("D", "Dstar", "f")
        Parameter ordering used when ``bounds`` is a mapping.
    """
    w = interval_widths(lower, upper)
    if w.ndim == 1:
        if isinstance(bounds, Mapping):
            lo_b, hi_b = next(iter(bounds.values()))
        else:
            b_arr = np.asarray(bounds, dtype=np.float64).ravel()
            lo_b, hi_b = float(b_arr[0]), float(b_arr[1])
        span = float(hi_b - lo_b)
        if span <= 0.0:
            raise ValueError(f"Bound span must be positive; got ({lo_b}, {hi_b}).")
        return float(np.mean(w) / span)

    if isinstance(bounds, Mapping):
        spans = np.array(
            [float(bounds[p][1] - bounds[p][0]) for p in param_names],
            dtype=np.float64,
        )
    else:
        b_mat = np.asarray(bounds, dtype=np.float64)
        spans = b_mat[:, 1] - b_mat[:, 0]
    if np.any(spans <= 0.0):
        raise ValueError(f"All parameter bound spans must be positive; got {spans}.")
    return np.mean(w, axis=0) / spans


def relative_width(
    lower: np.ndarray | Sequence[float],
    upper: np.ndarray | Sequence[float],
    y_true: np.ndarray | Sequence[float],
    eps: float = 1e-12,
) -> float | np.ndarray:
    """Mean interval width normalized per-voxel by ``max(|y_true|, eps)``."""
    w = interval_widths(lower, upper)
    y = np.asarray(y_true, dtype=np.float64)
    denom = np.maximum(np.abs(y), float(eps))
    ratio = w / denom
    if ratio.ndim == 1:
        return float(np.mean(ratio))
    return np.mean(ratio, axis=0)


# --------------------------------------------------------------------------- #
# Fisher Information Matrix and Cramér-Rao Lower Bound (CRLB)
# --------------------------------------------------------------------------- #
def ivim_jacobian(
    bvalues: np.ndarray | Sequence[float],
    params: np.ndarray | Sequence[float],
    s0: float | np.ndarray = 1.0,
    include_s0: bool = False,
) -> np.ndarray:
    """Analytical Jacobian ``dS/dtheta`` of the bi-exponential IVIM signal.

    Parameters
    ----------
    bvalues : (n_b,) array-like
        Diffusion b-values in ``s/mm^2``.
    params : (3,) or (n, 3) array-like
        Parameters ordered ``(D, Dstar, f)`` in ``(mm^2/s, mm^2/s, dimensionless)``.
    s0 : float or (n,) array-like, default=1.0
        Baseline ``b = 0`` signal intensity.
    include_s0 : bool, default=False
        If ``True``, appends ``dS/dS0`` as a 4th column to account for nuisance
        ``S0`` estimation uncertainty.

    Returns
    -------
    np.ndarray
        Jacobian array of shape ``(n, n_b, 3)`` (or ``(n, n_b, 4)`` when
        ``include_s0=True``; leading ``n`` squeezed if ``params`` was 1D).
    """
    b = np.asarray(bvalues, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    single = p.ndim == 1
    p2d = np.atleast_2d(p)
    if p2d.shape[1] != 3:
        raise ValueError(f"`params` must have 3 columns (D, Dstar, f); got {p2d.shape}.")

    n = p2d.shape[0]
    D = p2d[:, 0:1]
    Dstar = p2d[:, 1:2]
    f = p2d[:, 2:3]
    s0_col = np.broadcast_to(np.asarray(s0, dtype=np.float64), (n,))[:, None]
    bb = b[None, :]

    e_d = np.exp(-bb * D)
    e_ds = np.exp(-bb * Dstar)

    ds_dd = s0_col * (1.0 - f) * (-bb) * e_d
    ds_dds = s0_col * f * (-bb) * e_ds
    ds_df = s0_col * (e_ds - e_d)

    cols = [ds_dd, ds_dds, ds_df]
    if include_s0:
        ds_ds0 = f * e_ds + (1.0 - f) * e_d
        cols.append(ds_ds0)

    jac = np.stack(cols, axis=-1)
    return jac[0] if single else jac


def fisher_information(
    bvalues: np.ndarray | Sequence[float],
    params: np.ndarray | Sequence[float],
    snr: float | np.ndarray,
    s0: float | np.ndarray = 1.0,
    averages: np.ndarray | Sequence[int] | None = None,
    include_s0: bool = False,
) -> np.ndarray:
    """Compute per-voxel Gaussian Fisher Information Matrix ``I(theta)``."""
    b = np.asarray(bvalues, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    single = p.ndim == 1
    p2d = np.atleast_2d(p)
    n = p2d.shape[0]

    jac = ivim_jacobian(b, p2d, s0=s0, include_s0=include_s0)  # (n, n_b, P)
    w = (
        np.ones(b.size, dtype=np.float64)
        if averages is None
        else np.asarray(averages, dtype=np.float64)
    )
    s0_vec = np.broadcast_to(np.asarray(s0, dtype=np.float64), (n,))
    snr_vec = np.broadcast_to(np.asarray(snr, dtype=np.float64), (n,))
    sigma = s0_vec / snr_vec

    jac_weighted = jac * w[None, :, None]
    info = np.einsum("nbi,nbj->nij", jac_weighted, jac) / (sigma**2)[:, None, None]
    return info[0] if single else info


def crlb_std(
    bvalues: np.ndarray | Sequence[float],
    params: np.ndarray | Sequence[float],
    snr: float | np.ndarray,
    s0: float | np.ndarray = 1.0,
    averages: np.ndarray | Sequence[int] | None = None,
    include_s0: bool = False,
) -> np.ndarray:
    """Cramér-Rao lower bound standard error for ``(D, Dstar, f)`` in physical units.

    Singular or ill-conditioned Fisher matrices (condition number ``> 1e14``)
    yield ``np.inf`` for locally unidentifiable parameters.
    """
    p = np.asarray(params, dtype=np.float64)
    single = p.ndim == 1
    p2d = np.atleast_2d(p)
    n = p2d.shape[0]
    info = fisher_information(
        bvalues, p2d, snr=snr, s0=s0, averages=averages, include_s0=include_s0
    )
    out = np.full((n, 3), np.inf, dtype=np.float64)

    for i in range(n):
        mat = info[i]
        if not np.all(np.isfinite(mat)):
            continue
        cond = np.linalg.cond(mat)
        if not np.isfinite(cond) or cond > 1e14:
            continue
        try:
            cov = np.linalg.inv(mat)
            diag = np.diag(cov)[:3]
            valid = np.isfinite(diag) & (diag > 0.0)
            out[i, valid] = np.sqrt(diag[valid])
        except np.linalg.LinAlgError:
            continue

    return out[0] if single else out


def crlb_interval_width(
    bvalues: np.ndarray | Sequence[float],
    params: np.ndarray | Sequence[float],
    snr: float | np.ndarray,
    alpha: float = 0.10,
    s0: float | np.ndarray = 1.0,
    averages: np.ndarray | Sequence[int] | None = None,
    include_s0: bool = False,
) -> np.ndarray:
    """Theoretical ``(1 - alpha)`` asymptotic Gaussian CRLB interval width ``2 * z * sigma_CRLB``."""
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"`alpha` must lie in (0, 1); got {alpha}.")
    z_crit = float(stats.norm.ppf(1.0 - 0.5 * float(alpha)))
    sd = crlb_std(
        bvalues, params, snr=snr, s0=s0, averages=averages, include_s0=include_s0
    )
    return 2.0 * z_crit * sd


def crlb_efficiency_ratio(
    lower: np.ndarray,
    upper: np.ndarray,
    bvalues: np.ndarray | Sequence[float],
    y_true: np.ndarray,
    snr: float | np.ndarray,
    alpha: float = 0.10,
    s0: float | np.ndarray = 1.0,
    averages: np.ndarray | Sequence[int] | None = None,
    include_s0: bool = False,
) -> CRLBEfficiencyResult:
    """Compare predicted ``(1 - alpha)`` interval widths against the Cramér-Rao bound.

    Parameters
    ----------
    lower, upper : (n, 3) np.ndarray
        Predicted lower and upper endpoints for ``(D, Dstar, f)``.
    bvalues : (n_b,) array-like
        Acquisition b-value schedule in ``s/mm^2``.
    y_true : (n, 3) np.ndarray
        Ground-truth ``(D, Dstar, f)`` parameters.
    snr : float or (n,) array-like
        Signal-to-noise ratio at ``b = 0``.
    alpha : float, default=0.10
        Nominal interval miscoverage rate (``0.10`` for 90% central intervals).
    include_s0 : bool, default=False
        Whether to treat baseline ``S0`` as an unknown 4th nuisance parameter in
        the Fisher information matrix.

    Returns
    -------
    CRLBEfficiencyResult
        Structured summary with per-parameter median/mean sharpness-to-CRLB ratios,
        CRLB efficiencies, mean widths, and identifiable voxel fractions.
    """
    pred_w = interval_widths(lower, upper)
    truth = np.asarray(y_true, dtype=np.float64)
    if pred_w.ndim != 2 or pred_w.shape != truth.shape or truth.shape[1] != 3:
        raise ValueError(
            f"`lower`, `upper`, and `y_true` must share shape (n, 3); "
            f"got pred_w={pred_w.shape}, y_true={truth.shape}."
        )

    crlb_w = crlb_interval_width(
        bvalues,
        truth,
        snr=snr,
        alpha=alpha,
        s0=s0,
        averages=averages,
        include_s0=include_s0,
    )

    med_ratio = np.full(3, np.nan, dtype=np.float64)
    mean_ratio = np.full(3, np.nan, dtype=np.float64)
    efficiency = np.full(3, np.nan, dtype=np.float64)
    mean_pred_w = np.mean(pred_w, axis=0)
    mean_crlb_w = np.full(3, np.inf, dtype=np.float64)
    id_frac = np.zeros(3, dtype=np.float64)

    for j in range(3):
        valid = np.isfinite(crlb_w[:, j]) & (crlb_w[:, j] > 0.0) & (pred_w[:, j] > 0.0)
        id_frac[j] = float(np.mean(np.isfinite(crlb_w[:, j]) & (crlb_w[:, j] > 0.0)))
        if np.any(valid):
            ratios = pred_w[valid, j] / crlb_w[valid, j]
            effs = crlb_w[valid, j] / pred_w[valid, j]
            med_ratio[j] = float(np.median(ratios))
            mean_ratio[j] = float(np.mean(ratios))
            efficiency[j] = float(np.median(effs))
            mean_crlb_w[j] = float(np.mean(crlb_w[valid, j]))

    return CRLBEfficiencyResult(
        sharpness_to_crlb_ratio=med_ratio,
        mean_sharpness_to_crlb_ratio=mean_ratio,
        crlb_efficiency=efficiency,
        mean_predicted_width=mean_pred_w,
        mean_crlb_width=mean_crlb_w,
        identifiable_fraction=id_frac,
        alpha=float(alpha),
    )


def crlb_dstar_wall_ratio(
    bvalues: np.ndarray | Sequence[float],
    params: np.ndarray,
    snr: float | np.ndarray,
    include_s0: bool = False,
) -> float:
    """Ratio of mean ``CRLB(D*)`` SD in the top ``D*`` tercile to that tercile's span.

    Values ``>= 1.0`` indicate the information-theoretic ``D*`` identifiability
    wall where a single standard error exceeds the entire high-``D*`` bin width.
    """
    truth = np.asarray(params, dtype=np.float64)
    dstar = truth[:, 1]
    q_hi, d_max = np.quantile(dstar, [2.0 / 3.0, 1.0])
    span = float(d_max - q_hi)
    mask = dstar >= q_hi
    if span <= 0.0 or not np.any(mask):
        return float("inf")
    snr_sub = snr if np.ndim(snr) == 0 else np.asarray(snr, dtype=np.float64)[mask]
    sd = crlb_std(bvalues, truth[mask], snr=snr_sub, include_s0=include_s0)[:, 1]
    finite = sd[np.isfinite(sd)]
    if finite.size == 0:
        return float("inf")
    return float(np.mean(finite) / span)
