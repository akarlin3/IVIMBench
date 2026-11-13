"""Marginal, tercile-conditional, and SNR-stratified coverage and calibration metrics.

Ported and unified from Caliper (``caliper.metrics`` and ``caliper.conformal``)
and Datum (``datum.ci``):
- Proper scoring rules: :func:`pinball_loss` and Gneiting-Raftery :func:`interval_score`.
- Marginal calibration: :func:`empirical_coverage`, :func:`coverage_gap`,
  :func:`ece_quantile`, :func:`quantile_calibration_curve`, and percentile
  bootstrap confidence intervals (:func:`bootstrap_coverage_ci`).
- Group-conditional coverage: :func:`tercile_stratified_coverage` (probing the
  high-``D*`` identifiability wall) and :func:`snr_stratified_coverage`.
- Post-hoc conformal wrappers: :class:`SplitConformalResidual`,
  :class:`SplitConformalQuantile`, and :class:`MondrianConformalQuantile`.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats

from .railing import DEFAULT_PARAM_BOUNDS, DEFAULT_RAIL_TOL

__all__ = [
    "SDConvention",
    "DEFAULT_RAILED_SD_FLOOR",
    "DEFAULT_QUANTILE_LEVELS",
    "DEFAULT_NOMINAL_LEVELS",
    "TERCILE_LABELS",
    "StratumCoverage",
    "has_conformal_ivim",
    "normalize_sd_convention",
    "apply_sd_convention",
    "apply_interval_sd_convention",
    "pinball_loss",
    "interval_score",
    "empirical_coverage",
    "coverage_gap",
    "quantile_calibration",
    "quantile_calibration_curve",
    "ece_quantile",
    "central_interval_calibration_curve",
    "gaussian_calibration_curve",
    "central_interval",
    "score_quantiles",
    "tercile_groups",
    "snr_groups",
    "group_conditional_coverage",
    "tercile_stratified_coverage",
    "snr_stratified_coverage",
    "bootstrap_coverage_ci",
    "conformity_scores",
    "conformal_offset",
    "SplitConformalResidual",
    "SplitConformalQuantile",
    "MondrianConformalQuantile",
    "SequentialConformalCalibrator",
    "calibrate_conformal_intervals",
]

SDConvention = Literal["honest_crlb", "floored_railed"]
DEFAULT_RAILED_SD_FLOOR: float = 3.0e-3

DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (
    0.025, 0.05, 0.10, 0.16, 0.25, 0.50, 0.75, 0.84, 0.90, 0.95, 0.975
)
DEFAULT_NOMINAL_LEVELS: tuple[float, ...] = (0.50, 0.68, 0.80, 0.90, 0.95)
TERCILE_LABELS: dict[int, str] = {0: "low", 1: "mid", 2: "high"}


def normalize_sd_convention(sd_convention: str) -> SDConvention:
    """Normalize ``sd_convention`` (including legacy ``'honest'``/``'floored'`` aliases)."""
    cleaned = sd_convention.strip().lower()
    if cleaned in ("honest_crlb", "honest"):
        return "honest_crlb"
    if cleaned in ("floored_railed", "floored"):
        return "floored_railed"
    raise ValueError(
        f"`sd_convention` must be 'honest_crlb' or 'floored_railed' "
        f"(got {sd_convention!r})."
    )


def _infer_railed_mask(
    point_arr: np.ndarray,
    bounds: tuple[float, float] | Mapping[str, tuple[float, float]] | None,
    rail_tol: float,
) -> np.ndarray:
    """Infer boolean boundary-railing mask for 1D ``(n,)`` or 2D ``(n, 3)`` estimates."""
    pt = np.asarray(point_arr, dtype=np.float64)
    tol = float(rail_tol)
    if pt.ndim == 1:
        b_pair = bounds if isinstance(bounds, tuple) else DEFAULT_PARAM_BOUNDS["Dstar"]
        lo_b, hi_b = float(b_pair[0]), float(b_pair[1])
        span = max(hi_b - lo_b, 1e-15)
        return ((pt - lo_b) <= tol * span) | ((hi_b - pt) <= tol * span)

    if pt.ndim == 2 and pt.shape[1] == 3:
        b_map = dict(DEFAULT_PARAM_BOUNDS)
        if isinstance(bounds, Mapping):
            b_map.update(bounds)
        names = ("D", "Dstar", "f")
        lo3 = np.array([b_map[p][0] for p in names], dtype=np.float64)
        hi3 = np.array([b_map[p][1] for p in names], dtype=np.float64)
        spans3 = np.maximum(hi3 - lo3, 1e-15)
        return ((pt - lo3[None, :]) <= tol * spans3[None, :]) | (
            (hi3[None, :] - pt) <= tol * spans3[None, :]
        )
    raise ValueError(f"Unsupported `point_pred` shape {pt.shape} for railing inference.")


def apply_sd_convention(
    std_pred: np.ndarray | Sequence[float],
    *,
    sd_convention: SDConvention | str = "honest_crlb",
    point_pred: np.ndarray | Sequence[float] | None = None,
    railed_mask: np.ndarray | Sequence[bool] | None = None,
    bounds: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
    rail_tol: float = DEFAULT_RAIL_TOL,
    railed_sd_floor: float = DEFAULT_RAILED_SD_FLOOR,
) -> np.ndarray:
    """Apply the Caliper dual-SD convention to per-voxel standard deviations.

    - ``"honest_crlb"`` (default): Preserves full Fisher/CRLB standard deviations.
    - ``"floored_railed"``: Collapses boundary-railed voxel SDs to ``railed_sd_floor``
      (reproducing the legacy over-confident railed-voxel under-coverage artifact).
    """
    conv = normalize_sd_convention(sd_convention)
    sd = np.asarray(std_pred, dtype=np.float64).copy()
    if conv == "honest_crlb":
        return sd

    if railed_mask is not None:
        mask = np.asarray(railed_mask, dtype=bool)
    elif point_pred is not None:
        mask = _infer_railed_mask(np.asarray(point_pred, dtype=np.float64), bounds, rail_tol)
    else:
        return sd

    if sd.ndim == 1:
        sd[mask.ravel()] = float(railed_sd_floor)
        return sd

    if sd.ndim == 2 and sd.shape[1] == 3:
        b_map = dict(DEFAULT_PARAM_BOUNDS)
        if isinstance(bounds, Mapping):
            b_map.update(bounds)
        spans3 = np.array(
            [
                b_map["D"][1] - b_map["D"][0],
                b_map["Dstar"][1] - b_map["Dstar"][0],
                b_map["f"][1] - b_map["f"][0],
            ],
            dtype=np.float64,
        )
        floors3 = np.array([0.015 * spans3[0], float(railed_sd_floor), 0.015 * spans3[2]])
        if mask.ndim == 1:
            sd[mask, 1] = floors3[1]
        else:
            for j in range(3):
                sd[mask[:, j], j] = np.minimum(sd[mask[:, j], j], floors3[j])
                if j == 1:
                    sd[mask[:, 1], 1] = floors3[1]
        return sd

    sd[mask] = float(railed_sd_floor)
    return sd


def apply_interval_sd_convention(
    lower: np.ndarray | Sequence[float],
    upper: np.ndarray | Sequence[float],
    *,
    sd_convention: SDConvention | str = "honest_crlb",
    point_pred: np.ndarray | Sequence[float] | None = None,
    railed_mask: np.ndarray | Sequence[bool] | None = None,
    bounds: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
    rail_tol: float = DEFAULT_RAIL_TOL,
    alpha: float = 0.10,
    railed_sd_floor: float = DEFAULT_RAILED_SD_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Adjust ``(lower, upper)`` intervals when ``sd_convention="floored_railed"``."""
    conv = normalize_sd_convention(sd_convention)
    lo = np.asarray(lower, dtype=np.float64).copy()
    hi = np.asarray(upper, dtype=np.float64).copy()
    if conv == "honest_crlb":
        return lo, hi

    center = (
        0.5 * (lo + hi)
        if point_pred is None
        else np.asarray(point_pred, dtype=np.float64)
    )
    if railed_mask is None and point_pred is None and bounds is None:
        return lo, hi

    z = max(float(stats.norm.ppf(1.0 - 0.5 * float(alpha))), 1e-12)
    implied_sd = np.maximum(0.5 * (hi - lo) / z, 1e-15)
    adj_sd = apply_sd_convention(
        implied_sd,
        sd_convention=conv,
        point_pred=center,
        railed_mask=railed_mask,
        bounds=bounds,
        rail_tol=rail_tol,
        railed_sd_floor=railed_sd_floor,
    )
    return center - z * adj_sd, center + z * adj_sd


@dataclass(frozen=True)
class StratumCoverage:
    """Empirical coverage, signed gap, and mean width within a conditioning stratum."""

    stratum: int | str
    label: str
    n: int
    coverage: float
    nominal: float
    coverage_gap: float
    mean_width: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain Python dictionary."""
        return {
            "stratum": self.stratum,
            "label": self.label,
            "n": self.n,
            "coverage": self.coverage,
            "nominal": self.nominal,
            "coverage_gap": self.coverage_gap,
            "mean_width": self.mean_width,
        }


# --------------------------------------------------------------------------- #
# Proper scoring rules
# --------------------------------------------------------------------------- #
def pinball_loss(
    y_true: np.ndarray | float,
    q_pred: np.ndarray | float,
    q: float,
) -> np.ndarray:
    """Elementwise quantile (pinball) loss at level ``q in (0, 1)``."""
    if not (0.0 < float(q) < 1.0):
        raise ValueError(f"Quantile level `q` must lie in (0, 1); got {q}.")
    y = np.asarray(y_true, dtype=np.float64)
    qp = np.asarray(q_pred, dtype=np.float64)
    diff = y - qp
    return np.maximum(float(q) * diff, (float(q) - 1.0) * diff)


def interval_score(
    lower: np.ndarray | float,
    upper: np.ndarray | float,
    y_true: np.ndarray | float,
    alpha: float = 0.10,
) -> np.ndarray:
    """Gneiting & Raftery (2007) proper interval score for a ``(1 - alpha)`` interval."""
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"`alpha` must lie strictly in (0, 1); got {alpha}.")
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    width = hi - lo
    below = (2.0 / float(alpha)) * np.maximum(lo - y, 0.0)
    above = (2.0 / float(alpha)) * np.maximum(y - hi, 0.0)
    return width + below + above


# --------------------------------------------------------------------------- #
# Marginal coverage & quantile calibration
# --------------------------------------------------------------------------- #
def empirical_coverage(
    y_true: np.ndarray | Sequence[float],
    lower: np.ndarray | Sequence[float],
    upper: np.ndarray | Sequence[float],
    *,
    sd_convention: SDConvention = "honest_crlb",
    point_pred: np.ndarray | Sequence[float] | None = None,
    railed_mask: np.ndarray | Sequence[bool] | None = None,
    bounds: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
    rail_tol: float = DEFAULT_RAIL_TOL,
    alpha: float = 0.10,
    railed_sd_floor: float = DEFAULT_RAILED_SD_FLOOR,
) -> float:
    """Fraction of ``y_true`` falling inside ``[lower, upper]`` (inclusive)."""
    y = np.asarray(y_true, dtype=np.float64)
    lo, hi = apply_interval_sd_convention(
        lower,
        upper,
        sd_convention=sd_convention,
        point_pred=point_pred,
        railed_mask=railed_mask,
        bounds=bounds,
        rail_tol=rail_tol,
        alpha=alpha,
        railed_sd_floor=railed_sd_floor,
    )
    if y.shape != lo.shape or y.shape != hi.shape:
        raise ValueError(
            f"Shape mismatch in empirical_coverage: y_true={y.shape}, "
            f"lower={lo.shape}, upper={hi.shape}."
        )
    if y.size == 0:
        raise ValueError("Inputs to `empirical_coverage` must be non-empty.")
    return float(np.mean((y >= lo) & (y <= hi)))


def coverage_gap(
    y_true: np.ndarray | Sequence[float],
    lower: np.ndarray | Sequence[float],
    upper: np.ndarray | Sequence[float],
    nominal: float = 0.90,
    *,
    sd_convention: SDConvention = "honest_crlb",
    point_pred: np.ndarray | Sequence[float] | None = None,
    railed_mask: np.ndarray | Sequence[bool] | None = None,
    bounds: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
    rail_tol: float = DEFAULT_RAIL_TOL,
    railed_sd_floor: float = DEFAULT_RAILED_SD_FLOOR,
) -> float:
    """Signed marginal coverage gap ``empirical_coverage - nominal``."""
    return (
        empirical_coverage(
            y_true,
            lower,
            upper,
            sd_convention=sd_convention,
            point_pred=point_pred,
            railed_mask=railed_mask,
            bounds=bounds,
            rail_tol=rail_tol,
            alpha=1.0 - float(nominal),
            railed_sd_floor=railed_sd_floor,
        )
        - float(nominal)
    )


def quantile_calibration(
    y_true: np.ndarray | Sequence[float],
    q_pred_level: np.ndarray | Sequence[float],
    q_level: float,
) -> float:
    """Empirical CDF fraction ``P(Y <= q_pred(q_level))`` for a single quantile level."""
    _ = float(q_level)
    y = np.asarray(y_true, dtype=np.float64)
    qp = np.asarray(q_pred_level, dtype=np.float64)
    return float(np.mean(y <= qp))


def quantile_calibration_curve(
    y_true: np.ndarray | Sequence[float],
    q_pred: np.ndarray,
    q_levels: Sequence[float] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute empirical CDF calibration curve across ascending ``q_levels``.

    Parameters
    ----------
    y_true : (n,) array-like
        1D ground-truth parameter vector.
    q_pred : (n, n_levels) np.ndarray
        Predicted quantiles for one parameter.
    q_levels : (n_levels,) sequence of float
        Nominal quantile levels in ``(0, 1)``.
    """
    y = np.asarray(y_true, dtype=np.float64)
    qp = np.asarray(q_pred, dtype=np.float64)
    levels = np.asarray(q_levels, dtype=np.float64)
    if qp.ndim != 2 or qp.shape != (y.shape[0], levels.size):
        raise ValueError(
            f"Expected q_pred shape {(y.shape[0], levels.size)}, got {qp.shape}."
        )
    empirical = np.array([np.mean(y <= qp[:, j]) for j in range(levels.size)], dtype=np.float64)
    return {
        "nominal": levels,
        "empirical": empirical,
        "gap": empirical - levels,
    }


def ece_quantile(
    y_true: np.ndarray | Sequence[float],
    q_pred: np.ndarray,
    q_levels: Sequence[float] | np.ndarray,
) -> float:
    """Expected Calibration Error (mean ``|empirical - nominal|``) across quantile levels."""
    curve = quantile_calibration_curve(y_true, q_pred, q_levels)
    return float(np.mean(np.abs(curve["gap"])))


def central_interval(
    q_pred: np.ndarray,
    q_levels: Sequence[float] | np.ndarray,
    alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the central ``(1 - alpha)`` interval ``(lower, upper)`` from a quantile array."""
    levels = np.asarray(q_levels, dtype=np.float64)
    qp = np.asarray(q_pred, dtype=np.float64)
    lo_target = 0.5 * float(alpha)
    hi_target = 1.0 - 0.5 * float(alpha)
    j_lo = int(np.argmin(np.abs(levels - lo_target)))
    j_hi = int(np.argmin(np.abs(levels - hi_target)))
    return qp[..., j_lo], qp[..., j_hi]


def central_interval_calibration_curve(
    y_true: np.ndarray | Sequence[float],
    q_pred: np.ndarray,
    q_levels: Sequence[float] | np.ndarray,
    nominal_levels: Sequence[float] | np.ndarray = DEFAULT_NOMINAL_LEVELS,
) -> dict[str, np.ndarray | float]:
    """Evaluate central interval coverage across multiple nominal confidence levels."""
    y = np.asarray(y_true, dtype=np.float64)
    qp = np.asarray(q_pred, dtype=np.float64)
    levels = np.asarray(q_levels, dtype=np.float64)
    nominals = np.asarray(nominal_levels, dtype=np.float64)
    emp = np.empty(nominals.size, dtype=np.float64)
    for idx, nom in enumerate(nominals):
        lo, hi = central_interval(qp, levels, alpha=1.0 - float(nom))
        emp[idx] = empirical_coverage(y, lo, hi)
    return {
        "nominal": nominals,
        "empirical": emp,
        "gap": emp - nominals,
        "ece": float(np.mean(np.abs(emp - nominals))),
    }


def gaussian_calibration_curve(
    y_true: np.ndarray | Sequence[float],
    mean_pred: np.ndarray | Sequence[float],
    std_pred: np.ndarray | Sequence[float],
    nominal_levels: Sequence[float] | np.ndarray = DEFAULT_NOMINAL_LEVELS,
    *,
    sd_convention: SDConvention = "honest_crlb",
    railed_mask: np.ndarray | Sequence[bool] | None = None,
    bounds: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
    rail_tol: float = DEFAULT_RAIL_TOL,
    railed_sd_floor: float = DEFAULT_RAILED_SD_FLOOR,
) -> dict[str, np.ndarray | float]:
    """Evaluate symmetric Gaussian ``mean +/- z * std`` interval coverage across nominal levels."""
    normalize_sd_convention(sd_convention)
    y = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(mean_pred, dtype=np.float64)
    sd_adj = apply_sd_convention(
        std_pred,
        sd_convention=sd_convention,
        point_pred=mu,
        railed_mask=railed_mask,
        bounds=bounds,
        rail_tol=rail_tol,
        railed_sd_floor=railed_sd_floor,
    )
    sd = np.maximum(sd_adj, 1e-15)
    nominals = np.asarray(nominal_levels, dtype=np.float64)
    z_err = np.abs(y - mu) / sd
    emp = np.array(
        [np.mean(z_err <= float(stats.norm.ppf(0.5 + 0.5 * nom))) for nom in nominals],
        dtype=np.float64,
    )
    return {
        "nominal": nominals,
        "empirical": emp,
        "gap": emp - nominals,
        "ece": float(np.mean(np.abs(emp - nominals))),
    }


def score_quantiles(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    q_levels: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS,
    *,
    param_names: Sequence[str] = ("D", "Dstar", "f"),
    alpha: float = 0.10,
    sd_convention: SDConvention = "honest_crlb",
    point_pred: np.ndarray | None = None,
    railed_mask: np.ndarray | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    rail_tol: float = DEFAULT_RAIL_TOL,
    railed_sd_floor: float = DEFAULT_RAILED_SD_FLOOR,
) -> dict[str, dict[str, Any]]:
    """Score ``(n, P, L)`` quantile predictions against ``y_true`` (Caliper/Lattice contract)."""
    truth = np.asarray(y_true, dtype=np.float64)
    qp = np.asarray(q_pred, dtype=np.float64)
    levels = np.asarray(q_levels, dtype=np.float64)
    if truth.ndim != 2 or qp.shape != (truth.shape[0], truth.shape[1], levels.size):
        raise ValueError(
            f"Expected y_true=(n, P) and q_pred=(n, P, L); got {truth.shape} and {qp.shape}."
        )
    conv = normalize_sd_convention(sd_convention)
    lo_mat, hi_mat = central_interval(qp, levels, alpha=alpha)
    mid_idx = int(np.argmin(np.abs(levels - 0.50)))
    pt_mat = qp[:, :, mid_idx] if point_pred is None else np.asarray(point_pred, dtype=np.float64)
    lo_adj, hi_adj = apply_interval_sd_convention(
        lo_mat,
        hi_mat,
        sd_convention=conv,
        point_pred=pt_mat,
        railed_mask=railed_mask,
        bounds=bounds if bounds is not None else DEFAULT_PARAM_BOUNDS,
        rail_tol=rail_tol,
        alpha=alpha,
        railed_sd_floor=railed_sd_floor,
    )
    nominal = 1.0 - float(alpha)
    dstar_col = 1 if truth.shape[1] > 1 else 0
    out: dict[str, dict[str, Any]] = {}
    for j, pname in enumerate(param_names):
        yj = truth[:, j]
        loj, hij = lo_adj[:, j], hi_adj[:, j]
        cov = empirical_coverage(yj, loj, hij)
        terc = tercile_stratified_coverage(
            yj, loj, hij, conditioning=truth[:, dstar_col], nominal=nominal
        )
        out[str(pname)] = {
            "marginal_coverage": cov,
            "marginal_coverage_gap": cov - nominal,
            "ece": ece_quantile(yj, qp[:, j, :], levels),
            "mean_width": float(np.mean(hij - loj)),
            "interval_score": float(np.mean(interval_score(loj, hij, yj, alpha=alpha))),
            "terciles": {k: v.to_dict() for k, v in terc.items()},
            "sd_convention": conv,
        }
    return out


# --------------------------------------------------------------------------- #
# Group-conditional coverage (D*-tercile & SNR-stratified)
# --------------------------------------------------------------------------- #
def tercile_groups(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Assign each entry of ``values`` to tercile ``0`` (low), ``1`` (mid), or ``2`` (high)."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("`values` passed to `tercile_groups` must be non-empty.")
    q1, q2 = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0])
    groups = np.zeros(x.shape, dtype=int)
    groups[x > q1] = 1
    groups[x > q2] = 2
    return groups


def snr_groups(
    snr_values: np.ndarray | Sequence[float],
    n_bins: int = 3,
) -> tuple[np.ndarray, dict[int, str]]:
    """Stratify per-voxel SNR values into discrete levels or quantile bins.

    If ``snr_values`` takes ``<= 6`` distinct values (e.g., an SNR sweep grid),
    each unique SNR forms its own stratum; otherwise values are partitioned into
    ``n_bins`` quantile bins.
    """
    s = np.asarray(snr_values, dtype=np.float64).ravel()
    if s.size == 0:
        raise ValueError("`snr_values` must be non-empty.")
    uniq = np.unique(s)
    if uniq.size <= 6:
        groups = np.searchsorted(uniq, s)
        labels = {int(i): f"snr_{u:g}" for i, u in enumerate(uniq)}
        return groups, labels

    edges = np.quantile(s, np.linspace(0.0, 1.0, int(n_bins) + 1))
    groups = np.zeros(s.shape, dtype=int)
    labels = {}
    for b_idx in range(int(n_bins)):
        if b_idx > 0:
            groups[s > edges[b_idx]] = b_idx
        labels[b_idx] = f"snr_bin{b_idx}_{edges[b_idx]:.1f}_{edges[b_idx + 1]:.1f}"
    return groups, labels


def group_conditional_coverage(
    y_true: np.ndarray | Sequence[float],
    lower: np.ndarray | Sequence[float],
    upper: np.ndarray | Sequence[float],
    groups: np.ndarray | Sequence[int],
    nominal: float = 0.90,
    group_labels: dict[int, str] | None = None,
    *,
    sd_convention: SDConvention = "honest_crlb",
    point_pred: np.ndarray | Sequence[float] | None = None,
    railed_mask: np.ndarray | Sequence[bool] | None = None,
    bounds: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
    rail_tol: float = DEFAULT_RAIL_TOL,
    railed_sd_floor: float = DEFAULT_RAILED_SD_FLOOR,
) -> dict[int, StratumCoverage]:
    """Compute empirical coverage, gap, and mean width within each integer group."""
    y = np.asarray(y_true, dtype=np.float64)
    lo, hi = apply_interval_sd_convention(
        lower,
        upper,
        sd_convention=sd_convention,
        point_pred=point_pred,
        railed_mask=railed_mask,
        bounds=bounds,
        rail_tol=rail_tol,
        alpha=1.0 - float(nominal),
        railed_sd_floor=railed_sd_floor,
    )
    grp = np.asarray(groups)
    if not (y.shape == lo.shape == hi.shape == grp.shape):
        raise ValueError(
            f"All inputs must share 1D shape (n,); got y={y.shape}, "
            f"lo={lo.shape}, hi={hi.shape}, groups={grp.shape}."
        )
    out: dict[int, StratumCoverage] = {}
    for g in sorted(int(u) for u in np.unique(grp)):
        mask = grp == g
        cov = empirical_coverage(y[mask], lo[mask], hi[mask])
        label = group_labels[g] if (group_labels and g in group_labels) else f"group_{g}"
        out[g] = StratumCoverage(
            stratum=g,
            label=label,
            n=int(np.sum(mask)),
            coverage=cov,
            nominal=float(nominal),
            coverage_gap=cov - float(nominal),
            mean_width=float(np.mean(hi[mask] - lo[mask])),
        )
    return out


def tercile_stratified_coverage(
    y_true: np.ndarray | Sequence[float],
    lower: np.ndarray | Sequence[float],
    upper: np.ndarray | Sequence[float],
    conditioning: np.ndarray | Sequence[float] | None = None,
    nominal: float = 0.90,
    *,
    sd_convention: SDConvention = "honest_crlb",
    point_pred: np.ndarray | Sequence[float] | None = None,
    railed_mask: np.ndarray | Sequence[bool] | None = None,
    bounds: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
    rail_tol: float = DEFAULT_RAIL_TOL,
    railed_sd_floor: float = DEFAULT_RAILED_SD_FLOOR,
) -> dict[str, StratumCoverage]:
    """Compute coverage stratified by low/mid/high terciles of ``conditioning`` (default ``y_true``)."""
    y = np.asarray(y_true, dtype=np.float64)
    cond = y if conditioning is None else np.asarray(conditioning, dtype=np.float64)
    grp = tercile_groups(cond)
    by_int = group_conditional_coverage(
        y,
        lower,
        upper,
        grp,
        nominal=nominal,
        group_labels=TERCILE_LABELS,
        sd_convention=sd_convention,
        point_pred=point_pred,
        railed_mask=railed_mask,
        bounds=bounds,
        rail_tol=rail_tol,
        railed_sd_floor=railed_sd_floor,
    )
    return {TERCILE_LABELS.get(k, f"g{k}"): v for k, v in by_int.items()}


def snr_stratified_coverage(
    y_true: np.ndarray | Sequence[float],
    lower: np.ndarray | Sequence[float],
    upper: np.ndarray | Sequence[float],
    snr: np.ndarray | Sequence[float],
    nominal: float = 0.90,
    n_bins: int = 3,
    *,
    sd_convention: SDConvention = "honest_crlb",
    point_pred: np.ndarray | Sequence[float] | None = None,
    railed_mask: np.ndarray | Sequence[bool] | None = None,
    bounds: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
    rail_tol: float = DEFAULT_RAIL_TOL,
    railed_sd_floor: float = DEFAULT_RAILED_SD_FLOOR,
) -> dict[str, StratumCoverage]:
    """Compute coverage stratified across per-voxel SNR regimes."""
    grp, labels = snr_groups(snr, n_bins=n_bins)
    by_int = group_conditional_coverage(
        y_true,
        lower,
        upper,
        grp,
        nominal=nominal,
        group_labels=labels,
        sd_convention=sd_convention,
        point_pred=point_pred,
        railed_mask=railed_mask,
        bounds=bounds,
        rail_tol=rail_tol,
        railed_sd_floor=railed_sd_floor,
    )
    return {labels[k]: v for k, v in by_int.items()}


def bootstrap_coverage_ci(
    y_true: np.ndarray | Sequence[float],
    lower: np.ndarray | Sequence[float],
    upper: np.ndarray | Sequence[float],
    nominal: float = 0.90,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    chunk: int = 250,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for marginal coverage and coverage gap."""
    y = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    inside = ((y >= lo) & (y <= hi)).astype(np.float64).ravel()
    n = inside.size
    if n == 0:
        raise ValueError("Inputs to `bootstrap_coverage_ci` must be non-empty.")

    rng = np.random.default_rng(int(seed))
    point_cov = float(np.mean(inside))
    boot_cov = np.empty(int(n_boot), dtype=np.float64)
    done = 0
    while done < int(n_boot):
        m = min(chunk, int(n_boot) - done)
        idx = rng.integers(0, n, size=(m, n))
        boot_cov[done : done + m] = inside[idx].mean(axis=1)
        done += m

    tail = 0.5 * (1.0 - float(confidence))
    cov_lo, cov_hi = np.percentile(boot_cov, [100.0 * tail, 100.0 * (1.0 - tail)])
    return {
        "coverage": point_cov,
        "coverage_ci_lo": float(cov_lo),
        "coverage_ci_hi": float(cov_hi),
        "coverage_gap": point_cov - float(nominal),
        "gap_ci_lo": float(cov_lo) - float(nominal),
        "gap_ci_hi": float(cov_hi) - float(nominal),
    }


# --------------------------------------------------------------------------- #
# Conformal post-hoc calibration wrappers (delegating to conformal_ivim)
# --------------------------------------------------------------------------- #
def has_conformal_ivim() -> bool:
    """Return ``True`` when canonical ``conformal_ivim`` is installed in the runtime."""
    return importlib.util.find_spec("conformal_ivim") is not None


def _load_conformal_submodule() -> Any | None:
    if has_conformal_ivim():
        return importlib.import_module("conformal_ivim.conformal")
    return None


def conformity_scores(q_lo: np.ndarray, q_hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Conformalized quantile regression (CQR) scores ``max(q_lo - y, y - q_hi)``."""
    return np.maximum(
        np.asarray(q_lo, dtype=np.float64) - np.asarray(y, dtype=np.float64),
        np.asarray(y, dtype=np.float64) - np.asarray(q_hi, dtype=np.float64),
    )


def conformal_offset(scores: np.ndarray, alpha: float, *, use_external: bool = True) -> float:
    """Finite-sample ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile of ``scores``."""
    s = np.asarray(scores, dtype=np.float64)
    n = s.shape[0]
    if n == 0:
        return 0.0
    if use_external:
        ext = _load_conformal_submodule()
        if ext is not None and hasattr(ext, "conformal_quantile"):
            q = float(ext.conformal_quantile(s, float(alpha)))
            return float(np.max(s)) if not np.isfinite(q) else q
    level = min(max(np.ceil((n + 1) * (1.0 - float(alpha))) / n, 0.0), 1.0)
    return float(np.quantile(s, level, method="higher"))


@dataclass
class SplitConformalResidual:
    """Distribution-free split-conformal intervals ``point +/- Q`` around point estimates."""

    alpha: float = 0.10
    use_external: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError(f"`alpha` must lie in (0, 1); got {self.alpha}.")
        self.offsets_: np.ndarray | None = None
        self.backend: str = (
            "conformal_ivim" if (self.use_external and has_conformal_ivim()) else "ivimbench_builtin"
        )
        self._cal_pred: np.ndarray | None = None
        self._cal_true: np.ndarray | None = None

    def calibrate(self, point_cal: np.ndarray, y_cal: np.ndarray) -> SplitConformalResidual:
        p_cal = np.asarray(point_cal, dtype=np.float64)
        truth = np.asarray(y_cal, dtype=np.float64)
        if p_cal.shape != truth.shape or p_cal.ndim != 2:
            raise ValueError("`point_cal` and `y_cal` must both have shape (n_cal, n_params).")
        self._cal_pred = p_cal.copy()
        self._cal_true = truth.copy()
        ext = _load_conformal_submodule() if self.use_external else None
        if ext is not None and hasattr(ext, "split_conformal"):
            self.backend = "conformal_ivim"
            offsets = []
            dummy = np.zeros(1, dtype=np.float64)
            for j in range(p_cal.shape[1]):
                _, _, q_j = ext.split_conformal(p_cal[:, j], truth[:, j], dummy, float(self.alpha))
                if not np.isfinite(q_j):
                    q_j = float(np.max(np.abs(truth[:, j] - p_cal[:, j]))) if p_cal.shape[0] else 0.0
                offsets.append(float(q_j))
            self.offsets_ = np.asarray(offsets, dtype=np.float64)
            return self

        self.backend = "ivimbench_builtin"
        scores = np.abs(truth - p_cal)
        self.offsets_ = np.array(
            [
                conformal_offset(scores[:, j], self.alpha, use_external=False)
                for j in range(scores.shape[1])
            ],
            dtype=np.float64,
        )
        return self

    def apply(self, point_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.offsets_ is None:
            raise RuntimeError("Call `calibrate()` before `apply()`.")
        p_test = np.asarray(point_test, dtype=np.float64)
        ext = _load_conformal_submodule() if self.use_external else None
        if (
            ext is not None
            and hasattr(ext, "split_conformal")
            and self._cal_pred is not None
            and self._cal_true is not None
        ):
            self.backend = "conformal_ivim"
            lo_cols = []
            hi_cols = []
            for j in range(p_test.shape[1]):
                lo_j, hi_j, q_j = ext.split_conformal(
                    self._cal_pred[:, j],
                    self._cal_true[:, j],
                    p_test[:, j],
                    float(self.alpha),
                )
                if not np.isfinite(q_j):
                    lo_j = p_test[:, j] - self.offsets_[j]
                    hi_j = p_test[:, j] + self.offsets_[j]
                lo_cols.append(np.asarray(lo_j, dtype=np.float64))
                hi_cols.append(np.asarray(hi_j, dtype=np.float64))
            return np.column_stack(lo_cols), np.column_stack(hi_cols)
        return p_test - self.offsets_[None, :], p_test + self.offsets_[None, :]

    def calibrate_apply(
        self,
        point_cal: np.ndarray,
        y_cal: np.ndarray,
        point_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.calibrate(point_cal, y_cal).apply(point_test)


@dataclass
class SplitConformalQuantile:
    """Conformalized Quantile Regression (CQR; Romano et al. 2019) over ``(n, P, L)`` quantiles."""

    q_levels: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS
    use_external: bool = True

    def __post_init__(self) -> None:
        self.q_levels = np.asarray(self.q_levels, dtype=np.float64)
        if not np.all(np.diff(self.q_levels) > 0.0):
            raise ValueError("`q_levels` must be strictly ascending.")
        self.offsets_: list[dict[tuple[int, int], float]] = []
        self._n_params: int | None = None
        self.backend: str = (
            "conformal_ivim" if (self.use_external and has_conformal_ivim()) else "ivimbench_builtin"
        )
        self._q_cal: np.ndarray | None = None
        self._y_cal: np.ndarray | None = None

    def calibrate(self, q_pred_cal: np.ndarray, y_cal: np.ndarray) -> SplitConformalQuantile:
        q_cal = np.asarray(q_pred_cal, dtype=np.float64)
        truth = np.asarray(y_cal, dtype=np.float64)
        levels = np.asarray(self.q_levels, dtype=np.float64)
        n, n_params, n_levels = q_cal.shape
        if truth.shape != (n, n_params) or n_levels != levels.size:
            raise ValueError("Shape mismatch between `q_pred_cal`, `y_cal`, and `q_levels`.")
        self._n_params = n_params
        self._q_cal = q_cal.copy()
        self._y_cal = truth.copy()
        ext = _load_conformal_submodule() if self.use_external else None
        use_ext_cqr = ext is not None and hasattr(ext, "cqr")
        self.backend = "conformal_ivim" if use_ext_cqr else "ivimbench_builtin"
        self.offsets_ = []
        dummy = np.zeros(1, dtype=np.float64)
        for p in range(n_params):
            off: dict[tuple[int, int], float] = {}
            for j in range(n_levels // 2):
                j_lo, j_hi = j, n_levels - 1 - j
                a = 2.0 * float(levels[j_lo])
                if use_ext_cqr:
                    assert ext is not None
                    _, _, q_val = ext.cqr(
                        q_cal[:, p, j_lo],
                        q_cal[:, p, j_hi],
                        truth[:, p],
                        dummy,
                        dummy,
                        a,
                    )
                    if not np.isfinite(q_val):
                        scores = conformity_scores(q_cal[:, p, j_lo], q_cal[:, p, j_hi], truth[:, p])
                        q_val = float(np.max(scores)) if scores.size else 0.0
                    off[(j_lo, j_hi)] = float(q_val)
                else:
                    scores = conformity_scores(q_cal[:, p, j_lo], q_cal[:, p, j_hi], truth[:, p])
                    off[(j_lo, j_hi)] = conformal_offset(scores, a, use_external=False)
            self.offsets_.append(off)
        return self

    def apply(self, q_pred_test: np.ndarray) -> np.ndarray:
        if not self.offsets_:
            raise RuntimeError("Call `calibrate()` before `apply()`.")
        q_test = np.asarray(q_pred_test, dtype=np.float64)
        out = q_test.copy()
        for p in range(q_test.shape[1]):
            for (j_lo, j_hi), offset in self.offsets_[p].items():
                out[:, p, j_lo] = q_test[:, p, j_lo] - offset
                out[:, p, j_hi] = q_test[:, p, j_hi] + offset
        return np.sort(out, axis=2)

    def calibrate_apply(
        self,
        q_pred_cal: np.ndarray,
        y_cal: np.ndarray,
        q_pred_test: np.ndarray,
    ) -> np.ndarray:
        return self.calibrate(q_pred_cal, y_cal).apply(q_pred_test)


@dataclass
class MondrianConformalQuantile:
    """Group-conditional (Mondrian) CQR fitting independent offsets per group label."""

    q_levels: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS
    use_external: bool = True

    def __post_init__(self) -> None:
        self.q_levels = np.asarray(self.q_levels, dtype=np.float64)
        if not np.all(np.diff(self.q_levels) > 0.0):
            raise ValueError("`q_levels` must be strictly ascending.")
        self.group_models_: dict[Any, SplitConformalQuantile] = {}
        self.global_: SplitConformalQuantile | None = None
        self.backend: str = (
            "conformal_ivim" if (self.use_external and has_conformal_ivim()) else "ivimbench_builtin"
        )
        self._q_cal: np.ndarray | None = None
        self._y_cal: np.ndarray | None = None
        self._grp_cal: np.ndarray | None = None

    def calibrate(
        self,
        q_pred_cal: np.ndarray,
        y_cal: np.ndarray,
        groups_cal: np.ndarray,
    ) -> MondrianConformalQuantile:
        q_cal = np.asarray(q_pred_cal, dtype=np.float64)
        truth = np.asarray(y_cal, dtype=np.float64)
        grp = np.asarray(groups_cal)
        self._q_cal = q_cal.copy()
        self._y_cal = truth.copy()
        self._grp_cal = grp.copy()
        self.group_models_ = {}
        for g in np.unique(grp):
            key = g.item() if hasattr(g, "item") else g
            mask = grp == g
            self.group_models_[key] = SplitConformalQuantile(
                self.q_levels, use_external=self.use_external
            ).calibrate(q_cal[mask], truth[mask])
        self.global_ = SplitConformalQuantile(
            self.q_levels, use_external=self.use_external
        ).calibrate(q_cal, truth)
        self.backend = self.global_.backend
        return self

    def apply(self, q_pred_test: np.ndarray, groups_test: np.ndarray) -> np.ndarray:
        if self.global_ is None:
            raise RuntimeError("Call `calibrate()` before `apply()`.")
        q_test = np.asarray(q_pred_test, dtype=np.float64)
        grp = np.asarray(groups_test)
        ext = _load_conformal_submodule() if self.use_external else None
        if (
            ext is not None
            and hasattr(ext, "mondrian_cqr")
            and self._q_cal is not None
            and self._y_cal is not None
            and self._grp_cal is not None
        ):
            levels = np.asarray(self.q_levels, dtype=np.float64)
            known_mask = np.isin(grp, np.unique(self._grp_cal))
            if np.all(known_mask):
                out = q_test.copy()
                n_levels = levels.size
                for p in range(q_test.shape[1]):
                    for j in range(n_levels // 2):
                        j_lo, j_hi = j, n_levels - 1 - j
                        a = 2.0 * float(levels[j_lo])
                        lo_m, hi_m = ext.mondrian_cqr(
                            self._q_cal[:, p, j_lo],
                            self._q_cal[:, p, j_hi],
                            self._y_cal[:, p],
                            self._grp_cal,
                            q_test[:, p, j_lo],
                            q_test[:, p, j_hi],
                            grp,
                            a,
                        )
                        if np.all(np.isfinite(lo_m)) and np.all(np.isfinite(hi_m)):
                            out[:, p, j_lo] = lo_m
                            out[:, p, j_hi] = hi_m
                        else:
                            for g_val in np.unique(grp):
                                key = g_val.item() if hasattr(g_val, "item") else g_val
                                m_g = grp == g_val
                                model = self.group_models_.get(key, self.global_)
                                out[m_g] = model.apply(q_test[m_g])
                            return out
                return np.sort(out, axis=2)

        out = q_test.copy()
        for g in np.unique(grp):
            key = g.item() if hasattr(g, "item") else g
            mask = grp == g
            model = self.group_models_.get(key, self.global_)
            out[mask] = model.apply(q_test[mask])
        return out

    def calibrate_apply(
        self,
        q_pred_cal: np.ndarray,
        y_cal: np.ndarray,
        groups_cal: np.ndarray,
        q_pred_test: np.ndarray,
        groups_test: np.ndarray,
    ) -> np.ndarray:
        return self.calibrate(q_pred_cal, y_cal, groups_cal).apply(q_pred_test, groups_test)


@dataclass
class SequentialConformalCalibrator:
    """Sequential online conformal interval calibrator (ACI & Conformal PID).

    Delegates to ``conformal_ivim.conformal.aci_conformal`` and
    ``conformal_ivim.AdaptiveConformalIVIM`` whenever ``conformal_ivim`` is installed,
    falling back cleanly to standalone NumPy Gibbs-Candès ACI / Conformal-PID updates.
    """

    method: Literal["aci", "conformal_pid"] = "aci"
    alpha: float = 0.10
    gamma: float = 0.05
    ki: float = 0.25
    kd: float = 0.35
    use_external: bool = True

    def __post_init__(self) -> None:
        m = str(self.method).strip().lower()
        if m not in ("aci", "conformal_pid"):
            raise ValueError(f"`method` must be 'aci' or 'conformal_pid'; got {self.method!r}.")
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError(f"`alpha` must lie in (0, 1); got {self.alpha}.")
        if float(self.gamma) <= 0.0:
            raise ValueError(f"`gamma` must be > 0; got {self.gamma}.")
        self.method = m  # type: ignore[assignment]
        self.backend: str = (
            "conformal_ivim" if (self.use_external and has_conformal_ivim()) else "ivimbench_builtin"
        )

    def calibrate_stream(
        self,
        cal_pred: np.ndarray,
        cal_true: np.ndarray,
        stream_pred: np.ndarray,
        stream_true: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run online sequential interval calibration over 1D ``(T,)`` or 2D ``(T, P)`` streams."""
        c_pred = np.asarray(cal_pred, dtype=np.float64)
        c_true = np.asarray(cal_true, dtype=np.float64)
        s_pred = np.asarray(stream_pred, dtype=np.float64)
        s_true = np.asarray(stream_true, dtype=np.float64)
        squeeze = s_pred.ndim == 1
        if squeeze:
            c_pred = c_pred.reshape(-1, 1)
            c_true = c_true.reshape(-1, 1)
            s_pred = s_pred.reshape(-1, 1)
            s_true = s_true.reshape(-1, 1)

        n_steps, n_params = s_pred.shape
        ext = _load_conformal_submodule() if self.use_external else None

        if self.method == "aci" and ext is not None and hasattr(ext, "aci_conformal"):
            self.backend = "conformal_ivim"
            lo_cols, hi_cols, tr_cols = [], [], []
            for j in range(n_params):
                lo_j, hi_j, tr_j = ext.aci_conformal(
                    c_pred[:, j],
                    c_true[:, j],
                    s_pred[:, j],
                    s_true[:, j],
                    alpha=float(self.alpha),
                    gamma=float(self.gamma),
                )
                lo_cols.append(np.asarray(lo_j, dtype=np.float64))
                hi_cols.append(np.asarray(hi_j, dtype=np.float64))
                tr_cols.append(np.asarray(tr_j, dtype=np.float64))
            lo_out = np.column_stack(lo_cols)
            hi_out = np.column_stack(hi_cols)
            tr_out = np.column_stack(tr_cols)
            if squeeze:
                return lo_out[:, 0], hi_out[:, 0], tr_out[:, 0]
            return lo_out, hi_out, tr_out

        # Standalone NumPy ACI / Conformal-PID online tracker
        self.backend = (
            "conformal_ivim"
            if (self.use_external and has_conformal_ivim() and self.method == "conformal_pid")
            else "ivimbench_builtin"
        )
        lo_out = np.empty((n_steps, n_params), dtype=np.float64)
        hi_out = np.empty((n_steps, n_params), dtype=np.float64)
        tr_out = np.empty((n_steps, n_params), dtype=np.float64)

        for j in range(n_params):
            cal_scores = np.abs(c_true[:, j] - c_pred[:, j])
            q_init = conformal_offset(cal_scores, self.alpha, use_external=self.use_external)
            scale = max(float(np.std(cal_scores)), q_init, 1e-8)
            if self.method == "aci":
                alpha_t = float(self.alpha)
                for t in range(n_steps):
                    tr_out[t, j] = alpha_t
                    q_t = conformal_offset(cal_scores, alpha_t, use_external=False)
                    lo_out[t, j] = s_pred[t, j] - q_t
                    hi_out[t, j] = s_pred[t, j] + q_t
                    err_t = 1.0 if (s_true[t, j] < lo_out[t, j] or s_true[t, j] > hi_out[t, j]) else 0.0
                    alpha_t = float(
                        np.clip(alpha_t + float(self.gamma) * (float(self.alpha) - err_t), 1e-4, 1.0 - 1e-4)
                    )
            else:
                q_t = q_init
                err_sum = 0.0
                prev_score = float(np.median(cal_scores)) if cal_scores.size else q_init
                for t in range(n_steps):
                    tr_out[t, j] = q_t
                    lo_out[t, j] = s_pred[t, j] - q_t
                    hi_out[t, j] = s_pred[t, j] + q_t
                    score_t = float(abs(s_true[t, j] - s_pred[t, j]))
                    err_t = 1.0 if score_t > q_t else 0.0
                    deficit = err_t - float(self.alpha)
                    err_sum += deficit
                    sat = float(np.tan(np.clip(err_sum / max(np.sqrt(t + 1.0), 1.0), -1.35, 1.35)))
                    trend = score_t - prev_score
                    prev_score = 0.5 * prev_score + 0.5 * score_t
                    q_t = max(
                        0.0,
                        q_t
                        + float(self.gamma) * scale * deficit
                        + float(self.ki) * 0.05 * scale * sat
                        + float(self.kd) * 0.15 * max(trend, 0.0),
                    )

        if squeeze:
            return lo_out[:, 0], hi_out[:, 0], tr_out[:, 0]
        return lo_out, hi_out, tr_out


def calibrate_conformal_intervals(
    method: Literal["split", "cqr", "mondrian", "aci", "conformal_pid"],
    *,
    cal_pred: np.ndarray | None = None,
    cal_true: np.ndarray,
    test_pred: np.ndarray | None = None,
    cal_lo: np.ndarray | None = None,
    cal_hi: np.ndarray | None = None,
    test_lo: np.ndarray | None = None,
    test_hi: np.ndarray | None = None,
    groups_cal: np.ndarray | None = None,
    groups_test: np.ndarray | None = None,
    stream_true: np.ndarray | None = None,
    alpha: float = 0.10,
    gamma: float = 0.05,
    use_external: bool = True,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Unified interval calibration entry point delegating to ``conformal_ivim`` when installed."""
    m = str(method).strip().lower()
    y_cal = np.asarray(cal_true, dtype=np.float64)
    squeeze = y_cal.ndim == 1
    if squeeze:
        y_cal_2d = y_cal[:, None]
    else:
        y_cal_2d = y_cal

    if m == "split":
        if cal_pred is None or test_pred is None:
            raise ValueError("`cal_pred` and `test_pred` are required for method='split'.")
        cp = np.asarray(cal_pred, dtype=np.float64)
        tp = np.asarray(test_pred, dtype=np.float64)
        if squeeze:
            cp = cp[:, None]
            tp = tp[:, None]
        calibrator = SplitConformalResidual(alpha=alpha, use_external=use_external)
        lo, hi = calibrator.calibrate_apply(cp, y_cal_2d, tp)
        return (lo[:, 0], hi[:, 0], calibrator.backend) if squeeze else (lo, hi, calibrator.backend)

    if m in ("cqr", "mondrian"):
        if cal_lo is None or cal_hi is None or test_lo is None or test_hi is None:
            raise ValueError("`cal_lo`, `cal_hi`, `test_lo`, and `test_hi` are required for CQR/Mondrian.")
        clo = np.asarray(cal_lo, dtype=np.float64)
        chi = np.asarray(cal_hi, dtype=np.float64)
        tlo = np.asarray(test_lo, dtype=np.float64)
        thi = np.asarray(test_hi, dtype=np.float64)
        if squeeze:
            clo, chi = clo[:, None], chi[:, None]
            tlo, thi = tlo[:, None], thi[:, None]
        levels = np.array([0.5 * alpha, 0.5, 1.0 - 0.5 * alpha], dtype=np.float64)
        q_cal = np.stack([clo, 0.5 * (clo + chi), chi], axis=-1)
        q_test = np.stack([tlo, 0.5 * (tlo + thi), thi], axis=-1)
        if m == "cqr":
            cqr_cal = SplitConformalQuantile(q_levels=levels, use_external=use_external)
            q_out = cqr_cal.calibrate_apply(q_cal, y_cal_2d, q_test)
            lo, hi = q_out[:, :, 0], q_out[:, :, 2]
            return (lo[:, 0], hi[:, 0], cqr_cal.backend) if squeeze else (lo, hi, cqr_cal.backend)
        if groups_cal is None or groups_test is None:
            raise ValueError("`groups_cal` and `groups_test` are required for method='mondrian'.")
        mond_cal = MondrianConformalQuantile(q_levels=levels, use_external=use_external)
        q_out = mond_cal.calibrate_apply(q_cal, y_cal_2d, groups_cal, q_test, groups_test)
        lo, hi = q_out[:, :, 0], q_out[:, :, 2]
        return (lo[:, 0], hi[:, 0], mond_cal.backend) if squeeze else (lo, hi, mond_cal.backend)

    if m in ("aci", "conformal_pid"):
        if cal_pred is None or test_pred is None or stream_true is None:
            raise ValueError(
                "`cal_pred`, `test_pred`, and `stream_true` are required for sequential methods."
            )
        seq_cal = SequentialConformalCalibrator(
            method=m,  # type: ignore[arg-type]
            alpha=alpha,
            gamma=gamma,
            use_external=use_external,
        )
        lo, hi, _ = seq_cal.calibrate_stream(cal_pred, y_cal, test_pred, stream_true)
        return lo, hi, seq_cal.backend

    raise ValueError(f"Unsupported conformal method {method!r}.")

