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

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "DEFAULT_QUANTILE_LEVELS",
    "DEFAULT_NOMINAL_LEVELS",
    "TERCILE_LABELS",
    "StratumCoverage",
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
]

DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (
    0.025, 0.05, 0.10, 0.16, 0.25, 0.50, 0.75, 0.84, 0.90, 0.95, 0.975
)
DEFAULT_NOMINAL_LEVELS: tuple[float, ...] = (0.50, 0.68, 0.80, 0.90, 0.95)
TERCILE_LABELS: dict[int, str] = {0: "low", 1: "mid", 2: "high"}


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
) -> float:
    """Fraction of ``y_true`` falling inside ``[lower, upper]`` (inclusive)."""
    y = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
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
) -> float:
    """Signed marginal coverage gap ``empirical_coverage - nominal``."""
    return empirical_coverage(y_true, lower, upper) - float(nominal)


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
) -> dict[str, np.ndarray | float]:
    """Evaluate symmetric Gaussian ``mean +/- z * std`` interval coverage across nominal levels."""
    y = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(mean_pred, dtype=np.float64)
    sd = np.maximum(np.asarray(std_pred, dtype=np.float64), 1e-15)
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
) -> dict[int, StratumCoverage]:
    """Compute empirical coverage, gap, and mean width within each integer group."""
    y = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
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
) -> dict[str, StratumCoverage]:
    """Compute coverage stratified by low/mid/high terciles of ``conditioning`` (default ``y_true``)."""
    y = np.asarray(y_true, dtype=np.float64)
    cond = y if conditioning is None else np.asarray(conditioning, dtype=np.float64)
    grp = tercile_groups(cond)
    by_int = group_conditional_coverage(
        y, lower, upper, grp, nominal=nominal, group_labels=TERCILE_LABELS
    )
    return {TERCILE_LABELS.get(k, f"g{k}"): v for k, v in by_int.items()}


def snr_stratified_coverage(
    y_true: np.ndarray | Sequence[float],
    lower: np.ndarray | Sequence[float],
    upper: np.ndarray | Sequence[float],
    snr: np.ndarray | Sequence[float],
    nominal: float = 0.90,
    n_bins: int = 3,
) -> dict[str, StratumCoverage]:
    """Compute coverage stratified across per-voxel SNR regimes."""
    grp, labels = snr_groups(snr, n_bins=n_bins)
    by_int = group_conditional_coverage(
        y_true, lower, upper, grp, nominal=nominal, group_labels=labels
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
# Conformal post-hoc calibration wrappers (ported from Caliper)
# --------------------------------------------------------------------------- #
def conformity_scores(q_lo: np.ndarray, q_hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Conformalized quantile regression (CQR) scores ``max(q_lo - y, y - q_hi)``."""
    return np.maximum(
        np.asarray(q_lo, dtype=np.float64) - np.asarray(y, dtype=np.float64),
        np.asarray(y, dtype=np.float64) - np.asarray(q_hi, dtype=np.float64),
    )


def conformal_offset(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile of ``scores``."""
    s = np.asarray(scores, dtype=np.float64)
    n = s.shape[0]
    if n == 0:
        return 0.0
    level = min(max(np.ceil((n + 1) * (1.0 - float(alpha))) / n, 0.0), 1.0)
    return float(np.quantile(s, level, method="higher"))


@dataclass
class SplitConformalResidual:
    """Distribution-free split-conformal intervals ``point +/- Q`` around point estimates."""

    alpha: float = 0.10

    def __post_init__(self) -> None:
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError(f"`alpha` must lie in (0, 1); got {self.alpha}.")
        self.offsets_: np.ndarray | None = None

    def calibrate(self, point_cal: np.ndarray, y_cal: np.ndarray) -> "SplitConformalResidual":
        p_cal = np.asarray(point_cal, dtype=np.float64)
        truth = np.asarray(y_cal, dtype=np.float64)
        if p_cal.shape != truth.shape or p_cal.ndim != 2:
            raise ValueError("`point_cal` and `y_cal` must both have shape (n_cal, n_params).")
        scores = np.abs(truth - p_cal)
        self.offsets_ = np.array(
            [conformal_offset(scores[:, j], self.alpha) for j in range(scores.shape[1])],
            dtype=np.float64,
        )
        return self

    def apply(self, point_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.offsets_ is None:
            raise RuntimeError("Call `calibrate()` before `apply()`.")
        p_test = np.asarray(point_test, dtype=np.float64)
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

    def __post_init__(self) -> None:
        self.q_levels = np.asarray(self.q_levels, dtype=np.float64)
        if not np.all(np.diff(self.q_levels) > 0.0):
            raise ValueError("`q_levels` must be strictly ascending.")
        self.offsets_: list[dict[tuple[int, int], float]] = []
        self._n_params: int | None = None

    def calibrate(self, q_pred_cal: np.ndarray, y_cal: np.ndarray) -> "SplitConformalQuantile":
        q_cal = np.asarray(q_pred_cal, dtype=np.float64)
        truth = np.asarray(y_cal, dtype=np.float64)
        n, n_params, n_levels = q_cal.shape
        if truth.shape != (n, n_params) or n_levels != self.q_levels.size:
            raise ValueError("Shape mismatch between `q_pred_cal`, `y_cal`, and `q_levels`.")
        self._n_params = n_params
        self.offsets_ = []
        for p in range(n_params):
            off: dict[tuple[int, int], float] = {}
            for j in range(n_levels // 2):
                j_lo, j_hi = j, n_levels - 1 - j
                a = 2.0 * float(self.q_levels[j_lo])
                scores = conformity_scores(q_cal[:, p, j_lo], q_cal[:, p, j_hi], truth[:, p])
                off[(j_lo, j_hi)] = conformal_offset(scores, a)
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

    def __post_init__(self) -> None:
        self.q_levels = np.asarray(self.q_levels, dtype=np.float64)
        if not np.all(np.diff(self.q_levels) > 0.0):
            raise ValueError("`q_levels` must be strictly ascending.")
        self.group_models_: dict[Any, SplitConformalQuantile] = {}
        self.global_: SplitConformalQuantile | None = None

    def calibrate(
        self,
        q_pred_cal: np.ndarray,
        y_cal: np.ndarray,
        groups_cal: np.ndarray,
    ) -> "MondrianConformalQuantile":
        q_cal = np.asarray(q_pred_cal, dtype=np.float64)
        truth = np.asarray(y_cal, dtype=np.float64)
        grp = np.asarray(groups_cal)
        self.group_models_ = {}
        for g in np.unique(grp):
            key = g.item() if hasattr(g, "item") else g
            mask = grp == g
            self.group_models_[key] = SplitConformalQuantile(self.q_levels).calibrate(
                q_cal[mask], truth[mask]
            )
        self.global_ = SplitConformalQuantile(self.q_levels).calibrate(q_cal, truth)
        return self

    def apply(self, q_pred_test: np.ndarray, groups_test: np.ndarray) -> np.ndarray:
        if self.global_ is None:
            raise RuntimeError("Call `calibrate()` before `apply()`.")
        q_test = np.asarray(q_pred_test, dtype=np.float64)
        grp = np.asarray(groups_test)
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
