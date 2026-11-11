"""Clinical threshold decision-utility evaluation, Value of Information (VoI), and domain-shift checks.

Implements:
1. **Threshold Decision & VoI Pricing** (:func:`evaluate_decision_voi`,
   :func:`compare_decision_calibration`): Evaluates calibrated vs. miscalibrated
   prediction intervals across parameterized clinical action thresholds ``tau``
   and asymmetric error costs ``cost_ratio``, quantifying expected utility
   regret, the decision–calibration gap, first-order EVPI, and second-order
   value-of-information (value of uncertainty calibration).
2. **Two-Sample Observable Domain-Shift / Staleness Check**
   (:func:`check_domain_shift`): Ground-truth-free deployment safeguard combining
   Mahalanobis signal-space drift and forward-model residual-space drift with
   deterministic ``PCG64`` permutation testing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..baselines.segmented import SegmentedEstimator
from ..metrics.coverage import SplitConformalResidual
from ..phantoms.synthetic import DEFAULT_SEED, PARAM_NAMES, ivim_biexp
from ..protocols import AcquisitionProtocol, resolve_bvalues
from ..runner import _sanitize_json, dataframe_to_latex, dataframe_to_markdown

__all__ = [
    "DecisionVoIReport",
    "DomainShiftReport",
    "infer_exceedance_probability",
    "evaluate_decision_voi",
    "compare_decision_calibration",
    "check_domain_shift",
]


def infer_exceedance_probability(
    lower: np.ndarray,
    upper: np.ndarray,
    tau: float,
    *,
    point: np.ndarray | None = None,
    alpha: float = 0.10,
    direction: str = "less",
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer event probability ``P(Theta <= tau)`` (or ``>= tau``) from ``(1 - alpha)`` intervals.

    Uses a split-normal predictive approximation anchored at ``point`` (defaulting
    to interval midpoint) with left/right scales matched to ``(point - lower) / z``
    and ``(upper - point) / z`` so asymmetric conformal/quantile intervals are
    respected without distortion.

    Returns
    -------
    prob : np.ndarray
        Exceedance probability ``p_hat(tau)`` in ``[0, 1]``.
    loc : np.ndarray
        Predictive center ``m_hat``.
    eff_sigma : np.ndarray
        Effective predictive standard deviation ``sigma_hat``.
    """
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.maximum(np.asarray(upper, dtype=np.float64).ravel(), lo)
    loc = (
        0.5 * (lo + hi)
        if point is None
        else np.clip(np.asarray(point, dtype=np.float64).ravel(), lo, hi)
    )
    z_crit = float(stats.norm.ppf(1.0 - 0.5 * float(alpha)))
    sigma_l = np.maximum((loc - lo) / z_crit, eps)
    sigma_r = np.maximum((hi - loc) / z_crit, eps)
    eff_sigma = 0.5 * (sigma_l + sigma_r)

    delta = float(tau) - loc
    scale = np.where(delta <= 0.0, sigma_l, sigma_r)
    cdf_at_tau = stats.norm.cdf(delta / scale)

    dir_key = direction.strip().lower()
    if dir_key in ("less", "le", "<=", "<", "below"):
        prob = cdf_at_tau
    elif dir_key in ("greater", "ge", ">=", ">", "above"):
        prob = 1.0 - cdf_at_tau
    else:
        raise ValueError(f"`direction` must be 'less' or 'greater'; got {direction!r}.")
    return np.clip(prob, 0.0, 1.0), loc, eff_sigma


@dataclass
class DecisionVoIReport:
    """Threshold clinical decision-utility, calibration gap, and second-order VoI report."""

    dataframe: pd.DataFrame
    summary: dict[str, float | str]
    metadata: dict[str, Any]

    TABLE_COLS: tuple[str, ...] = (
        "tau",
        "prevalence",
        "action_rate",
        "expected_regret",
        "perceived_regret",
        "decision_calibration_gap",
        "calibrated_regret",
        "second_order_voi",
        "net_benefit",
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
        caption: str = "Clinical threshold utility regret, decision--calibration gap, and second-order VoI.",
        label: str = "tab:decision_voi",
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
                "records": self.dataframe.to_dict(orient="records"),
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def evaluate_decision_voi(
    y_true: np.ndarray | Sequence[float],
    lower: np.ndarray | Sequence[float],
    upper: np.ndarray | Sequence[float],
    *,
    point: np.ndarray | Sequence[float] | None = None,
    taus: Sequence[float] | np.ndarray | None = None,
    n_taus: int = 9,
    parameter: str | int = "D",
    direction: str = "less",
    cost_ratio: float = 0.20,
    alpha: float = 0.10,
    calibrated_intervals: tuple[np.ndarray, np.ndarray] | None = None,
    seed: int = DEFAULT_SEED,
) -> DecisionVoIReport:
    """Evaluate clinical threshold utility regret, decision–calibration gap, and second-order VoI.

    Parameters
    ----------
    y_true : (n,) or (n, 3) array-like
        Ground-truth parameter values. If 2D ``(n, 3)``, ``parameter`` selects
        the column (``"D"``, ``"Dstar"``/``"D_star"``, or ``"f"``).
    lower, upper : (n,) or (n, 3) array-like
        Evaluated ``(1 - alpha)`` lower and upper prediction interval bounds.
    point : (n,) or (n, 3) array-like or None, optional
        Optional point predictions (defaults to interval midpoint).
    taus : sequence of float or None, optional
        Clinical action thresholds ``tau`` in physical parameter units. Defaults
        to ``n_taus`` interior quantiles (15th to 85th percentile) of ``y_true``.
    direction : {"less", "greater"}, default="less"
        Clinical positive condition: ``y <= tau`` (``"less"``, e.g. restricted
        ADC/``D`` in malignancy or hypoperfusion ``f``) or ``y >= tau`` (``"greater"``).
    cost_ratio : float, default=0.20
        Relative false-positive harm ``c in (0, 1)`` vs. false-negative harm
        ``1 - c`` (Bayes optimal probability cutoff ``p* = c``).
    alpha : float, default=0.10
        Interval miscoverage level corresponding to ``(lower, upper)``.
    calibrated_intervals : (lo_cal, hi_cal) tuple or None, optional
        Reference calibrated intervals for pricing ``second_order_voi`` (the
        expected regret reduction from uncertainty calibration). When ``None``,
        a two-fold cross-fitted :class:`~ivimbench.metrics.coverage.SplitConformalResidual`
        reference is constructed deterministically from ``(point, y_true)``.
    seed : int, default=42
        Deterministic ``PCG64`` seed for cross-fitted conformal reference.
    """
    if not (0.0 < float(cost_ratio) < 1.0):
        raise ValueError(f"`cost_ratio` must lie strictly in (0, 1); got {cost_ratio}.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"`alpha` must lie strictly in (0, 1); got {alpha}.")

    col_idx: int | None = None
    if isinstance(parameter, int):
        col_idx = parameter
        param_label = PARAM_NAMES[parameter] if 0 <= parameter < 3 else f"param_{parameter}"
    else:
        p_norm = "Dstar" if parameter == "D_star" else parameter
        param_label = p_norm
        if p_norm in PARAM_NAMES:
            col_idx = PARAM_NAMES.index(p_norm)

    def _select_1d(arr_in: Any) -> np.ndarray:
        a = np.asarray(arr_in, dtype=np.float64)
        if a.ndim == 2:
            if col_idx is None:
                raise ValueError(f"Cannot index 2D array with parameter={parameter!r}.")
            return a[:, col_idx].ravel()
        return a.ravel()

    y = _select_1d(y_true)
    lo = _select_1d(lower)
    hi = np.maximum(_select_1d(upper), lo)
    pt = 0.5 * (lo + hi) if point is None else _select_1d(point)
    n = y.size
    if n < 4:
        raise ValueError("`evaluate_decision_voi` requires at least 4 samples.")

    # Reference calibrated intervals for pricing second-order VoI (value of calibration)
    if calibrated_intervals is not None:
        lo_cal = _select_1d(calibrated_intervals[0])
        hi_cal = np.maximum(_select_1d(calibrated_intervals[1]), lo_cal)
    else:
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        perm = rng.permutation(n)
        half = n // 2
        idx1, idx2 = perm[:half], perm[half:]
        lo_cal = np.empty_like(lo)
        hi_cal = np.empty_like(hi)
        scr1 = SplitConformalResidual(alpha=alpha).calibrate(pt[idx1, None], y[idx1, None])
        l2, u2 = scr1.apply(pt[idx2, None])
        lo_cal[idx2], hi_cal[idx2] = l2[:, 0], u2[:, 0]
        scr2 = SplitConformalResidual(alpha=alpha).calibrate(pt[idx2, None], y[idx2, None])
        l1, u1 = scr2.apply(pt[idx1, None])
        lo_cal[idx1], hi_cal[idx1] = l1[:, 0], u1[:, 0]

    if taus is None:
        tau_grid = np.quantile(y, np.linspace(0.15, 0.85, int(n_taus)))
    else:
        tau_grid = np.asarray(taus, dtype=np.float64).ravel()

    c = float(cost_ratio)
    dir_is_less = direction.strip().lower() in ("less", "le", "<=", "<", "below")

    records: list[dict[str, Any]] = []
    for tau_val in tau_grid:
        t = float(tau_val)
        event = (y <= t).astype(np.float64) if dir_is_less else (y >= t).astype(np.float64)

        p_raw, m_raw, sig_raw = infer_exceedance_probability(
            lo, hi, t, point=pt, alpha=alpha, direction=direction
        )
        p_cal, _, sig_cal = infer_exceedance_probability(
            lo_cal, hi_cal, t, point=pt, alpha=alpha, direction=direction
        )

        act_raw = (p_raw >= c).astype(np.float64)
        act_cal = (p_cal >= c).astype(np.float64)

        # Asymmetric decision loss L(a, E) = c * a * (1 - E) + (1 - c) * (1 - a) * E
        # Oracle action a* = E incurs zero classification loss -> regret == realized loss.
        loss_raw = c * act_raw * (1.0 - event) + (1.0 - c) * (1.0 - act_raw) * event
        loss_cal = c * act_cal * (1.0 - event) + (1.0 - c) * (1.0 - act_cal) * event

        regret_raw = float(np.mean(loss_raw))
        regret_cal = float(np.mean(loss_cal))

        # Perceived expected regret under the estimator's own predictive probability p_raw
        perceived_loss = c * act_raw * (1.0 - p_raw) + (1.0 - c) * (1.0 - act_raw) * p_raw
        perceived_regret = float(np.mean(perceived_loss))

        dec_cal_gap = abs(perceived_regret - regret_raw)
        signed_dec_gap = perceived_regret - regret_raw
        exceed_cal_gap = abs(float(np.mean(p_raw) - np.mean(event)))
        brier = float(np.mean((p_raw - event) ** 2))

        # Vickers Decision Curve Net Benefit: TP/N - (c / (1 - c)) * FP/N
        tp_rate = float(np.mean(act_raw * event))
        fp_rate = float(np.mean(act_raw * (1.0 - event)))
        net_benefit = tp_rate - (c / max(1.0 - c, 1e-12)) * fp_rate

        # Second-order VoI: realized utility gain of uncertainty calibration + local quadratic VoI proxy
        voi_second_order = regret_raw - regret_cal
        z_cal = (t - m_raw) / np.maximum(sig_cal, 1e-12)
        quad_voi = float(
            np.mean(
                0.5
                * np.abs(sig_cal**2 - sig_raw**2)
                * (np.abs(z_cal) / np.maximum(sig_cal**2, 1e-18))
                * stats.norm.pdf(z_cal)
                * min(c, 1.0 - c)
            )
        )

        straddle = float(np.mean((lo <= t) & (t <= hi)))

        records.append(
            {
                "parameter": param_label,
                "tau": t,
                "prevalence": float(np.mean(event)),
                "action_rate": float(np.mean(act_raw)),
                "straddle_rate": straddle,
                "expected_utility": -regret_raw,
                "expected_regret": regret_raw,
                "perceived_regret": perceived_regret,
                "decision_calibration_gap": dec_cal_gap,
                "signed_decision_cal_gap": signed_dec_gap,
                "exceedance_cal_gap": exceed_cal_gap,
                "brier_score": brier,
                "calibrated_regret": regret_cal,
                "evpi": regret_cal,
                "second_order_voi": voi_second_order,
                "quadratic_voi_proxy": quad_voi,
                "net_benefit": net_benefit,
            }
        )

    df = pd.DataFrame.from_records(records)
    summary: dict[str, float | str] = {
        "parameter": param_label,
        "mean_expected_regret": float(df["expected_regret"].mean()),
        "mean_calibrated_regret": float(df["calibrated_regret"].mean()),
        "mean_decision_calibration_gap": float(df["decision_calibration_gap"].mean()),
        "max_decision_calibration_gap": float(df["decision_calibration_gap"].max()),
        "mean_second_order_voi": float(df["second_order_voi"].mean()),
        "max_second_order_voi": float(df["second_order_voi"].max()),
        "mean_evpi": float(df["evpi"].mean()),
        "mean_net_benefit": float(df["net_benefit"].mean()),
    }
    return DecisionVoIReport(
        dataframe=df,
        summary=summary,
        metadata={
            "parameter": param_label,
            "direction": direction,
            "cost_ratio": c,
            "alpha": float(alpha),
            "n_voxels": int(n),
            "n_taus": int(tau_grid.size),
        },
    )


def compare_decision_calibration(
    y_true: np.ndarray | Sequence[float],
    methods: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    point_preds: dict[str, np.ndarray] | None = None,
    taus: Sequence[float] | np.ndarray | None = None,
    parameter: str | int = "D",
    direction: str = "less",
    cost_ratio: float = 0.20,
    alpha: float = 0.10,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Compare multiple interval predictors across clinical action thresholds ``taus``."""
    rows: list[dict[str, Any]] = []
    for name, (lo, hi) in methods.items():
        pt = None if point_preds is None else point_preds.get(name)
        rep = evaluate_decision_voi(
            y_true,
            lo,
            hi,
            point=pt,
            taus=taus,
            parameter=parameter,
            direction=direction,
            cost_ratio=cost_ratio,
            alpha=alpha,
            seed=seed,
        )
        rows.append({"method": name, **rep.summary})
    return pd.DataFrame.from_records(rows)


# --------------------------------------------------------------------------- #
# Two-sample observable domain-shift / staleness check
# --------------------------------------------------------------------------- #
def _mahalanobis_two_sample(
    x_ref: np.ndarray,
    x_tgt: np.ndarray,
    *,
    ridge_frac: float = 1e-3,
    outlier_quantile: float = 0.95,
    n_perm: int = 250,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Compute mean-shift Mahalanobis^2, excess outlier rate, and permutation p-value."""
    mu_ref = np.mean(x_ref, axis=0)
    mu_tgt = np.mean(x_tgt, axis=0)
    d = x_ref.shape[1]

    cov_ref = np.atleast_2d(np.cov(x_ref, rowvar=False))
    cov_tgt = np.atleast_2d(np.cov(x_tgt, rowvar=False))
    n1, n2 = x_ref.shape[0], x_tgt.shape[0]
    cov_pool = ((n1 - 1) * cov_ref + (n2 - 1) * cov_tgt) / max(n1 + n2 - 2, 1)

    tr = float(np.trace(cov_pool)) / max(d, 1)
    reg = max(ridge_frac * tr, 1e-10)
    cov_reg = cov_pool + reg * np.eye(d, dtype=np.float64)
    inv_cov = np.linalg.pinv(cov_reg)

    diff = mu_tgt - mu_ref
    mahal_sq = float(diff @ inv_cov @ diff)

    # Reference-anchored per-sample Mahalanobis outlier rate
    cov_ref_reg = cov_ref + max(ridge_frac * (float(np.trace(cov_ref)) / max(d, 1)), 1e-10) * np.eye(d)
    inv_ref = np.linalg.pinv(cov_ref_reg)
    xc_ref = x_ref - mu_ref[None, :]
    xc_tgt = x_tgt - mu_ref[None, :]
    d2_ref = np.einsum("ni,ij,nj->n", xc_ref, inv_ref, xc_ref)
    d2_tgt = np.einsum("ni,ij,nj->n", xc_tgt, inv_ref, xc_tgt)
    thresh = float(np.quantile(d2_ref, outlier_quantile))
    outlier_rate = float(np.mean(d2_tgt > thresh))
    excess_outlier = outlier_rate - (1.0 - float(outlier_quantile))

    # Permutation p-value on two-sample mean + scale summary statistic
    pooled = np.vstack([x_ref, x_tgt])
    n_tot = pooled.shape[0]
    obs_stat = mahal_sq + float(np.sum((np.std(x_tgt, axis=0) - np.std(x_ref, axis=0)) ** 2) / max(tr, 1e-10))

    exceed_count = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(n_tot)
        g1 = pooled[perm[:n1]]
        g2 = pooled[perm[n1:]]
        dm = np.mean(g2, axis=0) - np.mean(g1, axis=0)
        stat_p = float(dm @ inv_cov @ dm) + float(
            np.sum((np.std(g2, axis=0) - np.std(g1, axis=0)) ** 2) / max(tr, 1e-10)
        )
        if stat_p >= obs_stat - 1e-15:
            exceed_count += 1
    p_val = float((exceed_count + 1) / (int(n_perm) + 1))
    return mahal_sq, outlier_rate, excess_outlier, p_val


@dataclass(frozen=True)
class DomainShiftReport:
    """Two-sample observable signal-space and residual-space staleness report."""

    signal_mahalanobis_sq: float
    signal_outlier_rate: float
    signal_excess_outlier_rate: float
    signal_pvalue: float
    residual_mahalanobis_sq: float
    residual_rms_ratio: float
    residual_outlier_rate: float
    residual_pvalue: float
    is_stale: bool
    staleness_reasons: tuple[str, ...]
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_mahalanobis_sq": float(self.signal_mahalanobis_sq),
            "signal_outlier_rate": float(self.signal_outlier_rate),
            "signal_excess_outlier_rate": float(self.signal_excess_outlier_rate),
            "signal_pvalue": float(self.signal_pvalue),
            "residual_mahalanobis_sq": float(self.residual_mahalanobis_sq),
            "residual_rms_ratio": float(self.residual_rms_ratio),
            "residual_outlier_rate": float(self.residual_outlier_rate),
            "residual_pvalue": float(self.residual_pvalue),
            "is_stale": bool(self.is_stale),
            "staleness_reasons": list(self.staleness_reasons),
            "alpha": float(self.alpha),
        }


def _observable_residuals(signals: np.ndarray, bvalues: np.ndarray) -> np.ndarray:
    """Compute normalized bi-exponential fit residuals solely from observed DWI signals."""
    est = SegmentedEstimator(protocol=bvalues, uncertainty="homoscedastic")
    res = est.solve(signals)
    b0_idx = int(np.argmin(bvalues))
    s0 = np.maximum(signals[:, b0_idx], 1e-4)
    y_norm = signals / s0[:, None]
    pred_norm = ivim_biexp(bvalues, res.params[:, 0], res.params[:, 1], res.params[:, 2], s0=1.0)
    return y_norm - pred_norm


def check_domain_shift(
    signals_ref: np.ndarray,
    signals_target: np.ndarray,
    *,
    bvalues: Sequence[float] | np.ndarray | str | AcquisitionProtocol | None = None,
    alpha: float = 0.02,
    outlier_quantile: float = 0.95,
    rms_drift_tol: float = 0.25,
    n_perm: int = 250,
    seed: int = DEFAULT_SEED,
) -> DomainShiftReport:
    """Two-sample observable domain-shift / calibration-staleness detector.

    Operates strictly on observable multi-b DWI signals without requiring
    ground-truth ``(D, D*, f)`` labels, combining:
    1. **Signal-space Mahalanobis drift**: Detects anatomical prior shifts,
       protocol/TE mismatches, and global ``S0`` gain shifts.
    2. **Residual-space Mahalanobis & RMS drift**: Fits a fast reference
       bi-exponential curve per voxel and tests normalized fit residuals
       ``r = S/S0 - S_biexp(theta_hat)`` (and squared residual energy ``r^2``) to
       detect SNR changes, receiver-coil drift, and non-bi-exponential forward
       physics mismatch (e.g., diffusional kurtosis or tri-exponential flow).
    """
    s_ref = np.atleast_2d(np.asarray(signals_ref, dtype=np.float64))
    s_tgt = np.atleast_2d(np.asarray(signals_target, dtype=np.float64))
    if s_ref.shape[1] != s_tgt.shape[1]:
        raise ValueError(
            f"`signals_ref` ({s_ref.shape}) and `signals_target` ({s_tgt.shape}) must have matching b-axis."
        )
    if s_ref.shape[0] < 8 or s_tgt.shape[0] < 8:
        raise ValueError("`check_domain_shift` requires at least 8 voxels per sample.")

    n_b = s_ref.shape[1]
    if bvalues is not None:
        b_arr, _ = resolve_bvalues(bvalues)
    elif n_b == 12:
        b_arr, _ = resolve_bvalues("osipi")
    elif n_b == 4:
        b_arr, _ = resolve_bvalues("tcga_lihc")
    else:
        b_arr = np.linspace(0.0, 800.0, n_b, dtype=np.float64)

    rng = np.random.Generator(np.random.PCG64(int(seed)))

    # 1. Observable signal-space features: [log(S0), normalized decay curve]
    b0_idx = int(np.argmin(b_arr))
    s0_ref = np.maximum(s_ref[:, b0_idx : b0_idx + 1], 1e-4)
    s0_tgt = np.maximum(s_tgt[:, b0_idx : b0_idx + 1], 1e-4)
    # Exclude exact b0 column from normalized ratio if normalized to 1.0 (avoid rank-deficiency),
    # and prepend log(s0) to capture absolute amplitude / gain drift.
    non_b0 = [k for k in range(n_b) if k != b0_idx] or [0]
    feat_sig_ref = np.hstack([np.log(s0_ref), (s_ref / s0_ref)[:, non_b0]])
    feat_sig_tgt = np.hstack([np.log(s0_tgt), (s_tgt / s0_tgt)[:, non_b0]])

    sig_m2, sig_out, sig_ex_out, sig_p = _mahalanobis_two_sample(
        feat_sig_ref,
        feat_sig_tgt,
        outlier_quantile=outlier_quantile,
        n_perm=n_perm,
        rng=rng,
    )

    # 2. Observable forward-model residual-space features: [residuals, squared residuals]
    res_ref = _observable_residuals(s_ref, b_arr)
    res_tgt = _observable_residuals(s_tgt, b_arr)
    rms_ref = max(float(np.sqrt(np.mean(res_ref**2))), 1e-12)
    rms_tgt = float(np.sqrt(np.mean(res_tgt**2)))
    rms_ratio = rms_tgt / rms_ref

    feat_res_ref = np.hstack([res_ref, res_ref**2])
    feat_res_tgt = np.hstack([res_tgt, res_tgt**2])
    res_m2, res_out, _, res_p = _mahalanobis_two_sample(
        feat_res_ref,
        feat_res_tgt,
        outlier_quantile=outlier_quantile,
        n_perm=n_perm,
        rng=rng,
    )

    reasons: list[str] = []
    if sig_p <= float(alpha) and (sig_m2 > 0.12 or sig_ex_out > 0.08):
        reasons.append("signal_space_shift")
    if res_p <= float(alpha) and (res_m2 > 0.12 or abs(rms_ratio - 1.0) > float(rms_drift_tol)):
        reasons.append("residual_space_shift")
    if abs(rms_ratio - 1.0) > float(rms_drift_tol) and "residual_space_shift" not in reasons:
        reasons.append("residual_rms_drift")

    return DomainShiftReport(
        signal_mahalanobis_sq=sig_m2,
        signal_outlier_rate=sig_out,
        signal_excess_outlier_rate=sig_ex_out,
        signal_pvalue=sig_p,
        residual_mahalanobis_sq=res_m2,
        residual_rms_ratio=rms_ratio,
        residual_outlier_rate=res_out,
        residual_pvalue=res_p,
        is_stale=len(reasons) > 0,
        staleness_reasons=tuple(reasons),
        alpha=float(alpha),
    )
