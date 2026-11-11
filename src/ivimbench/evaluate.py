"""Unified evaluation harness (``BenchmarkSuite``) for IVIM uncertainty calibration.

Consolidates Datum's submission/scoring harness and Caliper's calibration ruler
into a single class :class:`BenchmarkSuite` that scores point estimates,
prediction intervals, or quantile arrays against ground-truth IVIM parameters
and returns both a structured dictionary (:class:`BenchmarkReport`) and a tidy
:class:`pandas.DataFrame`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .baselines import CRLBNormalEstimator, MLPEstimator, NLLSEstimator, SegmentedEstimator
from .metrics.coverage import (
    DEFAULT_NOMINAL_LEVELS,
    DEFAULT_QUANTILE_LEVELS,
    bootstrap_coverage_ci,
    central_interval,
    ece_quantile,
    empirical_coverage,
    gaussian_calibration_curve,
    interval_score,
    pinball_loss,
    snr_stratified_coverage,
    tercile_stratified_coverage,
)
from .metrics.railing import (
    DEFAULT_PARAM_BOUNDS,
    DEFAULT_RAIL_TOL,
    evaluate_parameter_railing,
)
from .metrics.sharpness import (
    absolute_width,
    crlb_dstar_wall_ratio,
    crlb_efficiency_ratio,
    normalized_width,
    relative_width,
)
from .phantoms.dro import DigitalReferenceObject
from .phantoms.synthetic import DEFAULT_SEED, PARAM_NAMES, SyntheticCohort
from .protocols import AcquisitionProtocol, resolve_bvalues

__all__ = ["BenchmarkReport", "BenchmarkSuite"]


def _coerce_param_matrix(
    obj: Any,
    param_names: Sequence[str] = PARAM_NAMES,
    name: str = "array",
) -> np.ndarray:
    """Convert a cohort, DRO, dict, or 2D array into an ``(n, 3)`` float64 matrix."""
    if isinstance(obj, SyntheticCohort):
        return np.asarray(obj.params, dtype=np.float64)
    if isinstance(obj, DigitalReferenceObject):
        _, params = obj.to_voxels(foreground_only=True)
        return np.asarray(params, dtype=np.float64)
    if isinstance(obj, Mapping):
        cols = []
        for p in param_names:
            if p in obj:
                cols.append(np.asarray(obj[p], dtype=np.float64).ravel())
            elif p == "Dstar" and "D_star" in obj:
                cols.append(np.asarray(obj["D_star"], dtype=np.float64).ravel())
            else:
                raise KeyError(f"Missing parameter {p!r} in {name}.")
        return np.column_stack(cols)
    arr = np.asarray(obj, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(param_names):
        raise ValueError(
            f"Expected {name} to have shape (n, {len(param_names)}), got {arr.shape}."
        )
    return arr


@dataclass
class BenchmarkReport(Mapping[str, Any]):
    """Dictionary-compatible benchmark report with DataFrame and text-table views.

    Behaves as a standard read-only dictionary (``report["Dstar"]``,
    ``report["summary"]``, ``dict(report)``) while exposing
    :meth:`to_dataframe` / :attr:`dataframe` for tabular analysis.
    """

    method: str
    protocol: str
    alpha: float
    nominal_coverage: float
    n_voxels: int
    parameters: dict[str, dict[str, Any]]
    railing: dict[str, Any]
    crlb_dstar_wall_ratio: float
    dataframe: pd.DataFrame

    def _as_mapping(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "protocol": self.protocol,
            "alpha": self.alpha,
            "nominal_coverage": self.nominal_coverage,
            "n_voxels": self.n_voxels,
            "parameters": self.parameters,
            "D": self.parameters.get("D", {}),
            "Dstar": self.parameters.get("Dstar", {}),
            "f": self.parameters.get("f", {}),
            "railing": self.railing,
            "crlb_dstar_wall_ratio": self.crlb_dstar_wall_ratio,
            "dataframe": self.dataframe,
        }

    def __getitem__(self, key: str) -> Any:
        mapping = self._as_mapping()
        if key in mapping:
            return mapping[key]
        raise KeyError(f"Unknown key {key!r} in BenchmarkReport.")

    def __iter__(self) -> Iterator[str]:
        return iter(self._as_mapping())

    def __len__(self) -> int:
        return len(self._as_mapping())

    @property
    def df(self) -> pd.DataFrame:
        """Shorthand alias for :attr:`dataframe`."""
        return self.dataframe

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the per-parameter benchmark :class:`pandas.DataFrame`."""
        return self.dataframe.copy()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation (excluding DataFrame object)."""
        out = self._as_mapping()
        out.pop("dataframe", None)
        out["records"] = self.dataframe.to_dict(orient="records")
        return out

    def format_table(self) -> str:
        """Render a concise fixed-width summary table of core benchmark metrics."""
        cols = [
            "method",
            "parameter",
            "marginal_coverage",
            "marginal_coverage_gap",
            "tercile_cov_high",
            "railing_pct",
            "sharpness_to_crlb_ratio",
            "crlb_efficiency",
        ]
        present = [c for c in cols if c in self.dataframe.columns]
        return self.dataframe[present].to_string(index=False, float_format=lambda v: f"{v:.3f}")


class BenchmarkSuite:
    """Standardized evaluation harness for IVIM parameter and uncertainty predictions.

    Evaluates marginal coverage, ``D*``-tercile and SNR-stratified conditional
    coverage gaps, boundary-railing percentages with bootstrap CIs, absolute and
    normalized interval widths, and sharpness/efficiency ratios relative to the
    Cramér-Rao lower bound (CRLB).

    Parameters
    ----------
    protocol : str, AcquisitionProtocol, or sequence of float, default="osipi"
        Default acquisition b-value schedule used for CRLB calculations when not
        inferred directly from a :class:`~ivimbench.phantoms.SyntheticCohort`.
    snr : float or sequence of float, default=30.0
        Default signal-to-noise ratio at ``b = 0`` used for CRLB calculations when
        not carried by ``y_true``.
    alpha : float, default=0.10
        Central prediction interval miscoverage rate (``0.10`` -> 90% nominal
        coverage).
    bounds : mapping of {param: (lo, hi)} or None, optional
        Parameter bounds used to classify boundary railing and compute domain-span
        normalized widths. Defaults to :data:`DEFAULT_PARAM_BOUNDS`.
    rail_tol : float, default=1e-2
        Relative tolerance as a fraction of ``hi - lo`` for boundary railing.
    n_boot : int, default=1000
        Number of bootstrap resamples for coverage and railing confidence intervals.
    seed : int, default=42
        Deterministic RNG seed for bootstrap confidence intervals.
    include_s0_in_crlb : bool, default=False
        Whether to include ``S0`` as a 4th unknown parameter in CRLB Fisher matrices.
    """

    def __init__(
        self,
        protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray = "osipi",
        snr: float | Sequence[float] | np.ndarray = 30.0,
        *,
        alpha: float = 0.10,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        rail_tol: float = DEFAULT_RAIL_TOL,
        n_boot: int = 1000,
        seed: int = DEFAULT_SEED,
        include_s0_in_crlb: bool = False,
    ) -> None:
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"`alpha` must lie strictly in (0, 1); got {alpha}.")
        self.bvalues, self.protocol_name = resolve_bvalues(protocol)
        self.snr = snr
        self.alpha = float(alpha)
        self.nominal = 1.0 - float(alpha)
        active_bounds = dict(DEFAULT_PARAM_BOUNDS)
        if bounds is not None:
            active_bounds.update(bounds)
        self.bounds = active_bounds
        self.rail_tol = float(rail_tol)
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        self.include_s0_in_crlb = bool(include_s0_in_crlb)
        self.param_names = PARAM_NAMES

    def _resolve_context(
        self,
        y_true: Any,
        bvalues: Sequence[float] | np.ndarray | None,
        snr: float | np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, str, float | np.ndarray]:
        truth = _coerce_param_matrix(y_true, param_names=self.param_names, name="y_true")
        if bvalues is not None:
            b_arr, proto_label = resolve_bvalues(bvalues)
        elif isinstance(y_true, (SyntheticCohort, DigitalReferenceObject)):
            b_arr = np.asarray(y_true.bvalues, dtype=np.float64)
            proto_label = str(y_true.protocol)
        else:
            b_arr = self.bvalues
            proto_label = self.protocol_name

        if snr is not None:
            snr_arr = np.asarray(snr, dtype=np.float64)
            snr_eff: float | np.ndarray = float(snr_arr.item()) if snr_arr.ndim == 0 else snr_arr
        elif isinstance(y_true, (SyntheticCohort, DigitalReferenceObject)) and y_true.snr is not None:
            snr_y = np.asarray(y_true.snr, dtype=np.float64)
            snr_eff = float(snr_y.item()) if snr_y.ndim == 0 else snr_y
        else:
            snr_s = np.asarray(self.snr, dtype=np.float64)
            snr_eff = float(snr_s.item()) if snr_s.ndim == 0 else snr_s
        return truth, b_arr, proto_label, snr_eff

    def evaluate(
        self,
        y_true: Any,
        *,
        point_pred: Any | None = None,
        intervals: tuple[Any, Any] | np.ndarray | None = None,
        lower: Any | None = None,
        upper: Any | None = None,
        std_pred: Any | None = None,
        q_pred: np.ndarray | None = None,
        q_levels: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS,
        estimator: Any | None = None,
        signals: np.ndarray | None = None,
        method: str = "model",
        bvalues: Sequence[float] | np.ndarray | None = None,
        snr: float | np.ndarray | None = None,
        as_dataframe: bool = False,
    ) -> BenchmarkReport | pd.DataFrame:
        """Run all UQ calibration, sharpness/CRLB, and railing metrics on predictions.

        Supports four flexible prediction formats:
        1. ``point_pred`` + ``intervals=(lower, upper)`` (or ``lower=..., upper=...``),
        2. ``point_pred`` + ``std_pred`` (symmetric Gaussian error bars),
        3. ``q_pred`` ``(n, 3, L)`` + ``q_levels`` (quantile regression / posterior draws),
        4. ``estimator`` object exposing ``predict_quantiles`` or ``predict_intervals``
           (reads signals from ``signals`` or from ``y_true`` if ``y_true`` is a cohort/DRO).

        Parameters
        ----------
        y_true : SyntheticCohort, DigitalReferenceObject, (n, 3) ndarray, or dict
            Ground-truth IVIM parameters ``(D, Dstar, f)``.
        as_dataframe : bool, default=False
            If ``True``, returns the :class:`pandas.DataFrame` directly instead of
            a :class:`BenchmarkReport`.

        Returns
        -------
        BenchmarkReport or pandas.DataFrame
        """
        truth, b_arr, proto_label, snr_eff = self._resolve_context(y_true, bvalues, snr)
        n = truth.shape[0]
        levels_arr = np.asarray(q_levels, dtype=np.float64)

        if estimator is not None:
            if signals is None:
                if isinstance(y_true, SyntheticCohort):
                    signals = y_true.signals
                elif isinstance(y_true, DigitalReferenceObject):
                    signals, _ = y_true.to_voxels(foreground_only=True)
                else:
                    raise ValueError("`signals` must be provided when scoring an `estimator` with raw y_true.")
            if hasattr(estimator, "predict_quantiles") and not isinstance(estimator, str):
                q_pred = estimator.predict_quantiles(signals, levels_arr)
            elif hasattr(estimator, "predict_intervals") and not isinstance(estimator, str):
                lower, upper = estimator.predict_intervals(signals, alpha=self.alpha)
                if hasattr(estimator, "predict"):
                    point_pred = estimator.predict(signals)
            else:
                from .registry import run_estimator

                norm_out = run_estimator(
                    estimator,
                    signals,
                    b_arr,
                    alpha=self.alpha,
                    bounds=self.bounds,
                )
                point_pred = norm_out.point
                lower, upper = norm_out.lower, norm_out.upper
                if norm_out.std is not None:
                    std_pred = norm_out.std

        # Resolve (point_mat, lower_mat, upper_mat)
        q_mat: np.ndarray | None = None
        std_mat: np.ndarray | None = None

        if q_pred is not None:
            q_mat = np.asarray(q_pred, dtype=np.float64)
            if q_mat.shape != (n, 3, levels_arr.size):
                raise ValueError(
                    f"Expected `q_pred` shape {(n, 3, levels_arr.size)}, got {q_mat.shape}."
                )
            lo_mat, hi_mat = central_interval(q_mat, levels_arr, alpha=self.alpha)
            if point_pred is None:
                mid_idx = int(np.argmin(np.abs(levels_arr - 0.50)))
                pt_mat = q_mat[:, :, mid_idx]
            else:
                pt_mat = _coerce_param_matrix(point_pred, self.param_names, "point_pred")
        else:
            if intervals is not None:
                if isinstance(intervals, tuple) and len(intervals) == 2:
                    lower, upper = intervals
                else:
                    iv_arr = np.asarray(intervals, dtype=np.float64)
                    if iv_arr.ndim == 3 and iv_arr.shape == (n, 3, 2):
                        lower, upper = iv_arr[:, :, 0], iv_arr[:, :, 1]
                    else:
                        raise ValueError(
                            "Expected `intervals` to be a (lower, upper) tuple or (n, 3, 2) array."
                        )

            if lower is not None and upper is not None:
                lo_mat = _coerce_param_matrix(lower, self.param_names, "lower")
                hi_mat = _coerce_param_matrix(upper, self.param_names, "upper")
                pt_mat = (
                    0.5 * (lo_mat + hi_mat)
                    if point_pred is None
                    else _coerce_param_matrix(point_pred, self.param_names, "point_pred")
                )
            elif point_pred is not None and std_pred is not None:
                pt_mat = _coerce_param_matrix(point_pred, self.param_names, "point_pred")
                std_mat = _coerce_param_matrix(std_pred, self.param_names, "std_pred")
                z_crit = float(stats.norm.ppf(1.0 - 0.5 * self.alpha))
                lo_mat = pt_mat - z_crit * std_mat
                hi_mat = pt_mat + z_crit * std_mat
            else:
                raise ValueError(
                    "Must supply one of: (`intervals` / `lower`+`upper`), "
                    "(`point_pred` + `std_pred`), `q_pred`, or `estimator`."
                )

        if np.any(hi_mat < lo_mat - 1e-12):
            raise ValueError("Predicted `upper` interval bounds must be >= `lower` bounds.")
        hi_mat = np.maximum(hi_mat, lo_mat)

        # 1. Boundary railing diagnostics
        railing_res = evaluate_parameter_railing(
            pt_mat,
            bounds=self.bounds,
            param_names=self.param_names,
            rel_tol=self.rail_tol,
            n_boot=self.n_boot,
            seed=self.seed,
        )

        # 2. Sharpness & CRLB efficiency
        norm_widths = np.asarray(
            normalized_width(lo_mat, hi_mat, bounds=self.bounds, param_names=self.param_names),
            dtype=np.float64,
        )
        rel_widths = np.asarray(relative_width(lo_mat, hi_mat, truth), dtype=np.float64)
        crlb_res = crlb_efficiency_ratio(
            lo_mat,
            hi_mat,
            bvalues=b_arr,
            y_true=truth,
            snr=snr_eff,
            alpha=self.alpha,
            include_s0=self.include_s0_in_crlb,
        )
        wall_ratio = crlb_dstar_wall_ratio(
            b_arr, truth, snr=snr_eff, include_s0=self.include_s0_in_crlb
        )

        # 3. Per-parameter coverage, conditional coverage, proper scores, and accuracy
        dstar_true = truth[:, 1]
        has_snr_array = np.ndim(snr_eff) == 1 and np.asarray(snr_eff).size == n

        param_details: dict[str, dict[str, Any]] = {}
        records: list[dict[str, Any]] = []

        for j, pname in enumerate(self.param_names):
            yj = truth[:, j]
            ptj = pt_mat[:, j]
            loj = lo_mat[:, j]
            hij = hi_mat[:, j]

            marg_cov = empirical_coverage(yj, loj, hij)
            marg_ci = bootstrap_coverage_ci(
                yj,
                loj,
                hij,
                nominal=self.nominal,
                n_boot=self.n_boot,
                seed=self.seed + 100 + j,
            )

            # Conditional coverage across true D* terciles (identifiability-wall probe)
            terciles = tercile_stratified_coverage(
                yj, loj, hij, conditioning=dstar_true, nominal=self.nominal
            )
            t_low, t_mid, t_high = terciles["low"], terciles["mid"], terciles["high"]
            max_abs_t_gap = max(
                abs(t_low.coverage_gap),
                abs(t_mid.coverage_gap),
                abs(t_high.coverage_gap),
            )
            hi_lo_width_ratio = (
                float(t_high.mean_width / t_low.mean_width)
                if t_low.mean_width > 0.0
                else float("nan")
            )

            snr_cond = (
                {
                    k: v.to_dict()
                    for k, v in snr_stratified_coverage(
                        yj, loj, hij, snr=np.asarray(snr_eff), nominal=self.nominal
                    ).items()
                }
                if has_snr_array
                else {}
            )

            # Calibration error (ECE) & pinball
            if q_mat is not None:
                ece_val = ece_quantile(yj, q_mat[:, j, :], levels_arr)
                mean_pinball = float(
                    np.mean(
                        [
                            np.mean(pinball_loss(yj, q_mat[:, j, k], float(levels_arr[k])))
                            for k in range(levels_arr.size)
                        ]
                    )
                )
            elif std_mat is not None:
                g_curve = gaussian_calibration_curve(
                    yj, ptj, std_mat[:, j], nominal_levels=DEFAULT_NOMINAL_LEVELS
                )
                ece_val = float(g_curve["ece"])
                mean_pinball = float("nan")
            else:
                ece_val = abs(marg_cov - self.nominal)
                mean_pinball = float("nan")

            iscore = float(np.mean(interval_score(loj, hij, yj, alpha=self.alpha)))
            abs_w = float(absolute_width(loj, hij))
            err = ptj - yj
            rmse = float(np.sqrt(np.mean(err * err)))
            mae = float(np.mean(np.abs(err)))
            bias = float(np.mean(err))

            r_info = railing_res[pname]
            detail = {
                "method": method,
                "parameter": pname,
                "n_voxels": n,
                "nominal_coverage": self.nominal,
                "marginal_coverage": marg_cov,
                "marginal_coverage_gap": marg_cov - self.nominal,
                "coverage_ci_lo": marg_ci["coverage_ci_lo"],
                "coverage_ci_hi": marg_ci["coverage_ci_hi"],
                "tercile_cov_low": t_low.coverage,
                "tercile_cov_mid": t_mid.coverage,
                "tercile_cov_high": t_high.coverage,
                "tercile_gap_low": t_low.coverage_gap,
                "tercile_gap_mid": t_mid.coverage_gap,
                "tercile_gap_high": t_high.coverage_gap,
                "max_abs_tercile_gap": max_abs_t_gap,
                "ece": ece_val,
                "interval_score": iscore,
                "mean_pinball": mean_pinball,
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "abs_width": abs_w,
                "norm_width": float(norm_widths[j]),
                "rel_width": float(rel_widths[j]),
                "crlb_width": float(crlb_res.mean_crlb_width[j]),
                "sharpness_to_crlb_ratio": float(crlb_res.sharpness_to_crlb_ratio[j]),
                "crlb_efficiency": float(crlb_res.crlb_efficiency[j]),
                "high_to_low_width_ratio": hi_lo_width_ratio,
                "railing_pct": float(r_info["railed_pct"]),
                "railing_lower_pct": float(r_info["lower_pct"]),
                "railing_upper_pct": float(r_info["upper_pct"]),
                "railing_ci_lo_pct": float(r_info["pct_ci_lo"]),
                "railing_ci_hi_pct": float(r_info["pct_ci_hi"]),
            }
            records.append(detail)
            param_details[pname] = {
                **detail,
                "terciles": {k: v.to_dict() for k, v in terciles.items()},
                "snr_strata": snr_cond,
            }

        df = pd.DataFrame.from_records(records)
        if as_dataframe:
            return df

        return BenchmarkReport(
            method=method,
            protocol=proto_label,
            alpha=self.alpha,
            nominal_coverage=self.nominal,
            n_voxels=n,
            parameters=param_details,
            railing={
                k: {rk: rv for rk, rv in v.items() if rk != "ci"}
                for k, v in railing_res.items()
            },
            crlb_dstar_wall_ratio=wall_ratio,
            dataframe=df,
        )

    __call__ = evaluate

    def compare(
        self,
        y_true: Any,
        methods: Mapping[str, Mapping[str, Any]],
        *,
        bvalues: Sequence[float] | np.ndarray | None = None,
        snr: float | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Evaluate multiple prediction sets side-by-side and return a combined DataFrame.

        Parameters
        ----------
        y_true : SyntheticCohort, DigitalReferenceObject, or (n, 3) ndarray
            Shared ground-truth cohort.
        methods : mapping of {method_name: kwargs_for_evaluate}
            Dictionary mapping each method's display name to a dict of keyword
            arguments accepted by :meth:`evaluate` (e.g.,
            ``{"MyNet": {"point_pred": pt, "intervals": (lo, hi)}}``).
        """
        frames: list[pd.DataFrame] = []
        for name, kw in methods.items():
            rep = self.evaluate(
                y_true,
                method=name,
                bvalues=bvalues,
                snr=snr,
                as_dataframe=True,
                **kw,
            )
            assert isinstance(rep, pd.DataFrame)
            frames.append(rep)
        return pd.concat(frames, ignore_index=True)

    def run_baselines(
        self,
        cohort: SyntheticCohort | DigitalReferenceObject,
        include: Sequence[str] = ("segmented", "nlls", "mlp"),
    ) -> pd.DataFrame:
        """Run the standard IVIMBench reference baselines on ``cohort``."""
        if isinstance(cohort, DigitalReferenceObject):
            signals, _ = cohort.to_voxels(foreground_only=True)
            b_arr = cohort.bvalues
            organ = cohort.organ
            snr_val = float(cohort.snr or 30.0)
        else:
            signals = cohort.signals
            b_arr = cohort.bvalues
            organ = cohort.organ
            snr_val = float(np.mean(cohort.snr))

        frames: list[pd.DataFrame] = []
        for key in include:
            norm = key.strip().lower()
            if norm == "segmented":
                est: Any = SegmentedEstimator(protocol=b_arr, bounds=self.bounds)
                label = "Segmented"
            elif norm == "nlls":
                est = NLLSEstimator(protocol=b_arr, bounds=self.bounds)
                label = "NLLS"
            elif norm in ("crlb", "crlb_normal"):
                est = CRLBNormalEstimator(protocol=b_arr, bounds=self.bounds)
                label = "CRLBNormal"
            elif norm == "mlp":
                est = MLPEstimator(protocol=b_arr, bounds=self.bounds, seed=self.seed)
                est.fit(organ=organ, snr=snr_val)
                label = "StandardMLP"
            else:
                raise ValueError(
                    f"Unknown baseline {key!r}; choose from 'segmented', 'nlls', 'crlb', 'mlp'."
                )

            df_m = self.evaluate(
                cohort,
                estimator=est,
                signals=signals,
                method=label,
                as_dataframe=True,
            )
            assert isinstance(df_m, pd.DataFrame)
            frames.append(df_m)
        return pd.concat(frames, ignore_index=True)
