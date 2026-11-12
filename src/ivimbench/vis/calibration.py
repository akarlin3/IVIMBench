"""Calibration reliability curves and D*-tercile conditional coverage plots.

Preserves full backwards compatibility with legacy ``ivimbench.vis`` while adding
the publication 2-panel :func:`plot_reliability_and_terciles` comparison across
``honest_crlb``, ``floored_railed``, and conformalized intervals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..metrics.coverage import (
    DEFAULT_NOMINAL_LEVELS,
    DEFAULT_QUANTILE_LEVELS,
    DEFAULT_RAILED_SD_FLOOR,
    apply_sd_convention,
    central_interval_calibration_curve,
    gaussian_calibration_curve,
    tercile_stratified_coverage,
)
from ..phantoms.synthetic import PARAM_NAMES

__all__ = [
    "plot_calibration_curves",
    "plot_tercile_coverage",
    "plot_reliability_and_terciles",
]


def _require_matplotlib() -> Any:
    try:
        import matplotlib

        if not matplotlib.get_backend():  # pragma: no cover
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Visualization requires matplotlib (and optionally seaborn). "
            "Install via: pip install 'ivimbench[vis]'"
        ) from exc
    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="notebook")
    except ImportError:  # pragma: no cover
        pass
    return plt


def _save_figure(fig: Any, out_path: str | Path | None, dpi: int = 200) -> None:
    if out_path is not None:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=dpi, bbox_inches="tight")


def _extract_y_true_matrix(
    y_true: Any,
    param_names: Sequence[str] = PARAM_NAMES,
) -> np.ndarray:
    if hasattr(y_true, "to_voxels") and callable(y_true.to_voxels):
        _, params = y_true.to_voxels(foreground_only=True)
        arr = np.asarray(params, dtype=np.float64)
    elif hasattr(y_true, "params"):
        arr = np.asarray(y_true.params, dtype=np.float64)
    elif isinstance(y_true, Mapping):
        cols = []
        for p in param_names:
            if p in y_true:
                cols.append(np.asarray(y_true[p], dtype=np.float64).ravel())
            elif p == "Dstar" and "D_star" in y_true:
                cols.append(np.asarray(y_true["D_star"], dtype=np.float64).ravel())
            else:
                raise KeyError(f"Missing parameter {p!r} in y_true mapping.")
        arr = np.column_stack(cols)
    else:
        arr = np.asarray(y_true, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(param_names):
        raise ValueError(f"Expected y_true shape (n, {len(param_names)}), got {arr.shape}.")
    return arr


def plot_calibration_curves(
    y_true: Any,
    predictions: np.ndarray | tuple[np.ndarray, np.ndarray] | Mapping[str, Any],
    *,
    q_levels: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS,
    nominal_levels: Sequence[float] | np.ndarray = DEFAULT_NOMINAL_LEVELS,
    mode: str = "quantile",
    param_names: Sequence[str] = PARAM_NAMES,
    axes: Sequence[Any] | None = None,
    figsize: tuple[float, float] = (12.0, 4.0),
    save_path: str | Path | None = None,
    out_path: str | Path | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot 3-panel reliability curves (Empirical vs. Nominal Coverage) for ``(D, D*, f)``."""
    plt = _require_matplotlib()
    truth = _extract_y_true_matrix(y_true, param_names=param_names)
    n_params = len(param_names)

    if isinstance(predictions, Mapping):
        pred_map: dict[str, Any] = dict(predictions)
    else:
        pred_map = {"Model": predictions}

    if axes is None:
        fig, ax_arr = plt.subplots(1, n_params, figsize=figsize, sharey=True)
        ax_arr = np.atleast_1d(ax_arr)
    else:
        ax_arr = np.asarray(axes)
        fig = ax_arr[0].figure

    levels_arr = np.asarray(q_levels, dtype=np.float64)
    nominals_arr = np.asarray(nominal_levels, dtype=np.float64)
    display_titles = {"D": "D (tissue)", "Dstar": "D* (pseudo-diffusion)", "f": "f (perfusion)"}

    for j, pname in enumerate(param_names):
        ax = ax_arr[j]
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", lw=1.1, label="Ideal")

        for label, pred in pred_map.items():
            if isinstance(pred, tuple) and len(pred) == 2:
                pt = np.asarray(pred[0], dtype=np.float64)
                sd = np.asarray(pred[1], dtype=np.float64)
                curve = gaussian_calibration_curve(
                    truth[:, j], pt[:, j], sd[:, j], nominal_levels=nominals_arr
                )
                nom_x = curve["nominal"]
                emp_y = curve["empirical"]
                ece_val = float(curve["ece"])
            else:
                q_arr = np.asarray(pred, dtype=np.float64)
                if mode == "central":
                    curve = central_interval_calibration_curve(
                        truth[:, j], q_arr[:, j, :], levels_arr, nominal_levels=nominals_arr
                    )
                    nom_x = curve["nominal"]
                    emp_y = curve["empirical"]
                    ece_val = float(curve["ece"])
                else:
                    emp_y = np.array(
                        [np.mean(truth[:, j] <= q_arr[:, j, k]) for k in range(levels_arr.size)],
                        dtype=np.float64,
                    )
                    nom_x = levels_arr
                    ece_val = float(np.mean(np.abs(emp_y - nom_x)))

            ax.plot(
                nom_x,
                emp_y,
                marker="o",
                markersize=4.5,
                lw=1.8,
                label=f"{label} (ECE={ece_val:.3f})",
            )

        ax.set_title(display_titles.get(pname, pname))
        ax.set_xlabel("Nominal probability")
        if j == 0:
            ax.set_ylabel("Empirical coverage")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, out_path if out_path is not None else save_path, dpi=150)
    return fig, ax_arr


def plot_tercile_coverage(
    data: Any,
    *,
    lower: np.ndarray | Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    upper: np.ndarray | None = None,
    alpha: float = 0.10,
    parameter: str = "Dstar",
    ax: Any | None = None,
    figsize: tuple[float, float] = (7.2, 4.5),
    save_path: str | Path | None = None,
    out_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot conditional empirical coverage stratified by low/mid/high ``D*`` terciles."""
    plt = _require_matplotlib()

    if hasattr(data, "to_dataframe"):
        df = data.to_dataframe()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        truth = _extract_y_true_matrix(data)
        dstar_cond = truth[:, 1]
        p_idx = list(PARAM_NAMES).index(parameter)
        nominal = 1.0 - float(alpha)

        if isinstance(lower, Mapping):
            intervals_map = dict(lower)
        elif lower is not None and upper is not None:
            intervals_map = {"Model": (lower, upper)}
        else:
            raise ValueError(
                "Provide either a BenchmarkReport/DataFrame or `y_true` with `lower` and `upper`."
            )

        rows = []
        for m_name, (lo_m, hi_m) in intervals_map.items():
            lo_arr = np.asarray(lo_m, dtype=np.float64)
            hi_arr = np.asarray(hi_m, dtype=np.float64)
            lo_p = lo_arr[:, p_idx] if lo_arr.ndim == 2 else lo_arr
            hi_p = hi_arr[:, p_idx] if hi_arr.ndim == 2 else hi_arr
            strat = tercile_stratified_coverage(
                truth[:, p_idx], lo_p, hi_p, conditioning=dstar_cond, nominal=nominal
            )
            rows.append(
                {
                    "method": m_name,
                    "parameter": parameter,
                    "nominal_coverage": nominal,
                    "tercile_cov_low": strat["low"].coverage,
                    "tercile_cov_mid": strat["mid"].coverage,
                    "tercile_cov_high": strat["high"].coverage,
                }
            )
        df = pd.DataFrame(rows)

    sub = df[df["parameter"] == parameter]
    if sub.empty:
        raise ValueError(f"No rows found for parameter {parameter!r} in benchmark table.")

    methods = list(sub["method"])
    nominal_val = float(sub["nominal_coverage"].iloc[0])
    x = np.arange(3, dtype=np.float64)
    width = 0.78 / max(len(methods), 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for i, (_, row) in enumerate(sub.iterrows()):
        vals = [
            float(row["tercile_cov_low"]),
            float(row["tercile_cov_mid"]),
            float(row["tercile_cov_high"]),
        ]
        offset = (i - 0.5 * (len(methods) - 1)) * width
        ax.bar(x + offset, vals, width=width, label=str(row["method"]))

    ax.axhline(
        nominal_val,
        linestyle="--",
        color="black",
        lw=1.2,
        label=f"Nominal ({nominal_val:.2f})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Low D* Tercile", "Mid D* Tercile", "High D* Tercile"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(f"Conditional coverage ({parameter})")
    ax.set_title(f"{parameter} Coverage Across Ground-Truth D* Terciles")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, out_path if out_path is not None else save_path, dpi=150)
    return fig, ax


def _build_three_convention_bundle(
    cohort_or_dro: Any,
    *,
    alpha: float = 0.10,
    railed_sd_floor: float = DEFAULT_RAILED_SD_FLOOR,
    nominal_levels: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute ``honest_crlb``, ``floored_railed``, and split-``conformal`` intervals on a cohort/DRO."""
    from ..registry import run_estimator

    if hasattr(cohort_or_dro, "to_voxels") and callable(cohort_or_dro.to_voxels):
        signals, truth = cohort_or_dro.to_voxels(foreground_only=True)
    else:
        signals = np.asarray(cohort_or_dro.signals, dtype=np.float64)
        truth = np.asarray(cohort_or_dro.params, dtype=np.float64)
    bvalues = np.asarray(cohort_or_dro.bvalues, dtype=np.float64)
    snr_raw = getattr(cohort_or_dro, "snr", 30.0)
    snr_val = float(np.mean(snr_raw)) if snr_raw is not None else 30.0

    n = truth.shape[0]
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    perm = rng.permutation(n)
    n_cal = max(min(n // 2, n - 6), 6)
    cal_idx = perm[:n_cal]
    test_idx = perm[n_cal:] if n > n_cal else perm

    out_honest = run_estimator(
        "crlb", signals, bvalues, alpha=alpha, snr=snr_val, sd_convention="honest_crlb"
    )
    pt = np.asarray(out_honest.point, dtype=np.float64)
    z_nom = float(stats.norm.ppf(1.0 - 0.5 * alpha))
    sd_honest = (
        np.asarray(out_honest.std, dtype=np.float64)
        if out_honest.std is not None
        else np.maximum((out_honest.upper - out_honest.lower) / (2.0 * z_nom), 1e-12)
    )
    sd_floored = apply_sd_convention(
        sd_honest,
        sd_convention="floored_railed",
        point_pred=pt,
        railed_sd_floor=railed_sd_floor,
    )

    # Studentized split-conformal calibration across nominal levels
    sd_safe = np.maximum(np.minimum(sd_honest, 1.0), 1e-8)
    cal_scores = np.abs(truth[cal_idx] - pt[cal_idx]) / sd_safe[cal_idx]
    q_mult_alpha = np.array(
        [
            float(
                np.quantile(
                    cal_scores[:, j],
                    min(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal, 1.0),
                    method="higher",
                )
            )
            for j in range(3)
        ],
        dtype=np.float64,
    )

    eval_y = truth[test_idx]
    eval_pt = pt[test_idx]
    eval_sd_h = sd_honest[test_idx]
    eval_sd_f = sd_floored[test_idx]
    eval_sd_c = sd_safe[test_idx]

    methods_curves: dict[str, dict[str, dict[str, Any]]] = {
        "honest_crlb": {},
        "floored_railed": {},
        "conformal": {},
    }
    tercile_rows: list[dict[str, Any]] = []
    nominal = 1.0 - float(alpha)

    for j, pname in enumerate(PARAM_NAMES):
        c_h = gaussian_calibration_curve(
            eval_y[:, j], eval_pt[:, j], eval_sd_h[:, j], nominal_levels=nominal_levels
        )
        c_f = gaussian_calibration_curve(
            eval_y[:, j], eval_pt[:, j], eval_sd_f[:, j], nominal_levels=nominal_levels
        )
        emp_conf = []
        for nom_l in nominal_levels:
            q_level = min(np.ceil((n_cal + 1) * float(nom_l)) / n_cal, 1.0)
            q_j = float(np.quantile(cal_scores[:, j], q_level, method="higher"))
            lo_l = eval_pt[:, j] - q_j * eval_sd_c[:, j]
            hi_l = eval_pt[:, j] + q_j * eval_sd_c[:, j]
            emp_conf.append(float(np.mean((eval_y[:, j] >= lo_l) & (eval_y[:, j] <= hi_l))))
        emp_conf_arr = np.asarray(emp_conf, dtype=np.float64)
        methods_curves["honest_crlb"][pname] = c_h
        methods_curves["floored_railed"][pname] = c_f
        methods_curves["conformal"][pname] = {
            "nominal": nominal_levels,
            "empirical": emp_conf_arr,
            "ece": float(np.mean(np.abs(emp_conf_arr - nominal_levels))),
        }

        intervals_at_alpha = {
            "honest_crlb": (
                eval_pt[:, j] - z_nom * eval_sd_h[:, j],
                eval_pt[:, j] + z_nom * eval_sd_h[:, j],
            ),
            "floored_railed": (
                eval_pt[:, j] - z_nom * eval_sd_f[:, j],
                eval_pt[:, j] + z_nom * eval_sd_f[:, j],
            ),
            "conformal": (
                eval_pt[:, j] - q_mult_alpha[j] * eval_sd_c[:, j],
                eval_pt[:, j] + q_mult_alpha[j] * eval_sd_c[:, j],
            ),
        }
        for m_label, (lo_j, hi_j) in intervals_at_alpha.items():
            strat = tercile_stratified_coverage(
                eval_y[:, j], lo_j, hi_j, conditioning=eval_y[:, 1], nominal=nominal
            )
            tercile_rows.append(
                {
                    "method": m_label,
                    "parameter": pname,
                    "nominal_coverage": nominal,
                    "marginal_coverage": float(np.mean((eval_y[:, j] >= lo_j) & (eval_y[:, j] <= hi_j))),
                    "tercile_cov_low": strat["low"].coverage,
                    "tercile_cov_mid": strat["mid"].coverage,
                    "tercile_cov_high": strat["high"].coverage,
                }
            )

    return {
        "curves": methods_curves,
        "dataframe": pd.DataFrame.from_records(tercile_rows),
        "nominal": nominal,
    }


def _approx_reliability_from_marginal(
    nominal_cov: float,
    empirical_cov: float,
    nominal_levels: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Smooth normal-scale reliability curve passing through ``(nominal_cov, empirical_cov)``."""
    nom_clamped = float(np.clip(nominal_cov, 1e-3, 1.0 - 1e-3))
    emp_clamped = float(np.clip(empirical_cov, 1e-4, 1.0 - 1e-4))
    z_nom = float(stats.norm.ppf(0.5 * (1.0 + nom_clamped)))
    z_emp = float(stats.norm.ppf(0.5 * (1.0 + emp_clamped)))
    scale = z_emp / max(z_nom, 1e-8)
    z_grid = stats.norm.ppf(0.5 * (1.0 + np.clip(nominal_levels, 1e-4, 1.0 - 1e-4)))
    emp_curve = np.asarray(2.0 * stats.norm.cdf(scale * z_grid) - 1.0, dtype=np.float64)
    ece = float(np.mean(np.abs(emp_curve - nominal_levels)))
    return emp_curve, ece


def plot_reliability_and_terciles(
    eval_result: Any,
    out_path: str | Path | None = None,
    *,
    parameter: str = "Dstar",
    nominal_levels: Sequence[float] | np.ndarray = (0.20, 0.40, 0.50, 0.68, 0.80, 0.90, 0.95),
    alpha: float = 0.10,
    seed: int = 42,
    figsize: tuple[float, float] = (12.2, 4.8),
    dpi: int = 200,
    save_path: str | Path | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot a 2-panel reliability diagram and ``D*``-tercile conditional coverage bar chart.

    Compares ``honest_crlb``, ``floored_railed``, and conformalized intervals (or any
    collection of evaluated methods supplied in ``eval_result``).

    Parameters
    ----------
    eval_result : SyntheticCohort, DigitalReferenceObject, BenchmarkReport, MultiTaskBenchmarkResult, DataFrame, or Mapping
        Benchmark evaluation object, cohort/DRO, or dictionary mapping method names to
        :class:`~ivimbench.evaluate.BenchmarkReport` / ``(point, std)`` / curve dicts.
    out_path : str, Path, or None, optional
        Output image path (PNG/PDF/SVG).
    parameter : {"D", "Dstar", "f"}, default="Dstar"
        Target IVIM parameter to display across both panels.
    """
    plt = _require_matplotlib()
    nom_arr = np.asarray(nominal_levels, dtype=np.float64)
    p_name = "Dstar" if parameter == "D_star" else parameter

    curves_by_method: dict[str, dict[str, Any]] = {}
    df_terciles: pd.DataFrame

    # Case 1: Cohort or DRO -> evaluate honest_crlb, floored_railed, and conformal directly
    if hasattr(eval_result, "signals") and hasattr(eval_result, "bvalues"):
        bundle = _build_three_convention_bundle(
            eval_result,
            alpha=alpha,
            nominal_levels=nom_arr,
            seed=seed,
        )
        for m_name, p_map in bundle["curves"].items():
            curves_by_method[m_name] = p_map[p_name]
        df_terciles = bundle["dataframe"]
    elif isinstance(eval_result, Mapping) and "curves" in eval_result and "dataframe" in eval_result:
        raw_curves = eval_result["curves"]
        for m_name, c_entry in raw_curves.items():
            curves_by_method[str(m_name)] = c_entry[p_name] if p_name in c_entry else c_entry
        df_terciles = pd.DataFrame(eval_result["dataframe"])
    elif isinstance(eval_result, Mapping) and "y_true" in eval_result and "methods" in eval_result:
        truth = _extract_y_true_matrix(eval_result["y_true"])
        p_idx = list(PARAM_NAMES).index(p_name)
        nominal = 1.0 - float(alpha)
        rows = []
        for m_name, spec in dict(eval_result["methods"]).items():
            if isinstance(spec, tuple) and len(spec) == 2:
                pt_m = np.asarray(spec[0], dtype=np.float64)
                sd_m = np.asarray(spec[1], dtype=np.float64)
                z_crit = float(stats.norm.ppf(1.0 - 0.5 * alpha))
                lo_p = pt_m[:, p_idx] - z_crit * sd_m[:, p_idx]
                hi_p = pt_m[:, p_idx] + z_crit * sd_m[:, p_idx]
                curves_by_method[str(m_name)] = gaussian_calibration_curve(
                    truth[:, p_idx], pt_m[:, p_idx], sd_m[:, p_idx], nominal_levels=nom_arr
                )
            elif isinstance(spec, Mapping):
                lo_arr = np.asarray(spec["lower"], dtype=np.float64)
                hi_arr = np.asarray(spec["upper"], dtype=np.float64)
                lo_p = lo_arr[:, p_idx] if lo_arr.ndim == 2 else lo_arr
                hi_p = hi_arr[:, p_idx] if hi_arr.ndim == 2 else hi_arr
                marg_c = float(np.mean((truth[:, p_idx] >= lo_p) & (truth[:, p_idx] <= hi_p)))
                if "std" in spec and spec["std"] is not None and "point" in spec:
                    pt_arr = np.asarray(spec["point"], dtype=np.float64)
                    sd_arr = np.asarray(spec["std"], dtype=np.float64)
                    curves_by_method[str(m_name)] = gaussian_calibration_curve(
                        truth[:, p_idx], pt_arr[:, p_idx], sd_arr[:, p_idx], nominal_levels=nom_arr
                    )
                else:
                    emp_c, ece_c = _approx_reliability_from_marginal(nominal, marg_c, nom_arr)
                    curves_by_method[str(m_name)] = {
                        "nominal": nom_arr,
                        "empirical": emp_c,
                        "ece": ece_c,
                    }
            else:
                raise TypeError(f"Unsupported method specification for {m_name!r}.")

            strat = tercile_stratified_coverage(
                truth[:, p_idx], lo_p, hi_p, conditioning=truth[:, 1], nominal=nominal
            )
            rows.append(
                {
                    "method": str(m_name),
                    "parameter": p_name,
                    "nominal_coverage": nominal,
                    "marginal_coverage": float(np.mean((truth[:, p_idx] >= lo_p) & (truth[:, p_idx] <= hi_p))),
                    "tercile_cov_low": strat["low"].coverage,
                    "tercile_cov_mid": strat["mid"].coverage,
                    "tercile_cov_high": strat["high"].coverage,
                }
            )
        df_terciles = pd.DataFrame.from_records(rows)
    else:
        # BenchmarkReport, MultiTaskBenchmarkResult, DataFrame, or {method: BenchmarkReport}
        if isinstance(eval_result, pd.DataFrame):
            raw_df = eval_result.copy()
        elif hasattr(eval_result, "dataframe") and isinstance(eval_result.dataframe, pd.DataFrame):
            raw_df = eval_result.dataframe.copy()
        elif isinstance(eval_result, Mapping) and "records" in eval_result:
            raw_df = pd.DataFrame(eval_result["records"])
        elif isinstance(eval_result, Mapping):
            frames = []
            for m_name, rep in eval_result.items():
                df_r = rep.to_dataframe() if hasattr(rep, "to_dataframe") else pd.DataFrame(rep)
                df_r = df_r.copy()
                df_r["method"] = str(m_name)
                frames.append(df_r)
            raw_df = pd.concat(frames, ignore_index=True)
        else:
            raise TypeError(
                "Unsupported `eval_result` type for `plot_reliability_and_terciles`."
            )

        sub_p = raw_df[raw_df["parameter"] == p_name]
        if sub_p.empty:
            raise ValueError(f"No benchmark rows found for parameter {p_name!r}.")

        agg_rows = []
        for m_name, grp in sub_p.groupby("method", sort=False):
            nom_c = (
                float(grp["nominal_coverage"].mean())
                if "nominal_coverage" in grp.columns
                else (1.0 - float(alpha))
            )
            marg_c = float(grp["marginal_coverage"].mean())
            emp_curve, ece_est = _approx_reliability_from_marginal(nom_c, marg_c, nom_arr)
            ece_val = float(grp["ece"].mean()) if "ece" in grp.columns and np.isfinite(grp["ece"].mean()) else ece_est
            curves_by_method[str(m_name)] = {
                "nominal": nom_arr,
                "empirical": emp_curve,
                "ece": ece_val,
                "anchor": (nom_c, marg_c),
            }
            agg_rows.append(
                {
                    "method": str(m_name),
                    "parameter": p_name,
                    "nominal_coverage": nom_c,
                    "marginal_coverage": marg_c,
                    "tercile_cov_low": float(grp["tercile_cov_low"].mean()),
                    "tercile_cov_mid": float(grp["tercile_cov_mid"].mean()),
                    "tercile_cov_high": float(grp["tercile_cov_high"].mean()),
                }
            )
        df_terciles = pd.DataFrame.from_records(agg_rows)

    sub = df_terciles[df_terciles["parameter"] == p_name]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_rel, ax_bar = axes[0], axes[1]

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    # Left panel: Reliability diagram
    ax_rel.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#222222", lw=1.2, label="Ideal (y = x)")
    for idx, (m_name, c_info) in enumerate(curves_by_method.items()):
        color = palette[idx % len(palette)]
        nom_x = np.asarray(c_info["nominal"], dtype=np.float64)
        emp_y = np.asarray(c_info["empirical"], dtype=np.float64)
        ece_v = float(c_info.get("ece", np.mean(np.abs(emp_y - nom_x))))
        ax_rel.plot(
            nom_x,
            emp_y,
            marker="o",
            markersize=4.8,
            lw=2.0,
            color=color,
            label=f"{m_name} (ECE={ece_v:.3f})",
        )
        if "anchor" in c_info:
            ax_rel.scatter(
                [c_info["anchor"][0]],
                [c_info["anchor"][1]],
                s=52,
                color=color,
                edgecolors="black",
                zorder=5,
            )

    ax_rel.set_xlim(0.0, 1.0)
    ax_rel.set_ylim(0.0, 1.03)
    ax_rel.set_xlabel("Nominal coverage probability")
    ax_rel.set_ylabel(f"Empirical coverage ({p_name})")
    ax_rel.set_title(f"(a) {p_name} Reliability Diagram")
    ax_rel.legend(fontsize=8.5, loc="lower right")
    ax_rel.grid(alpha=0.3)

    # Right panel: D*-tercile conditional coverage bar chart
    methods = list(sub["method"])
    nom_target = float(sub["nominal_coverage"].iloc[0]) if not sub.empty else (1.0 - float(alpha))
    x = np.arange(3, dtype=np.float64)
    bar_w = 0.78 / max(len(methods), 1)

    for idx, (_, row) in enumerate(sub.iterrows()):
        vals = [
            float(row["tercile_cov_low"]),
            float(row["tercile_cov_mid"]),
            float(row["tercile_cov_high"]),
        ]
        offset = (idx - 0.5 * (len(methods) - 1)) * bar_w
        ax_bar.bar(
            x + offset,
            vals,
            width=bar_w,
            color=palette[idx % len(palette)],
            alpha=0.90,
            edgecolor="white",
            label=str(row["method"]),
        )

    ax_bar.axhline(
        nom_target,
        linestyle="--",
        color="#222222",
        lw=1.3,
        label=f"Nominal ({nom_target:.2f})",
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["Low D* Tercile", "Mid D* Tercile", "High D* Tercile"])
    ax_bar.set_ylim(0.0, 1.06)
    ax_bar.set_ylabel(f"Conditional coverage ({p_name})")
    ax_bar.set_title(f"(b) {p_name} Conditional Coverage by D* Tercile")
    ax_bar.legend(fontsize=8.5, loc="lower left")
    ax_bar.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, out_path if out_path is not None else save_path, dpi=dpi)
    return fig, np.asarray(axes)
