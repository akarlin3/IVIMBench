"""Visualization utilities for IVIM uncertainty calibration and tercile coverage.

Requires the optional ``vis`` extra (``pip install ivimbench[vis]``). Imports
``matplotlib`` and ``seaborn`` lazily so core ``ivimbench`` remains lightweight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics.coverage import (
    DEFAULT_NOMINAL_LEVELS,
    DEFAULT_QUANTILE_LEVELS,
    central_interval_calibration_curve,
    gaussian_calibration_curve,
    tercile_stratified_coverage,
)
from .phantoms.synthetic import PARAM_NAMES

__all__ = ["plot_calibration_curves", "plot_tercile_coverage"]


def _require_matplotlib() -> Any:
    try:
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


def _extract_y_true_matrix(
    y_true: Any,
    param_names: Sequence[str] = PARAM_NAMES,
) -> np.ndarray:
    if hasattr(y_true, "params"):
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
) -> tuple[Any, np.ndarray]:
    """Plot 3-panel reliability curves (Empirical vs. Nominal Coverage) for ``(D, D*, f)``.

    Parameters
    ----------
    y_true : SyntheticCohort, (n, 3) ndarray, or dict
        Ground-truth IVIM parameters.
    predictions : (n, 3, L) ndarray, (point, std) tuple, or dict of {method: pred}
        Predicted quantile arrays ``(n, 3, L)`` or Gaussian ``(point, std)`` tuples.
    q_levels : sequence of float, default=DEFAULT_QUANTILE_LEVELS
        Quantile levels corresponding to the trailing axis when quantile arrays are given.
    nominal_levels : sequence of float, default=(0.50, 0.68, 0.80, 0.90, 0.95)
        Central confidence levels used when ``(point, std)`` tuples are passed or
        ``mode="central"``.
    mode : {"quantile", "central"}, default="quantile"
        For 3D quantile arrays, plot either full CDF quantile calibration
        (``"quantile"``) or central confidence interval coverage (``"central"``).
    param_names : sequence of str, default=("D", "Dstar", "f")
        Parameter titles for the subplot panels.
    axes : sequence of matplotlib Axes or None, optional
        Existing subplot axes of length ``len(param_names)``.
    figsize : tuple of float, default=(12.0, 4.0)
        Figure size when creating a new figure.
    save_path : str, Path, or None, optional
        Optional output path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
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
    if save_path is not None:
        fig.savefig(Path(save_path), dpi=150, bbox_inches="tight")
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
) -> tuple[Any, Any]:
    """Plot conditional empirical coverage stratified by low/mid/high ``D*`` terciles.

    Accepts either:
    1. A :class:`~ivimbench.evaluate.BenchmarkReport` or :class:`pandas.DataFrame`
       produced by :class:`~ivimbench.evaluate.BenchmarkSuite`, or
    2. Ground-truth ``data`` (``SyntheticCohort`` / ``(n, 3)`` array) alongside
       ``lower`` and ``upper`` interval arrays (or ``lower`` as a mapping
       ``{method_name: (lo_arr, hi_arr)}``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
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
    if save_path is not None:
        fig.savefig(Path(save_path), dpi=150, bbox_inches="tight")
    return fig, ax
