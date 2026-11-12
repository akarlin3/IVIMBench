"""Publication diagnostic figures for misspecification aliasing, Vernier Pareto schedules, and DRO slices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..baselines import SegmentedEstimator
from ..metrics.railing import DEFAULT_PARAM_BOUNDS, DEFAULT_RAIL_TOL, classify_railing
from .calibration import _require_matplotlib, _save_figure

__all__ = [
    "plot_misspecification_aliasing",
    "plot_vernier_pareto_frontier",
    "plot_dro_slice_montage",
]

DEFAULT_ALIASING_FAMILIES: tuple[str, ...] = (
    "biexp",
    "dispersion_gamma",
    "dispersion_lognormal",
    "stretched",
    "triexp",
)


def _coerce_misspec_summary(misspec_result: Any) -> tuple[pd.DataFrame, float]:
    """Normalize MisspecificationReport, MultiTaskBenchmarkResult, DataFrame, or dict into compartment summary."""
    nominal = 0.90
    if hasattr(misspec_result, "compartment_summary") and isinstance(
        misspec_result.compartment_summary, pd.DataFrame
    ):
        meta = getattr(misspec_result, "metadata", {}) or {}
        nominal = float(meta.get("nominal_coverage", 0.90))
        return misspec_result.compartment_summary.copy(), nominal

    if isinstance(misspec_result, Mapping):
        meta = dict(misspec_result.get("metadata", {}))
        nominal = float(meta.get("nominal_coverage", 0.90))
        if "compartment_summary" in misspec_result:
            return pd.DataFrame(misspec_result["compartment_summary"]), nominal
        if "records" in misspec_result:
            misspec_result = pd.DataFrame(misspec_result["records"])

    if hasattr(misspec_result, "dataframe") and isinstance(misspec_result.dataframe, pd.DataFrame):
        meta = getattr(misspec_result, "metadata", {}) or {}
        nominal = float(meta.get("nominal_coverage", 0.90))
        raw_df = misspec_result.dataframe.copy()
    elif isinstance(misspec_result, pd.DataFrame):
        raw_df = misspec_result.copy()
    else:
        raise TypeError(
            "Unsupported `misspec_result` type for `plot_misspecification_aliasing`."
        )

    if {"D_marginal_cov", "D_worst_cond_cov", "truth_family"}.issubset(raw_df.columns):
        return raw_df, nominal

    fam_col = "truth_family" if "truth_family" in raw_df.columns else "family"
    if fam_col not in raw_df.columns or "parameter" not in raw_df.columns:
        raise ValueError(
            "Expected compartment summary or per-parameter DataFrame with family and parameter columns."
        )

    if "nominal_coverage" in raw_df.columns:
        nominal = float(raw_df["nominal_coverage"].iloc[0])

    rows: list[dict[str, Any]] = []
    for (m_name, fam), grp in raw_df.groupby(["method", fam_col], sort=False):
        d_sub = grp[grp["parameter"] == "D"]
        ds_sub = grp[grp["parameter"] == "Dstar"]
        if d_sub.empty:
            continue
        d_marg = float(d_sub["marginal_coverage"].mean())
        d_worst = float(
            d_sub[["tercile_cov_low", "tercile_cov_mid", "tercile_cov_high"]].min(axis=1).mean()
        )
        ds_high = (
            float(ds_sub["tercile_cov_high"].mean()) if not ds_sub.empty else float("nan")
        )
        rows.append(
            {
                "method": str(m_name),
                "truth_family": str(fam),
                "D_marginal_cov": d_marg,
                "D_worst_cond_cov": d_worst,
                "Dstar_high_tercile_cov": ds_high,
            }
        )
    return pd.DataFrame.from_records(rows), nominal


def plot_misspecification_aliasing(
    misspec_result: Any,
    out_path: str | Path | None = None,
    *,
    families: Sequence[str] | None = DEFAULT_ALIASING_FAMILIES,
    figsize: tuple[float, float] = (12.4, 4.9),
    dpi: int = 200,
    save_path: str | Path | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot Procrustes marginal vs. conditional coverage curves on tissue ``D`` across forward families.

    Separates apparent global/marginal coverage on slow tissue diffusion ``D`` from
    worst-tercile conditional coverage degradation across ``dispersion_gamma``,
    ``dispersion_lognormal``, ``stretched``, and ``triexp`` forward signal physics.
    """
    plt = _require_matplotlib()
    comp_df, nominal = _coerce_misspec_summary(misspec_result)

    available_fams = list(dict.fromkeys(comp_df["truth_family"].astype(str)))
    if families is not None:
        ordered = [f for f in families if f in available_fams]
        if not ordered:
            ordered = available_fams
    else:
        ordered = available_fams

    short_labels = {
        "biexp": "biexp\n(matched)",
        "dispersion_gamma": "disp.\ngamma",
        "dispersion_lognormal": "disp.\nlognormal",
        "stretched": "stretched\n(CTRW)",
        "triexp": "triexp\n(3-comp)",
        "kurtosis": "kurtosis\n(DKI)",
        "heavy_tail": "heavy-tail\n(Student-t)",
    }
    x = np.arange(len(ordered), dtype=np.float64)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    ax_d, ax_gap = axes[0], axes[1]

    for idx, (m_name, grp) in enumerate(comp_df.groupby("method", sort=False)):
        color = palette[idx % len(palette)]
        indexed = grp.set_index("truth_family")
        present_idx = [i for i, f in enumerate(ordered) if f in indexed.index]
        present_fams = [ordered[i] for i in present_idx]
        xs = x[present_idx]

        d_marg = np.array([float(indexed.loc[f, "D_marginal_cov"]) for f in present_fams])
        d_cond = np.array([float(indexed.loc[f, "D_worst_cond_cov"]) for f in present_fams])

        ax_d.plot(
            xs,
            d_marg,
            marker="o",
            lw=2.0,
            color=color,
            linestyle="-",
            label=f"{m_name} (marginal D)",
        )
        ax_d.plot(
            xs,
            d_cond,
            marker="s",
            lw=1.8,
            color=color,
            linestyle="--",
            alpha=0.88,
            label=f"{m_name} (worst-tercile D)",
        )
        ax_d.fill_between(xs, d_cond, d_marg, color=color, alpha=0.10)

        ax_gap.plot(
            xs,
            d_cond,
            marker="s",
            lw=2.0,
            color=color,
            linestyle="-",
            label=f"{m_name} (tissue D cond.)",
        )
        if "Dstar_high_tercile_cov" in indexed.columns:
            ds_cond = np.array(
                [float(indexed.loc[f, "Dstar_high_tercile_cov"]) for f in present_fams]
            )
            ax_gap.plot(
                xs,
                ds_cond,
                marker="^",
                lw=1.7,
                color=color,
                linestyle=":",
                alpha=0.85,
                label=f"{m_name} (high-D* cond.)",
            )

    for ax in (ax_d, ax_gap):
        ax.axhline(
            nominal,
            linestyle="--",
            color="#222222",
            lw=1.2,
            label=f"Nominal ({nominal:.2f})",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([short_labels.get(f, f) for f in ordered], fontsize=8.5)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)

    ax_d.set_ylabel("Empirical coverage")
    ax_d.set_title("(a) Procrustes Marginal vs. Conditional Coverage on Tissue D")
    ax_d.legend(fontsize=7.8, loc="lower left")

    ax_gap.set_title("(b) Compartment Aliasing: Tissue D vs. High-D* Tercile")
    ax_gap.legend(fontsize=7.8, loc="lower left")

    fig.tight_layout()
    _save_figure(fig, out_path if out_path is not None else save_path, dpi=dpi)
    return fig, np.asarray(axes)


def _pareto_frontier_indices(delta_sharp: np.ndarray, delta_cond: np.ndarray) -> np.ndarray:
    """Return sorted indices on the non-dominated (min delta_sharp, max delta_cond) Pareto front."""
    n = delta_sharp.size
    is_optimal = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_or_eq = (delta_sharp[j] <= delta_sharp[i] + 1e-12) and (
                delta_cond[j] >= delta_cond[i] - 1e-12
            )
            strictly_better = (delta_sharp[j] < delta_sharp[i] - 1e-12) or (
                delta_cond[j] > delta_cond[i] + 1e-12
            )
            if better_or_eq and strictly_better:
                is_optimal[i] = False
                break
    front_idx = np.flatnonzero(is_optimal)
    order = np.argsort(delta_sharp[front_idx], kind="mergesort")
    return front_idx[order]


def plot_vernier_pareto_frontier(
    scheme_results: Any,
    out_path: str | Path | None = None,
    *,
    ax: Any | None = None,
    figsize: tuple[float, float] = (8.0, 5.4),
    dpi: int = 200,
    save_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot sharpness (``delta_sharp``) vs. high-``D*`` conditional coverage (``delta_cond``) Pareto frontier."""
    plt = _require_matplotlib()

    if hasattr(scheme_results, "summary_dataframe") and isinstance(
        scheme_results.summary_dataframe, pd.DataFrame
    ):
        df = scheme_results.summary_dataframe.copy()
    elif isinstance(scheme_results, pd.DataFrame):
        df = scheme_results.copy()
    elif isinstance(scheme_results, Mapping) and "summary" in scheme_results:
        df = pd.DataFrame(scheme_results["summary"])
    elif isinstance(scheme_results, Sequence):
        df = pd.DataFrame(list(scheme_results))
    else:
        raise TypeError("Unsupported `scheme_results` input for `plot_vernier_pareto_frontier`.")

    if df.empty:
        raise ValueError("`scheme_results` contains zero schedule rows.")

    if "scheme" not in df.columns and "protocol" in df.columns:
        df["scheme"] = df["protocol"]
    if "delta_sharp" not in df.columns:
        ref_w = max(float(df["norm_width"].iloc[0]), 1e-12)
        df["delta_sharp"] = (df["norm_width"] - ref_w) / ref_w
    if "delta_cond" not in df.columns:
        nom = float(df["nominal_coverage"].iloc[0]) if "nominal_coverage" in df.columns else 0.90
        df["delta_cond"] = df["tercile_cov_high"] - nom

    d_sharp = np.asarray(df["delta_sharp"], dtype=np.float64)
    d_cond = np.asarray(df["delta_cond"], dtype=np.float64)
    schemes = [str(s) for s in df["scheme"]]
    front_idx = _pareto_frontier_indices(d_sharp, d_cond)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.axhline(0.0, linestyle="--", color="#333333", lw=1.1, alpha=0.8, label="Nominal high-D* target (Δcond = 0)")
    ax.axvline(0.0, linestyle=":", color="#666666", lw=1.1, alpha=0.75, label="Reference sharpness (Δsharp = 0)")

    is_opt_sched = np.array(
        [("opt" in s.lower() or "vernier" in s.lower() or "cal" in s.lower()) for s in schemes],
        dtype=bool,
    )

    if np.any(~is_opt_sched):
        ax.scatter(
            d_sharp[~is_opt_sched],
            d_cond[~is_opt_sched],
            s=85,
            marker="o",
            color="#1f77b4",
            edgecolors="black",
            lw=0.8,
            zorder=4,
            label="Clinical b-value schedules",
        )
    if np.any(is_opt_sched):
        ax.scatter(
            d_sharp[is_opt_sched],
            d_cond[is_opt_sched],
            s=115,
            marker="D",
            color="#d62728",
            edgecolors="black",
            lw=0.9,
            zorder=5,
            label="Optimized / Vernier schedules",
        )

    if front_idx.size >= 1:
        ax.plot(
            d_sharp[front_idx],
            d_cond[front_idx],
            linestyle="-",
            lw=2.1,
            color="#2ca02c",
            marker="o",
            markersize=4.0,
            zorder=3,
            label="Pareto-optimal frontier",
        )

    for idx, label in enumerate(schemes):
        ax.annotate(
            label,
            xy=(float(d_sharp[idx]), float(d_cond[idx])),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold" if idx in set(front_idx.tolist()) else "normal",
        )

    ax.set_xlabel(r"Relative interval sharpness $\Delta_{\mathrm{sharp}}$ (lower = sharper)")
    ax.set_ylabel(r"High-$D^*$ conditional coverage gap $\Delta_{\mathrm{cond}}$ (0 = nominal)")
    ax.set_title("Vernier Acquisition Pareto Frontier: Sharpness vs. High-D* Coverage")
    ax.legend(fontsize=8.5, loc="best")
    ax.grid(alpha=0.32)

    fig.tight_layout()
    _save_figure(fig, out_path if out_path is not None else save_path, dpi=dpi)
    return fig, ax


def _extract_2d_slice(arr: np.ndarray, slice_idx: int | None) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        idx = a.shape[2] // 2 if slice_idx is None else int(slice_idx)
        return a[:, :, idx]
    raise ValueError(f"Expected 2D or 3D spatial map, got shape {a.shape}.")


def plot_dro_slice_montage(
    dro_or_maps: Any,
    out_path: str | Path | None = None,
    *,
    slice_idx: int | None = None,
    rail_bounds: tuple[float, float] | None = None,
    rail_tol: float = DEFAULT_RAIL_TOL,
    figsize: tuple[float, float] = (13.6, 3.6),
    dpi: int = 200,
    save_path: str | Path | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot a 4-panel spatial montage of ``D``, ``D*``, ``f``, and binary ``D*`` boundary-railing mask."""
    plt = _require_matplotlib()
    ds_bounds = rail_bounds if rail_bounds is not None else DEFAULT_PARAM_BOUNDS["Dstar"]

    if hasattr(dro_or_maps, "D_map") and hasattr(dro_or_maps, "Dstar_map") and hasattr(dro_or_maps, "f_map"):
        d_2d = _extract_2d_slice(dro_or_maps.D_map, slice_idx)
        ds_2d = _extract_2d_slice(dro_or_maps.Dstar_map, slice_idx)
        f_2d = _extract_2d_slice(dro_or_maps.f_map, slice_idx)
        fg_2d = (
            _extract_2d_slice(dro_or_maps.foreground_mask, slice_idx) > 0.5
            if hasattr(dro_or_maps, "foreground_mask")
            else np.isfinite(d_2d)
        )
        if hasattr(dro_or_maps, "signals") and hasattr(dro_or_maps, "bvalues"):
            sig_full = np.asarray(dro_or_maps.signals, dtype=np.float64)
            sig_2d = (
                sig_full
                if sig_full.ndim == 3
                else sig_full[:, :, sig_full.shape[2] // 2 if slice_idx is None else int(slice_idx), :]
            )
            est = SegmentedEstimator(
                protocol=np.asarray(dro_or_maps.bvalues, dtype=np.float64),
                bounds=dict(DEFAULT_PARAM_BOUNDS, Dstar=ds_bounds),
            )
            fg_coords = np.argwhere(fg_2d)
            rail_2d = np.zeros_like(d_2d, dtype=np.float64)
            if fg_coords.shape[0] > 0:
                pt_fg = est.predict(sig_2d[fg_2d])
                railed_fg = classify_railing(pt_fg[:, 1], bounds=ds_bounds, rel_tol=rail_tol)["railed_mask"]
                rail_2d[fg_2d] = railed_fg.astype(np.float64)
        else:
            railed_all = classify_railing(ds_2d.ravel(), bounds=ds_bounds, rel_tol=rail_tol)["railed_mask"]
            rail_2d = np.where(fg_2d, railed_all.reshape(ds_2d.shape).astype(np.float64), 0.0)
    elif isinstance(dro_or_maps, Mapping):
        d_raw = dro_or_maps["D"]
        ds_raw = dro_or_maps["Dstar"] if "Dstar" in dro_or_maps else dro_or_maps["D_star"]
        f_raw = dro_or_maps["f"]
        d_2d = _extract_2d_slice(d_raw, slice_idx)
        ds_2d = _extract_2d_slice(ds_raw, slice_idx)
        f_2d = _extract_2d_slice(f_raw, slice_idx)
        fg_2d = (
            _extract_2d_slice(dro_or_maps["mask"], slice_idx) > 0.5
            if "mask" in dro_or_maps
            else (np.isfinite(d_2d) & ((np.abs(d_2d) + np.abs(ds_2d) + np.abs(f_2d)) > 0.0))
        )
        if "railed_mask" in dro_or_maps:
            rail_2d = _extract_2d_slice(dro_or_maps["railed_mask"], slice_idx)
        elif "Dstar_railed" in dro_or_maps:
            rail_2d = _extract_2d_slice(dro_or_maps["Dstar_railed"], slice_idx)
        else:
            railed_flat = classify_railing(ds_2d.ravel(), bounds=ds_bounds, rel_tol=rail_tol)["railed_mask"]
            rail_2d = np.where(fg_2d, railed_flat.reshape(ds_2d.shape).astype(np.float64), 0.0)
    else:
        raise TypeError("Expected a DigitalReferenceObject or mapping of spatial parameter maps.")

    d_disp = np.where(fg_2d, d_2d * 1e3, np.nan)
    ds_disp = np.where(fg_2d, ds_2d * 1e3, np.nan)
    f_disp = np.where(fg_2d, f_2d, np.nan)
    rail_disp = np.where(fg_2d, rail_2d, np.nan)
    rail_pct = 100.0 * float(np.mean(rail_2d[fg_2d] > 0.5)) if np.any(fg_2d) else 0.0

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    panels = [
        (d_disp, "viridis", "D (×10⁻³ mm²/s)"),
        (ds_disp, "magma", "D* (×10⁻³ mm²/s)"),
        (f_disp, "plasma", "Perfusion f"),
        (rail_disp, "coolwarm", f"D* Railed Mask ({rail_pct:.1f}%)"),
    ]

    for ax, (img, cmap_name, title) in zip(axes, panels, strict=True):
        vmin = 0.0 if "Railed" in title else None
        vmax = 1.0 if "Railed" in title else None
        im = ax.imshow(img, cmap=cmap_name, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    _save_figure(fig, out_path if out_path is not None else save_path, dpi=dpi)
    return fig, np.asarray(axes)
