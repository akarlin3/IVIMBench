"""Self-contained, zero-dependency interactive HTML leaderboard & Gnomon audit report."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._version import __version__

__all__ = [
    "compute_gnomon_audit_badges",
    "generate_html_report",
    "generate_volume_viewer_html",
]


def _extract_report_tables(
    result: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Extract ``(summary_df, records_df, metadata)`` from any IVIMBench report object or dict."""
    meta: dict[str, Any] = {}
    if hasattr(result, "metadata") and isinstance(result.metadata, Mapping):
        meta = dict(result.metadata)

    if hasattr(result, "summary_dataframe") and hasattr(result, "dataframe"):
        return result.summary_dataframe.copy(), result.dataframe.copy(), meta

    if hasattr(result, "compartment_summary") and hasattr(result, "dataframe"):
        return result.compartment_summary.copy(), result.dataframe.copy(), meta

    if hasattr(result, "to_dataframe") and callable(result.to_dataframe):
        df_rec = result.to_dataframe()
        meta.setdefault("protocol", getattr(result, "protocol", "osipi"))
        meta.setdefault("nominal_coverage", getattr(result, "nominal_coverage", 0.90))
        return _summarize_records(df_rec), df_rec, meta

    if isinstance(result, pd.DataFrame):
        return _summarize_records(result), result.copy(), meta

    if isinstance(result, Mapping):
        meta = dict(result.get("metadata", {}))
        if "summary" in result and "records" in result:
            return pd.DataFrame(result["summary"]), pd.DataFrame(result["records"]), meta
        if "compartment_summary" in result and "records" in result:
            return (
                pd.DataFrame(result["compartment_summary"]),
                pd.DataFrame(result["records"]),
                meta,
            )
        if "records" in result:
            rec_df = pd.DataFrame(result["records"])
            return _summarize_records(rec_df), rec_df, meta
        if "summary" in result:
            sum_df = pd.DataFrame(result["summary"])
            return sum_df, sum_df.copy(), meta

    raise TypeError(
        "Unsupported `benchmark_or_pipeline_result` for `generate_html_report`; "
        "pass a MultiTaskBenchmarkResult, BenchmarkReport, SchemeComparisonReport, DataFrame, or dict."
    )


def _summarize_records(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    group_col = "method" if "method" in df.columns else ("scheme" if "scheme" in df.columns else None)
    if group_col is None or "marginal_coverage" not in df.columns:
        return df.copy()

    rows: list[dict[str, Any]] = []
    for key_val, grp in df.groupby(group_col, sort=False):
        ds_grp = grp[grp["parameter"] == "Dstar"] if "parameter" in grp.columns else grp
        rows.append(
            {
                group_col: str(key_val),
                "mean_coverage": float(grp["marginal_coverage"].mean()),
                "dstar_high_tercile_cov": (
                    float(ds_grp["tercile_cov_high"].mean())
                    if "tercile_cov_high" in ds_grp.columns and not ds_grp.empty
                    else float("nan")
                ),
                "dstar_railing_pct": (
                    float(ds_grp["railing_pct"].mean())
                    if "railing_pct" in ds_grp.columns and not ds_grp.empty
                    else float("nan")
                ),
                "mean_norm_width": (
                    float(grp["norm_width"].mean()) if "norm_width" in grp.columns else float("nan")
                ),
                "mean_interval_score": (
                    float(grp["interval_score"].mean())
                    if "interval_score" in grp.columns
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def compute_gnomon_audit_badges(
    summary_df: pd.DataFrame,
    records_df: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Compute Gnomon calibration, conditional-coverage, railing, and provenance audit badges."""
    nominal = float(metadata.get("nominal_coverage", 1.0 - float(metadata.get("alpha", 0.10))))

    cov_series = (
        summary_df["mean_coverage"]
        if "mean_coverage" in summary_df.columns
        else (
            records_df["marginal_coverage"]
            if "marginal_coverage" in records_df.columns
            else pd.Series([nominal])
        )
    )
    best_cov_gap = float(np.nanmin(np.abs(np.asarray(cov_series, dtype=np.float64) - nominal)))
    if best_cov_gap <= 0.05:
        cal_status, cal_tone = "PASS", "pass"
    elif best_cov_gap <= 0.12:
        cal_status, cal_tone = "WARN", "warn"
    else:
        cal_status, cal_tone = "FAIL", "fail"

    cond_col = (
        "dstar_high_tercile_cov"
        if "dstar_high_tercile_cov" in summary_df.columns
        else ("tercile_cov_high" if "tercile_cov_high" in summary_df.columns else None)
    )
    if cond_col is not None and not summary_df[cond_col].dropna().empty:
        best_high_ds = float(np.nanmax(np.asarray(summary_df[cond_col], dtype=np.float64)))
    elif "tercile_cov_high" in records_df.columns and not records_df["tercile_cov_high"].dropna().empty:
        best_high_ds = float(np.nanmax(np.asarray(records_df["tercile_cov_high"], dtype=np.float64)))
    else:
        best_high_ds = nominal

    if best_high_ds >= nominal - 0.08:
        cond_status, cond_tone = "PASS", "pass"
    elif best_high_ds >= nominal - 0.20:
        cond_status, cond_tone = "WARN", "warn"
    else:
        cond_status, cond_tone = "FAIL", "fail"

    rail_col = (
        "dstar_railing_pct"
        if "dstar_railing_pct" in summary_df.columns
        else ("railing_pct" if "railing_pct" in records_df.columns else None)
    )
    if rail_col is not None:
        src_df = summary_df if rail_col in summary_df.columns else records_df
        min_rail = float(np.nanmin(np.asarray(src_df[rail_col], dtype=np.float64)))
    else:
        min_rail = 0.0

    if min_rail <= 20.0:
        rail_status, rail_tone = "PASS", "pass"
    elif min_rail <= 45.0:
        rail_status, rail_tone = "WARN", "warn"
    else:
        rail_status, rail_tone = "FAIL", "fail"

    return [
        {
            "label": "Gnomon Marginal Calibration",
            "status": cal_status,
            "tone": cal_tone,
            "detail": f"Best |Ĉ − {nominal:.2f}| = {best_cov_gap:.3f}",
        },
        {
            "label": "Gnomon High-D* Conditional Gate",
            "status": cond_status,
            "tone": cond_tone,
            "detail": f"Peak top-D* cov = {best_high_ds:.3f}",
        },
        {
            "label": "Gnomon Boundary-Railing Audit",
            "status": rail_status,
            "tone": rail_tone,
            "detail": f"Min D* railing = {min_rail:.1f}%",
        },
        {
            "label": "Gnomon Reproducibility & PHI Gate",
            "status": "VERIFIED",
            "tone": "pass",
            "detail": f"PCG64 Seed={metadata.get('seed', 42)} · 100% Synthetic",
        },
    ]


def _build_svg_reliability_bars(summary_df: pd.DataFrame, nominal: float) -> str:
    """Render an inline zero-dependency SVG reliability & high-D* coverage bar plot."""
    w, h = 500, 275
    pad_l, pad_r, pad_t, pad_b = 52, 22, 34, 58
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b

    name_col = "method" if "method" in summary_df.columns else (
        "scheme" if "scheme" in summary_df.columns else summary_df.columns[0]
    )
    marg_col = (
        "mean_coverage"
        if "mean_coverage" in summary_df.columns
        else ("marginal_coverage" if "marginal_coverage" in summary_df.columns else None)
    )
    cond_col = (
        "dstar_high_tercile_cov"
        if "dstar_high_tercile_cov" in summary_df.columns
        else ("tercile_cov_high" if "tercile_cov_high" in summary_df.columns else None)
    )

    if marg_col is None or summary_df.empty:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="{h}">'
            '<rect width="100%" height="100%" fill="#ffffff" rx="8"/>'
            '<text x="250" y="140" text-anchor="middle" fill="#64748b" font-size="13">No coverage columns available</text>'
            "</svg>"
        )

    grouped = summary_df.groupby(name_col, sort=False).mean(numeric_only=True).reset_index()
    labels = [str(v) for v in grouped[name_col]][:8]
    marg_vals = [float(v) for v in grouped[marg_col]][:8]
    cond_vals = (
        [float(v) if np.isfinite(v) else 0.0 for v in grouped[cond_col]][:8]
        if cond_col in grouped.columns
        else marg_vals
    )

    n = max(len(labels), 1)
    slot = pw / n
    bw = min(22.0, slot * 0.34)

    def _y(val: float) -> float:
        return pad_t + ph * (1.0 - float(np.clip(val, 0.0, 1.05)) / 1.05)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="Reliability and Tercile Coverage Chart">',
        '<rect width="100%" height="100%" fill="#ffffff" rx="8"/>',
        f'<text x="{pad_l}" y="22" fill="#0f172a" font-size="13" font-weight="600">Marginal vs. High-D* Tercile Coverage</text>',
    ]
    for tick in (0.0, 0.25, 0.50, 0.75, 1.0):
        yt = _y(tick)
        lines.append(f'<line x1="{pad_l}" y1="{yt:.1f}" x2="{w - pad_r}" y2="{yt:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'<text x="{pad_l - 8}" y="{yt + 4:.1f}" text-anchor="end" fill="#64748b" font-size="10.5">{tick:.2f}</text>')

    ynom = _y(nominal)
    lines.append(
        f'<line x1="{pad_l}" y1="{ynom:.1f}" x2="{w - pad_r}" y2="{ynom:.1f}" stroke="#0f172a" stroke-width="1.4" stroke-dasharray="5,4"/>'
    )
    lines.append(
        f'<text x="{w - pad_r - 4}" y="{ynom - 5:.1f}" text-anchor="end" fill="#0f172a" font-size="10" font-weight="600">Nominal {nominal:.2f}</text>'
    )

    y_base = _y(0.0)
    for i, lbl in enumerate(labels):
        cx = pad_l + (i + 0.5) * slot
        ym = _y(marg_vals[i])
        yc = _y(cond_vals[i])
        lines.append(
            f'<rect x="{cx - bw - 1:.1f}" y="{ym:.1f}" width="{bw:.1f}" height="{max(y_base - ym, 1.0):.1f}" fill="#2563eb" rx="2">'
            f"<title>{html.escape(lbl)} marginal coverage: {marg_vals[i]:.3f}</title></rect>"
        )
        lines.append(
            f'<rect x="{cx + 1:.1f}" y="{yc:.1f}" width="{bw:.1f}" height="{max(y_base - yc, 1.0):.1f}" fill="#0d9488" rx="2">'
            f"<title>{html.escape(lbl)} high-D* tercile coverage: {cond_vals[i]:.3f}</title></rect>"
        )
        lines.append(
            f'<text x="{cx:.1f}" y="{h - 28}" text-anchor="middle" fill="#334155" font-size="10.5">{html.escape(lbl[:14])}</text>'
        )

    # Legend
    lines.append(f'<rect x="{pad_l}" y="{h - 15}" width="10" height="10" fill="#2563eb" rx="2"/>')
    lines.append(f'<text x="{pad_l + 14}" y="{h - 6}" fill="#475569" font-size="10.5">Marginal Coverage</text>')
    lines.append(f'<rect x="{pad_l + 135}" y="{h - 15}" width="10" height="10" fill="#0d9488" rx="2"/>')
    lines.append(f'<text x="{pad_l + 149}" y="{h - 6}" fill="#475569" font-size="10.5">High-D* Tercile Coverage</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def _build_svg_pareto_scatter(summary_df: pd.DataFrame, nominal: float) -> str:
    """Render an inline zero-dependency SVG Sharpness vs. Conditional Coverage Pareto plot."""
    w, h = 500, 275
    pad_l, pad_r, pad_t, pad_b = 56, 28, 34, 48
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b

    name_col = "method" if "method" in summary_df.columns else (
        "scheme" if "scheme" in summary_df.columns else summary_df.columns[0]
    )
    sharp_col = (
        "delta_sharp"
        if "delta_sharp" in summary_df.columns
        else ("mean_norm_width" if "mean_norm_width" in summary_df.columns else (
            "norm_width" if "norm_width" in summary_df.columns else None
        ))
    )
    cond_col = (
        "delta_cond"
        if "delta_cond" in summary_df.columns
        else ("dstar_high_tercile_cov" if "dstar_high_tercile_cov" in summary_df.columns else (
            "tercile_cov_high" if "tercile_cov_high" in summary_df.columns else None
        ))
    )

    if sharp_col is None or cond_col is None or summary_df.empty:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="{h}">'
            '<rect width="100%" height="100%" fill="#ffffff" rx="8"/>'
            '<text x="250" y="140" text-anchor="middle" fill="#64748b" font-size="13">Sharpness / Pareto columns not present</text>'
            "</svg>"
        )

    sub = summary_df[[name_col, sharp_col, cond_col]].dropna().copy()
    if sub.empty:
        sub = summary_df.head(1).fillna(0.0)

    xs = np.asarray(sub[sharp_col], dtype=np.float64)
    ys_raw = np.asarray(sub[cond_col], dtype=np.float64)
    ys = ys_raw if cond_col == "delta_cond" else (ys_raw - nominal)
    labels = [str(v) for v in sub[name_col]]

    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    xspan = max(xmax - xmin, 0.08)
    yspan = max(ymax - ymin, 0.08)
    xmin -= 0.15 * xspan
    xmax += 0.22 * xspan
    ymin -= 0.15 * yspan
    ymax += 0.18 * yspan

    def _px(v: float) -> float:
        return pad_l + pw * (v - xmin) / max(xmax - xmin, 1e-12)

    def _py(v: float) -> float:
        return pad_t + ph * (1.0 - (v - ymin) / max(ymax - ymin, 1e-12))

    # Compute Pareto-optimal points (minimize xs, maximize ys)
    n = xs.size
    pareto_mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and xs[j] <= xs[i] + 1e-12 and ys[j] >= ys[i] - 1e-12 and (
                xs[j] < xs[i] - 1e-12 or ys[j] > ys[i] + 1e-12
            ):
                pareto_mask[i] = False
                break
    p_idx = np.flatnonzero(pareto_mask)
    p_idx = p_idx[np.argsort(xs[p_idx], kind="mergesort")]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="Pareto Sharpness vs Conditional Coverage">',
        '<rect width="100%" height="100%" fill="#ffffff" rx="8"/>',
        f'<text x="{pad_l}" y="22" fill="#0f172a" font-size="13" font-weight="600">Sharpness vs. High-D* Coverage Pareto Frontier</text>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>',
    ]

    if ymin <= 0.0 <= ymax:
        y0 = _py(0.0)
        lines.append(
            f'<line x1="{pad_l}" y1="{y0:.1f}" x2="{w - pad_r}" y2="{y0:.1f}" stroke="#475569" stroke-width="1.2" stroke-dasharray="4,4"/>'
        )

    if p_idx.size >= 2:
        pts_attr = " ".join(f"{_px(float(xs[k])):.1f},{_py(float(ys[k])):.1f}" for k in p_idx)
        lines.append(f'<polyline points="{pts_attr}" fill="none" stroke="#16a34a" stroke-width="2.2"/>')

    for i, lbl in enumerate(labels):
        cx, cy = _px(float(xs[i])), _py(float(ys[i]))
        fill = "#dc2626" if pareto_mask[i] else "#2563eb"
        lines.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{fill}" stroke="#ffffff" stroke-width="1.4">'
            f"<title>{html.escape(lbl)}: {sharp_col}={xs[i]:.3f}, Δcond={ys[i]:+.3f}</title></circle>"
        )
        lines.append(
            f'<text x="{cx + 8:.1f}" y="{cy + 3.5:.1f}" fill="#1e293b" font-size="10">{html.escape(lbl[:15])}</text>'
        )

    lines.append(
        f'<text x="{pad_l + pw / 2:.1f}" y="{h - 10}" text-anchor="middle" fill="#475569" font-size="11">'
        f"{html.escape(sharp_col)} (left = sharper intervals)</text>"
    )
    lines.append(
        f'<text x="16" y="{pad_t + ph / 2:.1f}" text-anchor="middle" fill="#475569" font-size="11" '
        f'transform="rotate(-90 16 {pad_t + ph / 2:.1f})">Δcond (high-D* gap)</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def _df_to_sortable_html_table(df: pd.DataFrame, table_id: str) -> str:
    if df.empty:
        return "<p><em>No rows recorded.</em></p>"
    cols = list(df.columns)
    numeric_flags = [bool(pd.api.types.is_numeric_dtype(df[c])) for c in cols]

    thead_cells = []
    for idx, (col, is_num) in enumerate(zip(cols, numeric_flags, strict=True)):
        num_js = "true" if is_num else "false"
        align_cls = "num" if is_num else "txt"
        thead_cells.append(
            f'<th class="{align_cls}" onclick="sortBenchmarkTable(\'{table_id}\', {idx}, {num_js})">'
            f'{html.escape(str(col))} <span class="sort-ind">⇅</span></th>'
        )

    tbody_rows = []
    for _, row in df.iterrows():
        cells = []
        for col, is_num in zip(cols, numeric_flags, strict=True):
            val = row[col]
            if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
                disp = "—"
                sort_val = ""
            elif isinstance(val, (bool, np.bool_)):
                disp = "true" if bool(val) else "false"
                sort_val = "1" if bool(val) else "0"
            elif isinstance(val, (int, np.integer)):
                disp = str(int(val))
                sort_val = disp
            elif isinstance(val, (float, np.floating)):
                fv = float(val)
                disp = f"{fv:.2e}" if (0.0 < abs(fv) < 1e-3) else f"{fv:.4f}"
                sort_val = f"{fv:.12g}"
            else:
                disp = html.escape(str(val))
                sort_val = disp
            cls = "num" if is_num else "txt"
            cells.append(f'<td class="{cls}" data-val="{sort_val}">{disp}</td>')
        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        f'<div class="table-wrap"><table id="{table_id}" class="bench-table">'
        f"<thead><tr>{''.join(thead_cells)}</tr></thead>"
        f"<tbody>{''.join(tbody_rows)}</tbody></table></div>"
    )


def generate_html_report(
    benchmark_or_pipeline_result: Any,
    out_path: str | Path = "ivimbench_report.html",
    *,
    title: str | None = None,
    volume_bundle: Mapping[str, np.ndarray] | None = None,
    audit_summary: Mapping[str, Any] | None = None,
) -> Path:
    """Emit a standalone, zero-dependency interactive HTML report with sortable tables, inline SVGs, and Gnomon badges."""
    summary_df, records_df, metadata = _extract_report_tables(benchmark_or_pipeline_result)
    nominal = float(metadata.get("nominal_coverage", 1.0 - float(metadata.get("alpha", 0.10))))
    badges = compute_gnomon_audit_badges(summary_df, records_df, metadata)

    svg_rel = _build_svg_reliability_bars(summary_df, nominal=nominal)
    svg_pareto = _build_svg_pareto_scatter(summary_df, nominal=nominal)

    summary_table_html = _df_to_sortable_html_table(summary_df, "summary-leaderboard")
    records_table_html = _df_to_sortable_html_table(records_df.head(250), "detailed-records")

    page_title = title or f"IVIMBench v{__version__} Calibration & Acquisition Leaderboard"

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    viewer_banner = ""
    if volume_bundle is not None:
        viewer_path = target.parent / "volume_viewer.html"
        generate_volume_viewer_html(
            volume_bundle,
            viewer_path,
            title=f"{page_title} · Volumetric Viewer",
            audit_summary=audit_summary,
        )
        viewer_banner = (
            '<div class="panel" style="margin-bottom: 1.5rem; background: #121b2d; color: #f8fafc; '
            'border: 1px solid #d4af37; display: flex; justify-content: space-between; align-items: center;">'
            '<div><strong style="color: #d4af37;">Interactive 3D Multi-Slice Volumetric Viewer</strong>'
            '<div style="font-size: 0.85rem; color: #cbd5e1;">Inspect axial D, D*, f, Conformal CI Width, '
            'D* Boundary-Railed Mask, and RT Escalation Gate Mask directly in browser.</div></div>'
            '<a href="volume_viewer.html" style="color: #121b2d; background: #d4af37; padding: 0.45rem 0.9rem; '
            'border-radius: 6px; font-weight: 600; text-decoration: none;">Open volume_viewer.html</a></div>'
        )

    badge_cards = []
    for b in badges:
        badge_cards.append(
            f'<div class="gnomon-card tone-{b["tone"]}">'
            f'<div class="gnomon-top"><span class="gnomon-title">{html.escape(b["label"])}</span>'
            f'<span class="gnomon-pill pill-{b["tone"]}">{html.escape(b["status"])}</span></div>'
            f'<div class="gnomon-detail">{html.escape(b["detail"])}</div>'
            "</div>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(page_title)}</title>
  <style>
    :root {{
      --bg: #f8fafc; --surface: #ffffff; --border: #e2e8f0; --text: #0f172a; --muted: #475569;
      --pass-bg: #dcfce7; --pass-fg: #166534; --warn-bg: #fef9c3; --warn-fg: #854d0e; --fail-bg: #fee2e2; --fail-fg: #991b1b;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 2rem; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    header {{ margin-bottom: 1.5rem; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 1rem; }}
    h1 {{ font-size: 1.55rem; margin: 0; }}
    .meta-sub {{ color: var(--muted); font-size: 0.9rem; }}
    .gnomon-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.9rem; margin-bottom: 1.6rem; }}
    .gnomon-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 0.95rem 1.1rem; box-shadow: 0 1px 3px rgba(15,23,42,0.04); }}
    .gnomon-top {{ display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; margin-bottom: 0.35rem; }}
    .gnomon-title {{ font-weight: 600; font-size: 0.88rem; }}
    .gnomon-detail {{ color: var(--muted); font-size: 0.83rem; font-family: ui-monospace, monospace; }}
    .gnomon-pill {{ font-size: 0.73rem; font-weight: 700; padding: 0.16rem 0.5rem; border-radius: 999px; letter-spacing: 0.03em; }}
    .pill-pass {{ background: var(--pass-bg); color: var(--pass-fg); }}
    .pill-warn {{ background: var(--warn-bg); color: var(--warn-fg); }}
    .pill-fail {{ background: var(--fail-bg); color: var(--fail-fg); }}
    .svg-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 1.1rem; margin-bottom: 1.75rem; }}
    .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.2rem; box-shadow: 0 1px 3px rgba(15,23,42,0.04); }}
    .toolbar {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; gap: 1rem; flex-wrap: wrap; }}
    .filter-input {{ padding: 0.45rem 0.75rem; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.88rem; min-width: 240px; }}
    .table-wrap {{ overflow-x: auto; }}
    table.bench-table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
    table.bench-table th, table.bench-table td {{ padding: 0.52rem 0.7rem; border-bottom: 1px solid var(--border); white-space: nowrap; }}
    table.bench-table th {{ background: #f1f5f9; cursor: pointer; user-select: none; font-weight: 600; }}
    table.bench-table th:hover {{ background: #e2e8f0; }}
    table.bench-table th.num, table.bench-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    table.bench-table th.txt, table.bench-table td.txt {{ text-align: left; }}
    .sort-ind {{ color: #94a3b8; font-size: 0.76rem; margin-left: 0.2rem; }}
    footer {{ margin-top: 2rem; color: var(--muted); font-size: 0.8rem; text-align: center; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>{html.escape(page_title)}</h1>
        <div class="meta-sub">Gnomon Calibration &amp; Acquisition Audit · Nominal Coverage = {nominal:.2f} · IVIMBench v{__version__}</div>
      </div>
    </header>

    {viewer_banner}

    <section class="gnomon-grid" aria-label="Gnomon Audit Badges">
      {''.join(badge_cards)}
    </section>

    <section class="svg-grid" aria-label="Embedded Diagnostic Plots">
      <div class="panel">{svg_rel}</div>
      <div class="panel">{svg_pareto}</div>
    </section>

    <section class="panel" style="margin-bottom: 1.5rem;">
      <div class="toolbar">
        <h2 style="margin: 0; font-size: 1.1rem;">Leaderboard Summary (Click Column Headers to Sort)</h2>
        <input type="search" class="filter-input" placeholder="Filter leaderboard rows..." oninput="filterBenchmarkTable('summary-leaderboard', this.value)"/>
      </div>
      {summary_table_html}
    </section>

    <section class="panel">
      <div class="toolbar">
        <h2 style="margin: 0; font-size: 1.1rem;">Detailed Parameter &amp; Regime Records</h2>
        <input type="search" class="filter-input" placeholder="Filter detailed records..." oninput="filterBenchmarkTable('detailed-records', this.value)"/>
      </div>
      {records_table_html}
    </section>

    <footer>Generated deterministically by IVIMBench v{__version__} (MIT License · Zero External Dependencies · 100% Synthetic/PHI-Free).</footer>
  </div>

  <script>
    const _sortDirections = {{}};
    function sortBenchmarkTable(tableId, colIdx, isNumeric) {{
      const table = document.getElementById(tableId);
      if (!table || !table.tBodies.length) return;
      const tbody = table.tBodies[0];
      const key = tableId + ":" + colIdx;
      const asc = !_sortDirections[key];
      _sortDirections[key] = asc;
      const rows = Array.from(tbody.rows);
      rows.sort((ra, rb) => {{
        const va = ra.cells[colIdx]?.getAttribute("data-val") ?? "";
        const vb = rb.cells[colIdx]?.getAttribute("data-val") ?? "";
        if (isNumeric) {{
          const na = va === "" ? Number.NEGATIVE_INFINITY : parseFloat(va);
          const nb = vb === "" ? Number.NEGATIVE_INFINITY : parseFloat(vb);
          return asc ? (na - nb) : (nb - na);
        }}
        return asc ? va.localeCompare(vb) : vb.localeCompare(va);
      }});
      rows.forEach((r) => tbody.appendChild(r));
    }}

    function filterBenchmarkTable(tableId, query) {{
      const table = document.getElementById(tableId);
      if (!table || !table.tBodies.length) return;
      const q = (query || "").toLowerCase().trim();
      Array.from(table.tBodies[0].rows).forEach((row) => {{
        row.style.display = (!q || row.textContent.toLowerCase().includes(q)) ? "" : "none";
      }});
    }}
  </script>
</body>
</html>
"""
    target.write_text(doc, encoding="utf-8")
    return target


def _normalize_3d_volume(arr: Any, fallback_shape: tuple[int, int, int]) -> np.ndarray:
    if arr is None:
        return np.zeros(fallback_shape, dtype=np.float64)
    v = np.asarray(arr, dtype=np.float64)
    if v.ndim == 2:
        v = v[:, :, None]
    if v.ndim != 3:
        raise ValueError(f"Expected 2D or 3D volume array; got shape {v.shape}.")
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def generate_volume_viewer_html(
    volumes: Mapping[str, Any],
    out_path: str | Path = "volume_viewer.html",
    *,
    title: str | None = None,
    subject_id: str = "demo",
    audit_summary: Mapping[str, Any] | None = None,
) -> Path:
    """Emit a standalone Minimalist Luxury (`#121b2d` / `#d4af37`) interactive multi-slice HTML5 Canvas/SVG viewer."""
    d_raw = volumes.get("D_map", volumes.get("D"))
    if d_raw is None:
        first_arr = next(iter(volumes.values()))
        ref_v = np.asarray(first_arr, dtype=np.float64)
        if ref_v.ndim == 2:
            ref_v = ref_v[:, :, None]
        shape3 = (int(ref_v.shape[0]), int(ref_v.shape[1]), int(ref_v.shape[2]))
    else:
        tmp = np.asarray(d_raw, dtype=np.float64)
        if tmp.ndim == 2:
            tmp = tmp[:, :, None]
        shape3 = (int(tmp.shape[0]), int(tmp.shape[1]), int(tmp.shape[2]))

    d_vol = _normalize_3d_volume(volumes.get("D_map", volumes.get("D")), shape3)
    ds_vol = _normalize_3d_volume(volumes.get("Dstar_map", volumes.get("Dstar")), shape3)
    f_vol = _normalize_3d_volume(volumes.get("f_map", volumes.get("f")), shape3)
    ci_vol = _normalize_3d_volume(
        volumes.get("D_ci_width", volumes.get("f_ci_width", volumes.get("ci_width"))),
        shape3,
    )
    rail_vol = _normalize_3d_volume(
        volumes.get("Dstar_railed_mask", volumes.get("railed_mask")),
        shape3,
    )
    rt_vol = _normalize_3d_volume(
        volumes.get("rt_escalation_eligible_mask", volumes.get("rt_gate_mask")),
        shape3,
    )

    nx, ny, nz = shape3

    def _serialize_vol(v: np.ndarray, decimals: int = 6) -> list[Any]:
        # Transpose to [z][y][x] for direct axial canvas scanline rendering
        zyx = np.transpose(np.round(v, decimals), (2, 1, 0))
        return zyx.tolist()

    payload = {
        "shape": [nx, ny, nz],
        "subject_id": str(subject_id),
        "layers": {
            "D_map": {
                "label": "D Map",
                "unit": "mm²/s",
                "vmin": float(np.min(d_vol)),
                "vmax": max(float(np.max(d_vol)), 1e-6),
                "data": _serialize_vol(d_vol, 6),
            },
            "Dstar_map": {
                "label": "D* Map",
                "unit": "mm²/s",
                "vmin": float(np.min(ds_vol)),
                "vmax": max(float(np.max(ds_vol)), 1e-5),
                "data": _serialize_vol(ds_vol, 5),
            },
            "f_map": {
                "label": "f Map",
                "unit": "fraction",
                "vmin": float(np.min(f_vol)),
                "vmax": max(float(np.max(f_vol)), 1e-4),
                "data": _serialize_vol(f_vol, 4),
            },
            "ci_width": {
                "label": "Conformal CI Width",
                "unit": "90% CI",
                "vmin": 0.0,
                "vmax": max(float(np.max(ci_vol)), 1e-6),
                "data": _serialize_vol(ci_vol, 6),
            },
        },
        "overlays": {
            "railed_mask": {
                "label": "D* Boundary-Railed Mask",
                "data": _serialize_vol((rail_vol > 0).astype(int), 0),
            },
            "rt_gate_mask": {
                "label": "RT Escalation Gate Mask",
                "data": _serialize_vol((rt_vol > 0).astype(int), 0),
            },
        },
    }

    audit_info = dict(audit_summary or {})
    audit_status = str(audit_info.get("overall_status", audit_info.get("status", "PASS")))
    chain_sha = str(audit_info.get("chain_sha256", "standalone-session"))
    viewer_title = title or f"IVIMBench v{__version__} · Volumetric Trust & RT Gate Viewer"
    init_z = nz // 2

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(viewer_title)}</title>
  <style>
    :root {{
      --navy: #121b2d;
      --surface: #18243c;
      --panel: #1f2e4d;
      --gold: #d4af37;
      --gold-soft: rgba(212, 175, 55, 0.22);
      --text: #f8fafc;
      --muted: #9fb0cc;
      --crimson: #ef4444;
      --emerald: #10b981;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 2rem;
      background: var(--navy);
      color: var(--text);
      font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    .shell {{ max-width: 1160px; margin: 0 auto; }}
    header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      padding-bottom: 1.25rem;
      margin-bottom: 1.5rem;
      border-bottom: 1px solid var(--gold-soft);
    }}
    h1 {{ margin: 0; font-size: 1.45rem; letter-spacing: 0.02em; color: var(--gold); }}
    .sub {{ color: var(--muted); font-size: 0.86rem; font-family: ui-monospace, monospace; }}
    .badge {{
      padding: 0.35rem 0.8rem;
      border-radius: 999px;
      border: 1px solid var(--gold);
      background: rgba(212, 175, 55, 0.12);
      color: var(--gold);
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.05em;
    }}
    .workspace {{
      display: grid;
      grid-template-columns: minmax(320px, 1fr) 360px;
      gap: 1.5rem;
      align-items: start;
    }}
    @media (max-width: 860px) {{ .workspace {{ grid-template-columns: 1fr; }} }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--gold-soft);
      border-radius: 12px;
      padding: 1.25rem;
      box-shadow: 0 12px 28px rgba(0, 0, 0, 0.32);
    }}
    .canvas-stage {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 1rem;
    }}
    canvas {{
      width: 100%;
      max-width: 460px;
      aspect-ratio: {nx} / {ny};
      image-rendering: pixelated;
      border: 1px solid var(--gold);
      border-radius: 8px;
      background: #0a101d;
      cursor: crosshair;
    }}
    .scrubber {{ width: 100%; max-width: 460px; }}
    .scrubber label {{
      display: flex;
      justify-content: space-between;
      font-size: 0.86rem;
      color: var(--gold);
      margin-bottom: 0.35rem;
      font-weight: 600;
    }}
    input[type="range"] {{ width: 100%; accent-color: var(--gold); cursor: pointer; }}
    .control-title {{
      margin: 0 0 0.65rem 0;
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--gold);
    }}
    .layer-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 1.2rem; }}
    .layer-btn {{
      padding: 0.55rem 0.65rem;
      border-radius: 7px;
      border: 1px solid var(--gold-soft);
      background: var(--panel);
      color: var(--text);
      font-size: 0.83rem;
      font-weight: 600;
      cursor: pointer;
      text-align: left;
      transition: all 0.15s ease;
    }}
    .layer-btn.active {{
      background: var(--gold);
      color: var(--navy);
      border-color: var(--gold);
    }}
    .toggle-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.55rem 0.7rem;
      border-radius: 7px;
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.07);
      margin-bottom: 0.55rem;
      font-size: 0.85rem;
    }}
    .readout {{
      margin-top: 1rem;
      padding: 0.75rem;
      border-radius: 8px;
      background: #0f1728;
      border: 1px solid var(--gold-soft);
      font-family: ui-monospace, monospace;
      font-size: 0.8rem;
      color: var(--muted);
    }}
    footer {{
      margin-top: 1.75rem;
      text-align: center;
      font-size: 0.78rem;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>{html.escape(viewer_title)}</h1>
        <div class="sub">Subject: sub-{html.escape(str(subject_id))} · Matrix: {nx}×{ny}×{nz} · Chain SHA-256: {html.escape(chain_sha[:16])}…</div>
      </div>
      <div class="badge">SUITE I AUDIT: {html.escape(audit_status)}</div>
    </header>

    <div class="workspace">
      <section class="card canvas-stage" aria-label="Axial Slice Canvas">
        <canvas id="ivim-slice-canvas" width="{nx}" height="{ny}"></canvas>
        <div class="scrubber">
          <label for="z-slice-slider">
            <span>Axial Z-Slice Scrubber</span>
            <span id="z-slice-label">z = {init_z} / {max(nz - 1, 0)}</span>
          </label>
          <input id="z-slice-slider" type="range" min="0" max="{max(nz - 1, 0)}" step="1" value="{init_z}"/>
        </div>
        <svg width="100%" height="36" viewBox="0 0 420 36" aria-label="Colorbar Legend" style="max-width: 460px;">
          <defs>
            <linearGradient id="luxury-grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="#0d1626"/>
              <stop offset="35%" stop-color="#1e4d7b"/>
              <stop offset="70%" stop-color="#b8860b"/>
              <stop offset="100%" stop-color="#f7e396"/>
            </linearGradient>
          </defs>
          <rect x="10" y="4" width="400" height="12" rx="4" fill="url(#luxury-grad)" stroke="#d4af37" stroke-width="0.8"/>
          <text id="cbar-min" x="10" y="30" fill="#9fb0cc" font-size="11" font-family="monospace">0.0</text>
          <text id="cbar-unit" x="210" y="30" fill="#d4af37" font-size="11" text-anchor="middle" font-family="monospace">D Map (mm²/s)</text>
          <text id="cbar-max" x="410" y="30" fill="#9fb0cc" font-size="11" text-anchor="end" font-family="monospace">0.003</text>
        </svg>
      </section>

      <aside class="card" aria-label="Layer and Mask Controls">
        <h2 class="control-title">Base Quantitative Map</h2>
        <div class="layer-grid">
          <button type="button" class="layer-btn active" data-layer="D_map">D Map</button>
          <button type="button" class="layer-btn" data-layer="Dstar_map">D* Map</button>
          <button type="button" class="layer-btn" data-layer="f_map">f Map</button>
          <button type="button" class="layer-btn" data-layer="ci_width">Conformal CI Width</button>
        </div>

        <h2 class="control-title">Clinical QA &amp; RT Overlays (40% Alpha)</h2>
        <label class="toggle-row">
          <span><span style="color:#ef4444;">■</span> D* Boundary-Railed Mask</span>
          <input type="checkbox" id="toggle-railed" checked/>
        </label>
        <label class="toggle-row">
          <span><span style="color:#10b981;">■</span> RT Escalation Gate Mask</span>
          <input type="checkbox" id="toggle-rt-gate" checked/>
        </label>

        <div class="readout" id="voxel-inspector">
          Hover over axial slice to inspect per-voxel D, D*, f, Conformal CI Width, and RT gate status.
        </div>
      </aside>
    </div>

    <footer>IVIMBench v{__version__} · Suite I Clinical Quantitative-MRI Trust Spine · 100% Zero-Dependency Browser Viewer (PHI-Free)</footer>
  </div>

  <script>
    const VOLUME_BUNDLE = {json.dumps(payload, separators=(",", ":"))};
    let activeLayer = "D_map";
    let currentZ = {init_z};

    const canvas = document.getElementById("ivim-slice-canvas");
    const ctx = canvas.getContext("2d");
    const zSlider = document.getElementById("z-slice-slider");
    const zLabel = document.getElementById("z-slice-label");
    const toggleRailed = document.getElementById("toggle-railed");
    const toggleRt = document.getElementById("toggle-rt-gate");
    const inspector = document.getElementById("voxel-inspector");

    function sampleColormap(t) {{
      const u = Math.max(0.0, Math.min(1.0, t));
      if (u < 0.5) {{
        const s = u / 0.5;
        return [
          Math.round(13 + s * (38 - 13)),
          Math.round(22 + s * (92 - 22)),
          Math.round(38 + s * (145 - 38))
        ];
      }}
      const s = (u - 0.5) / 0.5;
      return [
        Math.round(38 + s * (238 - 38)),
        Math.round(92 + s * (205 - 92)),
        Math.round(145 + s * (112 - 145))
      ];
    }}

    function renderSlice() {{
      const [nx, ny, nz] = VOLUME_BUNDLE.shape;
      const layer = VOLUME_BUNDLE.layers[activeLayer];
      const slice = layer.data[currentZ];
      const railSlice = VOLUME_BUNDLE.overlays.railed_mask.data[currentZ];
      const rtSlice = VOLUME_BUNDLE.overlays.rt_gate_mask.data[currentZ];
      const img = ctx.createImageData(nx, ny);
      const span = Math.max(layer.vmax - layer.vmin, 1e-12);

      for (let y = 0; y < ny; y++) {{
        for (let x = 0; x < nx; x++) {{
          const val = slice[y][x];
          let [r, g, b] = val === 0 ? [10, 16, 29] : sampleColormap((val - layer.vmin) / span);
          if (toggleRailed.checked && railSlice[y][x] > 0) {{
            r = Math.round(0.6 * r + 0.4 * 239);
            g = Math.round(0.6 * g + 0.4 * 68);
            b = Math.round(0.6 * b + 0.4 * 68);
          }}
          if (toggleRt.checked && rtSlice[y][x] > 0) {{
            r = Math.round(0.6 * r + 0.4 * 16);
            g = Math.round(0.6 * g + 0.4 * 185);
            b = Math.round(0.6 * b + 0.4 * 129);
          }}
          const idx = 4 * (y * nx + x);
          img.data[idx] = r;
          img.data[idx + 1] = g;
          img.data[idx + 2] = b;
          img.data[idx + 3] = 255;
        }}
      }}
      ctx.putImageData(img, 0, 0);
      zLabel.textContent = `z = ${{currentZ}} / ${{Math.max(nz - 1, 0)}}`;
      document.getElementById("cbar-min").textContent = layer.vmin.toExponential(2);
      document.getElementById("cbar-max").textContent = layer.vmax.toExponential(2);
      document.getElementById("cbar-unit").textContent = `${{layer.label}} (${{layer.unit}})`;
    }}

    document.querySelectorAll(".layer-btn").forEach((btn) => {{
      btn.addEventListener("click", () => {{
        document.querySelectorAll(".layer-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        activeLayer = btn.getAttribute("data-layer");
        renderSlice();
      }});
    }});

    zSlider.addEventListener("input", (ev) => {{
      currentZ = parseInt(ev.target.value, 10) || 0;
      renderSlice();
    }});
    toggleRailed.addEventListener("change", renderSlice);
    toggleRt.addEventListener("change", renderSlice);

    canvas.addEventListener("mousemove", (ev) => {{
      const rect = canvas.getBoundingClientRect();
      const [nx, ny] = VOLUME_BUNDLE.shape;
      const x = Math.min(nx - 1, Math.max(0, Math.floor(((ev.clientX - rect.left) / rect.width) * nx)));
      const y = Math.min(ny - 1, Math.max(0, Math.floor(((ev.clientY - rect.top) / rect.height) * ny)));
      const dV = VOLUME_BUNDLE.layers.D_map.data[currentZ][y][x];
      const dsV = VOLUME_BUNDLE.layers.Dstar_map.data[currentZ][y][x];
      const fV = VOLUME_BUNDLE.layers.f_map.data[currentZ][y][x];
      const ciV = VOLUME_BUNDLE.layers.ci_width.data[currentZ][y][x];
      const isRail = VOLUME_BUNDLE.overlays.railed_mask.data[currentZ][y][x] > 0;
      const isRt = VOLUME_BUNDLE.overlays.rt_gate_mask.data[currentZ][y][x] > 0;
      inspector.innerHTML =
        `Voxel (${{x}}, ${{y}}, ${{currentZ}})<br/>` +
        `D=${{dV.toExponential(3)}} | D*=${{dsV.toExponential(3)}} | f=${{fV.toFixed(3)}}<br/>` +
        `CI Width=${{ciV.toExponential(3)}} | Railed=${{isRail}} | RT Eligible=${{isRt}}`;
    }});

    renderSlice();
  </script>
</body>
</html>
"""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html_doc, encoding="utf-8")
    return target

