"""Multi-task IVIM benchmark runner across protocols, forward physics families, and SNR regimes.

Evaluates registered estimators across clinical acquisition schedules
(``"osipi"``, ``"tcga_lihc"``, ``"acrin_6698"``), all five forward signal
families (``"biexp"``, ``"dispersion_gamma"``, ``"dispersion_lognormal"``,
``"stretched"``, ``"triexp"``), and SNR regimes, exporting reproducible
Markdown, JSON, and LaTeX summary tables without requiring optional templating
dependencies.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import BenchmarkSuite
from .phantoms.synthetic import DEFAULT_SEED, FAMILIES, generate_cohort
from .protocols import AcquisitionProtocol, get_protocol, resolve_bvalues
from .registry import DEFAULT_REGISTERED_ESTIMATORS, RegisteredEstimator, get_estimator, run_estimator

__all__ = [
    "DEFAULT_BENCHMARK_PROTOCOLS",
    "DEFAULT_BENCHMARK_FAMILIES",
    "DEFAULT_BENCHMARK_SNRS",
    "dataframe_to_markdown",
    "dataframe_to_latex",
    "MultiTaskBenchmarkResult",
    "run_benchmark_suite",
]

DEFAULT_BENCHMARK_PROTOCOLS: tuple[str, ...] = ("osipi", "tcga_lihc", "acrin_6698")
DEFAULT_BENCHMARK_FAMILIES: tuple[str, ...] = (
    "biexp",
    "dispersion_gamma",
    "dispersion_lognormal",
    "stretched",
    "triexp",
)
DEFAULT_BENCHMARK_SNRS: tuple[float, ...] = (20.0, 40.0)


def _format_cell(val: Any, float_fmt: str = ".3f", latex: bool = False) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    if isinstance(val, (float, np.floating)):
        if np.isinf(val):
            return r"$\infty$" if latex else "inf"
        if abs(float(val)) > 0.0 and abs(float(val)) < 1e-3:
            return f"{float(val):.2e}"
        return format(float(val), float_fmt)
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    text = str(val)
    if latex:
        text = text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    return text


def dataframe_to_markdown(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    float_fmt: str = ".3f",
) -> str:
    """Format a :class:`pandas.DataFrame` as a GitHub-flavored Markdown table (zero external deps)."""
    cols = [c for c in (columns or list(df.columns)) if c in df.columns]
    if not cols:
        return ""
    headers = [str(c) for c in cols]
    rows: list[list[str]] = []
    for _, row in df[cols].iterrows():
        rows.append([_format_cell(row[c], float_fmt=float_fmt, latex=False) for c in cols])

    widths = [
        max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)
    ]
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_boxes = []
    for i, c in enumerate(cols):
        is_num = pd.api.types.is_numeric_dtype(df[c])
        if is_num:
            sep_boxes.append("-" * max(widths[i] - 1, 2) + ":")
        else:
            sep_boxes.append(":" + "-" * max(widths[i] - 1, 2))
    sep_line = "| " + " | ".join(sep_boxes) + " |"
    body_lines = [
        "| "
        + " | ".join(
            r[i].rjust(widths[i]) if pd.api.types.is_numeric_dtype(df[cols[i]]) else r[i].ljust(widths[i])
            for i in range(len(cols))
        )
        + " |"
        for r in rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def dataframe_to_latex(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    caption: str = "IVIMBench multi-task uncertainty calibration summary.",
    label: str = "tab:ivimbench_summary",
    float_fmt: str = ".3f",
) -> str:
    """Format a :class:`pandas.DataFrame` as a publication-ready Booktabs LaTeX table."""
    cols = [c for c in (columns or list(df.columns)) if c in df.columns]
    if not cols:
        return ""
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in cols)
    header_cells = [str(c).replace("_", r"\_").replace("%", r"\%") for c in cols]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]
    for _, row in df[cols].iterrows():
        cells = [_format_cell(row[c], float_fmt=float_fmt, latex=True) for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return _sanitize_json(obj.tolist())
    return obj


@dataclass
class MultiTaskBenchmarkResult:
    """Structured multi-task benchmark results with Markdown, JSON, and LaTeX exporters."""

    dataframe: pd.DataFrame
    summary_dataframe: pd.DataFrame
    metadata: dict[str, Any]

    SUMMARY_COLS: tuple[str, ...] = (
        "method",
        "protocol",
        "family",
        "snr",
        "parameter",
        "marginal_coverage",
        "tercile_cov_high",
        "railing_pct",
        "norm_width",
        "sharpness_to_crlb_ratio",
        "rmse",
    )

    AGGREGATE_COLS: tuple[str, ...] = (
        "method",
        "protocol",
        "snr",
        "mean_coverage",
        "dstar_high_tercile_cov",
        "dstar_railing_pct",
        "mean_norm_width",
        "mean_sharpness_to_crlb",
        "mean_interval_score",
    )

    def to_markdown(self, *, aggregate: bool = True, float_fmt: str = ".3f") -> str:
        """Render a reproducible GitHub-flavored Markdown summary table."""
        if aggregate:
            return dataframe_to_markdown(
                self.summary_dataframe,
                columns=self.AGGREGATE_COLS,
                float_fmt=float_fmt,
            )
        return dataframe_to_markdown(
            self.dataframe,
            columns=self.SUMMARY_COLS,
            float_fmt=float_fmt,
        )

    def to_latex(
        self,
        *,
        aggregate: bool = True,
        caption: str = "IVIMBench multi-task calibration, sharpness, and railing benchmark summary.",
        label: str = "tab:ivimbench_multitask",
        float_fmt: str = ".3f",
    ) -> str:
        """Render a publication-ready Booktabs LaTeX summary table."""
        target_df = self.summary_dataframe if aggregate else self.dataframe
        cols = self.AGGREGATE_COLS if aggregate else self.SUMMARY_COLS
        return dataframe_to_latex(
            target_df,
            columns=cols,
            caption=caption,
            label=label,
            float_fmt=float_fmt,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of metadata, summary, and full records."""
        return _sanitize_json(
            {
                "metadata": self.metadata,
                "summary": self.summary_dataframe.to_dict(orient="records"),
                "records": self.dataframe.to_dict(orient="records"),
            }
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize metadata, aggregated summary, and full records to reproducible JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def export(
        self,
        out_dir: str | Path,
        prefix: str = "benchmark_summary",
    ) -> dict[str, Path]:
        """Export Markdown, JSON, LaTeX, and CSV benchmark tables into ``out_dir``."""
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        paths = {
            "markdown": target / f"{prefix}.md",
            "json": target / f"{prefix}.json",
            "latex": target / f"{prefix}.tex",
            "csv": target / f"{prefix}_records.csv",
        }
        paths["markdown"].write_text(self.to_markdown(aggregate=True) + "\n", encoding="utf-8")
        paths["json"].write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        paths["latex"].write_text(self.to_latex(aggregate=True) + "\n", encoding="utf-8")
        self.dataframe.to_csv(paths["csv"], index=False)
        return paths


def _resolve_estimator_map(
    estimators: Sequence[str | RegisteredEstimator | Callable[..., Any] | Any]
    | Mapping[str, Any]
    | None,
) -> dict[str, Any]:
    if estimators is None:
        return {name: get_estimator(name) for name in DEFAULT_REGISTERED_ESTIMATORS}
    if isinstance(estimators, Mapping):
        return dict(estimators)
    out: dict[str, Any] = {}
    for item in estimators:
        if isinstance(item, str):
            reg = get_estimator(item)
            out[reg.name] = reg
        elif isinstance(item, RegisteredEstimator):
            out[item.name] = item
        else:
            label = getattr(item, "__name__", item.__class__.__name__)
            out[str(label)] = item
    return out


def _resolve_organ_for_protocol(organ: str, protocol_key: str) -> str:
    if organ.strip().lower() in ("auto", "protocol_default"):
        try:
            proto = get_protocol(protocol_key)
            return proto.organ if proto.organ != "abdomen" else "liver"
        except KeyError:
            return "liver"
    return organ


def run_benchmark_suite(
    estimators: Sequence[str | RegisteredEstimator | Callable[..., Any] | Any]
    | Mapping[str, Any]
    | None = None,
    *,
    protocols: Sequence[str | AcquisitionProtocol | Sequence[float]] = DEFAULT_BENCHMARK_PROTOCOLS,
    families: Sequence[str] = DEFAULT_BENCHMARK_FAMILIES,
    snr_regimes: Sequence[float] | float = DEFAULT_BENCHMARK_SNRS,
    organ: str = "protocol_default",
    n_voxels: int = 200,
    alpha: float = 0.10,
    noise_type: str = "rician",
    n_coils: int = 1,
    n_boot: int = 200,
    seed: int = DEFAULT_SEED,
    out_dir: str | Path | None = None,
) -> MultiTaskBenchmarkResult:
    """Evaluate registered estimators across forward signal families, SNRs, and clinical protocols.

    Parameters
    ----------
    estimators : sequence or mapping of estimators, optional
        Estimator names registered via :func:`~ivimbench.registry.register_estimator`,
        callable functions ``(signals, bvalues) -> dict``, or estimator instances.
        Defaults to ``("segmented", "nlls", "crlb")``.
    protocols : sequence of protocol specs, default=("osipi", "tcga_lihc", "acrin_6698")
        Clinical b-value schedules to evaluate.
    families : sequence of str, default=5 forward families
        Forward signal physics models (``"biexp"``, ``"dispersion_gamma"``,
        ``"dispersion_lognormal"``, ``"stretched"``, ``"triexp"``).
    snr_regimes : float or sequence of float, default=(20.0, 40.0)
        Signal-to-noise ratio regimes at ``b = 0``.
    organ : str, default="protocol_default"
        Organ parameter prior (``"protocol_default"`` maps ``"acrin_6698"`` to
        ``"breast"`` and abdominal protocols to ``"liver"``).
    n_voxels : int, default=200
        Number of synthetic voxels per ``(protocol, family, snr)`` task.
    alpha : float, default=0.10
        Central prediction interval miscoverage rate (``0.10`` -> 90% nominal).
    out_dir : str, Path, or None, optional
        If provided, exports reproducible Markdown, JSON, LaTeX, and CSV tables.
    """
    est_map = _resolve_estimator_map(estimators)
    if not est_map:
        raise ValueError("At least one estimator must be provided to `run_benchmark_suite`.")

    snrs = (
        [float(snr_regimes)]
        if isinstance(snr_regimes, (int, float, np.floating))
        else [float(s) for s in snr_regimes]
    )
    if not snrs:
        raise ValueError("`snr_regimes` must be non-empty.")

    for fam in families:
        if fam not in FAMILIES:
            raise ValueError(f"Unknown forward family {fam!r}; choose from {list(FAMILIES)}.")

    task_frames: list[pd.DataFrame] = []
    task_idx = 0

    for proto in protocols:
        b_arr, proto_label = resolve_bvalues(proto)
        organ_eff = _resolve_organ_for_protocol(organ, proto_label)
        for snr_val in snrs:
            suite = BenchmarkSuite(
                protocol=b_arr,
                snr=snr_val,
                alpha=alpha,
                n_boot=n_boot,
                seed=int(seed) + task_idx,
            )
            for fam in families:
                # Deterministic per-task seed preserving exact parameter cohort across families at fixed protocol
                cohort = generate_cohort(
                    protocol=b_arr,
                    organ=organ_eff,
                    n_voxels=int(n_voxels),
                    snr=snr_val,
                    seed=int(seed),
                    noise_type=noise_type,
                    n_coils=n_coils,
                    family=fam,
                )
                for est_name, est_obj in est_map.items():
                    t0 = time.perf_counter()
                    norm_out = run_estimator(
                        est_obj,
                        cohort.signals,
                        b_arr,
                        alpha=alpha,
                        snr=snr_val,
                    )
                    elapsed_ms = 1000.0 * (time.perf_counter() - t0)

                    eval_kw: dict[str, Any] = {
                        "point_pred": norm_out.point,
                        "intervals": (norm_out.lower, norm_out.upper),
                    }
                    if norm_out.quantiles is not None and norm_out.q_levels is not None:
                        eval_kw = {
                            "point_pred": norm_out.point,
                            "q_pred": norm_out.quantiles,
                            "q_levels": norm_out.q_levels,
                        }
                    elif norm_out.std is not None:
                        eval_kw = {
                            "point_pred": norm_out.point,
                            "std_pred": norm_out.std,
                        }

                    df_task = suite.evaluate(
                        cohort,
                        method=est_name,
                        bvalues=b_arr,
                        snr=snr_val,
                        as_dataframe=True,
                        **eval_kw,
                    )
                    assert isinstance(df_task, pd.DataFrame)
                    df_task = df_task.copy()
                    df_task.insert(1, "protocol", proto_label)
                    df_task.insert(2, "organ", organ_eff)
                    df_task.insert(3, "family", fam)
                    df_task.insert(4, "snr", float(snr_val))
                    df_task["has_intervals"] = bool(norm_out.has_intervals)
                    df_task["runtime_ms"] = float(elapsed_ms)
                    task_frames.append(df_task)
                task_idx += 1

    full_df = pd.concat(task_frames, ignore_index=True)

    # Build concise aggregated summary grouped by (method, protocol, snr) across families
    summary_rows: list[dict[str, Any]] = []
    for (m_name, p_name, s_val), grp in full_df.groupby(
        ["method", "protocol", "snr"], sort=False
    ):
        dstar_grp = grp[grp["parameter"] == "Dstar"]
        summary_rows.append(
            {
                "method": m_name,
                "protocol": p_name,
                "snr": float(s_val),
                "n_families": int(grp["family"].nunique()),
                "mean_coverage": float(grp["marginal_coverage"].mean()),
                "mean_coverage_gap": float(grp["marginal_coverage_gap"].mean()),
                "dstar_high_tercile_cov": (
                    float(dstar_grp["tercile_cov_high"].mean()) if not dstar_grp.empty else float("nan")
                ),
                "dstar_railing_pct": (
                    float(dstar_grp["railing_pct"].mean()) if not dstar_grp.empty else float("nan")
                ),
                "mean_norm_width": float(grp["norm_width"].mean()),
                "mean_sharpness_to_crlb": float(grp["sharpness_to_crlb_ratio"].mean()),
                "mean_interval_score": float(grp["interval_score"].mean()),
                "mean_runtime_ms": float(grp["runtime_ms"].mean()),
            }
        )
    summary_df = pd.DataFrame.from_records(summary_rows)

    metadata = {
        "estimators": list(est_map.keys()),
        "protocols": [resolve_bvalues(p)[1] for p in protocols],
        "families": list(families),
        "snr_regimes": snrs,
        "organ": organ,
        "n_voxels": int(n_voxels),
        "alpha": float(alpha),
        "nominal_coverage": 1.0 - float(alpha),
        "noise_type": noise_type,
        "n_coils": int(n_coils),
        "seed": int(seed),
    }

    result = MultiTaskBenchmarkResult(
        dataframe=full_df,
        summary_dataframe=summary_df,
        metadata=metadata,
    )
    if out_dir is not None:
        result.export(out_dir=out_dir)
    return result
