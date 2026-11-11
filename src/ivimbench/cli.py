"""Command-line interface for IVIMBench (``ivimbench``).

Supports:
- ``ivimbench demo``: Fast (<10s) end-to-end synthetic DRO + multi-family
  calibration, repeatability, and decision-utility smoke benchmark.
- ``ivimbench benchmark --protocol osipi --snr 30 --out results/``: Multi-task
  runner exporting reproducible Markdown, JSON, and LaTeX summary tables.
- ``ivimbench compare-schemes --schemes osipi,acrin_6698``: Acquisition-design
  and CRLB/calibration comparison across diffusion schedules.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ._version import __version__
from .acquisition.design import SchemeEvaluator
from .decision.voi import check_domain_shift, evaluate_decision_voi
from .harnesses.repeatability import evaluate_repeatability, simulate_scan_rescan
from .phantoms.dro import generate_dro
from .phantoms.synthetic import FAMILIES, generate_cohort
from .registry import DEFAULT_REGISTERED_ESTIMATORS, run_estimator
from .runner import DEFAULT_BENCHMARK_FAMILIES, dataframe_to_markdown, run_benchmark_suite

__all__ = ["build_parser", "main"]


def _split_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _cmd_demo(args: argparse.Namespace) -> int:
    t_start = time.perf_counter()
    seed = int(args.seed)
    n_vox = int(args.n_voxels)

    # 1. Enriched 2D Anatomical DROs (Liver, Pancreas Core+Rim, Breast ACRIN-6698 + B1/SNR shading)
    dro_rows = []
    for organ_name, proto_name, b1_amp, snr_amp in (
        ("liver", "osipi", 0.0, 0.0),
        ("pancreas", "osipi", 0.15, 0.20),
        ("breast", "acrin_6698", 0.15, 0.20),
    ):
        dro = generate_dro(
            shape=(24, 24),
            organ=organ_name,
            protocol=proto_name,
            snr=30.0,
            b1_inhomogeneity=b1_amp,
            snr_inhomogeneity=snr_amp,
            seed=seed,
        )
        sig_fg, gt_fg = dro.to_voxels(foreground_only=True)
        pred_fg = run_estimator("crlb", sig_fg, dro.bvalues, alpha=0.10, snr=30.0)
        cov_d = float(np.mean((gt_fg[:, 0] >= pred_fg.lower[:, 0]) & (gt_fg[:, 0] <= pred_fg.upper[:, 0])))
        cov_ds = float(np.mean((gt_fg[:, 1] >= pred_fg.lower[:, 1]) & (gt_fg[:, 1] <= pred_fg.upper[:, 1])))
        dro_rows.append(
            {
                "organ": organ_name,
                "protocol": proto_name,
                "fg_voxels": int(sig_fg.shape[0]),
                "lesions": len(dro.lesion_names),
                "D_cov": cov_d,
                "Dstar_cov": cov_ds,
            }
        )

    # 2. Multi-family calibration smoke benchmark across all 5 forward signal families
    bench_res = run_benchmark_suite(
        estimators=("segmented", "crlb"),
        protocols=("osipi",),
        families=DEFAULT_BENCHMARK_FAMILIES,
        snr_regimes=(30.0,),
        organ="liver",
        n_voxels=n_vox,
        n_boot=80,
        seed=seed,
    )

    # 3. Synthetic Rician scan-rescan repeatability
    pair = simulate_scan_rescan(
        protocol="osipi",
        organ="liver",
        n_voxels=n_vox,
        snr=30.0,
        noise_type="rician",
        seed=seed,
    )
    rep_res = evaluate_repeatability(pair, estimators=("segmented", "crlb"), alpha=0.10)

    # 4. Decision VoI & observable domain-shift check (pancreas hypoperfusion f <= tau)
    cohort_ref = generate_cohort(protocol="osipi", organ="pancreas", n_voxels=n_vox, snr=25.0, seed=seed)
    cohort_shift = generate_cohort(
        protocol="osipi",
        organ="pancreas",
        n_voxels=n_vox,
        snr=12.0,
        family="triexp",
        seed=seed + 11,
    )
    est_b = run_estimator("crlb", cohort_ref.signals, cohort_ref.bvalues, alpha=0.10, snr=25.0)
    # Compare overconfident narrow intervals vs conformalized intervals on hypovascular f cutoff
    narrow_lo = est_b.point - 0.15 * (est_b.point - est_b.lower)
    narrow_hi = est_b.point + 0.15 * (est_b.upper - est_b.point)
    voi_res = evaluate_decision_voi(
        cohort_ref.params,
        narrow_lo,
        narrow_hi,
        point=est_b.point,
        parameter="f",
        direction="less",
        cost_ratio=0.15,
        seed=seed,
    )
    shift_res = check_domain_shift(
        cohort_ref.signals,
        cohort_shift.signals,
        bvalues=cohort_ref.bvalues,
        n_perm=100,
        seed=seed,
    )

    elapsed = time.perf_counter() - t_start

    if getattr(args, "json", False):
        payload = {
            "version": __version__,
            "elapsed_seconds": round(elapsed, 3),
            "dro_smoke": dro_rows,
            "multifamily_summary": bench_res.to_dict()["summary"],
            "repeatability_summary": rep_res.to_dict()["summary"],
            "decision_voi_summary": voi_res.summary,
            "domain_shift": shift_res.to_dict(),
        }
        print(json.dumps(payload, indent=2))
        return 0

    import pandas as pd

    print(f"=== IVIMBench v{__version__} End-to-End Smoke Benchmark ({elapsed:.2f}s) ===\n")
    print("[1] Spatial 2D Anatomical DROs (100% Synthetic, PCG64 Seeded)")
    print(dataframe_to_markdown(pd.DataFrame(dro_rows), float_fmt=".3f"))
    print("\n[2] Multi-Family Calibration Summary (5 Forward Families on OSIPI, SNR=30)")
    print(bench_res.to_markdown(aggregate=True))
    print("\n[3] Ground-Truth-Free Scan-Rescan Repeatability (Rician, OSIPI)")
    print(rep_res.to_markdown())
    print(
        "\n[4] Clinical Threshold Decision VoI & Observable Domain-Shift Safeguard\n"
        f"  - Mean Decision-Calibration Gap (f <= tau): {voi_res.summary['mean_decision_calibration_gap']:.4f}\n"
        f"  - Mean Second-Order VoI (Calibration Gain): {voi_res.summary['mean_second_order_voi']:.4f}\n"
        f"  - Observable Shift Detected (SNR 25->12 + triexp): is_stale={shift_res.is_stale} "
        f"(reasons={list(shift_res.staleness_reasons)}, RMS ratio={shift_res.residual_rms_ratio:.2f})"
    )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    protocols = _split_csv(args.protocol)
    snrs = [float(v) for v in _split_csv(str(args.snr))]
    families = _split_csv(args.families) if args.families else list(DEFAULT_BENCHMARK_FAMILIES)
    for f_name in families:
        if f_name not in FAMILIES:
            raise ValueError(f"Unknown family {f_name!r}; valid families: {list(FAMILIES)}.")
    estimators = _split_csv(args.estimators) if args.estimators else list(DEFAULT_REGISTERED_ESTIMATORS)

    res = run_benchmark_suite(
        estimators=estimators,
        protocols=protocols,
        families=families,
        snr_regimes=snrs,
        organ=args.organ,
        n_voxels=int(args.n_voxels),
        alpha=float(args.alpha),
        seed=int(args.seed),
        out_dir=args.out,
    )
    print(res.to_markdown(aggregate=True))
    if args.out:
        out_path = Path(args.out)
        print(
            f"\nExported Markdown, JSON, LaTeX, and CSV tables to: {out_path.as_posix()}"
        )
    return 0


def _cmd_compare_schemes(args: argparse.Namespace) -> int:
    schemes = _split_csv(args.schemes)
    evaluator = SchemeEvaluator(
        organ=args.organ,
        snr=float(args.snr),
        alpha=float(args.alpha),
        n_voxels=int(args.n_voxels),
        estimator=args.estimator,
        seed=int(args.seed),
    )
    report = evaluator.compare(
        schemes,
        include_optimized=bool(args.include_optimized),
        optimized_n_b=int(args.optimized_nb),
    )
    print(report.to_markdown())
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "scheme_comparison.md").write_text(report.to_markdown() + "\n", encoding="utf-8")
        (out_dir / "scheme_comparison.json").write_text(report.to_json() + "\n", encoding="utf-8")
        (out_dir / "scheme_comparison.tex").write_text(report.to_latex() + "\n", encoding="utf-8")
        print(f"\nSaved scheme comparison artifacts to: {out_dir.as_posix()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``ivimbench`` top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ivimbench",
        description="IVIMBench: Quantitative IVIM MRI uncertainty, acquisition-design, and decision-utility benchmark suite.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ivimbench {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. ivimbench demo
    p_demo = sub.add_parser(
        "demo",
        help="Run a fast (<10s) synthetic DRO, multi-family calibration, and repeatability smoke benchmark.",
    )
    p_demo.add_argument("--n-voxels", type=int, default=120, help="Voxels per smoke cohort (default: 120).")
    p_demo.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_demo.add_argument("--json", action="store_true", help="Emit structured JSON instead of tables.")
    p_demo.set_defaults(func=_cmd_demo)

    # 2. ivimbench benchmark
    p_bench = sub.add_parser(
        "benchmark",
        help="Run multi-task benchmark across protocols, forward families, and SNR regimes.",
    )
    p_bench.add_argument(
        "--protocol",
        type=str,
        default="osipi",
        help="Comma-separated clinical protocol keys (e.g. 'osipi,tcga_lihc,acrin_6698').",
    )
    p_bench.add_argument(
        "--snr",
        type=str,
        default="30",
        help="Comma-separated b=0 SNR regimes (e.g. '30' or '15,30,50').",
    )
    p_bench.add_argument(
        "--families",
        type=str,
        default="",
        help="Optional comma-separated forward families (default: all 5 families).",
    )
    p_bench.add_argument(
        "--estimators",
        type=str,
        default="segmented,nlls,crlb",
        help="Comma-separated registered estimators (default: 'segmented,nlls,crlb').",
    )
    p_bench.add_argument(
        "--organ",
        type=str,
        default="protocol_default",
        help="Organ prior ('liver', 'pancreas', 'breast', 'brain', 'generic', or 'protocol_default').",
    )
    p_bench.add_argument("--n-voxels", type=int, default=120, help="Number of voxels per task (default: 120).")
    p_bench.add_argument("--alpha", type=float, default=0.10, help="Miscoverage rate alpha (default: 0.10).")
    p_bench.add_argument("--seed", type=int, default=42, help="Master PCG64 seed (default: 42).")
    p_bench.add_argument("--out", type=str, default=None, help="Output directory for Markdown/JSON/LaTeX tables.")
    p_bench.set_defaults(func=_cmd_benchmark)

    # 3. ivimbench compare-schemes
    p_comp = sub.add_parser(
        "compare-schemes",
        help="Compare clinical b-value schedules on sharpness (delta_sharp) and high-D* coverage (delta_cond).",
    )
    p_comp.add_argument(
        "--schemes",
        type=str,
        required=True,
        help="Comma-separated protocol keys to compare (e.g. 'osipi,acrin_6698').",
    )
    p_comp.add_argument("--organ", type=str, default="liver", help="Organ prior (default: 'liver').")
    p_comp.add_argument("--snr", type=float, default=30.0, help="Nominal b=0 SNR (default: 30.0).")
    p_comp.add_argument("--alpha", type=float, default=0.10, help="Miscoverage rate alpha (default: 0.10).")
    p_comp.add_argument("--n-voxels", type=int, default=250, help="Evaluation cohort size (default: 250).")
    p_comp.add_argument("--estimator", type=str, default="crlb", help="Registered estimator name (default: 'crlb').")
    p_comp.add_argument(
        "--include-optimized",
        action="store_true",
        help="Also synthesize and compare CRLB-optimal and calibration-aware schedules.",
    )
    p_comp.add_argument("--optimized-nb", type=int, default=8, help="Number of b-values for optimized designs.")
    p_comp.add_argument("--seed", type=int, default=42, help="Master PCG64 seed (default: 42).")
    p_comp.add_argument("--out", type=str, default=None, help="Optional output directory for comparison tables.")
    p_comp.set_defaults(func=_cmd_compare_schemes)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point returning an integer exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
