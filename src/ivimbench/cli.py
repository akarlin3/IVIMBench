"""Command-line interface for IVIMBench (``ivimbench``).

Supports:
- ``ivimbench demo``: Fast (<10s) end-to-end synthetic DRO + multi-family
  calibration, repeatability, and decision-utility smoke benchmark.
- ``ivimbench selfcheck``: Clean-room continuity-limit residual and noiseless
  bi-exponential NLLS round-trip verification table.
- ``ivimbench benchmark --protocol osipi --snr 30 --out results/``: Multi-task
  runner exporting reproducible Markdown, JSON, and LaTeX summary tables.
- ``ivimbench compare-schemes --schemes osipi,acrin_6698``: Acquisition-design
  and CRLB/calibration comparison across diffusion schedules.
- ``ivimbench anomalous-wall --protocol osipi --snr-min 10 --snr-max 80``:
  Fractional-order anomalous diffusion ``(D, alpha)`` CRLB and NLLS collapse wall.
- ``ivimbench closed-loop --fractions 5 --snr 30``: Multi-fraction synthetic-twin
  adaptive radiotherapy harness comparing uncalibrated NLLS vs. conformal intervals.
- ``ivimbench pipeline --organ pancreas --protocol osipi --conformal cqr --fractions 5``:
  Unified 4-stage `Trust -> Value -> Action` cross-package pipeline runner.
- ``ivimbench fit-volume --dwi <path.nii.gz> --bval <path.bval> [--mask <mask.nii.gz>] --out <dir/>``:
  Affine-preserving 3D/4D NIfTI & DICOM volumetric fitting and parameter-map export
  (with ``--demo-volume`` synthetic 3D DRO round-trip mode).
- ``ivimbench reproduce-gnomon``: Clean-room reproduce-or-refute audit against
  pinned ``GNOMON_TARGETS``.
- ``ivimbench forge-ere-check``: 1.5T MR-Linac 7 MV FFF photon purity, Electron
  Return Effect (ERE) boundary perturbation, and MC simulation floor QA check.
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
from .anomalous.wall import map_recovery_collapse_wall
from .audit import run_audit, run_suite1_pipeline
from .bridges.avery_stack import run_trust_value_action_pipeline
from .closed_loop.ere_forge import (
    compute_ere_boundary_perturbation,
    estimate_mc_simulation_floor,
    generate_unity_7mv_fff_spectrum,
    verify_photon_beam_purity,
)
from .closed_loop.loop import compare_closed_loop_policies
from .decision.voi import check_domain_shift, evaluate_decision_voi
from .gnomon.reproduce import run_gnomon_cleanroom_audit
from .harnesses.repeatability import evaluate_repeatability, simulate_scan_rescan
from .io.volumes import export_dro_to_nifti, fit_dwi_volume
from .phantoms.dro import generate_dro
from .phantoms.selfcheck import run_nlls_roundtrip_selfcheck, verify_continuity_limits
from .phantoms.synthetic import FAMILIES, generate_cohort
from .registry import DEFAULT_REGISTERED_ESTIMATORS, run_estimator
from .reporting import export_bids_derivatives, generate_html_report
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


def _cmd_selfcheck(args: argparse.Namespace) -> int:
    t_start = time.perf_counter()
    seed = int(args.seed)
    cont = verify_continuity_limits(
        organ="generic",
        protocol="lattice_22b",
        n_voxels=int(args.n_voxels),
        seed=seed,
        check=False,
    )
    rt = run_nlls_roundtrip_selfcheck(
        organ="generic",
        protocol="lattice_22b",
        n_voxels=int(args.roundtrip_voxels),
        seed=seed,
        check=False,
    )
    elapsed = time.perf_counter() - t_start
    all_passed = bool(cont["passed"]) and bool(rt["passed"])

    if getattr(args, "json", False):
        payload = {
            "version": __version__,
            "passed": all_passed,
            "elapsed_seconds": round(elapsed, 4),
            "continuity": cont,
            "nlls_roundtrip": rt,
        }
        print(json.dumps(payload, indent=2))
        return 0 if all_passed else 1

    print(f"=== IVIMBench v{__version__} Clean-Room Self-Check ({elapsed:.2f}s) ===\n")
    rows = [
        "| Diagnostic Check | Target Limit | Observed Error | Tolerance | Status |",
        "| :--- | :--- | ---: | ---: | :---: |",
    ]
    for fam, info in cont["checks"].items():
        status = "PASS" if info["passed"] else "FAIL"
        rows.append(
            f"| Continuity: `{fam}` | {info['continuity_limit']} | "
            f"{info['max_abs_residual']:.3e} | <= {info['tolerance']:.3e} | {status} |"
        )
    rt_status = "PASS" if rt["passed"] else "FAIL"
    rows.append(
        f"| Noiseless `biexp` NLLS Round-Trip (`n={rt['n_voxels']}`) | "
        f"max rel error (D, D*, f) | {rt['max_rel_error']:.3e} | "
        f"<= {rt['tolerance']:.3e} | {rt_status} |"
    )
    print("\n".join(rows))
    overall_label = "PASSED" if all_passed else "FAILED"
    print(f"\nOverall Self-Check Status: {overall_label}")
    return 0 if all_passed else 1


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

    # 2. ivimbench selfcheck
    p_self = sub.add_parser(
        "selfcheck",
        help="Verify forward-signal continuity limits and noiseless bi-exponential NLLS round-trip recovery.",
    )
    p_self.add_argument(
        "--n-voxels",
        type=int,
        default=2000,
        help="Cohort size for continuity-limit verification (default: 2000).",
    )
    p_self.add_argument(
        "--roundtrip-voxels",
        type=int,
        default=60,
        help="Cohort size for noiseless NLLS round-trip recovery (default: 60).",
    )
    p_self.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_self.add_argument("--json", action="store_true", help="Emit structured JSON instead of table.")
    p_self.set_defaults(func=_cmd_selfcheck)

    # 3. ivimbench benchmark
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

    # 5. ivimbench anomalous-wall
    p_wall = sub.add_parser(
        "anomalous-wall",
        help="Map the fractional-order anomalous diffusion (D, alpha) CRLB and NLLS collapse wall.",
    )
    p_wall.add_argument(
        "--protocol",
        type=str,
        default="osipi",
        help="Comma-separated protocol keys (default: 'osipi').",
    )
    p_wall.add_argument("--snr-min", type=float, default=10.0, help="Minimum SNR on grid (default: 10.0).")
    p_wall.add_argument("--snr-max", type=float, default=80.0, help="Maximum SNR on grid (default: 80.0).")
    p_wall.add_argument("--n-snr", type=int, default=5, help="Number of SNR grid points (default: 5).")
    p_wall.add_argument(
        "--alphas",
        type=str,
        default="0.55,0.75,0.90,1.0",
        help="Comma-separated fractional orders alpha in (0, 1] (default: '0.55,0.75,0.90,1.0').",
    )
    p_wall.add_argument(
        "--n-replicates",
        type=int,
        default=36,
        help="Monte Carlo NLLS replicates per grid cell (default: 36).",
    )
    p_wall.add_argument("--n-boot", type=int, default=250, help="Bootstrap resamples for 95%% CIs (default: 250).")
    p_wall.add_argument(
        "--noise-type",
        type=str,
        default="rician",
        help="Magnitude noise model ('rician' or 'noncentral_chi', default: 'rician').",
    )
    p_wall.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_wall.add_argument("--out", type=str, default=None, help="Optional output directory for wall artifacts.")
    p_wall.add_argument("--json", action="store_true", help="Emit structured JSON instead of tables.")
    p_wall.set_defaults(func=_cmd_anomalous_wall)

    # 6. ivimbench closed-loop
    p_loop = sub.add_parser(
        "closed-loop",
        help="Run a synthetic-twin closed-loop adaptive radiotherapy simulation comparing uncalibrated vs. conformal UQ.",
    )
    p_loop.add_argument("--fractions", type=int, default=5, help="Number of RT fractions K (default: 5).")
    p_loop.add_argument("--snr", type=float, default=30.0, help="Nominal b=0 DWI SNR (default: 30.0).")
    p_loop.add_argument("--protocol", type=str, default="osipi", help="Acquisition protocol key (default: 'osipi').")
    p_loop.add_argument(
        "--substrate",
        type=str,
        default="organ",
        choices=("organ", "dose_geometry"),
        help="Synthetic twin geometry substrate ('organ' or 'dose_geometry', default: 'organ').",
    )
    p_loop.add_argument("--estimator", type=str, default="crlb", help="Registered estimator name (default: 'crlb').")
    p_loop.add_argument("--alpha", type=float, default=0.10, help="Miscoverage rate alpha (default: 0.10).")
    p_loop.add_argument(
        "--ere",
        action="store_true",
        help="Enable 1.5T MR-Linac 7 MV FFF Electron Return Effect (ERE) boundary dose perturbations.",
    )
    p_loop.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_loop.add_argument("--out", type=str, default=None, help="Optional output directory for loop artifacts.")
    p_loop.add_argument("--json", action="store_true", help="Emit structured JSON instead of tables.")
    p_loop.set_defaults(func=_cmd_closed_loop)

    # 7. ivimbench pipeline
    p_pipe = sub.add_parser(
        "pipeline",
        help="Run the unified 4-stage `Trust -> Value -> Action` cross-package pipeline.",
    )
    p_pipe.add_argument("--organ", type=str, default="pancreas", help="Organ prior (default: 'pancreas').")
    p_pipe.add_argument("--protocol", type=str, default="osipi", help="Acquisition protocol key (default: 'osipi').")
    p_pipe.add_argument("--snr", type=float, default=30.0, help="Nominal b=0 DWI SNR (default: 30.0).")
    p_pipe.add_argument(
        "--conformal",
        type=str,
        default="cqr",
        choices=("split", "cqr", "mondrian"),
        help="Conformal calibration method ('split', 'cqr', or 'mondrian', default: 'cqr').",
    )
    p_pipe.add_argument("--alpha", type=float, default=0.10, help="Miscoverage rate alpha (default: 0.10).")
    p_pipe.add_argument("--fractions", type=int, default=5, help="Closed-loop RT fractions (default: 5).")
    p_pipe.add_argument(
        "--ere",
        action="store_true",
        help="Enable 1.5T MR-Linac ERE dose perturbation in Stage 4 closed-loop action.",
    )
    p_pipe.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_pipe.add_argument("--out", type=str, default=None, help="Optional output directory for pipeline JSON/summary.")
    p_pipe.add_argument("--json", action="store_true", help="Emit structured JSON instead of summary table.")
    p_pipe.set_defaults(func=_cmd_pipeline)

    # 8. ivimbench fit-volume
    p_vol = sub.add_parser(
        "fit-volume",
        help="Fit a 3D/4D NIfTI or DICOM DWI volume and export affine-preserved parameter & UQ NIfTI maps.",
    )
    p_vol.add_argument("--dwi", type=str, default=None, help="Path to 4D DWI NIfTI (.nii/.nii.gz) or DICOM folder.")
    p_vol.add_argument("--bval", type=str, default=None, help="Path to FSL-style .bval file.")
    p_vol.add_argument("--mask", type=str, default=None, help="Optional path to 3D binary ROI mask NIfTI.")
    p_vol.add_argument(
        "--demo-volume",
        action="store_true",
        help="Synthesize a 3D DRO, export to NIfTI, fit it, and write the full clinical NIfTI bundle.",
    )
    p_vol.add_argument("--out", type=str, required=True, help="Output directory for fitted NIfTI maps.")
    p_vol.add_argument(
        "--estimator",
        type=str,
        default="conformal_ivim",
        help="Registered estimator name (default: 'conformal_ivim').",
    )
    p_vol.add_argument("--organ", type=str, default="pancreas", help="Organ prior for --demo-volume (default: 'pancreas').")
    p_vol.add_argument("--protocol", type=str, default="osipi", help="Protocol for --demo-volume (default: 'osipi').")
    p_vol.add_argument("--snr", type=float, default=30.0, help="SNR for --demo-volume (default: 30.0).")
    p_vol.add_argument("--alpha", type=float, default=0.10, help="Miscoverage rate alpha (default: 0.10).")
    p_vol.add_argument("--prefix", type=str, default="", help="Optional filename prefix for saved NIfTI maps.")
    p_vol.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_vol.add_argument("--json", action="store_true", help="Emit structured JSON summary.")
    p_vol.set_defaults(func=_cmd_fit_volume)

    # 9. ivimbench reproduce-gnomon
    p_gnom = sub.add_parser(
        "reproduce-gnomon",
        help="Run the Gnomon clean-room reproduce-or-refute audit against pinned manifest targets.",
    )
    p_gnom.add_argument("--n-voxels", type=int, default=2000, help="Clean-room voxel count (default: 2000).")
    p_gnom.add_argument("--alpha", type=float, default=0.10, help="Nominal miscoverage alpha (default: 0.10).")
    p_gnom.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_gnom.add_argument("--out", type=str, default=None, help="Optional output directory for audit report.")
    p_gnom.add_argument("--json", action="store_true", help="Emit structured JSON instead of summary table.")
    p_gnom.set_defaults(func=_cmd_reproduce_gnomon)

    # 10. ivimbench forge-ere-check
    p_ere = sub.add_parser(
        "forge-ere-check",
        help="Run the Forge 7 MV FFF 1.5T MR-Linac ERE boundary dose perturbation & photon-purity QA check.",
    )
    p_ere.add_argument("--b-field", type=float, default=1.5, help="Transverse magnetic field in Tesla (default: 1.5).")
    p_ere.add_argument("--n-bins", type=int, default=64, help="Energy spectrum bins (default: 64).")
    p_ere.add_argument("--n-cases", type=int, default=10000, help="MC benchmark case count (default: 10000).")
    p_ere.add_argument("--out", type=str, default=None, help="Optional output directory for QA JSON.")
    p_ere.add_argument("--json", action="store_true", help="Emit structured JSON instead of QA summary.")
    p_ere.set_defaults(func=_cmd_forge_ere_check)

    # 11. ivimbench audit
    p_audit = sub.add_parser(
        "audit",
        help="Run standardized Suite I Stage 3 AuditReport across DRO continuity, NLLS round-trip, Gnomon, and closed-loop RT.",
    )
    p_audit.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_audit.add_argument("--json", action="store_true", help="Emit standardized machine-readable AuditReport JSON.")
    p_audit.set_defaults(func=_cmd_audit)

    # 12. ivimbench report
    p_rep = sub.add_parser(
        "report",
        help="Generate a standalone interactive HTML leaderboard report with inline SVGs and Gnomon badges.",
    )
    p_rep.add_argument(
        "--out",
        type=str,
        default="ivimbench_report.html",
        help="Output HTML file path (default: 'ivimbench_report.html').",
    )
    p_rep.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional existing benchmark JSON summary path; if omitted, runs a fast benchmark first.",
    )
    p_rep.add_argument(
        "--protocol",
        type=str,
        default="osipi,acrin_6698",
        help="Comma-separated protocols when running on-the-fly (default: 'osipi,acrin_6698').",
    )
    p_rep.add_argument(
        "--estimators",
        type=str,
        default="segmented,crlb",
        help="Comma-separated estimators when running on-the-fly (default: 'segmented,crlb').",
    )
    p_rep.add_argument("--snr", type=str, default="30", help="Comma-separated SNRs (default: '30').")
    p_rep.add_argument("--n-voxels", type=int, default=80, help="Cohort size per task (default: 80).")
    p_rep.add_argument("--alpha", type=float, default=0.10, help="Miscoverage rate alpha (default: 0.10).")
    p_rep.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_rep.set_defaults(func=_cmd_report)

    # 13. ivimbench export-bids
    p_bids = sub.add_parser(
        "export-bids",
        help="Export BIDS-Derivatives (qMRI) NIfTI parameter maps, JSON sidecars, and QC summary.",
    )
    p_bids.add_argument("--subject", type=str, default="01", help="BIDS subject label without PHI (default: '01').")
    p_bids.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output BIDS derivatives root directory (e.g. './bids_derivatives/').",
    )
    p_bids.add_argument("--organ", type=str, default="liver", help="Synthetic DRO organ preset (default: 'liver').")
    p_bids.add_argument("--protocol", type=str, default="osipi", help="Acquisition protocol key (default: 'osipi').")
    p_bids.add_argument(
        "--estimator",
        type=str,
        default="segmented",
        help="Registered estimator used to fit the synthetic DRO (default: 'segmented').",
    )
    p_bids.add_argument("--snr", type=float, default=30.0, help="Nominal b=0 SNR (default: 30.0).")
    p_bids.add_argument(
        "--shape",
        type=str,
        default="16,16,8",
        help="Comma-separated 2D or 3D spatial DRO shape with dims >= 8 (default: '16,16,8').",
    )
    p_bids.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_bids.add_argument("--json", action="store_true", help="Emit JSON manifest to stdout.")
    p_bids.set_defaults(func=_cmd_export_bids)

    # 14. ivimbench suite-audit
    p_suite = sub.add_parser(
        "suite-audit",
        help="Run cross-package AveryCorp Suite I & Suite II audit across all 7 registered packages.",
    )
    p_suite.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=("all", "1", "2"),
        help="Suite filter ('all', '1' for Suite I, or '2' for Suite II; default: 'all').",
    )
    p_suite.add_argument(
        "--format",
        dest="output_format",
        type=str,
        default="table",
        choices=("table", "markdown", "latex", "html", "json"),
        help="Output rendering format (default: 'table').",
    )
    p_suite.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_suite.add_argument("--strict", action="store_true", help="Treat uninstalled companion packages as HALT.")
    p_suite.add_argument("--out", type=str, default=None, help="Optional output file path for rendered report.")
    p_suite.add_argument("--json", action="store_true", help="Emit machine-readable SuiteAuditReport JSON.")
    p_suite.set_defaults(func=_cmd_suite_audit)

    return parser


def _cmd_suite_audit(args: argparse.Namespace) -> int:
    from .suite_runner import (
        format_suite_html,
        format_suite_latex,
        format_suite_markdown,
        format_suite_table,
        run_suite_audit,
    )

    report = run_suite_audit(
        seed=int(args.seed),
        suite_filter=str(args.suite),
        strict=bool(getattr(args, "strict", False)),
    )
    fmt = "json" if getattr(args, "json", False) else str(args.output_format)
    if fmt == "json":
        rendered = report.to_json()
    elif fmt == "markdown":
        rendered = format_suite_markdown(report)
    elif fmt == "latex":
        rendered = format_suite_latex(report)
    elif fmt == "html":
        rendered = format_suite_html(report)
    else:
        rendered = format_suite_table(report)

    if getattr(args, "out", None):
        out_fp = Path(args.out)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_text(rendered + "\n", encoding="utf-8")

    print(rendered)
    return 0 if report.overall_status != "HALT" else 1


def _cmd_audit(args: argparse.Namespace) -> int:
    report = run_audit(seed=int(args.seed))
    if getattr(args, "json", False):
        print(report.to_json())
    else:
        print(report.summary_line())
        for k, v in report.metrics.items():
            print(f"  {k:<32}: {v}")
    return 0 if report.status != "HALT" else 1


def _cmd_anomalous_wall(args: argparse.Namespace) -> int:
    protocols = _split_csv(args.protocol)
    snr_min = float(args.snr_min)
    snr_max = float(args.snr_max)
    if snr_min <= 0.0 or snr_max < snr_min:
        raise ValueError(f"Require 0 < snr_min <= snr_max; got ({snr_min}, {snr_max}).")
    snr_grid = np.linspace(snr_min, snr_max, max(int(args.n_snr), 2))
    alpha_grid = [float(v) for v in _split_csv(args.alphas)]

    report = map_recovery_collapse_wall(
        b_schedules=protocols,
        snr_grid=snr_grid,
        alpha_grid=alpha_grid,
        seed=int(args.seed),
        n_replicates=int(args.n_replicates),
        noise_type=args.noise_type,
        n_boot=int(args.n_boot),
    )

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "anomalous_wall.md").write_text(report.to_markdown() + "\n", encoding="utf-8")
        (out_dir / "anomalous_wall.json").write_text(report.to_json() + "\n", encoding="utf-8")
        (out_dir / "anomalous_wall.tex").write_text(report.to_latex() + "\n", encoding="utf-8")
        report.dataframe.to_csv(out_dir / "anomalous_wall_records.csv", index=False)

    if getattr(args, "json", False):
        print(report.to_json())
        return 0

    print(
        f"=== IVIMBench v{__version__} Anomalous Fractional-Order Diffusion Recovery Wall ===\n"
    )
    print(report.to_markdown())
    print("\n[Collapse Wall Frontier (SNR x b_max)]")
    print(dataframe_to_markdown(report.wall_front, float_fmt=".3f"))
    if args.out:
        print(f"\nSaved anomalous wall artifacts to: {Path(args.out).as_posix()}")
    return 0


def _cmd_closed_loop(args: argparse.Namespace) -> int:
    substrate_mode = "organ" if args.substrate == "organ" else "dose_geometry"
    report = compare_closed_loop_policies(
        substrate=substrate_mode,  # type: ignore[arg-type]
        protocol=args.protocol,
        estimator=args.estimator,
        n_fractions=int(args.fractions),
        snr=float(args.snr),
        alpha=float(args.alpha),
        apply_mr_linac_ere=bool(getattr(args, "ere", False)),
        seed=int(args.seed),
    )

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "closed_loop_summary.md").write_text(report.to_markdown() + "\n", encoding="utf-8")
        (out_dir / "closed_loop_summary.json").write_text(report.to_json() + "\n", encoding="utf-8")
        (out_dir / "closed_loop_summary.tex").write_text(report.to_latex() + "\n", encoding="utf-8")
        report.policy_df.to_csv(out_dir / "closed_loop_policies.csv", index=False)

    if getattr(args, "json", False):
        print(report.to_json())
        return 0

    print(
        f"=== IVIMBench v{__version__} Closed-Loop Synthetic-Twin Adaptive RT "
        f"(K={int(args.fractions)}, SNR={float(args.snr):.1f}, substrate={args.substrate}) ===\n"
    )
    print("[Policy Outcome & Dose-Safety Summary]")
    print(report.to_markdown())
    print(
        f"\n  - Regret Reduction (Uncalibrated -> Conformal): "
        f"{report.summary['regret_reduction']:+.4f}\n"
        f"  - False-Escalation Reduction: "
        f"{report.summary['false_escalation_reduction']:+.4f}\n"
        f"  - Cumulative OAR Dose Sparing vs. Uncalibrated: "
        f"{report.summary['conformal_oar_sparing_vs_uncalibrated_gy']:+.4f} Gy"
    )
    if args.out:
        print(f"\nSaved closed-loop artifacts to: {Path(args.out).as_posix()}")
    return 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    report = run_trust_value_action_pipeline(
        protocol=args.protocol,
        organ=args.organ,
        snr=float(args.snr),
        conformal_method=args.conformal,
        alpha=float(args.alpha),
        fractions=int(args.fractions),
        seed=int(args.seed),
        apply_mr_linac_ere=bool(args.ere),
    )
    suite1_chain = run_suite1_pipeline(seed=int(args.seed))
    combined_dict = {
        **report.to_dict(),
        "suite": suite1_chain["suite"],
        "pipeline": suite1_chain["pipeline"],
        "ecosystem_complete": suite1_chain["ecosystem_complete"],
        "overall_status": suite1_chain["overall_status"],
        "stages": suite1_chain["stages"],
    }
    combined_json = json.dumps(combined_dict, indent=2, sort_keys=True)
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pipeline_report.json").write_text(combined_json + "\n", encoding="utf-8")
        (out_dir / "pipeline_summary.txt").write_text(
            report.format_summary_table() + "\n", encoding="utf-8"
        )
    if getattr(args, "json", False):
        print(combined_json)
        return 0
    print(report.format_summary_table())
    if args.out:
        print(f"\nSaved pipeline report to: {Path(args.out).as_posix()}")
    return 0


def _cmd_fit_volume(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo_volume:
        dro = generate_dro(
            shape=(10, 10, 8),
            organ=args.organ,
            protocol=args.protocol,
            snr=float(args.snr),
            seed=int(args.seed),
        )
        dro_dir = out_dir / "demo_dro_input"
        dro_paths = export_dro_to_nifti(dro, dro_dir)
        dwi_path = dro_paths["dwi"]
        bval_path: Path | str | None = dro_paths["bval"]
        mask_path: Path | str | None = dro_paths["roi_mask"]
    else:
        if not args.dwi:
            raise ValueError("Provide `--dwi <path>` or pass `--demo-volume`.")
        dwi_path = Path(args.dwi)
        bval_path = args.bval
        mask_path = args.mask

    vol, est_out, written = fit_dwi_volume(
        dwi_path,
        bval_path,
        out_dir,
        mask_path=mask_path,
        estimator=args.estimator,
        alpha=float(args.alpha),
        prefix=args.prefix,
        estimator_kwargs={"seed": int(args.seed)},
    )
    railed_rate = (
        float(np.mean(est_out.railed[:, 1])) if est_out.railed is not None else 0.0
    )
    payload = {
        "spatial_shape": list(vol.spatial_shape),
        "n_masked_voxels": int(vol.n_masked),
        "n_bvalues": int(vol.n_bvalues),
        "estimator": args.estimator,
        "dstar_railed_fraction": railed_rate,
        "files": {k: v.as_posix() for k, v in written.items()},
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
        return 0

    print(
        f"=== IVIMBench v{__version__} Volumetric 3D/4D Fit "
        f"(shape={vol.spatial_shape}, n_masked={vol.n_masked}, estimator={args.estimator}) ==="
    )
    print(f"  - D* Boundary-Railed Fraction: {100.0 * railed_rate:.1f}%")
    for key, fp in written.items():
        print(f"  - {key:<22}: {fp.as_posix()}")
    return 0


def _cmd_reproduce_gnomon(args: argparse.Namespace) -> int:
    verdict = run_gnomon_cleanroom_audit(
        n_voxels=int(args.n_voxels),
        seed=int(args.seed),
        alpha=float(args.alpha),
    )
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gnomon_audit.json").write_text(verdict.to_json() + "\n", encoding="utf-8")
        (out_dir / "gnomon_audit.txt").write_text(
            verdict.format_summary_table() + "\n", encoding="utf-8"
        )
    if getattr(args, "json", False):
        print(verdict.to_json())
        return 0
    print(verdict.format_summary_table())
    return 0


def _cmd_forge_ere_check(args: argparse.Namespace) -> int:
    spec = generate_unity_7mv_fff_spectrum(n_bins=int(args.n_bins))
    purity = verify_photon_beam_purity([spec.particle] * int(args.n_cases))

    # Construct 3D water-air-lung-tissue slab phantom (24 x 16 x 16) irradiated along +x
    nx, ny, nz = 24, 16, 16
    depth_idx = np.arange(nx, dtype=np.float64)[:, None, None]
    base_dose = np.broadcast_to(2.0 * np.exp(-0.018 * depth_idx), (nx, ny, nz)).copy()
    density = np.ones((nx, ny, nz), dtype=np.float64)
    density[9:14, :, :] = 0.0012   # Central air cavity (exit at i=8, re-entry at i=14)
    density[18:21, :, :] = 0.26    # Downstream lung slab

    ere_on = compute_ere_boundary_perturbation(
        base_dose, density, b_field_tesla=float(args.b_field), beam_axis=0
    )
    ere_zero = compute_ere_boundary_perturbation(
        base_dose, density, b_field_tesla=0.0, beam_axis=0
    )
    floor_rep = estimate_mc_simulation_floor(n_cases=int(args.n_cases))

    zero_b_max_abs = float(np.max(np.abs(ere_zero.delta_dose)))
    qa_pass = (
        bool(purity["zero_proton_contamination"])
        and zero_b_max_abs == 0.0
        and (
            float(args.b_field) == 0.0
            or (ere_on.peak_exit_enhancement_pct > 0.0 and ere_on.within_reference_band)
        )
    )

    payload = {
        "qa_status": "PASS" if qa_pass else "WARN",
        "spectrum": {
            "n_bins": spec.n_bins,
            "mean_energy_mev": spec.mean_energy_mev,
            "endpoint_mv": spec.endpoint_mv,
            "modality": spec.modality,
        },
        "photon_purity": purity,
        "ere_active_field": ere_on.to_dict(),
        "ere_zero_field_max_abs_gy": zero_b_max_abs,
        "mc_simulation_floor": floor_rep.to_dict(),
    }

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "forge_ere_qa.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(
        f"=== IVIMBench v{__version__} Forge 1.5T MR-Linac 7 MV FFF & ERE QA Check -> "
        f"{payload['qa_status']} ==="
    )
    print(
        f"  [1] 7 MV FFF Spectrum : {spec.n_bins} bins | mean E = {spec.mean_energy_mev:.3f} MeV | "
        f"zero_proton_contamination = {purity['zero_proton_contamination']} (n={purity['n_cases']})"
    )
    print(
        f"  [2] ERE @ B={ere_on.b_field_tesla:.2f} T     : "
        f"peak water->air exit hotspot = +{ere_on.peak_exit_enhancement_pct:.2f}% "
        f"(Unity Geant4 ref {ere_on.reference_band_pct[0]:.1f}-{ere_on.reference_band_pct[1]:.1f}% "
        f"+/- {ere_on.tolerance_pct:.1f}%, within_band={ere_on.within_reference_band})"
    )
    print(
        f"  [3] Zero-Field Control: max |Delta D_ERE(B=0 T)| = {zero_b_max_abs:.3e} Gy"
    )
    print(
        f"  [4] MC Floor ({floor_rep.n_cases:,} cases, {floor_rep.histories_per_case:.1e} hist/case): "
        f"{floor_rep.total_core_hours:,.1f} core-h ({floor_rep.wall_days_64_cores:.1f} d @ 64c) | "
        f"high-dose sigma = {floor_rep.estimated_high_dose_sigma_pct:.2f}% "
        f"(gate={floor_rep.ere_gate_status}, verdict={floor_rep.feasibility_verdict})"
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    if args.input:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        out_file = generate_html_report(payload, out_path=args.out)
    else:
        protocols = _split_csv(args.protocol)
        estimators = _split_csv(args.estimators)
        snrs = [float(v) for v in _split_csv(str(args.snr))]
        bench = run_benchmark_suite(
            estimators=estimators,
            protocols=protocols,
            families=("biexp", "dispersion_gamma", "triexp"),
            snr_regimes=snrs,
            n_voxels=int(args.n_voxels),
            alpha=float(args.alpha),
            n_boot=60,
            seed=int(args.seed),
        )
        out_file = generate_html_report(bench, out_path=args.out)
    print(f"Generated standalone interactive HTML report: {out_file.as_posix()}")
    return 0


def _cmd_export_bids(args: argparse.Namespace) -> int:
    shape_tuple = tuple(int(v) for v in _split_csv(args.shape))
    if len(shape_tuple) not in (2, 3):
        raise ValueError(f"`--shape` must have 2 or 3 integer dimensions; got {args.shape!r}.")
    dro = generate_dro(
        shape=shape_tuple,
        organ=args.organ,
        protocol=args.protocol,
        snr=float(args.snr),
        seed=int(args.seed),
    )
    sig_fg, _ = dro.to_voxels(foreground_only=True)
    pred = run_estimator(
        args.estimator,
        sig_fg,
        dro.bvalues,
        snr=float(args.snr),
    )
    res = export_bids_derivatives(
        output_dir=args.out,
        subject_id=args.subject,
        fit_result=pred,
        volume_meta={
            "mask": dro.foreground_mask,
            "shape": dro.spatial_shape,
            "bvalues": dro.bvalues,
            "estimation_algorithm": str(args.estimator),
        },
        protocol_name=args.protocol,
    )
    if getattr(args, "json", False):
        manifest = {
            "subject_id": res.subject_id,
            "output_dir": res.output_dir.as_posix(),
            "dataset_description": res.dataset_description.as_posix(),
            "nifti_paths": {k: p.as_posix() for k, p in res.nifti_paths.items()},
            "sidecar_paths": {k: p.as_posix() for k, p in res.sidecar_paths.items()},
            "svg_summary": res.svg_summary.as_posix(),
            "html_summary": res.html_summary.as_posix(),
            "summary_stats": res.summary_stats,
        }
        print(json.dumps(manifest, indent=2))
        return 0

    print(
        f"Exported BIDS-Derivatives qMRI dataset for sub-{res.subject_id} to: "
        f"{res.output_dir.as_posix()} ({len(res.nifti_paths)} NIfTIs + sidecars + QC SVG/HTML)"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point returning an integer exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
