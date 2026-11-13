"""Standalone console CLI for domain-agnostic anomalous transport & misspecification (``levy-wall``)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from .._version import __version__
from .core import (
    DEFAULT_Q_GRID,
    map_anomalous_identifiability_wall,
    probe_misspecification_conditional_aliasing,
    run_transport_audit,
)

__all__ = ["build_parser", "main"]


def _cmd_audit(args: argparse.Namespace) -> int:
    rep = run_transport_audit(seed=int(args.seed))
    if getattr(args, "json", False):
        print(rep.to_json())
    else:
        print(rep.summary_line())
        for k, v in rep.metrics.items():
            print(f"  {k:<34}: {v}")
    return 0 if rep.status != "HALT" else 1


def _cmd_crlb_wall(args: argparse.Namespace) -> int:
    snr_min = float(args.snr_min)
    snr_max = float(args.snr_max)
    if snr_min <= 0.0 or snr_max < snr_min:
        raise ValueError(f"Expected 0 < snr_min <= snr_max; got ({snr_min}, {snr_max}).")
    snr_grid = np.linspace(snr_min, snr_max, max(int(args.n_snr), 2))
    alpha_grid = [float(x.strip()) for x in str(args.alphas).split(",") if x.strip()]
    q_base = np.asarray(DEFAULT_Q_GRID, dtype=np.float64)
    q_scaled = q_base * (float(args.q_max) / float(np.max(q_base)))

    report = map_anomalous_identifiability_wall(
        q_schedules={"q_schedule": q_scaled},
        snr_grid=snr_grid,
        alpha_grid=alpha_grid,
        seed=int(args.seed),
        n_replicates=int(args.n_replicates),
        n_boot=int(args.n_boot),
    )
    if getattr(args, "json", False):
        print(report.to_json())
        return 0
    print(f"=== levy-wall v{__version__} Fractional Transport Identifiability Wall ===\n")
    print(report.to_markdown())
    return 0


def _cmd_procrustes_check(args: argparse.Namespace) -> int:
    report = probe_misspecification_conditional_aliasing(
        q=DEFAULT_Q_GRID,
        family=str(args.family),
        snr=float(args.snr),
        seed=int(args.seed),
        n_samples=int(args.n_samples),
        alpha=float(args.alpha),
    )
    if getattr(args, "json", False):
        print(report.to_json())
        return 0
    print(
        f"=== levy-wall v{__version__} Procrustes Misspecification Conditional Aliasing Check ===\n"
        f"  Family                      : {report.family} (SNR={report.snr:.1f}, n_test={report.n_samples})\n"
        f"  Global Marginal Coverage    : {report.global_marginal_coverage:.3f} (target {report.nominal_target_coverage:.2f})\n"
        f"  Tercile Coverages (low/m/hi): "
        f"{report.tercile_coverages['low_fast_weight']:.3f} / "
        f"{report.tercile_coverages['mid_fast_weight']:.3f} / "
        f"{report.tercile_coverages['high_fast_weight']:.3f}\n"
        f"  Conditional Aliasing Gap    : {report.conditional_coverage_aliasing_gap:.3f} "
        f"(aliasing_detected={report.aliasing_detected})"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for ``levy-wall`` CLI."""
    parser = argparse.ArgumentParser(
        prog="levy-wall",
        description="Domain-agnostic stretched-exponential / CTRW transport CRLB wall and Procrustes misspecification gate.",
    )
    parser.add_argument("--version", action="version", version=f"levy-wall (ivimbench {__version__})")
    sub = parser.add_subparsers(dest="command", required=True)

    p_aud = sub.add_parser("audit", help="Run deterministic transport & Procrustes audit.")
    p_aud.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_aud.add_argument("--json", action="store_true", help="Emit machine-readable AuditReport JSON.")
    p_aud.set_defaults(func=_cmd_audit)

    p_wall = sub.add_parser("crlb-wall", help="Map the (K, alpha) fractional transport CRLB & NLLS collapse wall.")
    p_wall.add_argument("--snr-min", type=float, default=10.0, help="Minimum SNR on grid (default: 10.0).")
    p_wall.add_argument("--snr-max", type=float, default=80.0, help="Maximum SNR on grid (default: 80.0).")
    p_wall.add_argument("--n-snr", type=int, default=3, help="Number of SNR grid points (default: 3).")
    p_wall.add_argument("--q-max", type=float, default=1000.0, help="Maximum wavevector/Laplace q (default: 1000.0).")
    p_wall.add_argument("--alphas", type=str, default="0.60,0.80,1.0", help="Comma-separated alpha grid.")
    p_wall.add_argument("--n-replicates", type=int, default=20, help="Monte Carlo replicates per cell (default: 20).")
    p_wall.add_argument("--n-boot", type=int, default=100, help="Bootstrap resamples (default: 100).")
    p_wall.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_wall.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_wall.set_defaults(func=_cmd_crlb_wall)

    p_proc = sub.add_parser(
        "procrustes-check",
        help="Measure conditional coverage aliasing on slow rate K_slow under continuous/heavy-tail decay.",
    )
    p_proc.add_argument("--family", type=str, default="dispersion_gamma", help="Decay family (default: dispersion_gamma).")
    p_proc.add_argument("--snr", type=float, default=40.0, help="Signal-to-noise ratio (default: 40.0).")
    p_proc.add_argument("--n-samples", type=int, default=360, help="Total calibration+test sample count (default: 360).")
    p_proc.add_argument("--alpha", type=float, default=0.10, help="Miscoverage rate alpha (default: 0.10).")
    p_proc.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_proc.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_proc.set_defaults(func=_cmd_procrustes_check)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console script entrypoint for ``levy-wall``."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
