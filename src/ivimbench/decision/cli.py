"""Standalone console CLI for domain-agnostic statistical decision theory & sequential stopping (``minos-gate``)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from .._version import __version__
from ..closed_loop.sentinel_stopping import evaluate_regret_vs_coverage_stopping
from .audit import run_decision_audit
from .voi import compute_decision_calibration_gap

__all__ = ["build_parser", "main"]


def _cmd_audit(args: argparse.Namespace) -> int:
    rep = run_decision_audit(seed=int(args.seed))
    if getattr(args, "json", False):
        print(rep.to_json())
    else:
        print(rep.summary_line())
        for k, v in rep.metrics.items():
            print(f"  {k:<34}: {v}")
    return 0 if rep.status != "HALT" else 1


def _cmd_gap_demo(args: argparse.Namespace) -> int:
    rng = np.random.Generator(np.random.PCG64(int(args.seed)))
    n = int(args.n_samples)
    truth = rng.normal(0.50, 0.18, size=n)
    pred = truth + rng.normal(0.0, 0.08, size=n)
    half_w = np.full(n, 0.09, dtype=np.float64)
    rep = compute_decision_calibration_gap(
        (pred - half_w, pred + half_w),
        truth,
        cost_asymmetry=float(args.cost_asymmetry),
        alpha=float(args.alpha),
        seed=int(args.seed),
    )
    if getattr(args, "json", False):
        print(rep.to_json())
        return 0
    auc_sep = rep.observable_shift_auc - rep.hidden_truth_shift_auc
    print(
        f"=== minos-gate v{__version__} Decision-Calibration Gap & Trust-Gate Pricing ===\n"
        f"  Cost Asymmetry (lambda)     : {rep.cost_asymmetry:.2f}\n"
        f"  Statistical Scale tau_stat  : {rep.tau_stat:.4f} (cov={rep.coverage_at_tau_stat:.3f})\n"
        f"  Utility-Optimal tau_star    : {rep.tau_star:.4f} (cov={rep.coverage_at_tau_star:.3f})\n"
        f"  Calibration Gap G           : {rep.gap:+.4f}\n"
        f"  Value of Calibration (VoC)  : {rep.value_of_calibration:.4f}\n"
        f"  Value of Trust-Gate (VoTG)  : {rep.value_of_trust_gate:.4f}\n"
        f"  Shift AUC (Obs vs Hidden)   : {rep.observable_shift_auc:.3f} vs {rep.hidden_truth_shift_auc:.3f} "
        f"(sep={auc_sep:+.3f})"
    )
    return 0


def _cmd_sentinel_stopping(args: argparse.Namespace) -> int:
    rep = evaluate_regret_vs_coverage_stopping(
        None,
        seed=int(args.seed),
        n_boot=int(args.n_boot),
        cost_asymmetry=float(args.cost_asymmetry),
    )
    if getattr(args, "json", False):
        print(rep.to_json())
        return 0
    print(
        f"=== minos-gate v{__version__} Sentinel Regret vs. Coverage Sequential Stopping ===\n"
        f"  Coverage Stabilization Step : F{rep.coverage_stop_fraction}\n"
        f"  Regret Stabilization Step   : F{rep.regret_stop_fraction}\n"
        f"  Stopping Step Gap (Delta F) : {rep.stopping_fraction_gap:+.2f} "
        f"95% CI [{rep.stopping_gap_ci95[0]:+.2f}, {rep.stopping_gap_ci95[1]:+.2f}]\n"
        f"  Excess Regret @ cov_stop    : {rep.excess_regret_of_coverage_stop:+.4f} "
        f"95% CI [{rep.excess_regret_ci95[0]:+.4f}, {rep.excess_regret_ci95[1]:+.4f}]\n"
        f"  Diagnostic Gate Verdict     : {rep.verdict}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for ``minos-gate``."""
    parser = argparse.ArgumentParser(
        prog="minos-gate",
        description="Domain-agnostic asymmetric decision-calibration gap pricer (Minos) & sequential stopping auditor (Sentinel).",
    )
    parser.add_argument("--version", action="version", version=f"minos-gate (ivimbench {__version__})")
    sub = parser.add_subparsers(dest="command", required=True)

    p_aud = sub.add_parser("audit", help="Run deterministic Minos + Sentinel decision audit.")
    p_aud.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_aud.add_argument("--json", action="store_true", help="Emit machine-readable AuditReport JSON.")
    p_aud.set_defaults(func=_cmd_audit)

    p_gap = sub.add_parser(
        "gap-demo",
        help="Compute G = tau* - tau_stat, VoC, VoTG, and observable vs. hidden shift AUC separation.",
    )
    p_gap.add_argument(
        "--cost-asymmetry",
        type=float,
        default=3.0,
        help="Cost asymmetry lambda > 0 (default: 3.0).",
    )
    p_gap.add_argument("--alpha", type=float, default=0.10, help="Miscoverage rate alpha (default: 0.10).")
    p_gap.add_argument("--n-samples", type=int, default=600, help="Sample size (default: 600).")
    p_gap.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_gap.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_gap.set_defaults(func=_cmd_gap_demo)

    p_sent = sub.add_parser(
        "sentinel-stopping",
        help="Compare rolling coverage-stability stopping vs. utility regret stabilization.",
    )
    p_sent.add_argument(
        "--cost-asymmetry",
        type=float,
        default=3.0,
        help="Cost asymmetry lambda > 0 (default: 3.0).",
    )
    p_sent.add_argument("--n-boot", type=int, default=250, help="Bootstrap resamples (default: 250).")
    p_sent.add_argument("--seed", type=int, default=42, help="Deterministic PCG64 seed (default: 42).")
    p_sent.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_sent.set_defaults(func=_cmd_sentinel_stopping)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console script entrypoint for ``minos-gate``."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
