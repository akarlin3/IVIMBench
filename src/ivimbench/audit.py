"""Standardized AveryCorp Suite I (Stage 3) clean-room benchmark & chained pipeline audit."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.stats import norm

from ._version import __version__
from .closed_loop.loop import SyntheticOrganTwin, compare_closed_loop_policies
from .metrics.coverage import DEFAULT_QUANTILE_LEVELS, ece_quantile, empirical_coverage
from .phantoms.selfcheck import run_nlls_roundtrip_selfcheck, verify_continuity_limits

SUITE_I_NAME: Literal["Suite I: Clinical Quantitative-MRI Trust Spine"] = (
    "Suite I: Clinical Quantitative-MRI Trust Spine"
)
SUITE_II_NAME: Literal["Suite II: Scientific Identifiability & Null-Audit Suite"] = (
    "Suite II: Scientific Identifiability & Null-Audit Suite"
)
SuiteLiteral = Literal[
    "Suite I: Clinical Quantitative-MRI Trust Spine",
    "Suite II: Scientific Identifiability & Null-Audit Suite",
]
StatusLiteral = Literal["PASS", "WARN", "HALT"]


def _to_builtin(val: Any) -> float | int | str | bool:
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return float(val)
    return str(val)


@dataclass(frozen=True)
class AuditReport:
    """Standardized AveryCorp scientific identifiability and diagnostic audit report."""

    package: str
    version: str
    suite: SuiteLiteral
    stage_role: str
    status: StatusLiteral
    verdict: str
    metrics: dict[str, float | int | str | bool]
    thresholds: dict[str, float | int | str]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize the audit report to a JSON-compatible Python dictionary."""
        return {
            "package": str(self.package),
            "version": str(self.version),
            "suite": str(self.suite),
            "stage_role": str(self.stage_role),
            "status": str(self.status),
            "verdict": str(self.verdict),
            "metrics": {str(k): _to_builtin(v) for k, v in self.metrics.items()},
            "thresholds": {str(k): _to_builtin(v) for k, v in self.thresholds.items()},
            "seed": int(self.seed),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the audit report to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary_line(self) -> str:
        """Return a concise one-line human-readable status summary."""
        return (
            f"[{self.status}] {self.package} v{self.version} "
            f"({self.suite} | {self.stage_role}) -> {self.verdict}"
        )


def _verify_gnomon_ruler(seed: int, n_samples: int = 2000) -> dict[str, float | bool]:
    """Verify clean-room Gnomon/Caliper coverage and quantile ECE rulers on an analytical Gaussian oracle."""
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    mu = rng.uniform(-1.0, 1.0, size=n_samples)
    sigma = rng.uniform(0.15, 0.65, size=n_samples)
    y_true = rng.normal(mu, sigma)

    z90 = float(norm.ppf(0.95))
    lo90 = mu - z90 * sigma
    hi90 = mu + z90 * sigma
    cov90 = empirical_coverage(y_true, lo90, hi90, alpha=0.10)

    q_levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)
    z_q = norm.ppf(q_levels)[None, :]
    q_pred = mu[:, None] + sigma[:, None] * z_q
    ece = ece_quantile(y_true, q_pred, q_levels)

    cov_err = abs(cov90 - 0.90)
    passed = (cov_err <= 0.03) and (ece <= 0.025)
    return {
        "gnomon_cov90": float(cov90),
        "gnomon_cov_abs_error": float(cov_err),
        "gnomon_quantile_ece": float(ece),
        "gnomon_passed": bool(passed),
    }


def run_audit(
    seed: int = 42,
    *,
    continuity_voxels: int = 250,
    roundtrip_voxels: int = 20,
    rt_fractions: int = 4,
    max_gnomon_ece: float = 0.025,
) -> AuditReport:
    """Run the Stage 3 Suite I clean-room DRO, Gnomon ruler, and closed-loop RT audit."""
    cont = verify_continuity_limits(
        organ="generic",
        protocol="lattice_22b",
        n_voxels=continuity_voxels,
        seed=int(seed),
        check=False,
    )
    rt = run_nlls_roundtrip_selfcheck(
        organ="generic",
        protocol="lattice_22b",
        n_voxels=roundtrip_voxels,
        seed=int(seed),
        check=False,
    )
    gnomon = _verify_gnomon_ruler(seed=int(seed))
    organ_twin = SyntheticOrganTwin(shape=(18, 18), organ="pancreas", seed=int(seed))
    loop_rep = compare_closed_loop_policies(
        substrate=organ_twin,
        protocol="osipi",
        estimator="segmented",
        n_fractions=int(rt_fractions),
        snr=25.0,
        alpha=0.10,
        cost_ratio=0.75,
        seed=int(seed),
    )

    max_cont_res = max(float(v["max_abs_residual"]) for v in cont["checks"].values())
    rt_max_rel = float(rt["max_rel_error"])
    gnomon_ece = float(gnomon["gnomon_quantile_ece"])
    regret_red = float(loop_rep.summary["regret_reduction"])
    oar_sparing = float(loop_rep.summary["conformal_oar_sparing_vs_uncalibrated_gy"])

    cont_ok = bool(cont["passed"])
    rt_ok = bool(rt["passed"])
    gnomon_ok = bool(gnomon["gnomon_passed"]) and (gnomon_ece <= max_gnomon_ece)
    loop_ok = (regret_red >= 0.0) and (oar_sparing > 0.0)

    if cont_ok and rt_ok and gnomon_ok and loop_ok:
        status: StatusLiteral = "PASS"
        verdict = (
            f"CLEAN_ROOM_VERIFIED: Forward continuity (max res {max_cont_res:.2e}), "
            f"noiseless NLLS round-trip ({rt_max_rel:.2e}), Gnomon ECE ({gnomon_ece:.4f}), "
            f"and closed-loop VoI regret reduction ({regret_red:+.4f}) all verified."
        )
    elif cont_ok and rt_ok and gnomon_ok:
        status = "WARN"
        verdict = (
            f"CLEAN_ROOM_PARTIAL: Physics/Gnomon checks passed, but closed-loop regret "
            f"reduction was non-positive ({regret_red:+.4f})."
        )
    else:
        status = "HALT"
        verdict = (
            f"CLEAN_ROOM_FAILURE: Continuity={cont_ok}, RoundTrip={rt_ok}, "
            f"Gnomon={gnomon_ok}, ClosedLoop={loop_ok}."
        )

    return AuditReport(
        package="ivimbench",
        version=__version__,
        suite=SUITE_I_NAME,
        stage_role="Stage 3: Canonical DRO Phantoms, Clean-Room Gnomon/Caliper Rulers & Closed-Loop RT Benchmark",
        status=status,
        verdict=verdict,
        metrics={
            "continuity_passed": cont_ok,
            "continuity_max_abs_residual": max_cont_res,
            "nlls_roundtrip_passed": rt_ok,
            "nlls_roundtrip_max_rel_error": rt_max_rel,
            "gnomon_passed": gnomon_ok,
            "gnomon_quantile_ece": round(gnomon_ece, 6),
            "closed_loop_regret_reduction": round(regret_red, 6),
            "closed_loop_oar_sparing_gy": round(oar_sparing, 6),
        },
        thresholds={
            "continuity_tolerance": 1.0e-12,
            "nlls_roundtrip_tolerance": float(rt["tolerance"]),
            "max_gnomon_quantile_ece": float(max_gnomon_ece),
            "min_closed_loop_regret_reduction": 0.0,
        },
        seed=int(seed),
    )


def run_suite1_pipeline(seed: int = 42) -> dict[str, Any]:
    """Execute the chained 3-stage Suite I clinical quantitative-MRI trust spine audit.

    Combines ``sextantIVIM -> conformalIVIM -> IVIMBench`` when companion Suite I
    packages are installed in the Python environment, degrading gracefully otherwise.
    """
    stages: list[dict[str, Any]] = []
    companion_specs = (
        ("sextant_core", "sextant-ivim", 1),
        ("conformal_ivim", "conformal-ivim", 2),
    )
    ecosystem_complete = True

    for mod_name, pkg_label, stage_num in companion_specs:
        try:
            mod = importlib.import_module(mod_name)
            rep = mod.run_audit(seed=int(seed))
            stage_dict = rep.to_dict()
            stage_dict["stage_index"] = stage_num
            stage_dict["installed"] = True
            stages.append(stage_dict)
        except ImportError:
            ecosystem_complete = False
            stages.append(
                {
                    "stage_index": stage_num,
                    "package": pkg_label,
                    "installed": False,
                    "status": "SKIPPED",
                    "verdict": f"{pkg_label} not installed; install ivimbench[ecosystem] to chain all 3 stages.",
                }
            )

    rep_stage3 = run_audit(seed=int(seed))
    s3_dict = rep_stage3.to_dict()
    s3_dict["stage_index"] = 3
    s3_dict["installed"] = True
    stages.append(s3_dict)

    installed_statuses = [s["status"] for s in stages if s.get("installed")]
    if "HALT" in installed_statuses:
        overall_status = "HALT"
    elif "WARN" in installed_statuses:
        overall_status = "WARN"
    else:
        overall_status = "PASS"

    return {
        "suite": SUITE_I_NAME,
        "pipeline": "sextantIVIM -> conformalIVIM -> IVIMBench",
        "seed": int(seed),
        "ecosystem_complete": ecosystem_complete,
        "overall_status": overall_status,
        "stages": stages,
    }
