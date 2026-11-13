"""Standardized AveryCorp Suite I (Stage 3) clean-room benchmark, cryptographic attestation & chained audit."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy
from scipy.stats import norm

from ._version import __version__
from .bridges.avery_stack import get_canonical_fitter, has_conformal_ivim
from .closed_loop.loop import SyntheticOrganTwin, compare_closed_loop_policies
from .metrics.coverage import DEFAULT_QUANTILE_LEVELS, ece_quantile, empirical_coverage
from .phantoms.selfcheck import run_nlls_roundtrip_selfcheck, verify_continuity_limits
from .phantoms.synthetic import generate_cohort

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


def get_platform_fingerprint() -> dict[str, str]:
    """Return Python, NumPy, and SciPy version fingerprint for audit provenance."""
    return {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
    }


def compute_input_sha256(data: np.ndarray | bytes | str | Path) -> str:
    """Compute deterministic SHA-256 digest of a canonical input signal/phantom tensor or NIfTI file."""
    hasher = hashlib.sha256()
    if isinstance(data, (bytes, bytearray)):
        hasher.update(bytes(data))
        return hasher.hexdigest()
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.is_file():
            hasher.update(path.read_bytes())
            return hasher.hexdigest()
        hasher.update(str(data).encode("utf-8"))
        return hasher.hexdigest()
    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    header = f"shape={arr.shape};dtype={arr.dtype.str};".encode()
    hasher.update(header)
    hasher.update(arr.astype("<f8", copy=False).tobytes(order="C"))
    return hasher.hexdigest()


def compute_report_sha256(
    *,
    package: str,
    version: str,
    suite: str,
    stage_role: str,
    status: str,
    verdict: str,
    seed: int,
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> str:
    """Compute canonical SHA-256 digest over sorted AuditReport core fields."""
    payload = {
        "metrics": {str(k): _to_builtin(v) for k, v in sorted(metrics.items())},
        "package": str(package),
        "seed": int(seed),
        "stage_role": str(stage_role),
        "status": str(status),
        "suite": str(suite),
        "thresholds": {str(k): _to_builtin(v) for k, v in sorted(thresholds.items())},
        "verdict": str(verdict),
        "version": str(version),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def compute_chain_sha256(stages: Sequence[Mapping[str, Any]]) -> str:
    """Compute canonical SHA-256 digest hashing ordered ``[stage1, stage2, stage3]`` ``report_sha256`` digests."""
    stage_digests: list[str] = []
    for st in stages:
        att = st.get("attestation")
        if isinstance(att, Mapping) and isinstance(att.get("report_sha256"), str):
            stage_digests.append(str(att["report_sha256"]))
        else:
            fallback = json.dumps(dict(st), sort_keys=True, separators=(",", ":")).encode("utf-8")
            stage_digests.append(hashlib.sha256(fallback).hexdigest())
    canonical = json.dumps(stage_digests, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class AuditAttestation:
    """Cryptographic provenance and numerical reproducibility attestation."""

    input_sha256: str
    report_sha256: str
    platform_fingerprint: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_sha256": str(self.input_sha256),
            "report_sha256": str(self.report_sha256),
            "platform_fingerprint": {
                str(k): str(v) for k, v in sorted(self.platform_fingerprint.items())
            },
        }


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
    attestation: AuditAttestation | None = None

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
            "attestation": self.attestation.to_dict() if self.attestation is not None else None,
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


@dataclass(frozen=True)
class AuditChain:
    """Chained 3-stage Suite I (`sextantIVIM -> conformalIVIM -> IVIMBench`) cryptographic audit chain."""

    suite: str
    pipeline: str
    seed: int
    ecosystem_complete: bool
    overall_status: str
    stages: list[dict[str, Any]]
    chain_sha256: str
    platform_fingerprint: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": str(self.suite),
            "pipeline": str(self.pipeline),
            "seed": int(self.seed),
            "ecosystem_complete": bool(self.ecosystem_complete),
            "overall_status": str(self.overall_status),
            "chain_sha256": str(self.chain_sha256),
            "platform_fingerprint": dict(self.platform_fingerprint),
            "stages": list(self.stages),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


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
    prefer_sextant: bool = True,
    prefer_conformal_ivim: bool = True,
) -> AuditReport:
    """Run the Stage 3 Suite I clean-room DRO, Gnomon ruler, and closed-loop RT audit."""
    ref_cohort = generate_cohort(
        protocol="osipi",
        organ="pancreas",
        n_voxels=32,
        snr=30.0,
        seed=int(seed),
    )
    input_digest = compute_input_sha256(ref_cohort.signals)
    fitter = get_canonical_fitter(prefer_sextant=prefer_sextant)
    fitter_backend = fitter.backend
    uq_backend = (
        "conformal_ivim" if (prefer_conformal_ivim and has_conformal_ivim()) else "ivimbench_builtin"
    )

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

    stage_role = (
        "Stage 3: Canonical DRO Phantoms, Clean-Room Gnomon/Caliper Rulers & Closed-Loop RT Benchmark"
    )
    metrics_dict: dict[str, float | int | str | bool] = {
        "continuity_passed": cont_ok,
        "continuity_max_abs_residual": max_cont_res,
        "nlls_roundtrip_passed": rt_ok,
        "nlls_roundtrip_max_rel_error": rt_max_rel,
        "gnomon_passed": gnomon_ok,
        "gnomon_quantile_ece": round(gnomon_ece, 6),
        "closed_loop_regret_reduction": round(regret_red, 6),
        "closed_loop_oar_sparing_gy": round(oar_sparing, 6),
        "fitter_backend": fitter_backend,
        "uq_backend": uq_backend,
    }
    thresholds_dict: dict[str, float | int | str] = {
        "continuity_tolerance": 1.0e-12,
        "nlls_roundtrip_tolerance": float(rt["tolerance"]),
        "max_gnomon_quantile_ece": float(max_gnomon_ece),
        "min_closed_loop_regret_reduction": 0.0,
    }
    report_digest = compute_report_sha256(
        package="ivimbench",
        version=__version__,
        suite=SUITE_I_NAME,
        stage_role=stage_role,
        status=status,
        verdict=verdict,
        seed=int(seed),
        metrics=metrics_dict,
        thresholds=thresholds_dict,
    )
    attestation = AuditAttestation(
        input_sha256=input_digest,
        report_sha256=report_digest,
        platform_fingerprint=get_platform_fingerprint(),
    )

    return AuditReport(
        package="ivimbench",
        version=__version__,
        suite=SUITE_I_NAME,
        stage_role=stage_role,
        status=status,
        verdict=verdict,
        metrics=metrics_dict,
        thresholds=thresholds_dict,
        seed=int(seed),
        attestation=attestation,
    )


def build_suite1_audit_chain(seed: int = 42) -> AuditChain:
    """Execute the chained 3-stage Suite I spine and return an immutable :class:`AuditChain`."""
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

    chain_digest = compute_chain_sha256(stages)
    return AuditChain(
        suite=SUITE_I_NAME,
        pipeline="sextantIVIM -> conformalIVIM -> IVIMBench",
        seed=int(seed),
        ecosystem_complete=ecosystem_complete,
        overall_status=overall_status,
        stages=stages,
        chain_sha256=chain_digest,
        platform_fingerprint=get_platform_fingerprint(),
    )


def run_suite1_pipeline(seed: int = 42) -> dict[str, Any]:
    """Execute the chained 3-stage Suite I clinical quantitative-MRI trust spine audit."""
    return build_suite1_audit_chain(seed=int(seed)).to_dict()


def write_audit_lock(report: AuditReport, lock_path: str | Path) -> Path:
    """Write a standalone Stage-3 cryptographic ``audit-lock.json`` file."""
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(indent=2) + "\n", encoding="utf-8")
    return path


def verify_audit_lock(report: AuditReport, lock_path: str | Path) -> dict[str, Any]:
    """Verify byte-identical ``input_sha256`` and ``report_sha256`` against ``lock_path``."""
    path = Path(lock_path)
    expected = json.loads(path.read_text(encoding="utf-8"))
    actual = report.to_dict()
    exp_att = expected.get("attestation") or {}
    act_att = actual.get("attestation") or {}
    if exp_att.get("input_sha256") != act_att.get("input_sha256"):
        raise AssertionError(
            f"Audit input_sha256 mismatch: expected {exp_att.get('input_sha256')}, "
            f"got {act_att.get('input_sha256')}."
        )
    if exp_att.get("report_sha256") != act_att.get("report_sha256"):
        raise AssertionError(
            f"Audit report_sha256 mismatch: expected {exp_att.get('report_sha256')}, "
            f"got {act_att.get('report_sha256')}."
        )
    return {"verified": True, "report_sha256": act_att.get("report_sha256")}


def write_suite_audit_lock(chain: AuditChain | Mapping[str, Any], lock_path: str | Path) -> Path:
    """Write the chained 3-stage Suite I cryptographic ``audit-lock.json`` file."""
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = chain.to_dict() if isinstance(chain, AuditChain) else dict(chain)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def verify_suite_audit_lock(
    chain: AuditChain | Mapping[str, Any], lock_path: str | Path
) -> dict[str, Any]:
    """Cryptographically verify chained Suite I ``chain_sha256`` and per-stage digests."""
    path = Path(lock_path)
    expected = json.loads(path.read_text(encoding="utf-8"))
    actual = chain.to_dict() if isinstance(chain, AuditChain) else dict(chain)
    act_stages = actual.get("stages", [])

    if "packages" in expected and "stages" not in expected:
        lock_pkgs = expected.get("packages") or {}
        suite_i_names = ("sextant-ivim", "conformal-ivim", "ivimbench")
        expected_chain_inputs = [
            str(lock_pkgs[name].get("report_sha256") or "")
            for name in suite_i_names
            if name in lock_pkgs
        ]
        recomputed_chain = hashlib.sha256(
            json.dumps(expected_chain_inputs, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        exp_chain_sha = expected.get("chain_sha256")
        if not exp_chain_sha or exp_chain_sha != recomputed_chain:
            raise AssertionError(
                f"Suite I chain_sha256 mismatch: expected {recomputed_chain!r}, got {exp_chain_sha!r}."
            )
        for idx, a_st in enumerate(act_stages, start=1):
            if not a_st.get("installed"):
                continue
            pkg_name = str(a_st.get("package", ""))
            if pkg_name not in lock_pkgs:
                raise AssertionError(f"Stage {idx} ({pkg_name}) missing from lockfile packages.")
            ref = lock_pkgs[pkg_name]
            exp_sha = ref.get("report_sha256")
            alt_sha = ref.get("standalone_report_sha256")
            valid_shas = {exp_sha} | ({alt_sha} if isinstance(alt_sha, str) and alt_sha else set())
            a_att = a_st.get("attestation") or {}
            obs_sha = a_att.get("report_sha256")
            if obs_sha not in valid_shas:
                raise AssertionError(
                    f"Stage {idx} ({pkg_name}) report_sha256 mismatch: expected {exp_sha}, got {obs_sha}."
                )
        return {
            "verified": True,
            "chain_sha256": str(exp_chain_sha),
            "n_stages": len(act_stages),
        }

    exp_chain_sha = expected.get("chain_sha256")
    act_chain_sha = actual.get("chain_sha256")
    if not exp_chain_sha or exp_chain_sha != act_chain_sha:
        raise AssertionError(
            f"Suite I chain_sha256 mismatch: expected {exp_chain_sha!r}, got {act_chain_sha!r}."
        )

    exp_stages = expected.get("stages", [])
    if len(exp_stages) != len(act_stages):
        raise AssertionError(
            f"Stage count mismatch: expected {len(exp_stages)}, got {len(act_stages)}."
        )

    for idx, (e_st, a_st) in enumerate(zip(exp_stages, act_stages, strict=True), start=1):
        e_att = e_st.get("attestation") or {}
        a_att = a_st.get("attestation") or {}
        if e_att.get("input_sha256") != a_att.get("input_sha256"):
            raise AssertionError(
                f"Stage {idx} input_sha256 mismatch: expected {e_att.get('input_sha256')}, "
                f"got {a_att.get('input_sha256')}."
            )
        if e_att.get("report_sha256") != a_att.get("report_sha256"):
            raise AssertionError(
                f"Stage {idx} report_sha256 mismatch: expected {e_att.get('report_sha256')}, "
                f"got {a_att.get('report_sha256')}."
            )

    return {
        "verified": True,
        "chain_sha256": str(act_chain_sha),
        "n_stages": len(act_stages),
    }


def assert_audit_baseline(
    report: AuditReport,
    baseline_path: str | Path,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """Compare ``report`` against a saved JSON baseline within numerical tolerance."""
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    actual = report.to_dict()
    for key in ("package", "suite", "stage_role", "status", "seed"):
        if actual.get(key) != baseline.get(key):
            raise AssertionError(
                f"Audit baseline mismatch on {key!r}: expected {baseline.get(key)!r}, "
                f"got {actual.get(key)!r}."
            )
    exp_metrics = baseline.get("metrics", {})
    act_metrics = actual.get("metrics", {})
    for k, exp_v in exp_metrics.items():
        if k not in act_metrics:
            raise AssertionError(f"Missing audit metric {k!r}.")
        act_v = act_metrics[k]
        if isinstance(exp_v, bool) or isinstance(act_v, bool):
            if bool(act_v) != bool(exp_v):
                raise AssertionError(f"Boolean metric {k!r} mismatch: {act_v} != {exp_v}.")
        elif isinstance(exp_v, (int, float)) and isinstance(act_v, (int, float)):
            if not np.isclose(float(act_v), float(exp_v), rtol=rtol, atol=atol):
                raise AssertionError(f"Numeric metric {k!r} mismatch: {act_v} vs {exp_v}.")
        elif str(act_v) != str(exp_v):
            raise AssertionError(f"String metric {k!r} mismatch: {act_v!r} != {exp_v!r}.")
