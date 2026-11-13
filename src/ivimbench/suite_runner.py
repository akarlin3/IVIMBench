"""Cross-package AveryCorp Suite I & Suite II audit runner, formatters, and lock/baseline manager.

Orchestrates the 7-package AveryCorp scientific software catalog across two suites:
- **Suite I: Clinical Quantitative-MRI Trust Spine** (3 stages):
  1. ``sextant-ivim`` (``sextant_core.audit``, AGPL-3.0-or-later, dynamic runtime discovery only)
  2. ``conformal-ivim`` (``conformal_ivim.audit``, MIT)
  3. ``ivimbench`` (``ivimbench.audit``, MIT)
- **Suite II: Scientific Identifiability & Null-Audit Suite** (4 pillars):
  4. ``weak-sindy`` (``weaksindy.audit``, MIT)
  5. ``chimera-anneal`` (``chimera_anneal.audit``, MIT)
  6. ``proteus-mine`` (``proteus_mine.audit``, MIT)
  7. ``limbo-audit`` (``limbo_audit.audit``, MIT)
"""

from __future__ import annotations

import hashlib
import html
import importlib
import importlib.metadata
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ._version import __version__
from .audit import SUITE_I_NAME, SUITE_II_NAME, SuiteLiteral, _to_builtin

__all__ = [
    "AVERYCORP_SUITE_REGISTRY",
    "DEFAULT_BASELINE_FILENAME",
    "DEFAULT_LOCK_FILENAME",
    "SuiteFilterLiteral",
    "SuitePackageSpec",
    "SuiteAuditReport",
    "get_suite_registry",
    "run_suite_audit",
    "format_suite_table",
    "format_suite_markdown",
    "format_suite_latex",
    "format_suite_html",
    "build_audit_baseline",
    "build_audit_lock",
    "refresh_suite_artifacts",
    "verify_against_baseline",
    "verify_against_lock",
]

SuiteIdLiteral = Literal["1", "2"]
SuiteFilterLiteral = Literal["all", "1", "2"]
SuiteStatusLiteral = Literal["PASS", "WARN", "HALT", "SKIPPED"]

DEFAULT_BASELINE_FILENAME = "audit-baseline.json"
DEFAULT_LOCK_FILENAME = "audit-lock.json"

REQUIRED_AUDIT_REPORT_KEYS: tuple[str, ...] = (
    "package",
    "version",
    "suite",
    "stage_role",
    "status",
    "verdict",
    "metrics",
    "thresholds",
    "seed",
)


@dataclass(frozen=True)
class SuitePackageSpec(Mapping[str, str]):
    """Registry metadata for a single package in the AveryCorp scientific suites catalog."""

    name: str
    suite: SuiteLiteral
    suite_id: SuiteIdLiteral
    module: str
    cli: str
    repo: str
    license: str
    stage: str

    def to_dict(self) -> dict[str, str]:
        """Serialize registry metadata to a plain string dictionary."""
        return {
            "name": self.name,
            "suite": self.suite,
            "suite_id": self.suite_id,
            "module": self.module,
            "cli": self.cli,
            "repo": self.repo,
            "license": self.license,
            "stage": self.stage,
        }

    def __getitem__(self, key: str) -> str:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return 8


AVERYCORP_SUITE_REGISTRY: tuple[SuitePackageSpec, ...] = (
    # --- Suite I: Clinical Quantitative-MRI Trust Spine (3 packages) ---
    SuitePackageSpec(
        name="sextant-ivim",
        suite=SUITE_I_NAME,
        suite_id="1",
        module="sextant_core.audit",
        cli="sextant-ivim",
        repo="akarlin3/sextantIVIM",
        license="AGPL-3.0-or-later",
        stage="Stage 1: Forward NLLS Fitting, Fisher CRLB & Assumption-Free D* Boundary-Railing Gate",
    ),
    SuitePackageSpec(
        name="conformal-ivim",
        suite=SUITE_I_NAME,
        suite_id="1",
        module="conformal_ivim.audit",
        cli="conformal-ivim",
        repo="akarlin3/conformalIVIM",
        license="MIT",
        stage="Stage 2: Split/CQR/Mondrian/ACI Conformal Calibration, Cohort Shift & Voxel OOD Gate",
    ),
    SuitePackageSpec(
        name="ivimbench",
        suite=SUITE_I_NAME,
        suite_id="1",
        module="ivimbench.audit",
        cli="ivimbench",
        repo="akarlin3/IVIMBench",
        license="MIT",
        stage="Stage 3: Canonical DRO Phantoms, Clean-Room Gnomon/Caliper Rulers & Closed-Loop RT Benchmark",
    ),
    # --- Suite II: Scientific Identifiability & Null-Audit Suite (4 packages) ---
    SuitePackageSpec(
        name="weak-sindy",
        suite=SUITE_II_NAME,
        suite_id="2",
        module="weaksindy.audit",
        cli="weaksindy",
        repo="akarlin3/weakSINDY",
        license="MIT",
        stage="Suite II: Weak-Form Fractional PDE/ODE Identifiability & Ensemble Inclusion Gate",
    ),
    SuitePackageSpec(
        name="chimera-anneal",
        suite=SUITE_II_NAME,
        suite_id="2",
        module="chimera_anneal.audit",
        cli="chimera-anneal",
        repo="akarlin3/chimeraAnneal",
        license="MIT",
        stage="Suite II: Absorbing Collapse vs. Transient Graze Separation & Finite-Size Hazard Gate",
    ),
    SuitePackageSpec(
        name="proteus-mine",
        suite=SUITE_II_NAME,
        suite_id="2",
        module="proteus_mine.audit",
        cli="proteus-mine",
        repo="akarlin3/proteusMine",
        license="MIT",
        stage="Suite II: Catalytic Triad Geometry Screening, pLDDT Stratification & TOST Non-Superiority Gate",
    ),
    SuitePackageSpec(
        name="limbo-audit",
        suite=SUITE_II_NAME,
        suite_id="2",
        module="limbo_audit.audit",
        cli="limbo-audit",
        repo="akarlin3/limboAudit",
        license="MIT",
        stage="Suite II: Zero-Phantom Citation & Manuscript Traceability Gate",
    ),
)


def _normalize_suite_filter(suite_filter: str) -> SuiteFilterLiteral:
    cleaned = str(suite_filter).strip().lower()
    if cleaned in ("all", "*"):
        return "all"
    if cleaned in (
        "1",
        "i",
        "suite1",
        "suite-1",
        "suite_1",
        "suite i",
        SUITE_I_NAME.lower(),
    ):
        return "1"
    if cleaned in (
        "2",
        "ii",
        "suite2",
        "suite-2",
        "suite_2",
        "suite ii",
        SUITE_II_NAME.lower(),
    ):
        return "2"
    raise ValueError(
        f"Unsupported `suite_filter` {suite_filter!r}; expected 'all', '1', or '2'."
    )


def get_suite_registry(
    suite_filter: str = "all",
    *,
    packages: Sequence[str] | None = None,
) -> tuple[SuitePackageSpec, ...]:
    """Return filtered ``SuitePackageSpec`` entries in canonical catalog order."""
    norm_filter = _normalize_suite_filter(suite_filter)
    selected = [
        spec
        for spec in AVERYCORP_SUITE_REGISTRY
        if norm_filter == "all" or spec.suite_id == norm_filter
    ]
    if packages is not None:
        wanted = {str(p).strip() for p in packages if str(p).strip()}
        known = {spec.name for spec in AVERYCORP_SUITE_REGISTRY}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(
                f"Unknown AveryCorp suite package(s): {unknown}; valid names: {sorted(known)}."
            )
        selected = [spec for spec in selected if spec.name in wanted]
    return tuple(selected)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _extract_core_audit_dict(raw_report: Any, spec: SuitePackageSpec, seed: int) -> dict[str, Any]:
    if hasattr(raw_report, "to_dict") and callable(raw_report.to_dict):
        rep_map = dict(raw_report.to_dict())
    elif isinstance(raw_report, Mapping):
        rep_map = dict(raw_report)
    else:
        raise TypeError(
            f"Audit runner `{spec.module}.run_audit` returned unsupported type {type(raw_report)!r}."
        )

    metrics_raw = rep_map.get("metrics", {})
    thresholds_raw = rep_map.get("thresholds", {})
    metrics_clean = (
        {str(k): _to_builtin(v) for k, v in metrics_raw.items()}
        if isinstance(metrics_raw, Mapping)
        else {}
    )
    thresholds_clean = (
        {str(k): _to_builtin(v) for k, v in thresholds_raw.items()}
        if isinstance(thresholds_raw, Mapping)
        else {}
    )
    return {
        "package": str(rep_map.get("package", spec.name)),
        "version": str(rep_map.get("version", "unknown")),
        "suite": str(rep_map.get("suite", spec.suite)),
        "stage_role": str(rep_map.get("stage_role", spec.stage)),
        "status": str(rep_map.get("status", "PASS")),
        "verdict": str(rep_map.get("verdict", "")),
        "metrics": metrics_clean,
        "thresholds": thresholds_clean,
        "seed": int(rep_map.get("seed", seed)),
    }


def _discover_git_commit(spec: SuitePackageSpec) -> str | None:
    """Inspect PEP 610 ``direct_url.json`` metadata if available to capture git commit SHA."""
    dist_candidates = [spec.name]
    if spec.name == "sextant-ivim":
        dist_candidates.insert(0, "sextant-core")
    for dist_name in dist_candidates:
        try:
            dist = importlib.metadata.distribution(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        direct_text = dist.read_text("direct_url.json")
        if not direct_text:
            continue
        try:
            direct_info = json.loads(direct_text)
            vcs_info = direct_info.get("vcs_info", {})
            commit_id = vcs_info.get("commit_id")
            if isinstance(commit_id, str) and commit_id:
                return commit_id
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class SuiteAuditReport(Mapping[str, Any]):
    """Structured cross-package diagnostic report across Suite I and Suite II."""

    seed: int
    suite_filter: SuiteFilterLiteral
    overall_status: SuiteStatusLiteral
    ecosystem_complete: bool
    catalog_size: int
    packages: tuple[dict[str, Any], ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the cross-suite report to a JSON-compatible dictionary."""
        return {
            "generator": "ivimbench.suite_runner",
            "ivimbench_version": __version__,
            "seed": int(self.seed),
            "suite_filter": str(self.suite_filter),
            "catalog_size": int(self.catalog_size),
            "ecosystem_complete": bool(self.ecosystem_complete),
            "overall_status": str(self.overall_status),
            "summary": dict(self.summary),
            "packages": [dict(entry) for entry in self.packages],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the cross-suite report to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def to_table(self) -> str:
        """Render an aligned plain-text diagnostic summary table."""
        return format_suite_table(self)

    def to_markdown(self) -> str:
        """Render a GitHub-Flavored Markdown cross-suite summary table."""
        return format_suite_markdown(self)

    def to_latex(self) -> str:
        """Render a publication-grade LaTeX ``booktabs`` table."""
        return format_suite_latex(self)

    def to_html(self) -> str:
        """Render a standalone zero-dependency HTML5 cross-suite dashboard."""
        return format_suite_html(self)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def run_suite_audit(
    seed: int = 42,
    suite_filter: str = "all",
    *,
    packages: Sequence[str] | None = None,
    strict: bool = False,
) -> SuiteAuditReport:
    """Execute ``run_audit(seed=...)`` across the AveryCorp Suite I / Suite II registry.

    Parameters
    ----------
    seed:
        Deterministic ``PCG64`` seed forwarded to each package's ``run_audit(seed=seed)``.
    suite_filter:
        ``"all"`` (all 7 packages: 3 in Suite I + 4 in Suite II), ``"1"`` (Suite I only),
        or ``"2"`` (Suite II only).
    packages:
        Optional explicit package-name subset filter.
    strict:
        When ``True``, uninstalled companion packages produce ``status="HALT"`` instead
        of ``status="SKIPPED"``.
    """
    norm_filter = _normalize_suite_filter(suite_filter)
    seed_int = int(seed)
    specs = get_suite_registry(norm_filter, packages=packages)

    records: list[dict[str, Any]] = []
    for spec in specs:
        try:
            mod = importlib.import_module(spec.module)
            raw_rep = mod.run_audit(seed=seed_int)
            core = _extract_core_audit_dict(raw_rep, spec, seed_int)
            digest = _canonical_sha256(core)
            git_commit = _discover_git_commit(spec)
            records.append(
                {
                    "name": spec.name,
                    "package": core["package"],
                    "version": core["version"],
                    "suite": core["suite"],
                    "suite_id": spec.suite_id,
                    "stage": spec.stage,
                    "stage_role": core["stage_role"],
                    "module": spec.module,
                    "cli": spec.cli,
                    "repo": spec.repo,
                    "license": spec.license,
                    "installed": True,
                    "status": core["status"],
                    "verdict": core["verdict"],
                    "metrics": core["metrics"],
                    "thresholds": core["thresholds"],
                    "seed": core["seed"],
                    "report_sha256": digest,
                    "git_commit": git_commit,
                }
            )
        except ImportError:
            missing_status: SuiteStatusLiteral = "HALT" if strict else "SKIPPED"
            if spec.license.startswith("AGPL"):
                install_hint = (
                    f"{spec.name} ({spec.module}) not installed; optional AGPL-3.0 runtime plugin "
                    f"(install separately from https://github.com/{spec.repo})."
                )
            else:
                install_hint = (
                    f"{spec.name} ({spec.module}) not installed; install `ivimbench[suite]` "
                    f"or `git+https://github.com/{spec.repo}.git`."
                )
            records.append(
                {
                    "name": spec.name,
                    "package": spec.name,
                    "version": "not-installed",
                    "suite": spec.suite,
                    "suite_id": spec.suite_id,
                    "stage": spec.stage,
                    "stage_role": spec.stage,
                    "module": spec.module,
                    "cli": spec.cli,
                    "repo": spec.repo,
                    "license": spec.license,
                    "installed": False,
                    "status": missing_status,
                    "verdict": install_hint,
                    "metrics": {},
                    "thresholds": {},
                    "seed": seed_int,
                    "report_sha256": None,
                    "git_commit": None,
                }
            )

    installed_count = sum(1 for r in records if r["installed"])
    pass_count = sum(1 for r in records if r["status"] == "PASS")
    warn_count = sum(1 for r in records if r["status"] == "WARN")
    halt_count = sum(1 for r in records if r["status"] == "HALT")
    skipped_count = sum(1 for r in records if r["status"] == "SKIPPED")
    suite_i_count = sum(1 for r in records if r["suite_id"] == "1")
    suite_ii_count = sum(1 for r in records if r["suite_id"] == "2")

    if halt_count > 0:
        overall: SuiteStatusLiteral = "HALT"
    elif warn_count > 0:
        overall = "WARN"
    elif pass_count > 0:
        overall = "PASS"
    else:
        overall = "SKIPPED"

    summary = {
        "total_packages": len(records),
        "installed_packages": installed_count,
        "suite_i_packages": suite_i_count,
        "suite_ii_packages": suite_ii_count,
        "pass_count": pass_count,
        "warn_count": warn_count,
        "halt_count": halt_count,
        "skipped_count": skipped_count,
    }

    return SuiteAuditReport(
        seed=seed_int,
        suite_filter=norm_filter,
        overall_status=overall,
        ecosystem_complete=(installed_count == len(records) and len(records) > 0),
        catalog_size=len(AVERYCORP_SUITE_REGISTRY),
        packages=tuple(records),
        summary=summary,
    )


def _coerce_report_dict(
    report: SuiteAuditReport | Mapping[str, Any] | None,
    *,
    seed: int = 42,
    suite_filter: str = "all",
) -> dict[str, Any]:
    if report is None:
        return run_suite_audit(seed=seed, suite_filter=suite_filter).to_dict()
    if isinstance(report, SuiteAuditReport):
        return report.to_dict()
    if isinstance(report, Mapping):
        return dict(report)
    raise TypeError(f"Expected `SuiteAuditReport` or mapping; got {type(report)!r}.")


def _short_suite_label(entry: Mapping[str, Any]) -> str:
    sid = str(entry.get("suite_id", ""))
    if sid == "1" or str(entry.get("suite", "")).startswith("Suite I:"):
        return "Suite I"
    if sid == "2" or str(entry.get("suite", "")).startswith("Suite II:"):
        return "Suite II"
    return str(entry.get("suite", "Suite"))


def _truncate(text: str, max_len: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= max_len:
        return clean
    return clean[: max(max_len - 3, 1)] + "..."


def format_suite_table(
    report: SuiteAuditReport | Mapping[str, Any] | None = None,
    *,
    seed: int = 42,
    suite_filter: str = "all",
) -> str:
    """Format a cross-suite audit report as an aligned terminal summary table."""
    data = _coerce_report_dict(report, seed=seed, suite_filter=suite_filter)
    pkgs = list(data.get("packages", []))
    summary = dict(data.get("summary", {}))

    header = (
        f"AveryCorp Cross-Suite Diagnostic Runner "
        f"(filter={data.get('suite_filter', 'all')}, seed={data.get('seed', seed)}, "
        f"overall={data.get('overall_status', 'UNKNOWN')})"
    )
    sep = "=" * 116
    thin = "-" * 116
    col_hdr = (
        f"{'Suite':<10} {'Package':<16} {'Version':<14} {'License':<18} "
        f"{'Status':<9} {'Stage / Diagnostic Verdict'}"
    )
    lines = [header, sep, col_hdr, thin]

    for entry in pkgs:
        s_lbl = _short_suite_label(entry)
        name = str(entry.get("name", entry.get("package", "")))
        ver = str(entry.get("version", "unknown"))
        lic = str(entry.get("license", ""))
        status = str(entry.get("status", "UNKNOWN"))
        stage = _truncate(str(entry.get("stage", entry.get("stage_role", ""))), 44)
        lines.append(
            f"{s_lbl:<10} {name:<16} {ver:<14} {lic:<18} {status:<9} {stage}"
        )

    lines.append(thin)
    lines.append(
        f"Packages: {summary.get('total_packages', len(pkgs))} "
        f"(Suite I={summary.get('suite_i_packages', 0)}, "
        f"Suite II={summary.get('suite_ii_packages', 0)}) | "
        f"Installed={summary.get('installed_packages', 0)} | "
        f"PASS={summary.get('pass_count', 0)} "
        f"WARN={summary.get('warn_count', 0)} "
        f"HALT={summary.get('halt_count', 0)} "
        f"SKIPPED={summary.get('skipped_count', 0)}"
    )
    lines.append(sep)
    return "\n".join(lines)


def _escape_md(text: Any) -> str:
    return " ".join(str(text).replace("|", "\\|").split())


def format_suite_markdown(
    report: SuiteAuditReport | Mapping[str, Any] | None = None,
    *,
    seed: int = 42,
    suite_filter: str = "all",
) -> str:
    """Format a cross-suite audit report as a GitHub-Flavored Markdown document."""
    data = _coerce_report_dict(report, seed=seed, suite_filter=suite_filter)
    pkgs = list(data.get("packages", []))
    summary = dict(data.get("summary", {}))

    lines = [
        "## AveryCorp Scientific Suites Diagnostic Summary",
        "",
        (
            f"- **Filter**: `{data.get('suite_filter', 'all')}` | "
            f"**Seed**: `{data.get('seed', seed)}` | "
            f"**Overall Status**: **{data.get('overall_status', 'UNKNOWN')}** | "
            f"**Installed**: `{summary.get('installed_packages', 0)}/{summary.get('total_packages', len(pkgs))}`"
        ),
        "",
        "| Suite | Package | CLI | Version | License | Stage Role | Status | Verdict |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |",
    ]

    for entry in pkgs:
        s_lbl = _escape_md(_short_suite_label(entry))
        name = _escape_md(entry.get("name", entry.get("package", "")))
        repo = str(entry.get("repo", "")).strip()
        pkg_cell = f"[`{name}`](https://github.com/{repo})" if repo else f"`{name}`"
        cli = f"`{_escape_md(entry.get('cli', name))}`"
        ver = f"`{_escape_md(entry.get('version', 'unknown'))}`"
        lic = _escape_md(entry.get("license", ""))
        stage = _escape_md(entry.get("stage", entry.get("stage_role", "")))
        status = f"**{_escape_md(entry.get('status', 'UNKNOWN'))}**"
        verdict = _escape_md(_truncate(str(entry.get("verdict", "")), 96))
        lines.append(
            f"| {s_lbl} | {pkg_cell} | {cli} | {ver} | {lic} | {stage} | {status} | {verdict} |"
        )

    return "\n".join(lines)


def _escape_latex(text: Any) -> str:
    raw = " ".join(str(text).split())
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in raw)


def format_suite_latex(
    report: SuiteAuditReport | Mapping[str, Any] | None = None,
    *,
    seed: int = 42,
    suite_filter: str = "all",
) -> str:
    """Format a cross-suite audit report as a publication-ready LaTeX ``booktabs`` table."""
    data = _coerce_report_dict(report, seed=seed, suite_filter=suite_filter)
    pkgs = list(data.get("packages", []))
    overall = _escape_latex(data.get("overall_status", "UNKNOWN"))
    seed_val = int(data.get("seed", seed))

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{AveryCorp Cross-Suite Identifiability \& Null-Audit Summary "
            rf"($\mathrm{{seed}}={seed_val}$, overall status: \texttt{{{overall}}}).}}"
        ),
        r"\label{tab:averycorp_suite_audit}",
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"Suite & Package & Version & License & Status & Stage Role \\",
        r"\midrule",
    ]

    for entry in pkgs:
        s_lbl = _escape_latex(_short_suite_label(entry))
        name = _escape_latex(entry.get("name", entry.get("package", "")))
        ver = _escape_latex(entry.get("version", "unknown"))
        lic = _escape_latex(entry.get("license", ""))
        status = _escape_latex(entry.get("status", "UNKNOWN"))
        stage = _escape_latex(_truncate(str(entry.get("stage", entry.get("stage_role", ""))), 68))
        lines.append(
            rf"{s_lbl} & \texttt{{{name}}} & \texttt{{{ver}}} & {lic} & \textbf{{{status}}} & {stage} \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _status_badge_class(status: str) -> str:
    s = status.upper()
    if s == "PASS":
        return "badge-pass"
    if s == "WARN":
        return "badge-warn"
    if s == "HALT":
        return "badge-halt"
    return "badge-skip"


def format_suite_html(
    report: SuiteAuditReport | Mapping[str, Any] | None = None,
    *,
    seed: int = 42,
    suite_filter: str = "all",
) -> str:
    """Format a cross-suite audit report as a self-contained HTML5 dashboard."""
    data = _coerce_report_dict(report, seed=seed, suite_filter=suite_filter)
    pkgs = list(data.get("packages", []))
    summary = dict(data.get("summary", {}))
    overall = str(data.get("overall_status", "UNKNOWN"))
    overall_cls = _status_badge_class(overall)

    row_fragments: list[str] = []
    for entry in pkgs:
        s_lbl = html.escape(_short_suite_label(entry))
        name = str(entry.get("name", entry.get("package", "")))
        name_esc = html.escape(name)
        repo = str(entry.get("repo", "")).strip()
        repo_link = (
            f'<a href="https://github.com/{html.escape(repo)}" target="_blank" rel="noopener">'
            f"<code>{name_esc}</code></a>"
            if repo
            else f"<code>{name_esc}</code>"
        )
        cli_esc = html.escape(str(entry.get("cli", name)))
        ver_esc = html.escape(str(entry.get("version", "unknown")))
        lic_esc = html.escape(str(entry.get("license", "")))
        stage_esc = html.escape(str(entry.get("stage", entry.get("stage_role", ""))))
        status_str = str(entry.get("status", "UNKNOWN"))
        badge_cls = _status_badge_class(status_str)
        verdict_esc = html.escape(str(entry.get("verdict", "")))
        row_fragments.append(
            "<tr>"
            f"<td>{s_lbl}</td>"
            f"<td>{repo_link}</td>"
            f"<td><code>{cli_esc}</code></td>"
            f"<td><code>{ver_esc}</code></td>"
            f"<td>{lic_esc}</td>"
            f"<td>{stage_esc}</td>"
            f'<td><span class="badge {badge_cls}">{html.escape(status_str)}</span></td>'
            f"<td>{verdict_esc}</td>"
            "</tr>"
        )

    rows_html = "\n          ".join(row_fragments)
    embedded_json = html.escape(json.dumps(data, indent=2))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AveryCorp Scientific Suites Audit Report</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --pass: #10b981;
      --warn: #f59e0b;
      --halt: #ef4444;
      --skip: #64748b;
      --accent: #38bdf8;
    }}
    body {{
      margin: 0;
      padding: 2rem;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    .container {{ max-width: 1280px; margin: 0 auto; }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
      margin-bottom: 1.5rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border);
    }}
    h1 {{ margin: 0; font-size: 1.5rem; }}
    .meta {{ color: var(--muted); font-size: 0.9rem; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      padding: 1rem;
    }}
    .card .label {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase; }}
    .card .value {{ font-size: 1.4rem; font-weight: 700; margin-top: 0.25rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      overflow: hidden;
      font-size: 0.9rem;
    }}
    th, td {{
      padding: 0.75rem 0.85rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{ background: rgba(15, 23, 42, 0.6); color: var(--muted); font-weight: 600; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85em; }}
    .badge {{
      display: inline-block;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 700;
    }}
    .badge-pass {{ background: rgba(16, 185, 129, 0.2); color: var(--pass); }}
    .badge-warn {{ background: rgba(245, 158, 11, 0.2); color: var(--warn); }}
    .badge-halt {{ background: rgba(239, 68, 68, 0.2); color: var(--halt); }}
    .badge-skip {{ background: rgba(100, 116, 139, 0.25); color: #cbd5e1; }}
    details {{ margin-top: 1.5rem; background: var(--panel); border: 1px solid var(--border); border-radius: 0.5rem; padding: 1rem; }}
    pre {{ overflow-x: auto; font-size: 0.8rem; color: #cbd5e1; margin: 0.75rem 0 0; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>AveryCorp Scientific Suites Cross-Package Audit</h1>
        <div class="meta">Filter: <code>{html.escape(str(data.get("suite_filter", "all")))}</code> &middot; Seed: <code>{int(data.get("seed", seed))}</code> &middot; Runner: <code>ivimbench v{html.escape(__version__)}</code></div>
      </div>
      <div><span class="badge {overall_cls}">OVERALL: {html.escape(overall)}</span></div>
    </header>
    <section class="cards">
      <div class="card"><div class="label">Catalog Packages</div><div class="value">{int(summary.get("total_packages", len(pkgs)))}</div></div>
      <div class="card"><div class="label">Suite I (Clinical MRI)</div><div class="value">{int(summary.get("suite_i_packages", 0))}</div></div>
      <div class="card"><div class="label">Suite II (Null-Audit)</div><div class="value">{int(summary.get("suite_ii_packages", 0))}</div></div>
      <div class="card"><div class="label">Installed</div><div class="value">{int(summary.get("installed_packages", 0))} / {int(summary.get("total_packages", len(pkgs)))}</div></div>
      <div class="card"><div class="label">PASS / WARN / HALT</div><div class="value">{int(summary.get("pass_count", 0))} / {int(summary.get("warn_count", 0))} / {int(summary.get("halt_count", 0))}</div></div>
    </section>
    <table>
      <thead>
        <tr>
          <th>Suite</th>
          <th>Package</th>
          <th>CLI</th>
          <th>Version</th>
          <th>License</th>
          <th>Stage Role</th>
          <th>Status</th>
          <th>Verdict</th>
        </tr>
      </thead>
      <tbody>
          {rows_html}
      </tbody>
    </table>
    <details>
      <summary>Machine-Readable Provenance JSON</summary>
      <pre>{embedded_json}</pre>
    </details>
  </div>
</body>
</html>
"""


def build_audit_baseline(
    report: SuiteAuditReport | None = None,
    *,
    seed: int = 42,
    existing_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic ``audit-baseline.json`` dictionary for all 7 AveryCorp packages."""
    active = report if report is not None else run_suite_audit(seed=seed, suite_filter="all")
    prev_pkgs: dict[str, Mapping[str, Any]] = {}
    if existing_baseline is not None and isinstance(existing_baseline.get("packages"), Mapping):
        prev_pkgs = {
            str(k): v
            for k, v in existing_baseline["packages"].items()
            if isinstance(v, Mapping)
        }

    packages_map: dict[str, dict[str, Any]] = {}
    for entry in active.packages:
        name = str(entry["name"])
        if entry["installed"]:
            packages_map[name] = {
                "name": name,
                "package": entry["package"],
                "version": entry["version"],
                "suite": entry["suite"],
                "suite_id": entry["suite_id"],
                "stage": entry["stage"],
                "stage_role": entry["stage_role"],
                "module": entry["module"],
                "cli": entry["cli"],
                "repo": entry["repo"],
                "license": entry["license"],
                "status": entry["status"],
                "verdict": entry["verdict"],
                "metrics": dict(entry["metrics"]),
                "thresholds": dict(entry["thresholds"]),
                "seed": int(entry["seed"]),
                "report_sha256": entry["report_sha256"],
            }
        elif name in prev_pkgs:
            preserved = dict(prev_pkgs[name])
            preserved["name"] = name
            preserved["suite"] = entry["suite"]
            preserved["suite_id"] = entry["suite_id"]
            preserved["stage"] = entry["stage"]
            preserved["module"] = entry["module"]
            preserved["cli"] = entry["cli"]
            preserved["repo"] = entry["repo"]
            preserved["license"] = entry["license"]
            packages_map[name] = preserved
        else:
            packages_map[name] = {
                "name": name,
                "package": name,
                "version": "not-installed",
                "suite": entry["suite"],
                "suite_id": entry["suite_id"],
                "stage": entry["stage"],
                "stage_role": entry["stage"],
                "module": entry["module"],
                "cli": entry["cli"],
                "repo": entry["repo"],
                "license": entry["license"],
                "status": "SKIPPED",
                "verdict": entry["verdict"],
                "metrics": {},
                "thresholds": {},
                "seed": int(entry["seed"]),
                "report_sha256": None,
            }

    statuses = [p["status"] for p in packages_map.values() if p["status"] != "SKIPPED"]
    if "HALT" in statuses:
        baseline_status = "HALT"
    elif "WARN" in statuses:
        baseline_status = "WARN"
    elif "PASS" in statuses:
        baseline_status = "PASS"
    else:
        baseline_status = "SKIPPED"

    return {
        "schema_version": "1.0.0",
        "generator": "ivimbench.suite_runner",
        "ivimbench_version": __version__,
        "seed": int(active.seed),
        "catalog_size": len(AVERYCORP_SUITE_REGISTRY),
        "suite_counts": {
            SUITE_I_NAME: sum(1 for s in AVERYCORP_SUITE_REGISTRY if s.suite_id == "1"),
            SUITE_II_NAME: sum(1 for s in AVERYCORP_SUITE_REGISTRY if s.suite_id == "2"),
        },
        "overall_status": baseline_status,
        "packages": packages_map,
    }


def build_audit_lock(
    report: SuiteAuditReport | None = None,
    *,
    seed: int = 42,
    existing_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic ``audit-lock.json`` lockfile pinning hashes & versions at ``seed=42``."""
    active = report if report is not None else run_suite_audit(seed=seed, suite_filter="all")
    prev_entries: dict[str, Mapping[str, Any]] = {}
    if existing_lock is not None:
        raw_prev = existing_lock.get("packages")
        if isinstance(raw_prev, Mapping):
            prev_entries = {str(k): v for k, v in raw_prev.items() if isinstance(v, Mapping)}
        elif isinstance(raw_prev, list):
            for item in raw_prev:
                if isinstance(item, Mapping) and "name" in item:
                    prev_entries[str(item["name"])] = item

    locked_packages: dict[str, dict[str, Any]] = {}
    for entry in active.packages:
        name = str(entry["name"])
        if entry["installed"]:
            pkg_lock: dict[str, Any] = {
                "name": name,
                "suite": entry["suite"],
                "suite_id": entry["suite_id"],
                "module": entry["module"],
                "cli": entry["cli"],
                "repo": entry["repo"],
                "license": entry["license"],
                "stage": entry["stage"],
                "version": entry["version"],
                "git_commit": entry.get("git_commit"),
                "status": entry["status"],
                "seed": int(entry["seed"]),
                "report_sha256": entry["report_sha256"],
            }
            if name == "ivimbench":
                from .audit import run_audit as _run_ivim_audit

                standalone_rep = _run_ivim_audit(
                    seed=int(entry["seed"]),
                    prefer_sextant=False,
                    prefer_conformal_ivim=False,
                )
                if standalone_rep.attestation is not None:
                    pkg_lock["standalone_report_sha256"] = standalone_rep.attestation.report_sha256
            locked_packages[name] = pkg_lock
        elif name in prev_entries:
            preserved = dict(prev_entries[name])
            preserved.update(
                {
                    "name": name,
                    "suite": entry["suite"],
                    "suite_id": entry["suite_id"],
                    "module": entry["module"],
                    "cli": entry["cli"],
                    "repo": entry["repo"],
                    "license": entry["license"],
                    "stage": entry["stage"],
                }
            )
            locked_packages[name] = preserved
        else:
            locked_packages[name] = {
                "name": name,
                "suite": entry["suite"],
                "suite_id": entry["suite_id"],
                "module": entry["module"],
                "cli": entry["cli"],
                "repo": entry["repo"],
                "license": entry["license"],
                "stage": entry["stage"],
                "version": "not-installed",
                "git_commit": None,
                "status": "SKIPPED",
                "seed": int(entry["seed"]),
                "report_sha256": None,
            }

    catalog_digest = _canonical_sha256(
        {
            "seed": int(active.seed),
            "packages": {
                k: {
                    "repo": v["repo"],
                    "module": v["module"],
                    "version": v["version"],
                    "status": v["status"],
                    "report_sha256": v["report_sha256"],
                }
                for k, v in locked_packages.items()
            },
        }
    )

    suite_i_digests = [
        str(locked_packages[s.name].get("report_sha256") or "")
        for s in AVERYCORP_SUITE_REGISTRY
        if s.suite_id == "1" and locked_packages.get(s.name)
    ]
    chain_digest = hashlib.sha256(
        json.dumps(suite_i_digests, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": "1.0.0",
        "generator": "ivimbench.suite_runner",
        "ivimbench_version": __version__,
        "seed": int(active.seed),
        "catalog_size": len(AVERYCORP_SUITE_REGISTRY),
        "suite_i_count": sum(1 for s in AVERYCORP_SUITE_REGISTRY if s.suite_id == "1"),
        "suite_ii_count": sum(1 for s in AVERYCORP_SUITE_REGISTRY if s.suite_id == "2"),
        "catalog_sha256": catalog_digest,
        "chain_sha256": chain_digest,
        "packages": locked_packages,
    }


def refresh_suite_artifacts(
    baseline_path: str | Path = DEFAULT_BASELINE_FILENAME,
    lock_path: str | Path = DEFAULT_LOCK_FILENAME,
    *,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run ``run_suite_audit(seed=seed)`` and write refreshed baseline & lock JSON files."""
    b_path = Path(baseline_path)
    l_path = Path(lock_path)

    existing_b = (
        json.loads(b_path.read_text(encoding="utf-8")) if b_path.is_file() else None
    )
    existing_l = (
        json.loads(l_path.read_text(encoding="utf-8")) if l_path.is_file() else None
    )

    report = run_suite_audit(seed=seed, suite_filter="all")
    baseline = build_audit_baseline(report, seed=seed, existing_baseline=existing_b)
    lock = build_audit_lock(report, seed=seed, existing_lock=existing_l)

    b_path.parent.mkdir(parents=True, exist_ok=True)
    l_path.parent.mkdir(parents=True, exist_ok=True)
    b_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    l_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return baseline, lock


def _load_json_or_mapping(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    return dict(json.loads(Path(source).read_text(encoding="utf-8")))


def verify_against_baseline(
    report: SuiteAuditReport | Mapping[str, Any],
    baseline: Mapping[str, Any] | str | Path = DEFAULT_BASELINE_FILENAME,
    *,
    atol: float = 1.0e-6,
) -> dict[str, Any]:
    """Compare installed package metrics and statuses in ``report`` against ``audit-baseline.json``."""
    rep_dict = _coerce_report_dict(report)
    base_dict = _load_json_or_mapping(baseline)
    base_pkgs = dict(base_dict.get("packages", {}))

    drifts: list[dict[str, Any]] = []
    checked = 0
    for entry in rep_dict.get("packages", []):
        if not entry.get("installed"):
            continue
        name = str(entry["name"])
        if name not in base_pkgs:
            drifts.append({"package": name, "reason": "missing_from_baseline"})
            continue
        ref = base_pkgs[name]
        if ref.get("status") == "SKIPPED":
            continue
        checked += 1
        if entry.get("status") != ref.get("status"):
            drifts.append(
                {
                    "package": name,
                    "reason": "status_mismatch",
                    "expected": ref.get("status"),
                    "observed": entry.get("status"),
                }
            )
        ref_metrics = dict(ref.get("metrics", {}))
        obs_metrics = dict(entry.get("metrics", {}))
        for m_key, m_exp in ref_metrics.items():
            if m_key not in obs_metrics:
                drifts.append({"package": name, "reason": "missing_metric", "metric": m_key})
                continue
            m_obs = obs_metrics[m_key]
            if (
                name == "ivimbench"
                and m_key in ("fitter_backend", "uq_backend")
                and m_obs in ("ivimbench_builtin", "sextant_core", "conformal_ivim")
            ):
                continue
            if isinstance(m_exp, (int, float)) and not isinstance(m_exp, bool) and isinstance(
                m_obs, (int, float)
            ):
                if abs(float(m_obs) - float(m_exp)) > atol:
                    drifts.append(
                        {
                            "package": name,
                            "reason": "metric_drift",
                            "metric": m_key,
                            "expected": m_exp,
                            "observed": m_obs,
                        }
                    )
            elif m_obs != m_exp:
                drifts.append(
                    {
                        "package": name,
                        "reason": "metric_drift",
                        "metric": m_key,
                        "expected": m_exp,
                        "observed": m_obs,
                    }
                )

    return {
        "passed": len(drifts) == 0,
        "checked_installed_packages": checked,
        "drifts": drifts,
    }


def verify_against_lock(
    report: SuiteAuditReport | Mapping[str, Any],
    lock: Mapping[str, Any] | str | Path = DEFAULT_LOCK_FILENAME,
) -> dict[str, Any]:
    """Verify installed package ``report_sha256`` and ``chain_sha256`` digests against ``audit-lock.json``."""
    rep_dict = _coerce_report_dict(report)
    lock_dict = _load_json_or_mapping(lock)
    lock_pkgs = dict(lock_dict.get("packages", {}))

    mismatches: list[dict[str, Any]] = []
    verified = 0

    if "chain_sha256" in lock_dict and lock_pkgs:
        expected_chain_inputs = [
            str(lock_pkgs[s.name].get("report_sha256") or "")
            for s in AVERYCORP_SUITE_REGISTRY
            if s.suite_id == "1" and s.name in lock_pkgs
        ]
        recomputed_chain = hashlib.sha256(
            json.dumps(expected_chain_inputs, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if lock_dict.get("chain_sha256") != recomputed_chain:
            mismatches.append(
                {
                    "package": "suite_i_chain",
                    "reason": "chain_sha256_mismatch",
                    "expected": recomputed_chain,
                    "observed": lock_dict.get("chain_sha256"),
                }
            )

    for entry in rep_dict.get("packages", []):
        if not entry.get("installed"):
            continue
        name = str(entry["name"])
        if name not in lock_pkgs:
            mismatches.append({"package": name, "reason": "missing_from_lock"})
            continue
        ref = lock_pkgs[name]
        exp_sha = ref.get("report_sha256")
        if exp_sha is None:
            continue
        verified += 1
        obs_sha = entry.get("report_sha256")
        alt_sha = ref.get("standalone_report_sha256")
        valid_shas = {exp_sha} | ({alt_sha} if isinstance(alt_sha, str) and alt_sha else set())
        if obs_sha not in valid_shas:
            mismatches.append(
                {
                    "package": name,
                    "reason": "sha256_mismatch",
                    "expected": exp_sha,
                    "observed": obs_sha,
                }
            )

    return {
        "passed": len(mismatches) == 0,
        "verified_installed_packages": verified,
        "mismatches": mismatches,
    }
