"""Tests for cross-package AveryCorp Suite I & II runner (`ivimbench.suite_runner`)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from ivimbench.audit import SUITE_I_NAME, SUITE_II_NAME, AuditReport
from ivimbench.cli import main, suite_main
from ivimbench.suite_runner import (
    AVERYCORP_SUITE_REGISTRY,
    build_audit_baseline,
    build_audit_lock,
    format_suite_html,
    format_suite_latex,
    format_suite_markdown,
    format_suite_table,
    get_suite_registry,
    refresh_suite_artifacts,
    run_suite_audit,
    verify_against_baseline,
    verify_against_lock,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_suite_registry_7_packages_and_limbo_audit_spec() -> None:
    assert len(AVERYCORP_SUITE_REGISTRY) == 7

    suite_i = get_suite_registry("1")
    suite_ii = get_suite_registry("2")
    suite_all = get_suite_registry("all")

    assert len(suite_i) == 3
    assert len(suite_ii) == 4
    assert len(suite_all) == 7

    assert [s.name for s in suite_i] == ["sextant-ivim", "conformal-ivim", "ivimbench"]
    assert [s.name for s in suite_ii] == [
        "weak-sindy",
        "chimera-anneal",
        "proteus-mine",
        "limbo-audit",
    ]

    limbo = suite_ii[-1]
    assert limbo.name == "limbo-audit"
    assert limbo.suite == SUITE_II_NAME
    assert limbo.suite_id == "2"
    assert limbo.module == "limbo_audit.audit"
    assert limbo.cli == "limbo-audit"
    assert limbo.repo == "akarlin3/limboAudit"
    assert limbo.license == "MIT"
    assert limbo.stage == "Suite II: Zero-Phantom Citation & Manuscript Traceability Gate"
    assert limbo["name"] == "limbo-audit"
    assert limbo["module"] == "limbo_audit.audit"
    assert limbo["repo"] == "akarlin3/limboAudit"

    with pytest.raises(ValueError, match="suite_filter"):
        get_suite_registry("3")

    with pytest.raises(ValueError, match="Unknown AveryCorp suite package"):
        get_suite_registry("all", packages=["nonexistent-pkg"])


def test_pyproject_suite_optional_dependencies() -> None:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    raw_text = pyproject_path.read_text(encoding="utf-8")
    assert "limbo-audit @ git+https://github.com/akarlin3/limboAudit.git" in raw_text

    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        tomllib = None

    if tomllib is not None:
        parsed = tomllib.loads(raw_text)
        suite_deps = parsed["project"]["optional-dependencies"]["suite"]
        assert "limbo-audit @ git+https://github.com/akarlin3/limboAudit.git" in suite_deps
        assert "conformal-ivim @ git+https://github.com/akarlin3/conformalIVIM.git" in suite_deps
        assert "weak-sindy @ git+https://github.com/akarlin3/weakSINDY.git" in suite_deps
        assert "chimera-anneal @ git+https://github.com/akarlin3/chimeraAnneal.git" in suite_deps
        assert "proteus-mine @ git+https://github.com/akarlin3/proteusMine.git" in suite_deps
        assert len(suite_deps) == 5


def test_mocked_limbo_audit_dispatch_formatters_and_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pkg: Any = types.ModuleType("limbo_audit")
    fake_audit: Any = types.ModuleType("limbo_audit.audit")

    def _fake_run_audit(seed: int = 42) -> AuditReport:
        return AuditReport(
            package="limbo-audit",
            version="1.0.0",
            suite=SUITE_II_NAME,
            stage_role="Suite II: Zero-Phantom Citation & Manuscript Traceability Gate",
            status="PASS",
            verdict="ZERO_PHANTOM_VERIFIED: 0 phantom citations and 0 unverified manuscript claims.",
            metrics={
                "phantom_citation_count": 0,
                "unverified_claim_count": 0,
                "traceability_pass": True,
            },
            thresholds={
                "max_phantom_citation_count": 0,
                "max_unverified_claim_count": 0,
            },
            seed=int(seed),
        )

    fake_audit.run_audit = _fake_run_audit
    fake_pkg.audit = fake_audit
    monkeypatch.setitem(sys.modules, "limbo_audit", fake_pkg)
    monkeypatch.setitem(sys.modules, "limbo_audit.audit", fake_audit)

    rep_s2 = run_suite_audit(seed=42, suite_filter="2")
    assert len(rep_s2.packages) == 4
    assert rep_s2.summary["suite_ii_packages"] == 4
    limbo_rec = next(p for p in rep_s2.packages if p["name"] == "limbo-audit")
    assert limbo_rec["installed"] is True
    assert limbo_rec["version"] == "1.0.0"
    assert limbo_rec["status"] == "PASS"
    assert limbo_rec["repo"] == "akarlin3/limboAudit"
    assert limbo_rec["license"] == "MIT"
    assert isinstance(limbo_rec["report_sha256"], str) and len(limbo_rec["report_sha256"]) == 64

    tbl = format_suite_table(rep_s2)
    md = format_suite_markdown(rep_s2)
    tex = format_suite_latex(rep_s2)
    htm = format_suite_html(rep_s2)

    assert "limbo-audit" in tbl
    assert "Zero-Phantom Citation" in tbl
    assert "[`limbo-audit`](https://github.com/akarlin3/limboAudit)" in md
    assert "`limbo-audit`" in md
    assert r"\texttt{limbo-audit}" in tex
    assert r"\toprule" in tex and r"\bottomrule" in tex
    assert "<!DOCTYPE html>" in htm
    assert "https://github.com/akarlin3/limboAudit" in htm
    assert "limbo-audit" in htm

    # Simulate uninstalled limbo_audit under default vs. strict mode
    monkeypatch.setitem(sys.modules, "limbo_audit", None)
    monkeypatch.setitem(sys.modules, "limbo_audit.audit", None)

    rep_skip = run_suite_audit(seed=42, suite_filter="2", packages=["limbo-audit"], strict=False)
    assert rep_skip.packages[0]["installed"] is False
    assert rep_skip.packages[0]["status"] == "SKIPPED"
    assert rep_skip.overall_status == "SKIPPED"

    rep_strict = run_suite_audit(seed=42, suite_filter="2", packages=["limbo-audit"], strict=True)
    assert rep_strict.packages[0]["installed"] is False
    assert rep_strict.packages[0]["status"] == "HALT"
    assert rep_strict.overall_status == "HALT"


def test_run_suite_audit_all_and_ivimbench_baseline_lock_verification(tmp_path: Path) -> None:
    rep_all_subset = run_suite_audit(
        seed=42,
        suite_filter="all",
        packages=["ivimbench", "limbo-audit"],
    )
    assert rep_all_subset.catalog_size == 7
    assert len(rep_all_subset.packages) == 2
    names = [p["name"] for p in rep_all_subset.packages]
    assert names == ["ivimbench", "limbo-audit"]
    assert rep_all_subset.packages[0]["status"] == "PASS"

    for rendered in (
        rep_all_subset.to_table(),
        rep_all_subset.to_markdown(),
        rep_all_subset.to_latex(),
        rep_all_subset.to_html(),
    ):
        assert "ivimbench" in rendered
        assert "limbo-audit" in rendered

    # Verify committed repository audit-baseline.json and audit-lock.json
    baseline_path = REPO_ROOT / "audit-baseline.json"
    lock_path = REPO_ROOT / "audit-lock.json"
    assert baseline_path.is_file()
    assert lock_path.is_file()

    baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))

    expected_names = [s.name for s in AVERYCORP_SUITE_REGISTRY]
    assert baseline_data["seed"] == 42
    assert baseline_data["catalog_size"] == 7
    assert baseline_data["suite_counts"][SUITE_I_NAME] == 3
    assert baseline_data["suite_counts"][SUITE_II_NAME] == 4
    assert list(baseline_data["packages"].keys()) == expected_names
    assert baseline_data["packages"]["limbo-audit"]["repo"] == "akarlin3/limboAudit"
    assert baseline_data["packages"]["limbo-audit"]["module"] == "limbo_audit.audit"
    assert baseline_data["packages"]["limbo-audit"]["version"] == "1.0.0"
    assert baseline_data["packages"]["limbo-audit"]["status"] == "PASS"
    assert baseline_data["packages"]["limbo-audit"]["metrics"]["verified_references"] == 59
    assert baseline_data["packages"]["limbo-audit"]["metrics"]["phantom_citations"] == 0
    assert len(baseline_data["packages"]["limbo-audit"]["report_sha256"]) == 64

    assert lock_data["seed"] == 42
    assert lock_data["catalog_size"] == 7
    assert lock_data["suite_i_count"] == 3
    assert lock_data["suite_ii_count"] == 4
    assert list(lock_data["packages"].keys()) == expected_names
    assert lock_data["packages"]["limbo-audit"]["cli"] == "limbo-audit"
    assert lock_data["packages"]["limbo-audit"]["version"] == "1.0.0"
    assert lock_data["packages"]["limbo-audit"]["status"] == "PASS"
    assert len(lock_data["packages"]["limbo-audit"]["report_sha256"]) == 64
    assert isinstance(lock_data["packages"]["limbo-audit"]["git_commit"], str)

    rep_ivim = run_suite_audit(seed=42, suite_filter="1", packages=["ivimbench"])
    base_check = verify_against_baseline(rep_ivim, baseline_data)
    lock_check = verify_against_lock(rep_ivim, lock_data)
    assert base_check["passed"] is True, base_check
    assert lock_check["passed"] is True, lock_check

    # Test artifact build & preservation of existing entries
    b_built = build_audit_baseline(rep_ivim, seed=42, existing_baseline=baseline_data)
    l_built = build_audit_lock(rep_ivim, seed=42, existing_lock=lock_data)
    assert b_built["packages"]["ivimbench"]["status"] == "PASS"
    assert l_built["packages"]["ivimbench"]["report_sha256"] == rep_ivim.packages[0]["report_sha256"]

    # Round-trip write in tmp_path on a fast 1-package mock or ivimbench check
    tmp_b = tmp_path / "audit-baseline.json"
    tmp_l = tmp_path / "audit-lock.json"
    tmp_b.write_text(json.dumps(baseline_data, indent=2) + "\n", encoding="utf-8")
    tmp_l.write_text(json.dumps(lock_data, indent=2) + "\n", encoding="utf-8")
    assert verify_against_baseline(rep_ivim, tmp_b)["passed"] is True
    assert verify_against_lock(rep_ivim, tmp_l)["passed"] is True

    # Ensure refresh_suite_artifacts symbol is callable
    assert callable(refresh_suite_artifacts)


def test_cli_suite_audit_subcommand(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Isolate companion modules so CLI test runs in milliseconds deterministically
    for mod_prefix in (
        "sextant_core",
        "sextant_core.audit",
        "conformal_ivim",
        "conformal_ivim.audit",
        "weaksindy",
        "weaksindy.audit",
        "chimera_anneal",
        "chimera_anneal.audit",
        "proteus_mine",
        "proteus_mine.audit",
        "limbo_audit",
        "limbo_audit.audit",
    ):
        monkeypatch.setitem(sys.modules, mod_prefix, None)

    rc_json = main(["suite-audit", "--suite", "2", "--seed", "42", "--json"])
    assert rc_json == 0
    out_json = json.loads(capsys.readouterr().out)
    assert out_json["suite_filter"] == "2"
    assert out_json["catalog_size"] == 7
    assert len(out_json["packages"]) == 4
    assert out_json["packages"][-1]["name"] == "limbo-audit"

    rc_runner = suite_main(["--suite", "2", "--seed", "42", "--json"])
    assert rc_runner == 0
    out_runner = json.loads(capsys.readouterr().out)
    assert out_runner["packages"][-1]["name"] == "limbo-audit"

    md_path = tmp_path / "suite2.md"
    rc_md = main(
        ["suite-audit", "--suite", "2", "--seed", "42", "--format", "markdown", "--out", str(md_path)]
    )
    assert rc_md == 0
    _ = capsys.readouterr()
    md_content = md_path.read_text(encoding="utf-8")
    assert "limbo-audit" in md_content
    assert "akarlin3/limboAudit" in md_content
