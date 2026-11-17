"""Tests for the unified deterministic ``--export-web-fixture`` verification payload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ivimbench.cli import suite_main
from ivimbench.suite_runner import (
    AUDIT_CHAIN_STATE_TABLE,
    AVERYCORP_SUITE_REGISTRY,
    REQUIRED_AUDIT_REPORT_KEYS,
    WEB_FIXTURE_SCHEMA_VERSION,
    _canonical_sha256,
    build_web_fixture,
    format_suite_latex,
    run_suite_audit,
    verify_against_lock,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "audit-lock.json"

SUITE_I = ("sextant-ivim", "conformal-ivim", "ivimbench")
SUITE_II = ("weak-sindy", "chimera-anneal", "proteus-mine", "limbo-audit")


@pytest.fixture(scope="module")
def report() -> Any:
    return run_suite_audit(seed=42, suite_filter="all")


@pytest.fixture(scope="module")
def fixture(report: Any) -> dict[str, Any]:
    return build_web_fixture(report, seed=42)


def test_registry_has_all_seven_packages() -> None:
    names = tuple(s.name for s in AVERYCORP_SUITE_REGISTRY)
    assert names == SUITE_I + SUITE_II
    assert {s.name for s in AVERYCORP_SUITE_REGISTRY if s.suite_id == "1"} == set(SUITE_I)
    assert {s.name for s in AVERYCORP_SUITE_REGISTRY if s.suite_id == "2"} == set(SUITE_II)


def test_top_level_schema(fixture: dict[str, Any]) -> None:
    assert fixture["schema_version"] == WEB_FIXTURE_SCHEMA_VERSION
    assert fixture["seed"] == 42
    assert set(fixture) >= {
        "suite_summary",
        "audit_chain_states",
        "sample_bids_sidecars",
        "fixture_sha256",
    }
    body = {k: v for k, v in fixture.items() if k != "fixture_sha256"}
    assert fixture["fixture_sha256"] == _canonical_sha256(body)
    # Round-trips through strict JSON (no NaN/Infinity, all builtins).
    assert json.loads(json.dumps(fixture, allow_nan=False)) == fixture


def test_suite_summary_ledger(fixture: dict[str, Any], report: Any) -> None:
    summ = fixture["suite_summary"]
    assert summ["seed"] == 42
    assert summ["catalog_size"] == 7
    assert [p["name"] for p in summ["packages"]] == list(SUITE_I + SUITE_II)
    assert list(summ["report_sha256"]) == list(SUITE_I + SUITE_II)
    for pkg in summ["packages"]:
        for key in REQUIRED_AUDIT_REPORT_KEYS:
            assert key in pkg
        assert pkg["seed"] == 42
        if pkg["installed"]:
            assert len(pkg["report_sha256"]) == 64
            assert summ["baseline_metrics"][pkg["name"]] == pkg["metrics"]
        else:
            assert pkg["report_sha256"] is None
    assert summ["latex_booktabs"] == format_suite_latex(report)
    assert r"\toprule" in summ["latex_booktabs"]
    assert r"\bottomrule" in summ["latex_booktabs"]


def test_audit_chain_states(fixture: dict[str, Any]) -> None:
    chain = fixture["audit_chain_states"]
    states = chain["states"]
    assert len(states) == len(AUDIT_CHAIN_STATE_TABLE) == 4
    combos = {
        (s["stage1_sextant_ivim"]["status"], s["stage2_conformal_ivim"]["status"]): s
        for s in states
    }
    assert set(combos) == {("PASS", "PASS"), ("PASS", "WARN"), ("WARN", "PASS"), ("WARN", "WARN")}
    assert combos[("PASS", "PASS")]["stage3_action"] == "PROCEED"
    assert combos[("PASS", "WARN")]["stage3_action"] == "WIDEN_MONDRIAN_STRATA"
    for key in (("WARN", "PASS"), ("WARN", "WARN")):
        assert combos[key]["stage3_action"] == "HOLD_BASELINE_DOSE"
        assert combos[key]["regret_vs_baseline_dose"] == 0.0
    for s in states:
        body = {k: v for k, v in s.items() if k != "state_sha256"}
        assert s["state_sha256"] == _canonical_sha256(body)

    live = chain["live_chain"]
    assert [st["package"] for st in live["stages"]] == list(SUITE_I)
    assert live["stage3_action"] in {"PROCEED", "WIDEN_MONDRIAN_STRATA", "HOLD_BASELINE_DOSE", "HALT"}


def _assert_bids_generated_by(entries: Any) -> None:
    assert isinstance(entries, list) and entries
    for gen in entries:
        assert isinstance(gen["Name"], str) and gen["Name"]
        assert isinstance(gen["Version"], str) and gen["Version"]
        assert gen["CodeURL"].startswith("https://")


def test_bids_sidecars_generated_by_compliance(fixture: dict[str, Any]) -> None:
    bids = fixture["sample_bids_sidecars"]
    desc = bids["dataset_description"]
    assert desc["DatasetType"] == "derivative"
    assert isinstance(desc["BIDSVersion"], str)
    _assert_bids_generated_by(desc["GeneratedBy"])

    for key in ("osipi_interior_pass", "tcga_lihc_railed_54pct_shift"):
        entry = bids[key]
        assert entry["filename"].endswith("_desc-ivim_audit.json")
        sub = entry["filename"].split("/")[0]
        assert sub.startswith("sub-")
        assert Path(entry["filename"]).name.startswith(f"{sub}_")
        side = entry["sidecar"]
        _assert_bids_generated_by(side["GeneratedBy"])
        assert all(s.startswith("bids::") for s in side["Sources"])
        body = {k: v for k, v in side.items() if k != "SidecarSHA256"}
        assert side["SidecarSHA256"] == _canonical_sha256(body)

    assert bids["osipi_interior_pass"]["filename"] == "sub-01/dwi/sub-01_desc-ivim_audit.json"
    osipi = bids["osipi_interior_pass"]["sidecar"]
    assert osipi["QCStatus"] == "PASS"
    assert osipi["railing_fraction_1d"] == 0.0
    assert osipi["DeploymentValidityMonitor"]["fires"] is False
    assert osipi["AuditChain"]["stage3_action"] == "PROCEED"

    tcga = bids["tcga_lihc_railed_54pct_shift"]["sidecar"]
    assert tcga["QCStatus"] == "WARN"
    assert tcga["railing_fraction_1d"] == 0.542
    assert tcga["DeploymentValidityMonitor"]["fires"] is True
    assert any("DEPLOYMENT_SHIFT" in w for w in tcga["Warnings"])
    assert any("COVERAGE_SHORTFALL" in w for w in tcga["Warnings"])
    assert tcga["ConformalCoverage"]["empirical_coverage"] < tcga["ConformalCoverage"]["nominal_coverage"]
    assert osipi["ConformalCoverage"]["empirical_coverage"] >= osipi["ConformalCoverage"]["nominal_coverage"]
    assert tcga["AuditChain"]["stage3_action"] == "HOLD_BASELINE_DOSE"
    assert tcga["AuditChain"]["regret_vs_baseline_dose"] == 0.0


def test_committed_lock_verifies(report: Any) -> None:
    res = verify_against_lock(report, LOCK_PATH)
    assert res["passed"], res["mismatches"]


def test_cli_export_is_deterministic_and_lock_passes(tmp_path: Path, report: Any) -> None:
    out_a = tmp_path / "a" / "web-fixture.json"
    out_b = tmp_path / "b" / "web-fixture.json"
    rc = suite_main(
        ["--seed", "42", "--verify-lock", str(LOCK_PATH), "--export-web-fixture", str(out_a)]
    )
    assert rc == 0
    assert suite_main(["--seed", "42", "--export-web-fixture", str(out_b)]) == 0
    assert out_a.read_bytes() == out_b.read_bytes()
    loaded = json.loads(out_a.read_text(encoding="utf-8"))
    assert loaded == build_web_fixture(report, seed=42)
