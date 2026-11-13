"""Tests for standardized AveryCorp AuditReport contract and Suite I pipeline in IVIMBench."""

from __future__ import annotations

import json

import pytest

from ivimbench import AuditReport, run_audit, run_suite1_pipeline
from ivimbench.cli import main

REQUIRED_AUDIT_KEYS = {
    "package",
    "version",
    "suite",
    "stage_role",
    "status",
    "verdict",
    "metrics",
    "thresholds",
    "seed",
    "attestation",
}


def test_audit_report_contract_and_determinism() -> None:
    rep1 = run_audit(seed=42)
    rep2 = run_audit(seed=42)
    assert isinstance(rep1, AuditReport)
    assert rep1 == rep2
    assert rep1.package == "ivimbench"
    assert rep1.suite == "Suite I: Clinical Quantitative-MRI Trust Spine"
    assert rep1.status == "PASS"
    assert "CLEAN_ROOM_VERIFIED" in rep1.verdict

    parsed = json.loads(rep1.to_json())
    assert set(parsed.keys()) == REQUIRED_AUDIT_KEYS
    assert parsed["seed"] == 42
    assert parsed["metrics"]["continuity_passed"] is True
    assert parsed["metrics"]["nlls_roundtrip_passed"] is True
    assert parsed["metrics"]["gnomon_passed"] is True
    assert parsed["metrics"]["closed_loop_regret_reduction"] > 0.0
    assert parsed["metrics"]["closed_loop_oar_sparing_gy"] > 0.0
    assert "ivimbench" in rep1.summary_line()


def test_cli_audit_and_pipeline_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc_audit = main(["audit", "--seed", "42", "--json"])
    assert rc_audit == 0
    out_audit = json.loads(capsys.readouterr().out)
    assert set(out_audit.keys()) == REQUIRED_AUDIT_KEYS
    assert out_audit["package"] == "ivimbench"
    assert out_audit["status"] == "PASS"

    rc_pipe = main(["pipeline", "--fractions", "2", "--seed", "42", "--json"])
    assert rc_pipe == 0
    out_pipe = json.loads(capsys.readouterr().out)
    assert out_pipe["suite"] == "Suite I: Clinical Quantitative-MRI Trust Spine"
    assert out_pipe["config"]["conformal_method"] == "cqr"
    assert len(out_pipe["stages"]) == 3
    assert out_pipe["stages"][2]["package"] == "ivimbench"
    assert out_pipe["stages"][2]["status"] == "PASS"

    direct_pipe = run_suite1_pipeline(seed=42)
    assert direct_pipe["stages"] == out_pipe["stages"]
