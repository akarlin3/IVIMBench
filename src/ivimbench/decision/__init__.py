"""Decision-theoretic utility regret, Value of Information, sequential stopping, and domain-shift safeguards."""

from .voi import (  # noqa: I001
    DecisionVoIReport,
    DomainShiftReport,
    MinosCalibrationGapReport,
    check_domain_shift,
    compare_decision_calibration,
    compute_decision_calibration_gap,
    evaluate_decision_voi,
    infer_exceedance_probability,
)
from ..closed_loop.sentinel_stopping import (
    SentinelStoppingReport,
    evaluate_regret_vs_coverage_stopping,
)
from .audit import run_decision_audit

__all__ = [
    "DecisionVoIReport",
    "DomainShiftReport",
    "MinosCalibrationGapReport",
    "SentinelStoppingReport",
    "infer_exceedance_probability",
    "evaluate_decision_voi",
    "compare_decision_calibration",
    "compute_decision_calibration_gap",
    "evaluate_regret_vs_coverage_stopping",
    "check_domain_shift",
    "run_decision_audit",
]
