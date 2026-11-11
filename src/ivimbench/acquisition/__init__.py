"""Acquisition-design optimization and NEX-weighted Fisher Information / CRLB analysis."""

from .crlb import (
    CRLBResult,
    compute_crlb,
    crlb_optimality_criterion,
    ivim_fisher_matrix,
    ivim_signal_jacobian,
    resolve_schedule_and_nex,
)
from .design import (
    OptimizedScheduleResult,
    SchemeComparisonReport,
    SchemeEvaluator,
    optimize_b_schedule,
)

__all__ = [
    "CRLBResult",
    "resolve_schedule_and_nex",
    "ivim_signal_jacobian",
    "ivim_fisher_matrix",
    "compute_crlb",
    "crlb_optimality_criterion",
    "OptimizedScheduleResult",
    "SchemeComparisonReport",
    "SchemeEvaluator",
    "optimize_b_schedule",
]
