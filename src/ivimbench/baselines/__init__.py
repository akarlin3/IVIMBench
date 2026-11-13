"""Standard reference IVIM estimators: NLLS, Segmented, CRLB, MLP, MAF, IVIMNet, and Quantile MLP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .crlb_estimator import (
    CRLBEstimator,
    CRLBFitResult,
    CRLBNormalEstimator,
    crlb_normal_intervals,
)
from .maf import MAFPosterior, MAFPosteriorEstimator
from .mlp import MLPEstimator, MLPIVIMEstimator
from .neural import HeteroscedasticQuantileMLP, IVIMNetBaseline
from .nlls import NLLSEstimator, NLLSFitResult, NLLSIVIMEstimator
from .segmented import SegmentedEstimator, SegmentedFitResult, SegmentedIVIMEstimator

if TYPE_CHECKING:
    from ..bridges.avery_stack import CanonicalFitter


def get_canonical_fitter(prefer_sextant: bool = True, **kwargs: Any) -> CanonicalFitter:
    """Return canonical forward fitter delegating to ``sextant_core`` when installed."""
    from ..bridges.avery_stack import get_canonical_fitter as _get_canonical_fitter

    return _get_canonical_fitter(prefer_sextant=prefer_sextant, **kwargs)


__all__ = [
    "CRLBFitResult",
    "CRLBNormalEstimator",
    "CRLBEstimator",
    "crlb_normal_intervals",
    "NLLSFitResult",
    "NLLSEstimator",
    "NLLSIVIMEstimator",
    "SegmentedFitResult",
    "SegmentedEstimator",
    "SegmentedIVIMEstimator",
    "MLPEstimator",
    "MLPIVIMEstimator",
    "MAFPosteriorEstimator",
    "MAFPosterior",
    "IVIMNetBaseline",
    "HeteroscedasticQuantileMLP",
    "get_canonical_fitter",
]
