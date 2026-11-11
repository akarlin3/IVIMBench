"""IVIMBench: Standardized phantoms, clinical protocols, and UQ calibration benchmarks."""

from . import baselines, metrics, phantoms, protocols
from .baselines import MLPEstimator, NLLSEstimator, SegmentedEstimator
from .evaluate import BenchmarkReport, BenchmarkSuite
from .phantoms import (
    PARAM_NAMES,
    DigitalReferenceObject,
    LesionSpec,
    SyntheticCohort,
    generate_cohort,
    generate_dro,
)
from .protocols import AcquisitionProtocol, PROTOCOLS, get_protocol, list_protocols
from .vis import plot_calibration_curves, plot_tercile_coverage

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "PARAM_NAMES",
    "generate_cohort",
    "generate_dro",
    "SyntheticCohort",
    "DigitalReferenceObject",
    "LesionSpec",
    "AcquisitionProtocol",
    "PROTOCOLS",
    "get_protocol",
    "list_protocols",
    "BenchmarkSuite",
    "BenchmarkReport",
    "SegmentedEstimator",
    "NLLSEstimator",
    "MLPEstimator",
    "plot_calibration_curves",
    "plot_tercile_coverage",
    "phantoms",
    "protocols",
    "metrics",
    "baselines",
]
