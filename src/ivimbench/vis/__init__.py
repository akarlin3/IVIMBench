"""Publication-grade diagnostic visualization subpackage for IVIMBench (``ivimbench[vis]``).

Imports ``matplotlib`` and ``seaborn`` lazily inside plotting functions so core
``ivimbench`` installs remain strictly lightweight (``numpy``, ``scipy``, ``pandas``).
"""

from __future__ import annotations

from .calibration import (
    _extract_y_true_matrix,
    _require_matplotlib,
    plot_calibration_curves,
    plot_reliability_and_terciles,
    plot_tercile_coverage,
)
from .publication import (
    plot_dro_slice_montage,
    plot_misspecification_aliasing,
    plot_vernier_pareto_frontier,
)

__all__ = [
    "_extract_y_true_matrix",
    "_require_matplotlib",
    "plot_calibration_curves",
    "plot_tercile_coverage",
    "plot_reliability_and_terciles",
    "plot_misspecification_aliasing",
    "plot_vernier_pareto_frontier",
    "plot_dro_slice_montage",
]
