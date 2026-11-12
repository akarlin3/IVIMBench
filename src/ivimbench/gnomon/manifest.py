"""Frozen clean-room verification manifest (`GNOMON_TARGETS`) for IVIM uncertainty claims.

Pins publication-independent quantitative verification targets so every empirical
claim regarding bounded NLLS ``D*`` boundary railing, conditional high-``D*``
Gaussian interval failure vs. quantile-interval recovery, and ``honest_crlb`` vs.
``floored_railed`` SD conventions can be audited from scratch in a deterministic
clean-room pass (`PASS` / `PARTIAL` / `REFUTE`) with a `KEEP` / `REFRAME` / `DROP`
claims ledger.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

__all__ = [
    "ClaimVerdictStatus",
    "LedgerDisposition",
    "GnomonTargetSpec",
    "GNOMON_TARGETS",
    "GNOMON_PROTOCOL_BVALUES",
    "GNOMON_BOX_BOUNDS",
]

ClaimVerdictStatus = Literal["PASS", "PARTIAL", "REFUTE"]
LedgerDisposition = Literal["KEEP", "REFRAME", "DROP"]

# Canonical 8-shell clinical IVIM b-value schedule (s/mm^2)
GNOMON_PROTOCOL_BVALUES: tuple[float, ...] = (
    0.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    400.0,
    800.0,
)

# Canonical box constraints for bounded bi-exponential NLLS
GNOMON_BOX_BOUNDS: dict[str, tuple[float, float]] = {
    "D": (1.0e-4, 5.0e-3),
    "Dstar": (3.0e-3, 0.150),
    "f": (1.0e-3, 0.700),
}


@dataclass(frozen=True)
class GnomonTargetSpec:
    """Pinned quantitative claim target in the Gnomon clean-room verification manifest."""

    claim_id: str
    title: str
    metric_name: str
    reference_target: float
    cleanroom_center: float
    verified_window: tuple[float, float]
    partial_window: tuple[float, float]
    nominal_alpha: float
    expected_disposition: LedgerDisposition
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verified_window"] = list(self.verified_window)
        d["partial_window"] = list(self.partial_window)
        return d


GNOMON_TARGETS: dict[str, GnomonTargetSpec] = {
    "nlls_dstar_railing_rate": GnomonTargetSpec(
        claim_id="C1_NLLS_DSTAR_RAILING",
        title="Bounded NLLS D* boundary-railing rate on clinical multi-b DWI",
        metric_name="dstar_railing_rate",
        reference_target=0.547,
        cleanroom_center=0.542,
        verified_window=(0.520, 0.564),
        partial_window=(0.480, 0.600),
        nominal_alpha=0.10,
        expected_disposition="KEEP",
        rationale=(
            "Reference open-data NLLS D* boundary-railing target is 54.7%, with "
            "independent clean-room reproduction centered at 54.2% inside [52.0%, 56.4%]."
        ),
    ),
    "high_dstar_gaussian_undercoverage": GnomonTargetSpec(
        claim_id="C2_HIGH_DSTAR_GAUSSIAN_SHAPE_FAILURE",
        title="Conditional high-D* tercile symmetric Gaussian under-coverage (Laplace-SD & MCMC-SD)",
        metric_name="high_dstar_max_gaussian_coverage",
        reference_target=0.680,
        cleanroom_center=0.650,
        verified_window=(0.200, 0.820),
        partial_window=(0.150, 0.875),
        nominal_alpha=0.10,
        expected_disposition="KEEP",
        rationale=(
            "In the upper D* tercile, weak high-D* identifiability and box truncation "
            "skew the likelihood/posterior so symmetric mean +/- z*SD intervals "
            "(both Laplace-SD and MCMC-SD) fall well below nominal 90% coverage."
        ),
    ),
    "high_dstar_quantile_recovery": GnomonTargetSpec(
        claim_id="C3_QUANTILE_INTERVAL_RECOVERY",
        title="Conditional high-D* tercile quantile-interval recovery (MCMC-quantile & conformal)",
        metric_name="high_dstar_min_quantile_coverage",
        reference_target=0.900,
        cleanroom_center=0.900,
        verified_window=(0.855, 0.965),
        partial_window=(0.820, 0.985),
        nominal_alpha=0.10,
        expected_disposition="KEEP",
        rationale=(
            "Replacing symmetric +/- z*SD intervals with equal-tailed posterior quantiles "
            "(MCMC-quantile) or conformalized quantile intervals restores ~90% coverage "
            "even in the upper D* tercile."
        ),
    ),
    "honest_crlb_marginal_regime": GnomonTargetSpec(
        claim_id="C4_HONEST_CRLB_MARGINAL_SEPARATION",
        title="Honest uncollapsed CRLB marginal D* coverage (~0.80-0.90) with high-D* conditional drop",
        metric_name="honest_crlb_marginal_coverage",
        reference_target=0.850,
        cleanroom_center=0.845,
        verified_window=(0.790, 0.915),
        partial_window=(0.740, 0.940),
        nominal_alpha=0.10,
        expected_disposition="REFRAME",
        rationale=(
            "Uncollapsed Fisher/CRLB SDs (`honest_crlb`) achieve ~0.80-0.90 marginal "
            "coverage rather than catastrophic marginal collapse, so claims must frame "
            "CRLB's genuine limitation around conditional high-D* Gaussian shape failure."
        ),
    ),
    "floored_railed_collapsed_artifact": GnomonTargetSpec(
        claim_id="C5_FLOORED_RAILED_ARTIFACT_SEPARATION",
        title="Floored-railed collapsed-SD artifact (~0.30-0.67) explicitly separated from honest CRLB",
        metric_name="floored_railed_marginal_coverage",
        reference_target=0.410,
        cleanroom_center=0.410,
        verified_window=(0.280, 0.680),
        partial_window=(0.200, 0.740),
        nominal_alpha=0.10,
        expected_disposition="DROP",
        rationale=(
            "Zeroing/flooring SD on boundary-railed voxels (`floored_railed`) collapses "
            "marginal coverage to ~0.30-0.67 purely as an active-set software artifact; "
            "drop headline claims attributing ~0.40 marginal collapse to physics-based CRLB."
        ),
    ),
}
