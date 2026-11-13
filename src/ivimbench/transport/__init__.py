"""Domain-agnostic anomalous transport, stretched-exponential CRLB wall, and Procrustes misspecification API."""

from .core import (
    DEFAULT_Q_GRID,
    FractionalTransportCRLB,
    MisspecificationAliasingReport,
    TransportWallReport,
    compute_fractional_transport_crlb,
    discrete_multiexponential_decay,
    map_anomalous_identifiability_wall,
    probe_misspecification_conditional_aliasing,
    run_transport_audit,
    stretched_exponential_decay,
)

__all__ = [
    "DEFAULT_Q_GRID",
    "FractionalTransportCRLB",
    "TransportWallReport",
    "MisspecificationAliasingReport",
    "stretched_exponential_decay",
    "discrete_multiexponential_decay",
    "compute_fractional_transport_crlb",
    "map_anomalous_identifiability_wall",
    "probe_misspecification_conditional_aliasing",
    "run_transport_audit",
]
