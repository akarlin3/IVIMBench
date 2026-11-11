"""Standardized clinical and research diffusion b-value acquisition protocols.

Consolidates open-source IVIM/DWI acquisition schedules used across quantitative
MRI benchmarks:
- **OSIPI TF2.4** (``"osipi"``): 12-b abdominal IVIM reference schedule
  (Zenodo DOI ``10.5281/zenodo.14605039``, CC-BY-4.0) and the 7-b synthetic DRO
  schedule (``"osipi_dro"``).
- **TCGA-LIHC** (``"tcga_lihc"`` / ``"tcga_lihc_3b"``): Clinical hepatic DWI
  4-b and sparse 3-b schedules from The Cancer Imaging Archive
  (DOI ``10.7937/K9/TCIA.2016.IMMQW8UQ``, CC-BY-3.0).
- **ACRIN 6698** (``"acrin_6698"``): Multi-center breast DWI clinical trial 4-b
  schedule (DOI ``10.7937/TCIA.2019.CWVLPD26``).
- **Dense Research 22-b** (``"lattice_22b"``): Dense low-b research protocol
  from the Lattice DRO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "DEFAULT_SECONDS_PER_VOLUME",
    "AcquisitionProtocol",
    "PROTOCOLS",
    "PROTOCOL_ALIASES",
    "get_protocol",
    "resolve_bvalues",
    "list_protocols",
]

DEFAULT_SECONDS_PER_VOLUME: float = 12.0


@dataclass(frozen=True)
class AcquisitionProtocol:
    """Immutable specification of an IVIM diffusion b-value acquisition schedule.

    Parameters
    ----------
    key : str
        Canonical registry identifier (e.g., ``"osipi"``, ``"tcga_lihc"``,
        ``"acrin_6698"``).
    title : str
        Human-readable protocol name.
    bvalues : tuple of float
        Diffusion weighting b-values in ``s/mm^2`` (non-negative, non-decreasing).
    averages : tuple of int or None, optional
        Number of signal excitations/repeats per b-value entry. Defaults to unit
        averages ``(1, ..., 1)`` when ``None``.
    organ : str, default="abdomen"
        Primary anatomical context associated with the clinical protocol.
    description : str, default=""
        Summary of clinical/benchmark provenance.
    doi : str or None, optional
        Reference dataset or benchmark DOI.
    """

    key: str
    title: str
    bvalues: tuple[float, ...]
    averages: tuple[int, ...] | None = None
    organ: str = "abdomen"
    description: str = ""
    doi: str | None = None

    def __post_init__(self) -> None:
        b = np.asarray(self.bvalues, dtype=np.float64)
        if b.ndim != 1 or b.size < 2:
            raise ValueError(f"Protocol {self.key!r} requires at least 2 b-values.")
        if np.any(b < 0.0):
            raise ValueError(f"Protocol {self.key!r} cannot contain negative b-values.")
        if np.any(np.diff(b) < 0.0):
            raise ValueError(f"Protocol {self.key!r} b-values must be non-decreasing.")
        if self.averages is not None:
            a = np.asarray(self.averages, dtype=int)
            if a.shape != b.shape:
                raise ValueError(
                    f"Protocol {self.key!r}: averages length ({a.size}) must match "
                    f"bvalues length ({b.size})."
                )
            if np.any(a < 1):
                raise ValueError(f"Protocol {self.key!r}: all averages must be >= 1.")

    @property
    def b(self) -> np.ndarray:
        """Return b-values as a 1D ``float64`` NumPy array in ``s/mm^2``."""
        return np.asarray(self.bvalues, dtype=np.float64)

    @property
    def bvals(self) -> np.ndarray:
        """Alias for :attr:`b`."""
        return self.b

    @property
    def averages_arr(self) -> np.ndarray:
        """Return per-b signal averages as a 1D integer NumPy array."""
        if self.averages is None:
            return np.ones(len(self.bvalues), dtype=int)
        return np.asarray(self.averages, dtype=int)

    @property
    def n_b(self) -> int:
        """Number of b-value entries in the schedule."""
        return len(self.bvalues)

    @property
    def n_unique_b(self) -> int:
        """Number of distinct b-values in the schedule."""
        return int(np.unique(self.b).size)

    @property
    def n_acquisitions(self) -> int:
        """Total acquired volumes (sum of signal averages)."""
        return int(self.averages_arr.sum())

    @property
    def has_b0(self) -> bool:
        """Whether the schedule includes an unweighted ``b = 0`` acquisition."""
        return bool(np.any(np.isclose(self.b, 0.0, atol=1e-6)))

    def scan_minutes(self, seconds_per_volume: float = DEFAULT_SECONDS_PER_VOLUME) -> float:
        """Nominal acquisition duration in minutes given per-volume TR overhead."""
        return float(self.n_acquisitions * float(seconds_per_volume) / 60.0)

    def supports_segmented_fit(
        self,
        high_b_min: float = 200.0,
        low_b_max: float = 50.0,
    ) -> bool:
        """Return ``True`` if the schedule has dedicated low-b and high-b windows."""
        b = self.b
        n_high = int(np.sum(b >= high_b_min))
        n_low = int(np.sum((b > 0.0) & (b <= low_b_max)))
        return n_high >= 2 and n_low >= 2


PROTOCOLS: dict[str, AcquisitionProtocol] = {
    "osipi": AcquisitionProtocol(
        key="osipi",
        title="OSIPI TF2.4 Abdominal IVIM 12-b Schedule",
        bvalues=(0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 250.0, 400.0, 600.0),
        averages=None,
        organ="liver",
        description=(
            "Standardized 12-b abdominal IVIM acquisition schedule from the OSIPI "
            "Taskforce 2.4 IVIM-MRI Code Collection (Philips 3T reference protocol)."
        ),
        doi="10.5281/zenodo.14605039",
    ),
    "osipi_dro": AcquisitionProtocol(
        key="osipi_dro",
        title="OSIPI TF2.4 Synthetic DRO Native 7-b Schedule",
        bvalues=(0.0, 10.0, 10.0, 25.0, 50.0, 700.0, 950.0),
        averages=None,
        organ="abdomen",
        description=(
            "Sparse 7-acquisition native b-value schedule shipped in Utilities/DRO.npy "
            "of the OSIPI TF2.4 IVIM reference DRO."
        ),
        doi="10.5281/zenodo.14605039",
    ),
    "tcga_lihc": AcquisitionProtocol(
        key="tcga_lihc",
        title="TCGA-LIHC Clinical Hepatic 4-b DWI Schedule",
        bvalues=(0.0, 50.0, 500.0, 800.0),
        averages=None,
        organ="liver",
        description=(
            "Routine clinical 4-b abdominal liver DWI schedule from The Cancer Genome "
            "Atlas Liver Hepatocellular Carcinoma (TCGA-LIHC) collection."
        ),
        doi="10.7937/K9/TCIA.2016.IMMQW8UQ",
    ),
    "tcga_lihc_3b": AcquisitionProtocol(
        key="tcga_lihc_3b",
        title="TCGA-LIHC Sparse Clinical Hepatic 3-b DWI Schedule",
        bvalues=(50.0, 400.0, 800.0),
        averages=None,
        organ="liver",
        description=(
            "Sparse 3-b clinical free-breathing liver DWI schedule without b=0 from "
            "TCGA-LIHC, exposing severe pseudo-diffusion under-identification."
        ),
        doi="10.7937/K9/TCIA.2016.IMMQW8UQ",
    ),
    "acrin_6698": AcquisitionProtocol(
        key="acrin_6698",
        title="ACRIN 6698 Multi-Center Breast DWI 4-b Schedule",
        bvalues=(0.0, 100.0, 600.0, 800.0),
        averages=None,
        organ="breast",
        description=(
            "Standardized 4-b clinical breast DWI acquisition schedule from the "
            "ACRIN 6698 / I-SPY 2 multi-center neoadjuvant breast MRI trial."
        ),
        doi="10.7937/TCIA.2019.CWVLPD26",
    ),
    "lattice_22b": AcquisitionProtocol(
        key="lattice_22b",
        title="Lattice Dense 22-b IVIM Reference Schedule",
        bvalues=(
            0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0,
            120.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1000.0,
        ),
        averages=None,
        organ="abdomen",
        description=(
            "Dense 22-point research IVIM schedule with rich low-b perfusion sampling "
            "(b <= 100 s/mm^2) from the Lattice UQ calibration DRO."
        ),
        doi=None,
    ),
}

PROTOCOL_ALIASES: dict[str, str] = {
    "osipi": "osipi",
    "osipi_tf24": "osipi",
    "osipi_tf2_4": "osipi",
    "osipi_abdomen": "osipi",
    "tf24": "osipi",
    "tf2_4": "osipi",
    "osipi_dro": "osipi_dro",
    "osipi_7b": "osipi_dro",
    "tcga_lihc": "tcga_lihc",
    "tcga_lihc_4b": "tcga_lihc",
    "lihc": "tcga_lihc",
    "lihc_4b": "tcga_lihc",
    "tcga_lihc_3b": "tcga_lihc_3b",
    "lihc_3b": "tcga_lihc_3b",
    "acrin_6698": "acrin_6698",
    "acrin6698": "acrin_6698",
    "acrin": "acrin_6698",
    "lattice": "lattice_22b",
    "lattice_22b": "lattice_22b",
    "dense_22b": "lattice_22b",
}


def _normalize_protocol_key(name: str) -> str:
    cleaned = name.strip().lower()
    for ch in ("-", ".", " ", "/"):
        cleaned = cleaned.replace(ch, "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def get_protocol(protocol: str | AcquisitionProtocol) -> AcquisitionProtocol:
    """Look up a clinical acquisition protocol by name or alias.

    Parameters
    ----------
    protocol : str or AcquisitionProtocol
        Protocol identifier (e.g., ``"osipi"``, ``"OSIPI TF2.4"``,
        ``"tcga_lihc"``, ``"ACRIN 6698"``) or an existing
        :class:`AcquisitionProtocol`.

    Returns
    -------
    AcquisitionProtocol
        Matching protocol specification.
    """
    if isinstance(protocol, AcquisitionProtocol):
        return protocol
    if not isinstance(protocol, str):
        raise TypeError(f"Expected protocol name string or AcquisitionProtocol, got {type(protocol)!r}.")
    norm = _normalize_protocol_key(protocol)
    canonical = PROTOCOL_ALIASES.get(norm, norm)
    if canonical not in PROTOCOLS:
        raise KeyError(
            f"Unknown acquisition protocol {protocol!r}. "
            f"Available protocols: {sorted(PROTOCOLS.keys())}"
        )
    return PROTOCOLS[canonical]


def resolve_bvalues(
    protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, str]:
    """Resolve a protocol identifier or custom b-value array into ``(bvalues, label)``.

    Parameters
    ----------
    protocol : str, AcquisitionProtocol, or array-like of float
        Named clinical protocol or explicit sequence of b-values in ``s/mm^2``.

    Returns
    -------
    bvalues : np.ndarray
        1D ``float64`` array of b-values in ``s/mm^2``.
    protocol_name : str
        Canonical protocol key or ``"custom"``.
    """
    if isinstance(protocol, (str, AcquisitionProtocol)):
        spec = get_protocol(protocol)
        return spec.b, spec.key
    b = np.asarray(protocol, dtype=np.float64)
    if b.ndim != 1 or b.size < 2:
        raise ValueError(f"Custom b-value schedule must be a 1D array with >=2 entries; got shape {b.shape}.")
    if np.any(b < 0.0):
        raise ValueError("Custom b-value schedule cannot contain negative values.")
    return b, "custom"


def list_protocols() -> Mapping[str, AcquisitionProtocol]:
    """Return a read-only view of all registered acquisition protocols."""
    return dict(PROTOCOLS)
