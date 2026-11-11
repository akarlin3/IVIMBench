"""Optional download-on-demand fetcher for the external OSIPI TF2.4 IVIM reference DRO.

``IVIMBench`` ships exclusively 100% synthetic, PHI-free data in-tree. The
external Open Source Initiative for Perfusion Imaging (OSIPI) Taskforce 2.4
IVIM reference Digital Reference Object (Zenodo ``10.5281/zenodo.14605039``,
CC-BY-4.0) can be fetched on demand via :func:`fetch_osipi_dro` with pinned MD5
checksum verification and auditable provenance metadata (consolidated from
``Lattice/lattice/osipi.py``).

Importing this module never initiates any network activity; network I/O occurs
solely when :func:`fetch_osipi_dro` is called on a cache miss.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "OSIPI_DRO_FILENAME",
    "OSIPI_DRO_MD5",
    "OSIPI_DRO_URL",
    "OSIPI_SOURCE",
    "ExternalDRO",
    "compute_md5",
    "compute_sha256",
    "verify_md5_checksum",
    "provenance_record",
    "load_osipi_dro",
    "load_dro",
    "fetch_osipi_dro",
    "fetch",
]

OSIPI_DRO_FILENAME: str = "OSIPI_TF24_data_phantoms.zip"
OSIPI_DRO_MD5: str = "e7b3fe1d811a7a45c5aaf6c604c82793"
OSIPI_DRO_URL: str = (
    "https://zenodo.org/records/14605039/files/OSIPI_TF24_data_phantoms.zip?download=1"
)

# External source coordinates (CC-BY-4.0 data; Apache-2.0 code upstream).
OSIPI_SOURCE: dict[str, str] = {
    "name": "OSIPI TF2.4 IVIM reference DRO",
    "zenodo_doi": "10.5281/zenodo.14605039",
    "zenodo_url": "https://zenodo.org/records/14605039",
    "asset_filename": OSIPI_DRO_FILENAME,
    "download_url": OSIPI_DRO_URL,
    "md5": OSIPI_DRO_MD5,
    "data_license": "CC-BY-4.0",
    "code_license": "Apache-2.0",
    "citation": (
        "OSIPI Taskforce 2.4. OSIPI TF2.4 IVIM-MRI Code Collection. "
        "Zenodo; doi:10.5281/zenodo.14605039."
    ),
}


@dataclass
class ExternalDRO:
    """Ground-truth parameters and native signals loaded from an external reference DRO."""

    params: np.ndarray  # (N, 3) ordered (D, Dstar, f)
    signals: np.ndarray  # (N, n_b)
    bvalues: np.ndarray  # (n_b,)
    source: dict[str, Any]

    def __len__(self) -> int:
        return int(self.params.shape[0])

    def to_scorer_inputs(
        self,
        q_pred: np.ndarray,
        q_levels: Sequence[float] | np.ndarray = (
            0.025, 0.05, 0.10, 0.16, 0.25, 0.50, 0.75, 0.84, 0.90, 0.95, 0.975
        ),
    ) -> dict[str, Any]:
        """Package external DRO ground truth and predicted quantiles for calibration scoring."""
        qp = np.asarray(q_pred, dtype=np.float64)
        levels = np.asarray(q_levels, dtype=np.float64)
        expected = (self.params.shape[0], self.params.shape[1], levels.size)
        if qp.shape != expected:
            raise ValueError(f"q_pred shape {qp.shape} != expected {expected}.")
        return {
            "y_true": np.asarray(self.params, dtype=np.float64),
            "q_pred": qp,
            "q_levels": levels,
            "param_names": ["D", "Dstar", "f"],
        }


def compute_md5(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute lowercase hexadecimal MD5 digest of ``path``."""
    digest = hashlib.md5()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def compute_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute lowercase hexadecimal SHA-256 digest of ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_md5_checksum(path: str | Path, expected_md5: str = OSIPI_DRO_MD5) -> str:
    """Verify MD5 checksum of ``path`` against ``expected_md5`` (raises ``ValueError`` on mismatch)."""
    actual = compute_md5(path)
    expected = expected_md5.strip().lower()
    if actual != expected:
        raise ValueError(
            f"MD5 checksum mismatch for {Path(path).name!r}: "
            f"expected {expected!r}, got {actual!r}."
        )
    return actual


def provenance_record(
    local_path: str | Path,
    *,
    md5: str | None = None,
    sha256: str | None = None,
    fetched: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic provenance manifest dictionary for a fetched OSIPI artifact.

    ``fetched`` (an optional ISO-8601 timestamp string) is supplied by the caller
    rather than read from the wall clock so manifests remain reproducible and
    auditable across builds.
    """
    return {
        "source": dict(OSIPI_SOURCE),
        "local_path": str(local_path),
        "md5": md5,
        "sha256": sha256,
        "fetched": fetched,
        "note": "External CC-BY-4.0 data; fetched on demand and never redistributed in-tree.",
    }


def load_osipi_dro(path: str | Path) -> ExternalDRO:
    """Load an external DRO ``.npz`` bundle containing ``params``, ``signals``, and ``bvalues``."""
    target = Path(path)
    with np.load(target) as z:
        return ExternalDRO(
            params=np.asarray(z["params"], dtype=np.float64),
            signals=np.asarray(z["signals"], dtype=np.float64),
            bvalues=np.asarray(z["bvalues"], dtype=np.float64),
            source=dict(OSIPI_SOURCE),
        )


load_dro = load_osipi_dro


def _default_download(url: str, dest: Path, timeout: float) -> None:  # pragma: no cover - network
    req = urllib.request.Request(url, headers={"User-Agent": "ivimbench-osipi-fetcher"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out_fh:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out_fh.write(chunk)


def fetch_osipi_dro(
    cache_dir: str | Path | None = None,
    *,
    verify_md5: bool = True,
    expected_md5: str = OSIPI_DRO_MD5,
    url: str = OSIPI_DRO_URL,
    filename: str = OSIPI_DRO_FILENAME,
    force_download: bool = False,
    fetched: str | None = None,
    timeout: float = 60.0,
    downloader: Callable[[str, Path], None] | None = None,
) -> Path:
    """Fetch and cache the external OSIPI TF2.4 IVIM DRO archive with MD5 verification.

    Parameters
    ----------
    cache_dir : str, Path, or None, optional
        Destination directory for the cached archive and its ``.provenance.json``
        manifest. Defaults to ``~/.cache/ivimbench/osipi``.
    verify_md5 : bool, default=True
        Verify the downloaded or cached archive against ``expected_md5``
        (:data:`OSIPI_DRO_MD5`). Raises ``ValueError`` (and deletes any partial
        download) if the checksum fails.
    expected_md5 : str, default=OSIPI_DRO_MD5
        Pinned Zenodo MD5 digest for ``OSIPI_TF24_data_phantoms.zip``.
    url : str, default=OSIPI_DRO_URL
        Direct Zenodo record asset URL.
    filename : str, default=OSIPI_DRO_FILENAME
        Local cache filename inside ``cache_dir``.
    force_download : bool, default=False
        Re-download even when ``cache_dir / filename`` already exists.
    fetched : str or None, optional
        Optional caller-supplied ISO-8601 timestamp recorded in provenance.
    timeout : float, default=60.0
        Socket timeout in seconds when performing HTTP download.
    downloader : callable ``(url, dest_path) -> None`` or None, optional
        Optional transport hook for offline testing or custom mirrors.

    Returns
    -------
    pathlib.Path
        Path to the verified local archive inside ``cache_dir``.
    """
    dest_dir = (
        Path.home() / ".cache" / "ivimbench" / "osipi"
        if cache_dir is None
        else Path(cache_dir)
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_path = dest_dir / filename
    prov_path = dest_dir / f"{filename}.provenance.json"

    if not target_path.exists() or force_download:
        tmp_path = dest_dir / f".{filename}.part"
        try:
            if downloader is not None:
                downloader(url, tmp_path)
            else:  # pragma: no cover - live network path
                _default_download(url, tmp_path, timeout=float(timeout))
            if verify_md5:
                verify_md5_checksum(tmp_path, expected_md5=expected_md5)
            tmp_path.replace(target_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    md5_hex = (
        verify_md5_checksum(target_path, expected_md5=expected_md5)
        if verify_md5
        else compute_md5(target_path)
    )
    sha256_hex = compute_sha256(target_path)
    record = provenance_record(
        target_path,
        md5=md5_hex,
        sha256=sha256_hex,
        fetched=fetched,
    )
    prov_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return target_path


fetch = fetch_osipi_dro
