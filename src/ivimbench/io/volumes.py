"""Affine-preserving 3D/4D NIfTI-1 and DICOM clinical image I/O for IVIMBench.

Provides:
- Lazy-imported ``nibabel`` and ``pydicom`` adapters (raising a clean
  ``ImportError("Install ivimbench[io] for NIfTI/DICOM support")`` when optional
  clinical readers are required but absent).
- A standards-compliant pure-NumPy/gzip NIfTI-1 (``.nii`` / ``.nii.gz``) fallback
  reader and writer supporting ``uint8``, ``int16``, ``int32``, ``float32``, and
  ``float64`` 3D/4D arrays with exact ``4 x 4`` spatial ``affine`` preservation.
- :func:`load_dwi_volume`: Load 4D ``(X, Y, Z, B)`` NIfTI or DICOM series + FSL
  ``.bval`` + optional 3D ROI mask into a :class:`DWIVolumeData` bundle.
- :func:`save_parameter_maps`: Unmask 1D fitted parameter/interval arrays back to
  3D ``(X, Y, Z)`` NIfTI maps (``D_map``, ``Dstar_map``, ``f_map``, CI widths,
  ``uint8`` ``Dstar_railed_mask``, and ``exchangeability_report.json``).
- :func:`export_dro_to_nifti`: Export any 2D/3D :class:`~ivimbench.phantoms.dro.DigitalReferenceObject`
  directly to ``dwi.nii.gz`` + ``dwi.bval`` + ``roi_mask.nii.gz`` + ground-truth
  parameter NIfTIs for 3D Slicer, ITK-SNAP, FSL, or MATLAB testing.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..decision.voi import check_domain_shift
from ..metrics.railing import DEFAULT_PARAM_BOUNDS, DEFAULT_RAIL_TOL, classify_railing
from ..phantoms.dro import DigitalReferenceObject
from ..registry import NormalizedEstimatorOutput, run_estimator
from ..runner import _sanitize_json

__all__ = [
    "IO_EXTRA_INSTALL_MSG",
    "has_nibabel",
    "has_pydicom",
    "DWIVolumeData",
    "read_bvals",
    "write_bvals",
    "read_nifti",
    "write_nifti",
    "load_dwi_volume",
    "save_parameter_maps",
    "export_dro_to_nifti",
    "fit_dwi_volume",
]

IO_EXTRA_INSTALL_MSG = "Install ivimbench[io] for NIfTI/DICOM support"

# NIfTI-1 datatype codes: (datatype_code, bitpix, numpy_dtype)
_NP_TO_NIFTI_DTYPE: dict[np.dtype[Any], tuple[int, int]] = {
    np.dtype(np.uint8): (2, 8),
    np.dtype(np.int16): (4, 16),
    np.dtype(np.int32): (8, 32),
    np.dtype(np.float32): (16, 32),
    np.dtype(np.float64): (64, 64),
}

_NIFTI_CODE_TO_NP: dict[int, str] = {
    2: "u1",    # DT_UINT8
    4: "i2",    # DT_INT16
    8: "i4",    # DT_INT32
    16: "f4",   # DT_FLOAT32
    64: "f8",   # DT_FLOAT64
    256: "i1",  # DT_INT8
    512: "u2",  # DT_UINT16
    768: "u4",  # DT_UINT32
}

_NIFTI_ECODE_COMMENT = 6
_EXT_MAGIC_PREFIX = "IVIMBENCH_META_V1:"


def has_nibabel() -> bool:
    """Return ``True`` if ``nibabel`` is importable in the active Python environment."""
    return importlib.util.find_spec("nibabel") is not None


def has_pydicom() -> bool:
    """Return ``True`` if ``pydicom`` is importable in the active Python environment."""
    return importlib.util.find_spec("pydicom") is not None


def _require_nibabel() -> Any:
    if not has_nibabel():
        raise ImportError(IO_EXTRA_INSTALL_MSG)
    import nibabel as nib  # type: ignore[import-not-found]

    return nib


def _require_pydicom() -> Any:
    if not has_pydicom():
        raise ImportError(IO_EXTRA_INSTALL_MSG)
    import pydicom  # type: ignore[import-not-found]

    return pydicom


def _validate_affine(affine: np.ndarray | None, voxel_size_mm: Sequence[float] = (2.0, 2.0, 4.0)) -> np.ndarray:
    if affine is None:
        vx, vy, vz = (float(voxel_size_mm[0]), float(voxel_size_mm[1]), float(voxel_size_mm[2]))
        return np.diag([vx, vy, vz, 1.0]).astype(np.float64)
    mat = np.asarray(affine, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"`affine` must have shape (4, 4); got {mat.shape}.")
    if not np.all(np.isfinite(mat)):
        raise ValueError("`affine` must contain finite entries.")
    return mat.copy()


def _voxel_sizes_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    norms = np.linalg.norm(affine[:3, :3], axis=0)
    return (
        float(norms[0]) if norms[0] > 0 else 1.0,
        float(norms[1]) if norms[1] > 0 else 1.0,
        float(norms[2]) if norms[2] > 0 else 1.0,
    )


def read_bvals(bval_source: str | Path | Sequence[float] | np.ndarray) -> np.ndarray:
    """Load diffusion b-values (s/mm^2) from an FSL ``.bval`` file or sequence."""
    if isinstance(bval_source, (str, Path)):
        p = Path(bval_source)
        if not p.is_file():
            raise FileNotFoundError(f"b-value file not found: {p}")
        text = p.read_text(encoding="utf-8").replace(",", " ")
        tokens = text.split()
        if not tokens:
            raise ValueError(f"b-value file is empty: {p}")
        bvals = np.array([float(tok) for tok in tokens], dtype=np.float64)
    else:
        bvals = np.asarray(bval_source, dtype=np.float64).ravel()
    if bvals.size == 0 or np.any(~np.isfinite(bvals)) or np.any(bvals < 0.0):
        raise ValueError("`bvalues` must be a non-empty array of finite non-negative numbers.")
    return bvals


def write_bvals(bvalues: Sequence[float] | np.ndarray, bval_path: str | Path) -> Path:
    """Write diffusion b-values in single-line FSL ``.bval`` format."""
    b_arr = np.asarray(bvalues, dtype=np.float64).ravel()
    out_p = Path(bval_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    formatted = " ".join(
        f"{int(round(v))}" if abs(v - round(v)) < 1.0e-6 else f"{v:.6g}" for v in b_arr
    )
    out_p.write_text(formatted + "\n", encoding="utf-8")
    return out_p


def _write_nifti1_numpy(
    data: np.ndarray,
    affine: np.ndarray,
    path: Path,
    *,
    descrip: str = "IVIMBench v1.4 PHI-free NIfTI-1",
) -> Path:
    """Pure-NumPy NIfTI-1.1 single-file writer (`.nii` / `.nii.gz`) with exact 4x4 affine."""
    arr = np.asarray(data)
    if arr.ndim not in (3, 4):
        raise ValueError(f"NIfTI volume must be 3D or 4D; got shape {arr.shape}.")

    dt = arr.dtype
    if dt == np.dtype(bool):
        arr = arr.astype(np.uint8)
        dt = arr.dtype
    elif dt not in _NP_TO_NIFTI_DTYPE:
        if np.issubdtype(dt, np.integer):
            arr = arr.astype(np.int32)
        else:
            arr = arr.astype(np.float64)
        dt = arr.dtype

    le_dt = np.dtype("<" + dt.str[1:])
    datatype_code, bitpix = _NP_TO_NIFTI_DTYPE[np.dtype(dt.type)]
    aff = _validate_affine(affine)
    vx, vy, vz = _voxel_sizes_from_affine(aff)

    # Check if float32 srow_x/y/z loses any float64 precision; if so, attach a standard
    # 16-byte-aligned NIfTI-1.1 comment extension (ecode=6) preserving full float64 affine.
    aff_f32 = aff.astype(np.float32)
    needs_f64_ext = not np.array_equal(aff_f32.astype(np.float64), aff)
    ext_bytes = b""
    extender = b"\x00\x00\x00\x00"
    vox_offset = 352.0
    if needs_f64_ext:
        payload_json = (
            _EXT_MAGIC_PREFIX
            + json.dumps({"affine_f64": aff.tolist()}, separators=(",", ":"))
        ).encode("utf-8") + b"\x00"
        # Total extension block size (8-byte header + payload) must be a multiple of 16
        raw_len = 8 + len(payload_json)
        esize = ((raw_len + 15) // 16) * 16
        padded_payload = payload_json.ljust(esize - 8, b"\x00")
        ext_bytes = struct.pack("<ii", esize, _NIFTI_ECODE_COMMENT) + padded_payload
        extender = b"\x01\x00\x00\x00"
        vox_offset = float(352 + esize)

    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)  # sizeof_hdr

    ndim = arr.ndim
    dims = [ndim, int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]), 1, 1, 1, 1]
    if ndim == 4:
        dims[4] = int(arr.shape[3])
    if any(d > 32767 for d in dims[1:]):
        raise ValueError(f"NIfTI-1 axis lengths must be <= 32767; got shape {arr.shape}.")
    struct.pack_into("<8h", hdr, 40, *dims)

    struct.pack_into("<hh", hdr, 70, datatype_code, bitpix)
    pixdim = [1.0, vx, vy, vz, 1.0, 0.0, 0.0, 0.0]
    struct.pack_into("<8f", hdr, 76, *pixdim)
    struct.pack_into("<f", hdr, 108, vox_offset)
    struct.pack_into("<ff", hdr, 112, 1.0, 0.0)  # scl_slope=1.0, scl_inter=0.0
    struct.pack_into("<B", hdr, 123, 2)  # xyzt_units: NIFTI_UNITS_MM (2)

    desc_bytes = descrip.encode("ascii", errors="ignore")[:79]
    hdr[148 : 148 + len(desc_bytes)] = desc_bytes

    # qform_code=0, sform_code=1 (NIFTI_XFORM_SCANNER_ANAT)
    struct.pack_into("<hh", hdr, 252, 0, 1)
    struct.pack_into("<fff", hdr, 268, float(aff[0, 3]), float(aff[1, 3]), float(aff[2, 3]))
    struct.pack_into("<4f", hdr, 280, *[float(v) for v in aff[0]])
    struct.pack_into("<4f", hdr, 296, *[float(v) for v in aff[1]])
    struct.pack_into("<4f", hdr, 312, *[float(v) for v in aff[2]])
    hdr[344:348] = b"n+1\x00"

    # Column-major (Fortran 'F') voxel buffer as required by NIfTI-1 specification
    vox_bytes = np.asfortranarray(arr, dtype=le_dt).tobytes(order="F")
    full_stream = bytes(hdr) + extender + ext_bytes + vox_bytes

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.lower().endswith(".nii.gz"):
        buf = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buf, compresslevel=6, mtime=0) as gz:
            gz.write(full_stream)
        path.write_bytes(buf.getvalue())
    else:
        path.write_bytes(full_stream)
    return path


def _read_nifti1_numpy(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Pure-NumPy NIfTI-1.1 single-file reader (`.nii` / `.nii.gz`)."""
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    if path.name.lower().endswith(".nii.gz"):
        with gzip.open(path, "rb") as gz:
            raw = gz.read()
    else:
        raw = path.read_bytes()

    if len(raw) < 352:
        raise ValueError(f"Invalid or truncated NIfTI-1 file ({len(raw)} bytes): {path}")

    sizeof_le = struct.unpack_from("<i", raw, 0)[0]
    sizeof_be = struct.unpack_from(">i", raw, 0)[0]
    if sizeof_le == 348:
        endian = "<"
    elif sizeof_be == 348:
        endian = ">"
    else:
        raise ValueError(f"Unrecognized NIfTI-1 header size ({sizeof_le}) in {path}.")

    magic = raw[344:348]
    if magic not in (b"n+1\x00", b"ni1\x00"):
        raise ValueError(f"Unsupported NIfTI magic {magic!r} in {path}.")
    if magic == b"ni1\x00":
        raise ValueError(f"Two-file (.hdr/.img) NIfTI is not supported in fallback reader: {path}")

    dims_all = struct.unpack_from(f"{endian}8h", raw, 40)
    ndim = int(dims_all[0])
    if ndim < 1 or ndim > 7:
        raise ValueError(f"Invalid NIfTI dimension count dim[0]={ndim} in {path}.")
    shape = tuple(int(max(dims_all[i], 1)) for i in range(1, ndim + 1))

    datatype_code, bitpix = struct.unpack_from(f"{endian}hh", raw, 70)
    if datatype_code not in _NIFTI_CODE_TO_NP:
        if not has_nibabel():
            raise ImportError(IO_EXTRA_INSTALL_MSG)
        raise ValueError(f"Unsupported NIfTI datatype code {datatype_code} in {path}.")

    pixdim = struct.unpack_from(f"{endian}8f", raw, 76)
    vox_offset = int(round(struct.unpack_from(f"{endian}f", raw, 108)[0]))
    if vox_offset < 352:
        vox_offset = 352
    scl_slope, scl_inter = struct.unpack_from(f"{endian}ff", raw, 112)
    xyzt_units = struct.unpack_from("B", raw, 123)[0]
    descrip = raw[148:228].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
    qform_code, sform_code = struct.unpack_from(f"{endian}hh", raw, 252)

    if sform_code > 0:
        srow_x = struct.unpack_from(f"{endian}4f", raw, 280)
        srow_y = struct.unpack_from(f"{endian}4f", raw, 296)
        srow_z = struct.unpack_from(f"{endian}4f", raw, 312)
        affine = np.array(
            [
                list(srow_x),
                list(srow_y),
                list(srow_z),
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    elif qform_code > 0:
        qb, qc, qd, qx, qy, qz = struct.unpack_from(f"{endian}6f", raw, 256)
        qa_sq = max(1.0 - (qb * qb + qc * qc + qd * qd), 0.0)
        qa = float(np.sqrt(qa_sq))
        rot = np.array(
            [
                [
                    qa * qa + qb * qb - qc * qc - qd * qd,
                    2.0 * (qb * qc - qa * qd),
                    2.0 * (qb * qd + qa * qc),
                ],
                [
                    2.0 * (qb * qc + qa * qd),
                    qa * qa + qc * qc - qb * qb - qd * qd,
                    2.0 * (qc * qd - qa * qb),
                ],
                [
                    2.0 * (qb * qd - qa * qc),
                    2.0 * (qc * qd + qa * qb),
                    qa * qa + qd * qd - qb * qb - qc * qc,
                ],
            ],
            dtype=np.float64,
        )
        qfac = -1.0 if pixdim[0] < 0.0 else 1.0
        scales = np.array([pixdim[1], pixdim[2], qfac * pixdim[3]], dtype=np.float64)
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = rot * scales[None, :]
        affine[:3, 3] = [qx, qy, qz]
    else:
        vx = float(pixdim[1]) if pixdim[1] > 0 else 1.0
        vy = float(pixdim[2]) if pixdim[2] > 0 else 1.0
        vz = float(pixdim[3]) if pixdim[3] > 0 else 1.0
        affine = np.diag([vx, vy, vz, 1.0]).astype(np.float64)

    # Parse optional NIfTI-1.1 comment extension for lossless float64 affine round-trip
    if raw[348] != 0 and vox_offset >= 360:
        pos = 352
        while pos + 8 <= min(vox_offset, len(raw)):
            esize, ecode = struct.unpack_from(f"{endian}ii", raw, pos)
            if esize < 8 or pos + esize > vox_offset:
                break
            if ecode == _NIFTI_ECODE_COMMENT:
                txt = (
                    raw[pos + 8 : pos + esize]
                    .split(b"\x00", 1)[0]
                    .decode("utf-8", errors="ignore")
                )
                if txt.startswith(_EXT_MAGIC_PREFIX):
                    try:
                        ext_obj = json.loads(txt[len(_EXT_MAGIC_PREFIX) :])
                        if "affine_f64" in ext_obj:
                            affine = np.asarray(ext_obj["affine_f64"], dtype=np.float64)
                    except json.JSONDecodeError:
                        pass
            pos += esize

    np_dt = np.dtype(endian + _NIFTI_CODE_TO_NP[datatype_code])
    n_vox = int(np.prod(shape))
    byte_len = n_vox * np_dt.itemsize
    if vox_offset + byte_len > len(raw):
        raise ValueError(
            f"Truncated voxel payload in {path}: expected {vox_offset + byte_len} bytes, got {len(raw)}."
        )

    arr_f: np.ndarray = np.ndarray(
        shape=shape,
        dtype=np_dt,
        buffer=memoryview(raw)[vox_offset : vox_offset + byte_len],
        order="F",
    ).copy()
    arr = np.ascontiguousarray(arr_f)

    if (
        np.isfinite(scl_slope)
        and scl_slope != 0.0
        and not (abs(scl_slope - 1.0) < 1.0e-7 and abs(scl_inter) < 1.0e-7)
    ):
        arr = arr.astype(np.float64) * float(scl_slope) + float(scl_inter)

    meta: dict[str, Any] = {
        "source_format": "nifti1",
        "backend": "numpy_fallback",
        "datatype_code": int(datatype_code),
        "bitpix": int(bitpix),
        "voxel_sizes_mm": list(_voxel_sizes_from_affine(affine)),
        "xyzt_units": int(xyzt_units),
        "sform_code": int(sform_code),
        "qform_code": int(qform_code),
        "descrip": descrip,
    }
    return arr, affine, meta


def read_nifti(
    path: str | Path,
    *,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read a 3D or 4D NIfTI volume (`.nii` / `.nii.gz`), returning ``(data, affine, header_meta)``."""
    p = Path(path)
    if backend == "nibabel":
        nib = _require_nibabel()
        img = nib.load(str(p))
        arr = np.ascontiguousarray(img.get_fdata(dtype=np.float64))
        aff = np.asarray(img.affine, dtype=np.float64).copy()
        hdr = img.header
        meta = {
            "source_format": "nifti1",
            "backend": "nibabel",
            "datatype_code": int(hdr["datatype"]),
            "bitpix": int(hdr["bitpix"]),
            "voxel_sizes_mm": [float(z) for z in hdr.get_zooms()[:3]],
            "sform_code": int(hdr["sform_code"]),
            "qform_code": int(hdr["qform_code"]),
            "descrip": bytes(hdr["descrip"]).split(b"\x00", 1)[0].decode("ascii", errors="ignore"),
        }
        return arr, aff, meta

    if backend == "auto" and has_nibabel():
        # Prefer pure-NumPy reader when our lossless float64 affine extension is present,
        # otherwise delegate to nibabel while preserving integer dtype for masks.
        nib = _require_nibabel()
        img = nib.load(str(p))
        arr = np.ascontiguousarray(np.asarray(img.dataobj))
        aff = np.asarray(img.affine, dtype=np.float64).copy()
        for ext in img.header.extensions:
            if ext.get_code() == _NIFTI_ECODE_COMMENT:
                content = ext.get_content()
                txt = (
                    content.decode("utf-8", errors="ignore")
                    if isinstance(content, bytes)
                    else str(content)
                ).split("\x00", 1)[0]
                if txt.startswith(_EXT_MAGIC_PREFIX):
                    try:
                        ext_obj = json.loads(txt[len(_EXT_MAGIC_PREFIX) :])
                        if "affine_f64" in ext_obj:
                            aff = np.asarray(ext_obj["affine_f64"], dtype=np.float64)
                    except json.JSONDecodeError:
                        pass
        hdr = img.header
        meta = {
            "source_format": "nifti1",
            "backend": "nibabel",
            "datatype_code": int(hdr["datatype"]),
            "bitpix": int(hdr["bitpix"]),
            "voxel_sizes_mm": list(_voxel_sizes_from_affine(aff)),
            "sform_code": int(hdr["sform_code"]),
            "qform_code": int(hdr["qform_code"]),
            "descrip": bytes(hdr["descrip"]).split(b"\x00", 1)[0].decode("ascii", errors="ignore"),
        }
        return arr, aff, meta

    if backend in ("auto", "numpy"):
        return _read_nifti1_numpy(p)

    raise ValueError(f"Unknown NIfTI backend {backend!r}; choose 'auto', 'nibabel', or 'numpy'.")


def write_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    path: str | Path,
    *,
    backend: str = "auto",
    descrip: str = "IVIMBench v1.4 PHI-free NIfTI-1",
) -> Path:
    """Write a 3D or 4D NIfTI-1 volume (`.nii` / `.nii.gz`) with preserved ``4 x 4`` ``affine``."""
    out_p = Path(path)
    aff = _validate_affine(affine)
    arr = np.asarray(data)
    if arr.dtype == np.dtype(bool):
        arr = arr.astype(np.uint8)

    if backend == "nibabel":
        nib = _require_nibabel()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        img = nib.Nifti1Image(arr, aff)
        img.header.set_sform(aff, code=1)
        img.header["descrip"] = descrip.encode("ascii", errors="ignore")[:79]
        if not np.array_equal(aff.astype(np.float32).astype(np.float64), aff):
            ext_txt = _EXT_MAGIC_PREFIX + json.dumps({"affine_f64": aff.tolist()}, separators=(",", ":"))
            img.header.extensions.append(
                nib.nifti1.Nifti1Extension(_NIFTI_ECODE_COMMENT, ext_txt.encode("utf-8"))
            )
        nib.save(img, str(out_p))
        return out_p

    if backend in ("auto", "numpy"):
        return _write_nifti1_numpy(arr, aff, out_p, descrip=descrip)

    raise ValueError(f"Unknown NIfTI backend {backend!r}; choose 'auto', 'nibabel', or 'numpy'.")


def _extract_dicom_bvalue(ds: Any) -> float | None:
    """Extract diffusion b-value from standard or vendor DICOM tags (strictly PHI-free)."""
    # Standard DICOM Diffusion b-value tag (0018, 9087)
    if (0x0018, 0x9087) in ds:
        try:
            return float(ds[(0x0018, 0x9087)].value)
        except (TypeError, ValueError):
            pass
    # Common vendor private diffusion b-value tags: Siemens (0019,100C), GE (0043,1039), Philips (2001,1003)
    for group, elem in ((0x0019, 0x100C), (0x0043, 0x1039), (0x2001, 0x1003)):
        if (group, elem) in ds:
            raw_val = ds[(group, elem)].value
            try:
                if isinstance(raw_val, (list, tuple)):
                    return float(raw_val[0])
                if isinstance(raw_val, bytes):
                    txt = raw_val.decode("ascii", errors="ignore").strip().split("\\")[0]
                    return float(txt)
                return float(str(raw_val).split("\\")[0])
            except (TypeError, ValueError, IndexError):
                continue
    return None


def _load_dicom_series_4d(
    dicom_dir: Path,
    bvals_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a 4D DWI volume from a DICOM directory via ``pydicom`` without reading any PHI."""
    pydicom = _require_pydicom()

    files = sorted(
        p
        for p in dicom_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() not in (".bval", ".bvec", ".json")
    )
    if not files:
        raise FileNotFoundError(f"No DICOM slice files found in directory: {dicom_dir}")

    records: list[tuple[float, float, int, np.ndarray, Any]] = []
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp))
        except Exception:
            continue
        if not hasattr(ds, "pixel_array"):
            continue
        arr2d = np.asarray(ds.pixel_array, dtype=np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        arr2d = arr2d * slope + inter

        ipp = [float(v) for v in getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0])]
        iop = [float(v) for v in getattr(ds, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])]
        row_cos = np.array(iop[:3], dtype=np.float64)
        col_cos = np.array(iop[3:6], dtype=np.float64)
        slice_norm = np.cross(row_cos, col_cos)
        z_proj = float(np.dot(np.array(ipp, dtype=np.float64), slice_norm))

        b_tag = _extract_dicom_bvalue(ds)
        b_sort = float(b_tag) if b_tag is not None else -1.0
        inst_num = int(getattr(ds, "InstanceNumber", 0))
        records.append((b_sort, z_proj, inst_num, arr2d, ds))

    if not records:
        raise ValueError(f"No valid DICOM image slices could be parsed in {dicom_dir}.")

    # Group unique slice positions along slice normal
    z_positions = sorted({round(r[1], 3) for r in records})
    n_z = len(z_positions)
    n_total = len(records)
    if bvals_override is not None:
        n_b = int(bvals_override.size)
        if n_z * n_b != n_total:
            raise ValueError(
                f"DICOM slice count ({n_total}) does not factor into n_slices ({n_z}) * n_bvals ({n_b})."
            )
        # Sort by InstanceNumber to group volumes cleanly when bval tags are absent, else by b_sort
        if all(r[0] >= 0.0 for r in records):
            records.sort(key=lambda r: (r[0], round(r[1], 3), r[2]))
        else:
            records.sort(key=lambda r: (r[2], round(r[1], 3)))
        bvals = bvals_override.copy()
    else:
        if any(r[0] < 0.0 for r in records):
            raise ValueError(
                "DICOM slices lack diffusion b-value tags; provide an explicit `bval_path`."
            )
        b_unique = sorted({round(r[0], 4) for r in records})
        n_b = len(b_unique)
        if n_z * n_b != n_total:
            raise ValueError(
                f"DICOM slice count ({n_total}) != n_slices ({n_z}) * unique b-values ({n_b})."
            )
        records.sort(key=lambda r: (round(r[0], 4), round(r[1], 3), r[2]))
        bvals = np.array(b_unique, dtype=np.float64)

    first_arr = records[0][3]
    rows, cols = first_arr.shape
    # Standard NIfTI/DICOM convention: (X=cols, Y=rows, Z=slices, B=bvals)
    dwi_4d = np.zeros((cols, rows, n_z, n_b), dtype=np.float64)
    z_lookup = {z_val: zi for zi, z_val in enumerate(z_positions)}
    for b_idx in range(n_b):
        block = records[b_idx * n_z : (b_idx + 1) * n_z]
        block.sort(key=lambda r: round(r[1], 3))
        for z_idx_fallback, rec in enumerate(block):
            zi = z_lookup.get(round(rec[1], 3), z_idx_fallback)
            dwi_4d[:, :, zi, b_idx] = rec[3].T

    # Construct RAS+ 4x4 affine from DICOM LPS orientation & spacing (flipping L->R, P->A)
    ds0 = records[0][4]
    px_sp = [float(v) for v in getattr(ds0, "PixelSpacing", [2.0, 2.0])]  # [row_spacing, col_spacing]
    row_sp, col_sp = px_sp[0], px_sp[1]
    iop0 = [float(v) for v in getattr(ds0, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])]
    row_cos = np.array(iop0[:3], dtype=np.float64)  # direction along columns
    col_cos = np.array(iop0[3:6], dtype=np.float64)  # direction along rows
    slice_norm = np.cross(row_cos, col_cos)

    z_slices_sorted = sorted(records[:n_z], key=lambda r: round(r[1], 3))
    ipp_first = np.array(
        [float(v) for v in getattr(z_slices_sorted[0][4], "ImagePositionPatient", [0.0, 0.0, 0.0])],
        dtype=np.float64,
    )
    slice_vec: np.ndarray
    if n_z > 1:
        ipp_second = np.array(
            [float(v) for v in getattr(z_slices_sorted[1][4], "ImagePositionPatient", [0.0, 0.0, 1.0])],
            dtype=np.float64,
        )
        slice_vec = ipp_second - ipp_first
    else:
        dz = float(getattr(ds0, "SliceThickness", 4.0))
        slice_vec = np.asarray(slice_norm * dz, dtype=np.float64)

    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = lps_to_ras @ (row_cos * col_sp)
    affine[:3, 1] = lps_to_ras @ (col_cos * row_sp)
    affine[:3, 2] = lps_to_ras @ slice_vec
    affine[:3, 3] = lps_to_ras @ ipp_first

    # PHI-free geometric metadata only
    meta: dict[str, Any] = {
        "source_format": "dicom_series",
        "backend": "pydicom",
        "n_dicom_files": int(n_total),
        "voxel_sizes_mm": list(_voxel_sizes_from_affine(affine)),
        "phi_scrubbed": True,
    }
    return dwi_4d, bvals, affine, meta


@dataclass
class DWIVolumeData:
    """Affine-preserving 4D clinical DWI volume and masked 2D fitting matrix."""

    dwi_4d: np.ndarray
    signals: np.ndarray
    bvalues: np.ndarray
    mask: np.ndarray
    affine: np.ndarray
    spatial_shape: tuple[int, int, int]
    header_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_masked(self) -> int:
        """Number of masked foreground voxels in ``signals``."""
        return int(self.signals.shape[0])

    @property
    def n_bvalues(self) -> int:
        """Number of diffusion b-value shells."""
        return int(self.bvalues.size)

    def unmask(
        self,
        values: np.ndarray,
        *,
        fill_value: float = 0.0,
        dtype: Any = np.float64,
    ) -> np.ndarray:
        """Restore a 1D ``(N_masked,)`` or 2D ``(N_masked, K)`` array back to 3D/4D spatial shape."""
        arr = np.asarray(values)
        if arr.ndim == 1:
            if arr.shape[0] != self.n_masked:
                raise ValueError(
                    f"Cannot unmask length {arr.shape[0]} into mask with {self.n_masked} active voxels."
                )
            vol = np.full(self.spatial_shape, fill_value, dtype=dtype)
            vol[self.mask] = arr.astype(dtype)
            return vol
        if arr.ndim == 2:
            if arr.shape[0] != self.n_masked:
                raise ValueError(
                    f"Cannot unmask shape {arr.shape} into mask with {self.n_masked} active voxels."
                )
            vol4 = np.full(self.spatial_shape + (arr.shape[1],), fill_value, dtype=dtype)
            vol4[self.mask] = arr.astype(dtype)
            return vol4
        raise ValueError(f"`values` must be 1D or 2D; got shape {arr.shape}.")


def load_dwi_volume(
    dwi_path: str | Path,
    bval_path: str | Path | Sequence[float] | np.ndarray | None = None,
    mask_path: str | Path | np.ndarray | None = None,
    *,
    backend: str = "auto",
) -> DWIVolumeData:
    """Load a 4D ``(X, Y, Z, B)`` NIfTI file or DICOM series directory + ``.bval`` + optional 3D mask.

    Preserves the ``4 x 4`` spatial ``affine`` matrix and header metadata while
    flattening masked foreground voxels to ``(N_masked, B)`` for vectorized IVIM fitting.
    """
    p = Path(dwi_path)
    bvals_opt = read_bvals(bval_path) if bval_path is not None else None

    if p.is_dir():
        dwi_raw, bvalues, affine, meta = _load_dicom_series_4d(p, bvals_override=bvals_opt)
    else:
        if bvals_opt is None:
            raise ValueError("`bval_path` is required when loading a NIfTI DWI volume.")
        bvalues = bvals_opt
        dwi_raw, affine, meta = read_nifti(p, backend=backend)

    if dwi_raw.ndim == 3 and dwi_raw.shape[2] == bvalues.size:
        # Promote single-slice (X, Y, B) volume to 4D (X, Y, 1, B)
        dwi_4d = np.asarray(dwi_raw[:, :, None, :], dtype=np.float64)
    elif dwi_raw.ndim == 4:
        dwi_4d = np.asarray(dwi_raw, dtype=np.float64)
    else:
        raise ValueError(
            f"Expected 4D DWI volume `(X, Y, Z, B)`, got shape {dwi_raw.shape} from {p}."
        )

    if dwi_4d.shape[3] != bvalues.size:
        raise ValueError(
            f"DWI 4th dimension ({dwi_4d.shape[3]}) does not match b-value count ({bvalues.size})."
        )

    spatial_shape = (int(dwi_4d.shape[0]), int(dwi_4d.shape[1]), int(dwi_4d.shape[2]))

    mask_3d: np.ndarray
    if mask_path is None:
        mask_3d = np.ones(spatial_shape, dtype=bool)
    elif isinstance(mask_path, (str, Path)):
        mask_raw, _, _ = read_nifti(mask_path, backend=backend)
        if mask_raw.ndim == 2 and spatial_shape[2] == 1:
            mask_raw = mask_raw[:, :, None]
        if mask_raw.shape != spatial_shape:
            raise ValueError(
                f"ROI mask shape {mask_raw.shape} does not match DWI spatial shape {spatial_shape}."
            )
        mask_3d = np.asarray(mask_raw > 0, dtype=bool)
    else:
        mask_arr = np.asarray(mask_path)
        if mask_arr.ndim == 2 and spatial_shape[2] == 1:
            mask_arr = mask_arr[:, :, None]
        if mask_arr.shape != spatial_shape:
            raise ValueError(
                f"ROI mask shape {mask_arr.shape} does not match DWI spatial shape {spatial_shape}."
            )
        mask_3d = np.asarray(mask_arr > 0, dtype=bool)

    if not np.any(mask_3d):
        raise ValueError("ROI mask contains zero foreground voxels.")

    signals = np.ascontiguousarray(dwi_4d[mask_3d], dtype=np.float64)
    return DWIVolumeData(
        dwi_4d=dwi_4d,
        signals=signals,
        bvalues=bvalues,
        mask=mask_3d,
        affine=affine,
        spatial_shape=spatial_shape,
        header_meta=meta,
    )


def _extract_volume_meta(
    volume_meta: DWIVolumeData | Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], dict[str, Any], DWIVolumeData | None]:
    if isinstance(volume_meta, DWIVolumeData):
        return (
            volume_meta.mask,
            volume_meta.affine,
            volume_meta.spatial_shape,
            dict(volume_meta.header_meta),
            volume_meta,
        )
    if isinstance(volume_meta, Mapping):
        mask = np.asarray(volume_meta["mask"], dtype=bool)
        if mask.ndim != 3:
            raise ValueError(f"`volume_meta['mask']` must be 3D; got shape {mask.shape}.")
        aff = _validate_affine(np.asarray(volume_meta.get("affine", np.eye(4)), dtype=np.float64))
        shp = (int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2]))
        hdr = dict(volume_meta.get("header_meta", {}))
        return mask, aff, shp, hdr, None
    raise TypeError(
        f"`volume_meta` must be a DWIVolumeData or mapping; got {type(volume_meta).__name__}."
    )


def _extract_fit_arrays(
    fit_result: NormalizedEstimatorOutput | Mapping[str, Any] | np.ndarray,
    n_masked: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Normalize NormalizedEstimatorOutput, dict, or array into (point, lower, upper, dstar_railed, extras)."""
    extras: dict[str, Any] = {}
    if isinstance(fit_result, NormalizedEstimatorOutput):
        point = np.asarray(fit_result.point, dtype=np.float64)
        lower = np.asarray(fit_result.lower, dtype=np.float64)
        upper = np.asarray(fit_result.upper, dtype=np.float64)
        if fit_result.railed is not None:
            dstar_railed = np.asarray(fit_result.railed[:, 1], dtype=bool)
        else:
            dstar_railed = classify_railing(
                point[:, 1], DEFAULT_PARAM_BOUNDS["Dstar"], DEFAULT_RAIL_TOL
            )["railed_mask"]
        extras = dict(fit_result.extras)
    elif isinstance(fit_result, Mapping):
        if "point" in fit_result:
            point = np.asarray(fit_result["point"], dtype=np.float64)
        elif all(k in fit_result for k in ("D", "Dstar", "f")):
            point = np.column_stack(
                [
                    np.asarray(fit_result["D"], dtype=np.float64).ravel(),
                    np.asarray(fit_result["Dstar"], dtype=np.float64).ravel(),
                    np.asarray(fit_result["f"], dtype=np.float64).ravel(),
                ]
            )
        else:
            raise ValueError("`fit_result` mapping must provide `'point'` or `('D', 'Dstar', 'f')`.")

        if "lower" in fit_result and "upper" in fit_result:
            lower = np.asarray(fit_result["lower"], dtype=np.float64)
            upper = np.asarray(fit_result["upper"], dtype=np.float64)
        elif "ci_width" in fit_result:
            hw = 0.5 * np.asarray(fit_result["ci_width"], dtype=np.float64)
            lower = point - hw
            upper = point + hw
        else:
            lower = point.copy()
            upper = point.copy()

        if "railed" in fit_result and fit_result["railed"] is not None:
            r_arr = np.asarray(fit_result["railed"], dtype=bool)
            dstar_railed = r_arr[:, 1] if r_arr.ndim == 2 else r_arr.ravel()
        elif "Dstar_railed" in fit_result:
            dstar_railed = np.asarray(fit_result["Dstar_railed"], dtype=bool).ravel()
        else:
            dstar_railed = classify_railing(
                point[:, 1], DEFAULT_PARAM_BOUNDS["Dstar"], DEFAULT_RAIL_TOL
            )["railed_mask"]
        extras = dict(fit_result.get("extras", {}))
    else:
        point = np.asarray(fit_result, dtype=np.float64)
        lower = point.copy()
        upper = point.copy()
        dstar_railed = classify_railing(
            point[:, 1] if point.ndim == 2 else point.ravel(),
            DEFAULT_PARAM_BOUNDS["Dstar"],
            DEFAULT_RAIL_TOL,
        )["railed_mask"]

    if point.shape != (n_masked, 3):
        raise ValueError(
            f"`fit_result` point estimates shape {point.shape} must be ({n_masked}, 3)."
        )
    if lower.shape != (n_masked, 3) or upper.shape != (n_masked, 3):
        raise ValueError("`lower` and `upper` interval arrays must have shape (N_masked, 3).")
    if dstar_railed.shape != (n_masked,):
        raise ValueError(f"`dstar_railed` mask shape {dstar_railed.shape} != ({n_masked},).")

    return point, lower, upper, dstar_railed, extras


def save_parameter_maps(
    output_dir: str | Path,
    fit_result: NormalizedEstimatorOutput | Mapping[str, Any] | np.ndarray,
    volume_meta: DWIVolumeData | Mapping[str, Any],
    prefix: str = "",
    *,
    outside_fill: float = 0.0,
    exchangeability_report: Mapping[str, Any] | None = None,
    backend: str = "auto",
) -> dict[str, Path]:
    """Unmask fitted 1D voxel arrays back to 3D ``(X, Y, Z)`` NIfTI maps with preserved affines.

    Writes the clinical viewer and RT-planning bundle:
    - Point maps: ``{prefix}D_map.nii.gz``, ``{prefix}Dstar_map.nii.gz``, ``{prefix}f_map.nii.gz``
    - Uncertainty & QA maps: ``{prefix}D_ci_width.nii.gz``, ``{prefix}Dstar_ci_width.nii.gz``,
      ``{prefix}f_ci_width.nii.gz``, ``{prefix}Dstar_railed_mask.nii.gz`` (``uint8``),
      and ``{prefix}exchangeability_report.json``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask, affine, spatial_shape, hdr_meta, dwi_obj = _extract_volume_meta(volume_meta)
    n_masked = int(np.sum(mask))
    point, lower, upper, dstar_railed, extras = _extract_fit_arrays(fit_result, n_masked)
    ci_width = np.maximum(upper - lower, 0.0)

    def _unmask_float(col: np.ndarray) -> np.ndarray:
        vol = np.full(spatial_shape, float(outside_fill), dtype=np.float64)
        vol[mask] = col.astype(np.float64)
        return vol

    railed_vol_u8 = np.zeros(spatial_shape, dtype=np.uint8)
    railed_vol_u8[mask] = dstar_railed.astype(np.uint8)

    paths: dict[str, Path] = {}
    map_specs: list[tuple[str, str, np.ndarray]] = [
        ("D_map", f"{prefix}D_map.nii.gz", _unmask_float(point[:, 0])),
        ("Dstar_map", f"{prefix}Dstar_map.nii.gz", _unmask_float(point[:, 1])),
        ("f_map", f"{prefix}f_map.nii.gz", _unmask_float(point[:, 2])),
        ("D_ci_width", f"{prefix}D_ci_width.nii.gz", _unmask_float(ci_width[:, 0])),
        ("Dstar_ci_width", f"{prefix}Dstar_ci_width.nii.gz", _unmask_float(ci_width[:, 1])),
        ("f_ci_width", f"{prefix}f_ci_width.nii.gz", _unmask_float(ci_width[:, 2])),
        ("Dstar_railed_mask", f"{prefix}Dstar_railed_mask.nii.gz", railed_vol_u8),
    ]
    for key, fname, vol_arr in map_specs:
        paths[key] = write_nifti(vol_arr, affine, out_dir / fname, backend=backend)

    # Assemble exchangeability & QA report JSON
    shift_info: dict[str, Any]
    if exchangeability_report is not None:
        shift_info = dict(exchangeability_report)
    elif "shift_monitor" in extras and isinstance(extras["shift_monitor"], Mapping):
        shift_info = dict(extras["shift_monitor"])
    elif dwi_obj is not None and dwi_obj.n_masked >= 16:
        mid = dwi_obj.n_masked // 2
        rep = check_domain_shift(
            dwi_obj.signals[:mid],
            dwi_obj.signals[mid:],
            bvalues=dwi_obj.bvalues,
            n_perm=80,
            seed=42,
        )
        shift_info = {
            "mode": "within_volume_split_half_check",
            "is_stale": bool(rep.is_stale),
            "mahalanobis_stat": float(rep.signal_mahalanobis_sq),
            "p_value": float(min(rep.signal_pvalue, rep.residual_pvalue)),
            "residual_rms_ratio": float(rep.residual_rms_ratio),
            "staleness_reasons": list(rep.staleness_reasons),
        }
    else:
        shift_info = {
            "mode": "metadata_only",
            "is_stale": False,
        }

    report_payload = _sanitize_json(
        {
            "spatial_shape": list(spatial_shape),
            "n_total_voxels": int(np.prod(spatial_shape)),
            "n_masked_voxels": n_masked,
            "outside_fill": None if np.isnan(outside_fill) else float(outside_fill),
            "affine": affine.tolist(),
            "voxel_sizes_mm": list(_voxel_sizes_from_affine(affine)),
            "dstar_railed_voxels": int(np.sum(dstar_railed)),
            "dstar_railed_fraction": float(np.mean(dstar_railed)),
            "mean_point_estimates": {
                "D": float(np.mean(point[:, 0])),
                "Dstar": float(np.mean(point[:, 1])),
                "f": float(np.mean(point[:, 2])),
            },
            "median_ci_widths": {
                "D": float(np.median(ci_width[:, 0])),
                "Dstar": float(np.median(ci_width[:, 1])),
                "f": float(np.median(ci_width[:, 2])),
            },
            "exchangeability": shift_info,
            "header_meta": hdr_meta,
        }
    )
    rep_path = out_dir / f"{prefix}exchangeability_report.json"
    rep_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["exchangeability_report"] = rep_path
    return paths


def export_dro_to_nifti(
    dro: DigitalReferenceObject,
    output_dir: str | Path,
    *,
    affine: np.ndarray | None = None,
    voxel_size_mm: Sequence[float] = (2.0, 2.0, 4.0),
    backend: str = "auto",
) -> dict[str, Path]:
    """Export a 2D or 3D :class:`DigitalReferenceObject` to NIfTI-1 + FSL ``.bval`` files.

    Writes ``dwi.nii.gz``, ``dwi.bval``, ``roi_mask.nii.gz``, ``gt_D_map.nii.gz``,
    ``gt_Dstar_map.nii.gz``, ``gt_f_map.nii.gz``, ``gt_S0_map.nii.gz``,
    ``region_labels.nii.gz``, and ``dro_metadata.json`` so external tools
    (3D Slicer, ITK-SNAP, FSL, or MATLAB) can evaluate synthetic IVIM benchmarks directly.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aff = _validate_affine(affine, voxel_size_mm=voxel_size_mm)

    if len(dro.spatial_shape) == 2:
        dwi_4d = dro.signals[:, :, None, :]
        params_4d = dro.params_map[:, :, None, :]
        s0_3d = dro.S0_map[:, :, None]
        reg_3d = dro.region_mask[:, :, None].astype(np.uint8)
    elif len(dro.spatial_shape) == 3:
        dwi_4d = dro.signals
        params_4d = dro.params_map
        s0_3d = dro.S0_map
        reg_3d = dro.region_mask.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported DRO spatial shape {dro.spatial_shape}.")

    roi_mask_u8 = (reg_3d > 0).astype(np.uint8)
    region_labels_map: dict[str, str] = {"0": "background", "1": "parenchyma"}
    for k, v in dro.lesion_names.items():
        region_labels_map[str(k)] = str(v)

    paths: dict[str, Path] = {
        "dwi": write_nifti(dwi_4d, aff, out_dir / "dwi.nii.gz", backend=backend),
        "bval": write_bvals(dro.bvalues, out_dir / "dwi.bval"),
        "roi_mask": write_nifti(roi_mask_u8, aff, out_dir / "roi_mask.nii.gz", backend=backend),
        "gt_D_map": write_nifti(params_4d[..., 0], aff, out_dir / "gt_D_map.nii.gz", backend=backend),
        "gt_Dstar_map": write_nifti(
            params_4d[..., 1], aff, out_dir / "gt_Dstar_map.nii.gz", backend=backend
        ),
        "gt_f_map": write_nifti(params_4d[..., 2], aff, out_dir / "gt_f_map.nii.gz", backend=backend),
        "gt_S0_map": write_nifti(s0_3d, aff, out_dir / "gt_S0_map.nii.gz", backend=backend),
        "region_labels": write_nifti(
            reg_3d, aff, out_dir / "region_labels.nii.gz", backend=backend
        ),
    }

    meta_path = out_dir / "dro_metadata.json"
    meta_payload = _sanitize_json(
        {
            "organ": dro.organ,
            "protocol": dro.protocol,
            "family": dro.family,
            "noise_type": dro.noise_type,
            "snr": None if dro.snr is None else float(dro.snr),
            "original_dro_shape": list(dro.spatial_shape),
            "nifti_spatial_shape": list(dwi_4d.shape[:3]),
            "n_bvalues": int(dro.bvalues.size),
            "bvalues": dro.bvalues.tolist(),
            "affine": aff.tolist(),
            "region_labels": region_labels_map,
            "seed": int(dro.seed),
        }
    )
    meta_path.write_text(json.dumps(meta_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["metadata"] = meta_path
    return paths


def fit_dwi_volume(
    dwi_path: str | Path,
    bval_path: str | Path | Sequence[float] | np.ndarray | None,
    output_dir: str | Path,
    *,
    mask_path: str | Path | np.ndarray | None = None,
    estimator: str = "conformal_ivim",
    alpha: float = 0.10,
    prefix: str = "",
    outside_fill: float = 0.0,
    estimator_kwargs: Mapping[str, Any] | None = None,
    backend: str = "auto",
) -> tuple[DWIVolumeData, NormalizedEstimatorOutput, dict[str, Path]]:
    """Load a clinical/DRO 4D volume, run vectorized fitting + intervals, and write 3D NIfTI maps."""
    vol = load_dwi_volume(dwi_path, bval_path, mask_path=mask_path, backend=backend)
    kw = dict(estimator_kwargs or {})
    est_out = run_estimator(estimator, vol.signals, vol.bvalues, alpha=alpha, **kw)
    written = save_parameter_maps(
        output_dir,
        est_out,
        vol,
        prefix=prefix,
        outside_fill=outside_fill,
        backend=backend,
    )
    return vol, est_out, written
