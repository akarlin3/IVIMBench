"""BIDS-Derivatives (qMRI) NIfTI-1, JSON sidecar, and SVG/HTML QC exporter.

Implements a zero-external-dependency pure-NumPy/gzip NIfTI-1 reader/writer
(fully interoperable with ``nibabel``) alongside :func:`export_bids_derivatives`
producing BIDS-qMRI compliant quantitative IVIM parameter maps:
- ``sub-<id>_param-D_dwimap.nii.gz``
- ``sub-<id>_param-Dstar_dwimap.nii.gz``
- ``sub-<id>_param-f_dwimap.nii.gz``
- ``sub-<id>_desc-railed_mask.nii.gz``
plus ``dataset_description.json``, per-map JSON sidecars (``BValues``,
``EstimationAlgorithm``, ``SoftwareVersion``), and an embedded SVG/HTML visual QC summary.
"""

from __future__ import annotations

import gzip
import json
import re
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .._version import __version__
from ..metrics.railing import DEFAULT_PARAM_BOUNDS, DEFAULT_RAIL_TOL, classify_railing
from ..protocols import resolve_bvalues

__all__ = [
    "BIDSExportResult",
    "write_nifti1",
    "read_nifti1",
    "export_bids_derivatives",
]

_SUBJECT_RE = re.compile(r"^[A-Za-z0-9]+$")


def _sanitize_subject_label(subject_id: str) -> str:
    raw = str(subject_id).strip()
    if raw.startswith("sub-"):
        raw = raw[4:]
    if not raw or not _SUBJECT_RE.match(raw):
        raise ValueError(
            f"BIDS `subject_id` must be non-empty alphanumeric (PHI-free, e.g. '01'); got {subject_id!r}."
        )
    return raw


def write_nifti1(
    data: np.ndarray,
    path: str | Path,
    *,
    affine: np.ndarray | None = None,
    dtype: str = "float32",
    descrip: str = "IVIMBench qMRI derivative",
) -> Path:
    """Write a 3D or 4D array to a standard NIfTI-1 ``.nii`` or ``.nii.gz`` file using pure NumPy."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.ndim not in (3, 4):
        raise ValueError(f"NIfTI-1 volume must be 2D, 3D, or 4D; got shape {arr.shape}.")

    np_dtype: np.dtype[Any]
    if dtype == "uint8":
        np_dtype = np.dtype("<u1")
        datatype_code = 2
        bitpix = 8
    else:
        np_dtype = np.dtype("<f4")
        datatype_code = 16
        bitpix = 32

    arr_cast = np.asarray(arr, dtype=np_dtype)
    aff = np.eye(4, dtype=np.float64) if affine is None else np.asarray(affine, dtype=np.float64)
    if aff.shape != (4, 4):
        raise ValueError(f"Expected 4x4 affine matrix, got {aff.shape}.")

    ndim = arr_cast.ndim
    dim = [ndim, *list(arr_cast.shape), *([1] * (7 - ndim))]
    zooms = [float(np.linalg.norm(aff[:3, k])) or 1.0 for k in range(3)]
    pixdim = [1.0, zooms[0], zooms[1], zooms[2], 1.0, 0.0, 0.0, 0.0]

    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)  # sizeof_hdr
    struct.pack_into("<8h", header, 40, *dim)
    struct.pack_into("<h", header, 70, datatype_code)
    struct.pack_into("<h", header, 72, bitpix)
    struct.pack_into("<8f", header, 76, *pixdim)
    struct.pack_into("<f", header, 108, 352.0)  # vox_offset (348 + 4-byte extender)
    struct.pack_into("<f", header, 112, 1.0)  # scl_slope
    struct.pack_into("<f", header, 116, 0.0)  # scl_inter
    struct.pack_into("<B", header, 123, 2)  # xyzt_units = millimeters

    desc_bytes = descrip.encode("ascii", errors="ignore")[:79]
    header[148 : 148 + len(desc_bytes)] = desc_bytes

    struct.pack_into("<h", header, 252, 1)  # qform_code = scanner_anat
    struct.pack_into("<h", header, 254, 1)  # sform_code = scanner_anat
    struct.pack_into("<3f", header, 268, float(aff[0, 3]), float(aff[1, 3]), float(aff[2, 3]))
    struct.pack_into("<4f", header, 280, *[float(v) for v in aff[0]])
    struct.pack_into("<4f", header, 296, *[float(v) for v in aff[1]])
    struct.pack_into("<4f", header, 312, *[float(v) for v in aff[2]])
    header[344:348] = b"n+1\x00"

    extender = b"\x00\x00\x00\x00"
    payload = arr_cast.tobytes(order="F")
    blob = bytes(header) + extender + payload

    if target.name.endswith(".nii.gz"):
        with gzip.GzipFile(filename="", mode="wb", fileobj=target.open("wb"), mtime=0) as gz:
            gz.write(blob)
    else:
        target.write_bytes(blob)
    return target


def read_nifti1(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a NIfTI-1 ``.nii`` or ``.nii.gz`` file into ``(array, affine)`` using pure NumPy."""
    src = Path(path)
    raw = gzip.decompress(src.read_bytes()) if src.name.endswith(".nii.gz") else src.read_bytes()
    if len(raw) < 352:
        raise ValueError(f"Invalid NIfTI-1 file {src}: shorter than 352 bytes.")

    sizeof_hdr = struct.unpack_from("<i", raw, 0)[0]
    endian = "<" if sizeof_hdr == 348 else ">"
    dim = struct.unpack_from(f"{endian}8h", raw, 40)
    ndim = max(int(dim[0]), 1)
    shape = tuple(int(dim[i]) for i in range(1, ndim + 1))
    datatype_code = struct.unpack_from(f"{endian}h", raw, 70)[0]
    vox_offset = int(struct.unpack_from(f"{endian}f", raw, 108)[0]) or 352
    scl_slope = struct.unpack_from(f"{endian}f", raw, 112)[0]
    scl_inter = struct.unpack_from(f"{endian}f", raw, 116)[0]

    dtype_map = {
        2: f"{endian}u1",
        4: f"{endian}i2",
        8: f"{endian}i4",
        16: f"{endian}f4",
        64: f"{endian}f8",
        512: f"{endian}u2",
    }
    if datatype_code not in dtype_map:
        raise ValueError(f"Unsupported NIfTI-1 datatype code {datatype_code}.")

    dt = np.dtype(dtype_map[datatype_code])
    count = int(np.prod(shape))
    arr = np.frombuffer(raw, dtype=dt, count=count, offset=vox_offset).reshape(shape, order="F").copy()
    if scl_slope not in (0.0, 1.0) or scl_inter != 0.0:
        slope = 1.0 if scl_slope == 0.0 else float(scl_slope)
        arr = arr.astype(np.float64) * slope + float(scl_inter)

    sform_code = struct.unpack_from(f"{endian}h", raw, 254)[0]
    aff = np.eye(4, dtype=np.float64)
    if sform_code > 0:
        aff[0] = struct.unpack_from(f"{endian}4f", raw, 280)
        aff[1] = struct.unpack_from(f"{endian}4f", raw, 296)
        aff[2] = struct.unpack_from(f"{endian}4f", raw, 312)
    else:
        pixdim = struct.unpack_from(f"{endian}8f", raw, 76)
        aff[0, 0] = pixdim[1] or 1.0
        aff[1, 1] = pixdim[2] or 1.0
        aff[2, 2] = pixdim[3] or 1.0
    return arr, aff


@dataclass
class BIDSExportResult(Mapping[str, Any]):
    """Dictionary-compatible manifest of exported BIDS-Derivatives qMRI files."""

    output_dir: Path
    subject_id: str
    subject_dir: Path
    dataset_description: Path
    nifti_paths: dict[str, Path]
    sidecar_paths: dict[str, Path]
    svg_summary: Path
    html_summary: Path
    summary_stats: dict[str, Any]

    def _as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "subject_id": self.subject_id,
            "subject_dir": self.subject_dir,
            "dataset_description": self.dataset_description,
            "nifti_paths": self.nifti_paths,
            "sidecar_paths": self.sidecar_paths,
            "svg_summary": self.svg_summary,
            "html_summary": self.html_summary,
            "summary_stats": self.summary_stats,
            **self.nifti_paths,
        }

    def __getitem__(self, key: str) -> Any:
        mapping = self._as_dict()
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._as_dict())

    def __len__(self) -> int:
        return len(self._as_dict())


def _unflatten_param_vector(
    vec: np.ndarray,
    spatial_shape: tuple[int, int, int],
    fg_mask: np.ndarray | None,
) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    if v.ndim == 2:
        return v[:, :, np.newaxis]
    if v.ndim == 3:
        return v
    v_flat = v.ravel()
    if fg_mask is not None and int(np.sum(fg_mask)) == v_flat.size:
        out = np.zeros(spatial_shape, dtype=np.float64)
        out[fg_mask] = v_flat
        return out
    if int(np.prod(spatial_shape)) == v_flat.size:
        return v_flat.reshape(spatial_shape)
    raise ValueError(
        f"Cannot reshape parameter vector of size {v_flat.size} onto spatial shape {spatial_shape}."
    )


def _extract_spatial_maps(
    fit_result: Any,
    volume_meta: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return 3D ``(D, Dstar, f, railed_mask, fg_mask, algorithm_name)``."""
    meta = dict(volume_meta or {})
    algo = str(meta.get("estimation_algorithm", meta.get("algorithm", "IVIMBench-Estimator")))

    fg_mask_meta: np.ndarray | None = None
    if "mask" in meta and meta["mask"] is not None:
        m_arr = np.asarray(meta["mask"]) > 0
        fg_mask_meta = m_arr[:, :, np.newaxis] if m_arr.ndim == 2 else m_arr

    shape_meta: tuple[int, int, int] | None = None
    if "shape" in meta and meta["shape"] is not None:
        raw_s = tuple(int(x) for x in meta["shape"])
        if len(raw_s) == 2:
            shape_meta = (raw_s[0], raw_s[1], 1)
        elif len(raw_s) == 3:
            shape_meta = raw_s
    elif fg_mask_meta is not None:
        shape_meta = (fg_mask_meta.shape[0], fg_mask_meta.shape[1], fg_mask_meta.shape[2])

    railed_raw: np.ndarray | None = None

    if hasattr(fit_result, "D_map") and hasattr(fit_result, "Dstar_map") and hasattr(fit_result, "f_map"):
        d_raw = np.asarray(fit_result.D_map, dtype=np.float64)
        ds_raw = np.asarray(fit_result.Dstar_map, dtype=np.float64)
        f_raw = np.asarray(fit_result.f_map, dtype=np.float64)
        d_3d = d_raw[:, :, np.newaxis] if d_raw.ndim == 2 else d_raw
        ds_3d = ds_raw[:, :, np.newaxis] if ds_raw.ndim == 2 else ds_raw
        f_3d = f_raw[:, :, np.newaxis] if f_raw.ndim == 2 else f_raw
        if hasattr(fit_result, "foreground_mask"):
            fg_raw = np.asarray(fit_result.foreground_mask) > 0
            fg_mask_meta = fg_raw[:, :, np.newaxis] if fg_raw.ndim == 2 else fg_raw
        algo = str(meta.get("estimation_algorithm", "DigitalReferenceObject"))
    elif hasattr(fit_result, "point"):
        pt = np.asarray(fit_result.point, dtype=np.float64)
        algo = str(getattr(fit_result, "name", algo))
        if pt.ndim == 2 and pt.shape[1] == 3:
            sp_shape = shape_meta or (pt.shape[0], 1, 1)
            d_3d = _unflatten_param_vector(pt[:, 0], sp_shape, fg_mask_meta)
            ds_3d = _unflatten_param_vector(pt[:, 1], sp_shape, fg_mask_meta)
            f_3d = _unflatten_param_vector(pt[:, 2], sp_shape, fg_mask_meta)
        elif pt.ndim in (3, 4) and pt.shape[-1] == 3:
            p3 = pt[:, :, np.newaxis, :] if pt.ndim == 3 else pt
            d_3d, ds_3d, f_3d = p3[..., 0], p3[..., 1], p3[..., 2]
        else:
            raise ValueError(f"Unsupported `fit_result.point` shape {pt.shape}.")
    elif isinstance(fit_result, Mapping):
        if "algorithm" in fit_result:
            algo = str(fit_result["algorithm"])
        d_in = np.asarray(fit_result["D"], dtype=np.float64)
        ds_in = np.asarray(
            fit_result["Dstar"] if "Dstar" in fit_result else fit_result["D_star"],
            dtype=np.float64,
        )
        f_in = np.asarray(fit_result["f"], dtype=np.float64)
        if d_in.ndim == 1:
            sp_shape = shape_meta or (d_in.size, 1, 1)
            d_3d = _unflatten_param_vector(d_in, sp_shape, fg_mask_meta)
            ds_3d = _unflatten_param_vector(ds_in, sp_shape, fg_mask_meta)
            f_3d = _unflatten_param_vector(f_in, sp_shape, fg_mask_meta)
        else:
            d_3d = d_in[:, :, np.newaxis] if d_in.ndim == 2 else d_in
            ds_3d = ds_in[:, :, np.newaxis] if ds_in.ndim == 2 else ds_in
            f_3d = f_in[:, :, np.newaxis] if f_in.ndim == 2 else f_in
        if "railed_mask" in fit_result and fit_result["railed_mask"] is not None:
            rm = np.asarray(fit_result["railed_mask"], dtype=np.float64)
            if rm.ndim == 1:
                railed_raw = _unflatten_param_vector(rm, d_3d.shape, fg_mask_meta)
            else:
                railed_raw = rm[:, :, np.newaxis] if rm.ndim == 2 else rm
    else:
        arr = np.asarray(fit_result, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 3:
            sp_shape = shape_meta or (arr.shape[0], 1, 1)
            d_3d = _unflatten_param_vector(arr[:, 0], sp_shape, fg_mask_meta)
            ds_3d = _unflatten_param_vector(arr[:, 1], sp_shape, fg_mask_meta)
            f_3d = _unflatten_param_vector(arr[:, 2], sp_shape, fg_mask_meta)
        elif arr.ndim in (3, 4) and arr.shape[-1] == 3:
            p3 = arr[:, :, np.newaxis, :] if arr.ndim == 3 else arr
            d_3d, ds_3d, f_3d = p3[..., 0], p3[..., 1], p3[..., 2]
        else:
            raise ValueError(f"Unsupported `fit_result` array shape {arr.shape}.")

    fg_3d = (
        fg_mask_meta
        if fg_mask_meta is not None
        else (np.isfinite(d_3d) & ((np.abs(d_3d) + np.abs(ds_3d) + np.abs(f_3d)) > 0.0))
    )

    if railed_raw is not None:
        rail_3d = ((railed_raw > 0.5) & fg_3d).astype(np.uint8)
    else:
        ds_bounds = tuple(meta.get("dstar_bounds", DEFAULT_PARAM_BOUNDS["Dstar"]))
        rail_tol = float(meta.get("rail_tol", DEFAULT_RAIL_TOL))
        railed_flat = classify_railing(
            ds_3d.ravel(),
            bounds=(float(ds_bounds[0]), float(ds_bounds[1])),
            rel_tol=rail_tol,
        )["railed_mask"]
        rail_3d = (railed_flat.reshape(ds_3d.shape) & fg_3d).astype(np.uint8)

    d_clean = np.where(fg_3d, np.nan_to_num(d_3d, nan=0.0), 0.0)
    ds_clean = np.where(fg_3d, np.nan_to_num(ds_3d, nan=0.0), 0.0)
    f_clean = np.where(fg_3d, np.nan_to_num(f_3d, nan=0.0), 0.0)
    return d_clean, ds_clean, f_clean, rail_3d, fg_3d, algo


def _value_to_hex(norm_val: float, palette: str) -> str:
    t = float(np.clip(norm_val, 0.0, 1.0))
    if palette == "binary":
        return "#d62728" if t >= 0.5 else "#1f2937"
    if palette == "viridis":
        r = int(68 + 185 * (t**1.4))
        g = int(1 + 215 * (t**0.85))
        b = int(84 + 90 * np.sin(np.pi * t) - 45 * t)
    elif palette == "magma":
        r = int(15 + 235 * (t**0.85))
        g = int(10 + 175 * (t**1.55))
        b = int(45 + 170 * np.sin(0.85 * np.pi * t))
    else:  # plasma
        r = int(25 + 225 * (t**0.75))
        g = int(8 + 190 * (t**1.7))
        b = int(135 + 85 * (1.0 - t))
    return f"#{np.clip(r, 0, 255):02x}{np.clip(g, 0, 255):02x}{np.clip(b, 0, 255):02x}"


def _render_qc_svg(
    sub_id: str,
    d_3d: np.ndarray,
    ds_3d: np.ndarray,
    f_3d: np.ndarray,
    rail_3d: np.ndarray,
    fg_3d: np.ndarray,
    stats: Mapping[str, Any],
) -> str:
    z_mid = d_3d.shape[2] // 2
    slices = [
        ("D (×10⁻³ mm²/s)", d_3d[:, :, z_mid] * 1e3, "viridis"),
        ("D* (×10⁻³ mm²/s)", ds_3d[:, :, z_mid] * 1e3, "magma"),
        ("f (perfusion)", f_3d[:, :, z_mid], "plasma"),
        (f"Railed Mask ({stats['railed_pct']:.1f}%)", rail_3d[:, :, z_mid].astype(float), "binary"),
    ]
    fg_2d = fg_3d[:, :, z_mid]

    panel_w = 175
    pad = 18
    total_w = 4 * panel_w + 5 * pad
    total_h = 250

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {total_h}" width="{total_w}" height="{total_h}">',
        '<rect width="100%" height="100%" fill="#0f172a" rx="10"/>',
        f'<text x="{pad}" y="26" fill="#f8fafc" font-family="system-ui, sans-serif" font-size="14" font-weight="600">'
        f'BIDS-qMRI Derivative QC — sub-{sub_id} (axial slice z={z_mid}, IVIMBench v{__version__})</text>',
    ]

    for idx, (title, slc, pal) in enumerate(slices):
        x0 = pad + idx * (panel_w + pad)
        y0 = 46
        box = 156
        parts.append(
            f'<rect x="{x0}" y="{y0}" width="{box}" height="{box}" fill="#020617" stroke="#334155" stroke-width="1"/>'
        )
        # Downsample large slices to <= 32x32 grid so SVG stays compact and crisp
        h, w = slc.shape
        step_r = max(1, int(np.ceil(h / 32)))
        step_c = max(1, int(np.ceil(w / 32)))
        sub_s = slc[::step_r, ::step_c]
        sub_fg = fg_2d[::step_r, ::step_c]
        nr, nc = sub_s.shape
        cw = box / max(nc, 1)
        ch = box / max(nr, 1)

        if pal == "binary":
            vmin, vmax = 0.0, 1.0
        elif np.any(sub_fg):
            vals = sub_s[sub_fg]
            vmin = float(np.percentile(vals, 5))
            vmax = float(np.percentile(vals, 95))
            if vmax <= vmin + 1e-12:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0

        for r in range(nr):
            for c in range(nc):
                if not sub_fg[r, c]:
                    continue
                nv = (float(sub_s[r, c]) - vmin) / max(vmax - vmin, 1e-12)
                color = _value_to_hex(nv, pal)
                rx = x0 + c * cw
                ry = y0 + (nr - 1 - r) * ch
                parts.append(
                    f'<rect x="{rx:.2f}" y="{ry:.2f}" width="{cw + 0.35:.2f}" height="{ch + 0.35:.2f}" fill="{color}"/>'
                )

        parts.append(
            f'<text x="{x0 + box / 2:.1f}" y="{y0 + box + 20}" fill="#cbd5e1" text-anchor="middle" '
            f'font-family="system-ui, sans-serif" font-size="11.5">{title}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def export_bids_derivatives(
    output_dir: str | Path,
    subject_id: str,
    fit_result: Any,
    volume_meta: Mapping[str, Any] | None = None,
    protocol_name: str = "osipi",
) -> BIDSExportResult:
    """Export BIDS-Derivatives (``qMRI``) quantitative IVIM parameter NIfTIs, JSON sidecars, and QC report.

    Writes under ``<output_dir>``:
    - ``dataset_description.json``
    - ``sub-<id>/dwi/sub-<id>_param-D_dwimap.nii.gz`` (+ ``.json``)
    - ``sub-<id>/dwi/sub-<id>_param-Dstar_dwimap.nii.gz`` (+ ``.json``)
    - ``sub-<id>/dwi/sub-<id>_param-f_dwimap.nii.gz`` (+ ``.json``)
    - ``sub-<id>/dwi/sub-<id>_desc-railed_mask.nii.gz`` (+ ``.json``)
    - ``sub-<id>/dwi/sub-<id>_desc-summary_dwimap.svg`` & ``sub-<id>_desc-summary_report.html``
    """
    sub_label = _sanitize_subject_label(subject_id)
    meta = dict(volume_meta or {})
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    ses_raw = meta.get("session_id")
    if ses_raw:
        ses_clean = _sanitize_subject_label(str(ses_raw).removeprefix("ses-"))
        dwi_dir = out_root / f"sub-{sub_label}" / f"ses-{ses_clean}" / "dwi"
        entity_prefix = f"sub-{sub_label}_ses-{ses_clean}"
    else:
        dwi_dir = out_root / f"sub-{sub_label}" / "dwi"
        entity_prefix = f"sub-{sub_label}"
    dwi_dir.mkdir(parents=True, exist_ok=True)

    if "bvalues" in meta and meta["bvalues"] is not None:
        b_arr, proto_resolved = resolve_bvalues(meta["bvalues"])
        proto_key = str(protocol_name or proto_resolved)
    else:
        try:
            b_arr, proto_key = resolve_bvalues(protocol_name)
        except Exception:
            b_arr = np.array([0.0, 50.0, 200.0, 800.0], dtype=np.float64)
            proto_key = str(protocol_name)

    affine = (
        np.asarray(meta["affine"], dtype=np.float64)
        if "affine" in meta and meta["affine"] is not None
        else np.eye(4, dtype=np.float64)
    )
    d_3d, ds_3d, f_3d, rail_3d, fg_3d, algo_name = _extract_spatial_maps(fit_result, meta)

    n_fg = int(np.sum(fg_3d))
    railed_pct = 100.0 * float(np.mean(rail_3d[fg_3d] > 0)) if n_fg > 0 else 0.0
    summary_stats = {
        "subject_id": sub_label,
        "protocol_name": proto_key,
        "estimation_algorithm": algo_name,
        "software_version": __version__,
        "spatial_shape": list(d_3d.shape),
        "foreground_voxels": n_fg,
        "mean_D_mm2_s": float(np.mean(d_3d[fg_3d])) if n_fg > 0 else 0.0,
        "mean_Dstar_mm2_s": float(np.mean(ds_3d[fg_3d])) if n_fg > 0 else 0.0,
        "mean_f": float(np.mean(f_3d[fg_3d])) if n_fg > 0 else 0.0,
        "railed_voxels": int(np.sum(rail_3d[fg_3d] > 0)) if n_fg > 0 else 0,
        "railed_pct": railed_pct,
    }

    # 1. Top-level BIDS dataset_description.json
    ds_desc_path = out_root / "dataset_description.json"
    ds_desc = {
        "Name": "IVIMBench Quantitative Diffusion-Perfusion MRI Derivatives",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "License": "MIT",
        "GeneratedBy": [
            {
                "Name": "IVIMBench",
                "Version": __version__,
                "Description": "Quantitative IVIM uncertainty, acquisition-design, and BIDS-qMRI pipeline",
                "CodeURL": "https://github.com/akarlin3/IVIMBench",
            }
        ],
    }
    ds_desc_path.write_text(json.dumps(ds_desc, indent=2) + "\n", encoding="utf-8")

    # 2. Write quantitative parameter NIfTIs + JSON sidecars
    map_specs: list[tuple[str, str, np.ndarray, str, str, str]] = [
        (
            "D_map",
            f"{entity_prefix}_param-D_dwimap",
            d_3d,
            "float32",
            "mm^2/s",
            "Slow parenchymal tissue diffusion coefficient D",
        ),
        (
            "Dstar_map",
            f"{entity_prefix}_param-Dstar_dwimap",
            ds_3d,
            "float32",
            "mm^2/s",
            "Microvascular pseudo-diffusion coefficient D*",
        ),
        (
            "f_map",
            f"{entity_prefix}_param-f_dwimap",
            f_3d,
            "float32",
            "1",
            "Intravoxel microvascular perfusion fraction f",
        ),
        (
            "railed_mask",
            f"{entity_prefix}_desc-railed_mask",
            rail_3d,
            "uint8",
            "binary",
            "Binary D* optimizer boundary-railing indicator mask (1 = railed, 0 = interior)",
        ),
    ]

    nifti_paths: dict[str, Path] = {}
    sidecar_paths: dict[str, Path] = {}
    based_on = list(meta.get("based_on", [f"bids::sub-{sub_label}/dwi/{entity_prefix}_dwi.nii.gz"]))

    for key_name, stem, vol, dt_name, units_str, desc_text in map_specs:
        nii_path = write_nifti1(vol, dwi_dir / f"{stem}.nii.gz", affine=affine, dtype=dt_name, descrip=desc_text)
        json_path = dwi_dir / f"{stem}.json"
        sidecar = {
            "Description": desc_text,
            "Units": units_str,
            "BValues": [float(b) for b in b_arr],
            "ProtocolName": proto_key,
            "EstimationAlgorithm": algo_name,
            "SoftwareName": "IVIMBench",
            "SoftwareVersion": __version__,
            "BasedOn": based_on,
            "SpatialShape": list(vol.shape),
            "ForegroundVoxels": n_fg,
            "DstarRailedFraction": round(railed_pct / 100.0, 6),
        }
        json_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        nifti_paths[key_name] = nii_path
        sidecar_paths[key_name] = json_path

    # 3. Clean SVG & standalone HTML visual summary
    svg_markup = _render_qc_svg(sub_label, d_3d, ds_3d, f_3d, rail_3d, fg_3d, summary_stats)
    svg_path = dwi_dir / f"{entity_prefix}_desc-summary_dwimap.svg"
    svg_path.write_text(svg_markup + "\n", encoding="utf-8")

    html_path = dwi_dir / f"{entity_prefix}_desc-summary_report.html"
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>BIDS-qMRI Derivative Summary — sub-{sub_label}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #f8fafc; color: #0f172a; margin: 2rem; }}
    .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.5rem; max-width: 840px; box-shadow: 0 2px 8px rgba(15,23,42,0.05); }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; font-size: 0.92rem; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 0.55rem 0.75rem; text-align: left; }}
    th {{ background: #f1f5f9; font-weight: 600; }}
    .badge {{ display: inline-block; padding: 0.2rem 0.55rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; background: #dcfce7; color: #166534; }}
  </style>
</head>
<body>
  <div class="card">
    <h2>BIDS-Derivatives qMRI Summary: <code>sub-{sub_label}</code> <span class="badge">PHI-Free · v{__version__}</span></h2>
    <p>Protocol: <strong>{proto_key}</strong> &nbsp;|&nbsp; Estimator: <strong>{algo_name}</strong> &nbsp;|&nbsp; Foreground Voxels: <strong>{n_fg}</strong></p>
    {svg_markup}
    <table>
      <thead><tr><th>Derivative Metric</th><th>Observed Value</th><th>Units</th></tr></thead>
      <tbody>
        <tr><td>Mean Tissue Diffusivity (D)</td><td>{summary_stats['mean_D_mm2_s']:.4e}</td><td>mm²/s</td></tr>
        <tr><td>Mean Pseudo-Diffusion (D*)</td><td>{summary_stats['mean_Dstar_mm2_s']:.4e}</td><td>mm²/s</td></tr>
        <tr><td>Mean Perfusion Fraction (f)</td><td>{summary_stats['mean_f']:.4f}</td><td>1</td></tr>
        <tr><td>D* Boundary-Railed Voxels</td><td>{summary_stats['railed_voxels']} ({railed_pct:.2f}%)</td><td>binary mask</td></tr>
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")

    return BIDSExportResult(
        output_dir=out_root,
        subject_id=sub_label,
        subject_dir=dwi_dir,
        dataset_description=ds_desc_path,
        nifti_paths=nifti_paths,
        sidecar_paths=sidecar_paths,
        svg_summary=svg_path,
        html_summary=html_path,
        summary_stats=summary_stats,
    )
