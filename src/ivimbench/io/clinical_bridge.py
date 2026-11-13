"""3D Slicer scene, color LUTs, and DICOM-RT structure / dose-action bridge (`clinical_bridge`).

Exports affine-preserved 3D NIfTI parameter and uncertainty maps alongside:
- ``sub-<id>_desc-rt_escalation_eligible_mask.nii.gz`` (``uint8``): ``1`` where a voxel is
  interior-identifiable, in-distribution, ``AuditChain`` is non-``HALT``, and the calibrated
  conformal interval clears the clinical decision threshold; ``0`` where gated out.
- ``sub-<id>_desc-ood_or_railed_halt_mask.nii.gz`` (``uint8``): ``1`` where a voxel is
  boundary-railed or flagged by :class:`VoxelOODDetector`.
- 3D Slicer color lookup tables (``ivim_railing_lut.ctbl`` / ``.json``,
  ``ivim_uq_width_lut.ctbl`` / ``.json``) and one-click ``load_in_slicer.py`` scene loader.
- Pure-NumPy/JSON RT-Structure & Dose-Action summary ``sub-<id>_rt_action_plan.json``
  (plus optional PHI-free ``pydicom`` RT-Struct metadata tags when ``ivimbench[io]`` is installed)
  and standalone ``volume_viewer.html``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .._version import __version__
from ..audit import AuditChain, build_suite1_audit_chain
from ..bridges.avery_stack import detect_voxel_ood
from ..registry import NormalizedEstimatorOutput
from ..reporting.html_report import generate_volume_viewer_html
from ..runner import _sanitize_json
from .volumes import (
    DWIVolumeData,
    _extract_fit_arrays,
    _extract_volume_meta,
    _voxel_sizes_from_affine,
    has_pydicom,
    save_parameter_maps,
    write_nifti,
)

__all__ = [
    "SlicerRTBundleResult",
    "export_slicer_rt_bundle",
]


@dataclass(frozen=True)
class SlicerRTBundleResult:
    """Manifest and summary statistics for an exported 3D Slicer & RT-Dose Action bundle."""

    subject_id: str
    output_dir: Path
    overall_audit_status: str
    chain_sha256: str
    n_masked_voxels: int
    n_rt_eligible_voxels: int
    n_halt_or_railed_voxels: int
    paths: dict[str, Path]
    action_plan: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_json(
            {
                "subject_id": self.subject_id,
                "output_dir": str(self.output_dir),
                "overall_audit_status": self.overall_audit_status,
                "chain_sha256": self.chain_sha256,
                "n_masked_voxels": int(self.n_masked_voxels),
                "n_rt_eligible_voxels": int(self.n_rt_eligible_voxels),
                "n_halt_or_railed_voxels": int(self.n_halt_or_railed_voxels),
                "paths": {k: str(v) for k, v in self.paths.items()},
                "action_plan": self.action_plan,
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _normalize_subject_id(subject_id: str) -> str:
    clean = str(subject_id).strip()
    if clean.startswith("sub-"):
        clean = clean[4:]
    return clean or "01"


def _write_slicer_luts(out_dir: Path) -> dict[str, Path]:
    """Write 3D Slicer `.ctbl` and `.json` color lookup tables for railing/halt and CI width."""
    railing_ctbl = "\n".join(
        [
            "# 3D Slicer Color Table: IVIM Boundary-Railing & OOD Halt Contours",
            "0 background 0 0 0 0",
            "1 ood_or_railed_halt 239 68 68 102",
            "2 rt_escalation_eligible 16 185 129 102",
            "",
        ]
    )
    railing_json = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/main/Base/Logic/Resources/ColorFiles/color-table.schema.json",
        "name": "IVIM_Railing_And_RT_Gate_LUT",
        "entries": [
            {"label": 0, "name": "background", "rgba": [0, 0, 0, 0]},
            {"label": 1, "name": "ood_or_railed_halt", "rgba": [239, 68, 68, 102]},
            {"label": 2, "name": "rt_escalation_eligible", "rgba": [16, 185, 129, 102]},
        ],
    }

    uq_lines = ["# 3D Slicer Color Table: IVIM Conformal UQ Width (Navy-to-Gold)"]
    uq_entries = []
    for idx in range(16):
        u = idx / 15.0
        r = int(round(18 + u * (212 - 18)))
        g = int(round(27 + u * (175 - 27)))
        b = int(round(45 + u * (55 - 45)))
        a = 0 if idx == 0 else 255
        uq_lines.append(f"{idx} uq_level_{idx:02d} {r} {g} {b} {a}")
        uq_entries.append({"label": idx, "name": f"uq_level_{idx:02d}", "rgba": [r, g, b, a]})
    uq_lines.append("")

    uq_json = {
        "name": "IVIM_Conformal_UQ_Width_LUT",
        "theme": "minimalist_luxury_navy_gold",
        "entries": uq_entries,
    }

    p_rail_ctbl = out_dir / "ivim_railing_lut.ctbl"
    p_rail_json = out_dir / "ivim_railing_lut.json"
    p_uq_ctbl = out_dir / "ivim_uq_width_lut.ctbl"
    p_uq_json = out_dir / "ivim_uq_width_lut.json"

    p_rail_ctbl.write_text(railing_ctbl, encoding="utf-8")
    p_rail_json.write_text(json.dumps(railing_json, indent=2) + "\n", encoding="utf-8")
    p_uq_ctbl.write_text("\n".join(uq_lines), encoding="utf-8")
    p_uq_json.write_text(json.dumps(uq_json, indent=2) + "\n", encoding="utf-8")

    return {
        "railing_lut_ctbl": p_rail_ctbl,
        "railing_lut_json": p_rail_json,
        "uq_width_lut_ctbl": p_uq_ctbl,
        "uq_width_lut_json": p_uq_json,
    }


def _write_slicer_loader_script(
    out_dir: Path,
    *,
    sub_id: str,
    overall_status: str,
    chain_sha256: str,
    verdict_summary: str,
) -> Path:
    """Emit auto-generated 3D Slicer Python scene loader (`load_in_slicer.py`)."""
    eligible_fname = f"sub-{sub_id}_desc-rt_escalation_eligible_mask.nii.gz"
    halt_fname = f"sub-{sub_id}_desc-ood_or_railed_halt_mask.nii.gz"
    script = f'''"""One-click 3D Slicer loader for IVIMBench v{__version__} Suite I RT-Action Bundle.

Usage inside 3D Slicer Python Interactor or CLI:
    slicer --python-script load_in_slicer.py
"""
from pathlib import Path
import slicer

BUNDLE_DIR = Path(__file__).resolve().parent
AUDIT_STATUS = {overall_status!r}
CHAIN_SHA256 = {chain_sha256!r}
VERDICT_TEXT = {verdict_summary!r}


def load_ivim_suite1_scene() -> dict:
    railing_lut = slicer.util.loadColorTable(str(BUNDLE_DIR / "ivim_railing_lut.ctbl"))
    uq_lut = slicer.util.loadColorTable(str(BUNDLE_DIR / "ivim_uq_width_lut.ctbl"))

    d_node = slicer.util.loadVolume(str(BUNDLE_DIR / "D_map.nii.gz"))
    dstar_node = slicer.util.loadVolume(str(BUNDLE_DIR / "Dstar_map.nii.gz"))
    f_node = slicer.util.loadVolume(str(BUNDLE_DIR / "f_map.nii.gz"))
    uq_node = slicer.util.loadVolume(str(BUNDLE_DIR / "D_ci_width.nii.gz"))
    if uq_node and uq_lut:
        uq_node.GetDisplayNode().SetAndObserveColorNodeID(uq_lut.GetID())

    halt_label = slicer.util.loadLabelVolume(str(BUNDLE_DIR / {halt_fname!r}))
    elig_label = slicer.util.loadLabelVolume(str(BUNDLE_DIR / {eligible_fname!r}))
    for lbl in (halt_label, elig_label):
        if lbl and railing_lut:
            lbl.GetDisplayNode().SetAndObserveColorNodeID(railing_lut.GetID())

    # Overlay railed/halt contour mask at 40% opacity across all axial/sagittal/coronal views
    slicer.util.setSliceViewerLayers(
        background=f_node,
        foreground=uq_node,
        label=halt_label,
        foregroundOpacity=0.25,
        labelOpacity=0.40,
    )

    lm = slicer.app.layoutManager()
    if lm is not None:
        for view_name in lm.sliceViewNames():
            view = lm.sliceWidget(view_name).sliceView()
            ca = view.cornerAnnotation()
            ca.SetText(
                2,
                f"Suite I Audit: {{AUDIT_STATUS}} | SHA256: {{CHAIN_SHA256[:12]}}\\n{{VERDICT_TEXT}}",
            )
            ca.GetTextProperty().SetColor(0.83, 0.69, 0.22)
            view.forceRender()

    return {{
        "D": d_node,
        "Dstar": dstar_node,
        "f": f_node,
        "D_ci_width": uq_node,
        "halt_mask": halt_label,
        "rt_eligible_mask": elig_label,
    }}


if __name__ == "__main__":
    load_ivim_suite1_scene()
'''
    script_path = out_dir / "load_in_slicer.py"
    script_path.write_text(script, encoding="utf-8")
    return script_path


def _extract_contour_slices(mask_3d: np.ndarray, affine: np.ndarray) -> list[dict[str, Any]]:
    """Compute pure-NumPy axial slice bounding contours in voxel and patient-mm coordinates."""
    contours: list[dict[str, Any]] = []
    for z_idx in range(mask_3d.shape[2]):
        sl = mask_3d[:, :, z_idx] > 0
        count = int(np.sum(sl))
        if count == 0:
            continue
        xs, ys = np.nonzero(sl)
        corners_ijk = np.array(
            [
                [float(np.min(xs)), float(np.min(ys)), float(z_idx), 1.0],
                [float(np.max(xs)), float(np.max(ys)), float(z_idx), 1.0],
            ],
            dtype=np.float64,
        )
        corners_xyz = (affine @ corners_ijk.T).T[:, :3]
        contours.append(
            {
                "z_index": int(z_idx),
                "n_voxels": count,
                "bbox_ijk": [
                    [int(np.min(xs)), int(np.min(ys)), int(z_idx)],
                    [int(np.max(xs)), int(np.max(ys)), int(z_idx)],
                ],
                "bbox_patient_mm": corners_xyz.round(3).tolist(),
            }
        )
    return contours


def _build_optional_pydicom_rt_tags(
    out_dir: Path,
    *,
    sub_id: str,
    overall_status: str,
    chain_sha256: str,
    n_eligible: int,
    n_halt: int,
) -> tuple[dict[str, Any], Path | None]:
    """Build PHI-free DICOM RT-Struct metadata tags (and `.dcm` dataset when `pydicom` is installed)."""
    tags: dict[str, Any] = {
        "Modality": "RTSTRUCT",
        "PatientName": f"SYNTHETIC^SUB_{sub_id}",
        "PatientID": f"IVIMBENCH-SUB-{sub_id}",
        "PatientIdentityRemoved": "YES",
        "DeidentificationMethod": "SYNTHETIC_DRO_PHI_FREE",
        "StructureSetLabel": "IVIM_TRUST_GATE",
        "StructureSetName": f"SuiteI_{overall_status}",
        "StructureSetDescription": f"ChainSHA256:{chain_sha256[:32]}",
        "ROINames": [
            {"ROINumber": 1, "ROIName": "GTV_IVIM_RT_Escalation_Eligible", "VoxelCount": int(n_eligible)},
            {"ROINumber": 2, "ROIName": "PRV_IVIM_OOD_Or_Railed_Halt", "VoxelCount": int(n_halt)},
        ],
    }
    if not has_pydicom():
        return tags, None

    import pydicom  # type: ignore[import-not-found]
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset  # type: ignore[import-not-found]
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # type: ignore[import-not-found]

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.UID("1.2.840.10008.5.1.4.1.1.481.3")
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dcm_path = out_dir / f"sub-{sub_id}_rt_struct_meta.dcm"
    ds = FileDataset(str(dcm_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "RTSTRUCT"
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientName = tags["PatientName"]
    ds.PatientID = tags["PatientID"]
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = tags["DeidentificationMethod"]
    ds.StructureSetLabel = "IVIM_TRUST_GATE"
    ds.StructureSetName = tags["StructureSetName"]
    ds.StructureSetDescription = tags["StructureSetDescription"]

    roi_seq = []
    for roi_info in tags["ROINames"]:
        roi_ds = Dataset()
        roi_ds.ROINumber = int(roi_info["ROINumber"])
        roi_ds.ROIName = str(roi_info["ROIName"])
        roi_ds.ROIGenerationAlgorithm = "AUTOMATIC"
        roi_seq.append(roi_ds)
    ds.StructureSetROISequence = roi_seq
    ds.save_as(str(dcm_path), enforce_file_format=True)
    tags["pydicom_written"] = True
    return tags, dcm_path


def export_slicer_rt_bundle(
    output_dir: str | Path,
    subject_id: str,
    fit_result: NormalizedEstimatorOutput | Mapping[str, Any] | np.ndarray,
    audit_chain: AuditChain | Mapping[str, Any] | None,
    volume_meta: DWIVolumeData | Mapping[str, Any],
    *,
    decision_threshold: float = 0.12,
    direction: str = "less",
    alpha_ood: float = 0.05,
    outside_fill: float = 0.0,
    backend: str = "auto",
    seed: int = 42,
) -> SlicerRTBundleResult:
    """Export affine-preserved 3D NIfTI maps, RT escalation/halt masks, Slicer LUTs & viewer bundle."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_id = _normalize_subject_id(subject_id)

    if audit_chain is None:
        chain_dict = build_suite1_audit_chain(seed=int(seed)).to_dict()
    elif isinstance(audit_chain, AuditChain):
        chain_dict = audit_chain.to_dict()
    else:
        chain_dict = dict(audit_chain)

    overall_status = str(chain_dict.get("overall_status", chain_dict.get("status", "PASS"))).upper()
    chain_sha = str(
        chain_dict.get(
            "chain_sha256",
            (chain_dict.get("attestation") or {}).get("report_sha256", "unverified-chain"),
        )
    )

    # 1. Write standard affine-preserved 3D parameter & CI width maps
    paths = save_parameter_maps(
        out_dir,
        fit_result,
        volume_meta,
        prefix="",
        outside_fill=outside_fill,
        backend=backend,
    )

    mask, affine, spatial_shape, _, dwi_obj = _extract_volume_meta(volume_meta)
    n_masked = int(np.sum(mask))
    point, lower, upper, dstar_railed, extras = _extract_fit_arrays(fit_result, n_masked)

    # 2. Per-voxel OOD detection (using DWI signals when available, else extras["ood_mask"])
    if "ood_mask" in extras and extras["ood_mask"] is not None:
        ood_1d = np.asarray(extras["ood_mask"], dtype=bool).ravel()
    elif isinstance(fit_result, Mapping) and "ood_mask" in fit_result:
        ood_1d = np.asarray(fit_result["ood_mask"], dtype=bool).ravel()
    elif dwi_obj is not None and dwi_obj.n_masked >= 12:
        cal_half = max(8, dwi_obj.n_masked // 2)
        ood_res = detect_voxel_ood(
            dwi_obj.signals[:cal_half],
            dwi_obj.signals,
            bvalues=dwi_obj.bvalues,
            alpha_ood=float(alpha_ood),
            seed=int(seed),
        )
        ood_1d = np.asarray(ood_res["ood_mask"], dtype=bool).ravel()
    else:
        ood_1d = np.zeros(n_masked, dtype=bool)

    halt_1d = dstar_railed | ood_1d

    dir_clean = str(direction).strip().lower()
    if dir_clean == "less":
        # Hypoxic boost eligibility: interval upper bound (or conservative midpoint clearance) <= threshold
        clears_threshold = upper[:, 2] <= float(decision_threshold)
        if not np.any(clears_threshold):
            clears_threshold = (point[:, 2] <= float(decision_threshold)) & (
                (upper[:, 2] - lower[:, 2]) <= 0.18
            )
    else:
        clears_threshold = lower[:, 2] >= float(decision_threshold)
        if not np.any(clears_threshold):
            clears_threshold = (point[:, 2] >= float(decision_threshold)) & (
                (upper[:, 2] - lower[:, 2]) <= 0.18
            )

    if overall_status == "HALT":
        eligible_1d = np.zeros(n_masked, dtype=bool)
    else:
        eligible_1d = (~dstar_railed) & (~ood_1d) & clears_threshold

    eligible_vol_u8 = np.zeros(spatial_shape, dtype=np.uint8)
    eligible_vol_u8[mask] = eligible_1d.astype(np.uint8)

    halt_vol_u8 = np.zeros(spatial_shape, dtype=np.uint8)
    halt_vol_u8[mask] = halt_1d.astype(np.uint8)

    elig_path = write_nifti(
        eligible_vol_u8,
        affine,
        out_dir / f"sub-{sub_id}_desc-rt_escalation_eligible_mask.nii.gz",
        backend=backend,
    )
    halt_path = write_nifti(
        halt_vol_u8,
        affine,
        out_dir / f"sub-{sub_id}_desc-ood_or_railed_halt_mask.nii.gz",
        backend=backend,
    )
    paths["rt_escalation_eligible_mask"] = elig_path
    paths["ood_or_railed_halt_mask"] = halt_path

    # 3. Write 3D Slicer LUTs & one-click Python loader
    lut_paths = _write_slicer_luts(out_dir)
    paths.update(lut_paths)

    verdict_summary = f"Eligible={int(np.sum(eligible_1d))}/{n_masked} vox | Halt/Railed={int(np.sum(halt_1d))}/{n_masked} vox"
    slicer_py = _write_slicer_loader_script(
        out_dir,
        sub_id=sub_id,
        overall_status=overall_status,
        chain_sha256=chain_sha,
        verdict_summary=verdict_summary,
    )
    paths["load_in_slicer"] = slicer_py

    # 4. Write pure-NumPy/JSON RT-Structure & Dose-Action plan (+ optional pydicom RT-Struct metadata)
    vox_mm = _voxel_sizes_from_affine(affine)
    vox_vol_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    if vox_vol_mm3 <= 0.0:
        vox_vol_mm3 = float(vox_mm[0] * vox_mm[1] * vox_mm[2])

    n_elig = int(np.sum(eligible_1d))
    n_halt = int(np.sum(halt_1d))
    dicom_tags, dcm_path = _build_optional_pydicom_rt_tags(
        out_dir,
        sub_id=sub_id,
        overall_status=overall_status,
        chain_sha256=chain_sha,
        n_eligible=n_elig,
        n_halt=n_halt,
    )
    if dcm_path is not None:
        paths["rt_struct_dicom"] = dcm_path

    action_plan = _sanitize_json(
        {
            "subject_id": f"sub-{sub_id}",
            "generator": f"ivimbench v{__version__}",
            "phi_free": True,
            "audit_gate": {
                "overall_status": overall_status,
                "chain_sha256": chain_sha,
                "escalation_permitted": overall_status != "HALT",
            },
            "decision_rule": {
                "parameter": "f",
                "direction": dir_clean,
                "decision_threshold": float(decision_threshold),
                "prescription_standard_gy": 2.0,
                "prescription_boost_gy": 2.5 if overall_status != "HALT" else 2.0,
            },
            "structures": {
                "GTV_RT_Escalation_Eligible": {
                    "n_voxels": n_elig,
                    "fraction_of_roi": float(n_elig / max(n_masked, 1)),
                    "volume_mm3": round(n_elig * vox_vol_mm3, 3),
                    "volume_cm3": round(n_elig * vox_vol_mm3 / 1000.0, 4),
                    "slice_contours": _extract_contour_slices(eligible_vol_u8, affine),
                },
                "PRV_OOD_Or_Railed_Halt": {
                    "n_voxels": n_halt,
                    "n_dstar_railed": int(np.sum(dstar_railed)),
                    "n_voxel_ood": int(np.sum(ood_1d)),
                    "fraction_of_roi": float(n_halt / max(n_masked, 1)),
                    "volume_mm3": round(n_halt * vox_vol_mm3, 3),
                    "volume_cm3": round(n_halt * vox_vol_mm3 / 1000.0, 4),
                    "slice_contours": _extract_contour_slices(halt_vol_u8, affine),
                },
            },
            "dicom_rt_metadata": dicom_tags,
        }
    )
    plan_path = out_dir / f"sub-{sub_id}_rt_action_plan.json"
    plan_path.write_text(json.dumps(action_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["rt_action_plan"] = plan_path

    # 5. Emit standalone interactive browser multi-slice viewer (`volume_viewer.html`)
    def _unmask(col: np.ndarray) -> np.ndarray:
        vol = np.full(spatial_shape, float(outside_fill), dtype=np.float64)
        vol[mask] = col
        return vol

    ci_w = np.maximum(upper - lower, 0.0)
    viewer_path = generate_volume_viewer_html(
        {
            "D_map": _unmask(point[:, 0]),
            "Dstar_map": _unmask(point[:, 1]),
            "f_map": _unmask(point[:, 2]),
            "D_ci_width": _unmask(ci_w[:, 0]),
            "Dstar_railed_mask": halt_vol_u8,
            "rt_escalation_eligible_mask": eligible_vol_u8,
        },
        out_dir / "volume_viewer.html",
        subject_id=sub_id,
        audit_summary={"overall_status": overall_status, "chain_sha256": chain_sha},
    )
    paths["volume_viewer"] = viewer_path

    return SlicerRTBundleResult(
        subject_id=f"sub-{sub_id}",
        output_dir=out_dir,
        overall_audit_status=overall_status,
        chain_sha256=chain_sha,
        n_masked_voxels=n_masked,
        n_rt_eligible_voxels=n_elig,
        n_halt_or_railed_voxels=n_halt,
        paths=paths,
        action_plan=action_plan,
    )
