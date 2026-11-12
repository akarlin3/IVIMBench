"""Deterministic unit tests for v1.5.0 publication figure suite, BIDS-Derivatives qMRI exporter, HTML leaderboard, and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ivimbench import (
    BenchmarkSuite,
    SchemeEvaluator,
    __version__,
    evaluate_misspecification,
    export_bids_derivatives,
    generate_cohort,
    generate_dro,
    generate_html_report,
    plot_calibration_curves,
    plot_dro_slice_montage,
    plot_misspecification_aliasing,
    plot_reliability_and_terciles,
    plot_tercile_coverage,
    plot_vernier_pareto_frontier,
    read_nifti1,
    run_benchmark_suite,
    run_estimator,
)
from ivimbench.cli import main as cli_main


def test_publication_figure_suite_and_legacy_vis_backcompat(tmp_path: Path) -> None:
    cohort = generate_cohort(protocol="osipi", organ="liver", n_voxels=90, snr=30.0, seed=42)
    est = run_estimator("crlb", cohort.signals, cohort.bvalues, alpha=0.10, snr=30.0)
    assert est.std is not None

    # 1. Backwards-compatible legacy vis functions
    p_cal = tmp_path / "legacy_cal.png"
    fig_cal, axes_cal = plot_calibration_curves(
        cohort,
        {"CRLBNormal": (est.point, est.std)},
        save_path=p_cal,
    )
    assert p_cal.is_file() and p_cal.stat().st_size > 1000
    assert len(axes_cal) == 3
    plt.close(fig_cal)

    suite = BenchmarkSuite(protocol="osipi", snr=30.0, alpha=0.10, n_boot=40, seed=42)
    rep = suite.evaluate(
        cohort,
        point_pred=est.point,
        std_pred=est.std,
        method="honest_crlb",
    )
    p_terc = tmp_path / "legacy_tercile.png"
    fig_t, _ = plot_tercile_coverage(rep, save_path=p_terc)
    assert p_terc.is_file() and p_terc.stat().st_size > 1000
    plt.close(fig_t)

    # 2. 2-panel reliability + D*-terciles comparing honest_crlb, floored_railed, and conformal
    p_rel = tmp_path / "reliability_and_terciles.png"
    fig_r, axes_r = plot_reliability_and_terciles(cohort, out_path=p_rel, seed=42)
    assert p_rel.is_file() and p_rel.stat().st_size > 2000
    assert axes_r.shape == (2,)
    legend_labels = [t.get_text() for t in axes_r[1].get_legend().get_texts()]
    assert "honest_crlb" in legend_labels
    assert "floored_railed" in legend_labels
    assert "conformal" in legend_labels
    plt.close(fig_r)

    # Also verify BenchmarkReport input path
    p_rel2 = tmp_path / "reliability_from_report.png"
    fig_r2, _ = plot_reliability_and_terciles(rep, out_path=p_rel2)
    assert p_rel2.is_file()
    plt.close(fig_r2)

    # 3. Misspecification Procrustes marginal vs. conditional coverage curves on tissue D
    misspec = evaluate_misspecification(
        estimators=("segmented", "crlb"),
        protocol="osipi",
        organ="liver",
        snr=30.0,
        n_voxels=70,
        n_boot=30,
        truth_families=("biexp", "dispersion_gamma", "dispersion_lognormal", "stretched", "triexp"),
        seed=42,
    )
    p_mis = tmp_path / "misspec_aliasing.png"
    fig_m, axes_m = plot_misspecification_aliasing(misspec, out_path=p_mis)
    assert p_mis.is_file() and p_mis.stat().st_size > 2000
    assert axes_m.shape == (2,)
    plt.close(fig_m)

    # 4. Vernier sharpness vs. high-D* conditional coverage Pareto frontier
    scheme_rep = SchemeEvaluator(
        organ="liver", snr=30.0, n_voxels=70, n_boot=30, seed=42
    ).compare(["osipi", "acrin_6698"], include_optimized=True, optimized_n_b=6)
    p_par = tmp_path / "vernier_pareto.png"
    fig_p, _ = plot_vernier_pareto_frontier(scheme_rep, out_path=p_par)
    assert p_par.is_file() and p_par.stat().st_size > 2000
    plt.close(fig_p)

    # 5. 4-panel DRO spatial slice montage (D, D*, f, binary D* boundary-railing mask)
    dro_3d = generate_dro(shape=(16, 16, 8), organ="pancreas", protocol="osipi", snr=25.0, seed=42)
    p_dro = tmp_path / "dro_montage.png"
    fig_d, axes_d = plot_dro_slice_montage(dro_3d, out_path=p_dro)
    assert p_dro.is_file() and p_dro.stat().st_size > 2000
    assert axes_d.shape == (4,)
    plt.close(fig_d)


def test_bids_derivatives_qmri_export_schema_and_nifti_roundtrip(tmp_path: Path) -> None:
    dro = generate_dro(shape=(12, 14, 8), organ="liver", protocol="osipi", snr=28.0, seed=42)
    sig_fg, _ = dro.to_voxels(foreground_only=True)
    pred = run_estimator("segmented", sig_fg, dro.bvalues, snr=28.0)

    affine = np.diag([2.0, 2.0, 4.0, 1.0])
    affine[:3, 3] = [-14.0, -16.0, 8.0]

    bids_dir = tmp_path / "bids_derivatives"
    res = export_bids_derivatives(
        output_dir=bids_dir,
        subject_id="01",
        fit_result=pred,
        volume_meta={
            "mask": dro.foreground_mask,
            "shape": dro.spatial_shape,
            "affine": affine,
            "bvalues": dro.bvalues,
            "estimation_algorithm": "SegmentedEstimator",
        },
        protocol_name="osipi",
    )

    assert res.dataset_description.is_file()
    ds_meta = json.loads(res.dataset_description.read_text(encoding="utf-8"))
    assert ds_meta["DatasetType"] == "derivative"
    assert ds_meta["GeneratedBy"][0]["Version"] == __version__

    sub_dwi = bids_dir / "sub-01" / "dwi"
    expected_niftis = {
        "D_map": sub_dwi / "sub-01_param-D_dwimap.nii.gz",
        "Dstar_map": sub_dwi / "sub-01_param-Dstar_dwimap.nii.gz",
        "f_map": sub_dwi / "sub-01_param-f_dwimap.nii.gz",
        "railed_mask": sub_dwi / "sub-01_desc-railed_mask.nii.gz",
    }
    for key, nii_path in expected_niftis.items():
        assert res.nifti_paths[key] == nii_path
        assert nii_path.is_file()
        arr, aff_read = read_nifti1(nii_path)
        assert arr.shape == (12, 14, 8)
        np.testing.assert_allclose(aff_read, affine, atol=1e-5)

        sidecar_path = res.sidecar_paths[key]
        assert sidecar_path.is_file()
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert sidecar["ProtocolName"] == "osipi"
        assert sidecar["EstimationAlgorithm"] == "SegmentedEstimator"
        assert sidecar["SoftwareVersion"] == __version__
        assert sidecar["BValues"] == [float(b) for b in dro.bvalues]

    # Verify binary mask values and nibabel compatibility if nibabel is installed
    mask_arr, _ = read_nifti1(expected_niftis["railed_mask"])
    assert set(np.unique(mask_arr)).issubset({0, 1})

    try:
        import nibabel as nib

        nib_img = nib.load(str(expected_niftis["D_map"]))
        pure_arr, _ = read_nifti1(expected_niftis["D_map"])
        np.testing.assert_allclose(np.asarray(nib_img.dataobj), pure_arr, atol=1e-6)
    except ImportError:
        pass

    assert res.svg_summary.is_file() and "<svg" in res.svg_summary.read_text(encoding="utf-8")
    assert res.html_summary.is_file() and "sub-01" in res.html_summary.read_text(encoding="utf-8")

    # Reject non-alphanumeric / unsafe subject identifiers
    with pytest.raises(ValueError, match="subject_id"):
        export_bids_derivatives(bids_dir, "patient_john_doe!", pred)


def test_html_report_and_cli_report_and_export_bids(tmp_path: Path) -> None:
    bench = run_benchmark_suite(
        estimators=("segmented", "crlb"),
        protocols=("osipi", "acrin_6698"),
        families=("biexp", "triexp"),
        snr_regimes=(30.0,),
        n_voxels=50,
        n_boot=30,
        seed=42,
    )
    report_path = tmp_path / "ivimbench_report.html"
    out = generate_html_report(bench, out_path=report_path)
    assert out == report_path and report_path.is_file()
    html_text = report_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text
    assert "Gnomon Marginal Calibration" in html_text
    assert "Gnomon High-D* Conditional Gate" in html_text
    assert "sortBenchmarkTable" in html_text
    assert html_text.count("<svg") >= 2

    # Verify CLI: ivimbench report --out ...
    cli_html = tmp_path / "cli_benchmark_report.html"
    rc_rep = cli_main(
        [
            "report",
            "--out",
            str(cli_html),
            "--protocol",
            "osipi",
            "--estimators",
            "segmented,crlb",
            "--n-voxels",
            "45",
            "--seed",
            "42",
        ]
    )
    assert rc_rep == 0
    assert cli_html.is_file() and "Gnomon" in cli_html.read_text(encoding="utf-8")

    # Verify CLI: ivimbench export-bids --subject 01 --out ...
    cli_bids = tmp_path / "cli_bids_out"
    rc_bids = cli_main(
        [
            "export-bids",
            "--subject",
            "01",
            "--out",
            str(cli_bids),
            "--shape",
            "12,12,8",
            "--seed",
            "42",
        ]
    )
    assert rc_bids == 0
    assert (cli_bids / "dataset_description.json").is_file()
    assert (cli_bids / "sub-01" / "dwi" / "sub-01_param-D_dwimap.nii.gz").is_file()
    assert (cli_bids / "sub-01" / "dwi" / "sub-01_param-Dstar_dwimap.nii.gz").is_file()
    assert (cli_bids / "sub-01" / "dwi" / "sub-01_param-f_dwimap.nii.gz").is_file()
    assert (cli_bids / "sub-01" / "dwi" / "sub-01_desc-railed_mask.nii.gz").is_file()
