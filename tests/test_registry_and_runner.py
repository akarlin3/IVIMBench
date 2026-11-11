"""Tests for the extensible estimator registry (@register_estimator) and multi-task runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ivimbench import (
    DEFAULT_REGISTERED_ESTIMATORS,
    CRLBNormalEstimator,
    MultiTaskBenchmarkResult,
    NormalizedEstimatorOutput,
    generate_cohort,
    get_estimator,
    list_estimators,
    register_estimator,
    run_benchmark_suite,
    run_estimator,
    unregister_estimator,
)
from ivimbench.phantoms.synthetic import FAMILIES


def test_five_line_custom_estimator_registration_and_cleanup() -> None:
    cohort = generate_cohort(protocol="osipi", organ="liver", n_voxels=40, snr=30.0, seed=42)

    @register_estimator("toy_five_line", overwrite=True)
    def fit_toy(signals: np.ndarray, bvalues: np.ndarray) -> dict[str, object]:
        d_hat = np.clip(-np.log(np.clip(signals[:, -1] / np.clip(signals[:, 0], 1e-4, None), 1e-4, 1.0)) / bvalues[-1], 1e-4, 4e-3)
        pt = np.column_stack([d_hat, np.full_like(d_hat, 35e-3), np.full_like(d_hat, 0.20)])
        return {"D": pt[:, 0], "D_star": pt[:, 1], "f": pt[:, 2], "intervals": (0.8 * pt, 1.2 * pt)}

    try:
        assert "toy_five_line" in list_estimators()
        reg = get_estimator("toy_five_line")
        out = reg.predict(cohort.signals, cohort.bvalues, alpha=0.10)
        assert isinstance(out, NormalizedEstimatorOutput)
        assert out.point.shape == (40, 3)
        assert out.lower.shape == (40, 3)
        assert out.upper.shape == (40, 3)
        assert out.has_intervals is True
        np.testing.assert_allclose(out.D_star, out.Dstar)

        res = run_benchmark_suite(
            estimators=["toy_five_line", "segmented"],
            protocols=["osipi"],
            families=["biexp"],
            snr_regimes=[30.0],
            n_voxels=30,
            n_boot=50,
            seed=42,
        )
        assert set(res.dataframe["method"]) == {"toy_five_line", "segmented"}
    finally:
        unregister_estimator("toy_five_line")
        assert "toy_five_line" not in list_estimators()


def test_register_class_estimator_and_point_only_fallback() -> None:
    cohort = generate_cohort(protocol="osipi", organ="liver", n_voxels=24, snr=30.0, seed=7)

    @register_estimator("toy_point_class", overwrite=True)
    class ToyPointEstimator:
        def fit_predict(self, signals: np.ndarray, bvalues: np.ndarray) -> dict[str, np.ndarray]:
            n = signals.shape[0]
            return {
                "D": np.full(n, 1.1e-3),
                "Dstar": np.full(n, 30.0e-3),
                "f": np.full(n, 0.18),
            }

    try:
        out = run_estimator("toy_point_class", cohort.signals, cohort.bvalues)
        assert out.has_intervals is False
        # Degenerate zero-width intervals when estimator omits intervals
        np.testing.assert_array_equal(out.lower, out.point)
        np.testing.assert_array_equal(out.upper, out.point)
    finally:
        unregister_estimator("toy_point_class")


def test_builtin_deterministic_baselines_and_crlb_estimator() -> None:
    for builtin_name in DEFAULT_REGISTERED_ESTIMATORS:
        assert builtin_name in list_estimators()

    cohort = generate_cohort(protocol="osipi", organ="liver", n_voxels=60, snr=35.0, seed=42)

    out_seg = run_estimator("segmented", cohort.signals, cohort.bvalues, alpha=0.10, snr=35.0)
    out_nlls = run_estimator("nlls", cohort.signals, cohort.bvalues, alpha=0.10, snr=35.0)
    out_crlb = run_estimator("crlb", cohort.signals, cohort.bvalues, alpha=0.10, snr=35.0)

    for out in (out_seg, out_nlls, out_crlb):
        assert out.point.shape == (60, 3)
        assert np.all(out.upper >= out.lower)
        assert out.railed is not None and out.railed.shape == (60, 3)

    # Direct CRLBNormalEstimator achieves calibrated (~90%) coverage on well-specified biexp at SNR=35
    crlb_est = CRLBNormalEstimator(protocol="osipi", snr=35.0)
    lo_c, hi_c = crlb_est.predict_intervals(cohort.signals, alpha=0.10)
    d_cov = float(np.mean((cohort.params[:, 0] >= lo_c[:, 0]) & (cohort.params[:, 0] <= hi_c[:, 0])))
    assert d_cov >= 0.80


def test_multi_task_runner_across_five_families_protocols_and_exports(tmp_path: Path) -> None:
    res = run_benchmark_suite(
        estimators=("segmented", "crlb"),
        protocols=("osipi", "tcga_lihc", "acrin_6698"),
        families=tuple(FAMILIES.keys()),
        snr_regimes=(20.0, 40.0),
        n_voxels=30,
        n_boot=40,
        seed=42,
        out_dir=tmp_path / "bench_out",
    )
    assert isinstance(res, MultiTaskBenchmarkResult)
    # 2 estimators * 3 protocols * 5 families * 2 SNRs * 3 parameters = 180 records
    assert len(res.dataframe) == 180
    assert set(res.dataframe["family"]) == set(FAMILIES.keys())
    assert set(res.dataframe["protocol"]) == {"osipi", "tcga_lihc", "acrin_6698"}
    assert set(res.dataframe["snr"]) == {20.0, 40.0}

    # Determinism check
    res_repeat = run_benchmark_suite(
        estimators=("segmented", "crlb"),
        protocols=("osipi",),
        families=("biexp",),
        snr_regimes=(20.0,),
        n_voxels=30,
        n_boot=40,
        seed=42,
    )
    sub_orig = res.dataframe[
        (res.dataframe["protocol"] == "osipi")
        & (res.dataframe["family"] == "biexp")
        & (res.dataframe["snr"] == 20.0)
    ].reset_index(drop=True)
    np.testing.assert_allclose(
        sub_orig["marginal_coverage"].to_numpy(),
        res_repeat.dataframe["marginal_coverage"].to_numpy(),
    )

    # Reproducible Markdown, JSON, LaTeX, and CSV export verification
    md_text = res.to_markdown(aggregate=True)
    tex_text = res.to_latex(aggregate=True)
    json_obj = json.loads(res.to_json())
    assert "| segmented" in md_text and "| crlb" in md_text
    assert "\\begin{table}" in tex_text and "\\end{table}" in tex_text
    assert len(json_obj["records"]) == 180

    for fname in (
        "benchmark_summary.md",
        "benchmark_summary.json",
        "benchmark_summary.tex",
        "benchmark_summary_records.csv",
    ):
        fpath = tmp_path / "bench_out" / fname
        assert fpath.exists() and fpath.stat().st_size > 100


def test_duplicate_registration_guard() -> None:
    with pytest.raises(ValueError, match="already registered"):
        @register_estimator("segmented", overwrite=False)
        def _dup(signals: np.ndarray, bvalues: np.ndarray) -> dict[str, np.ndarray]:
            return {"D": signals[:, 0], "D_star": signals[:, 0], "f": signals[:, 0]}
