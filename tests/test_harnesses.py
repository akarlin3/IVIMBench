"""Property and unit tests for misspecification and scan-rescan repeatability harnesses."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import stats

from ivimbench import register_estimator, unregister_estimator
from ivimbench.harnesses import (
    MISSPEC_FAMILIES,
    MisspecificationHarness,
    MisspecificationReport,
    RepeatabilityHarness,
    RepeatabilityReport,
    evaluate_misspecification,
    evaluate_repeatability,
    misspecification_continuity_residual,
    simulate_scan_rescan,
)
from ivimbench.phantoms import generate_dro, ivim_biexp, ivim_kurtosis


@pytest.mark.parametrize("family", MISSPEC_FAMILIES)
def test_all_misspecification_families_reduce_to_biexp_at_continuity_limit(family: str) -> None:
    res = misspecification_continuity_residual(family, n_voxels=300, seed=42)
    if family == "dispersion_gamma":
        assert res < 1e-6
    else:
        assert res == pytest.approx(0.0, abs=1e-13)


def test_ivim_kurtosis_monotonicity_and_zero_limit() -> None:
    b = np.array([0.0, 50.0, 200.0, 500.0, 800.0, 1200.0])
    s_bi = ivim_biexp(b, 1.2e-3, 35.0e-3, 0.20, s0=1.0)
    s_k0 = ivim_kurtosis(b, 1.2e-3, 35.0e-3, 0.20, K=0.0, s0=1.0)
    np.testing.assert_allclose(s_k0, s_bi, rtol=0.0, atol=1e-14)

    # Positive kurtosis elevates high-b signal above pure Gaussian decay while remaining non-increasing
    s_pos = ivim_kurtosis(b, 1.2e-3, 35.0e-3, 0.20, K=0.9, s0=1.0)
    assert np.all(np.diff(s_pos) <= 1e-12)
    assert np.all(s_pos[2:] > s_bi[2:])


def test_misspecification_harness_separates_d_vs_dstar_degradation() -> None:
    harness = MisspecificationHarness(
        protocol="osipi",
        organ="liver",
        snr=40.0,
        alpha=0.10,
        n_voxels=150,
        n_boot=50,
        seed=42,
    )
    report = harness.evaluate(
        estimators=["crlb"],
        families=("biexp", "dispersion_lognormal", "stretched", "triexp", "kurtosis", "heavy_tail"),
    )
    assert isinstance(report, MisspecificationReport)
    df = report.compartment_summary
    assert set(df["truth_family"]) == {
        "biexp",
        "dispersion_lognormal",
        "stretched",
        "triexp",
        "kurtosis",
        "heavy_tail",
    }

    # Matched biexp baseline achieves near-nominal global coverage and near-zero degradation
    row_bi = df.loc[df["truth_family"] == "biexp"].iloc[0]
    assert row_bi["global_marginal_cov"] >= 0.84
    assert row_bi["D_cov_drop"] == pytest.approx(0.0, abs=1e-12)
    assert row_bi["Dstar_cov_drop"] == pytest.approx(0.0, abs=1e-12)

    # Under non-Gaussian tissue kurtosis (K=0.85), slow tissue diffusion D suffers sharp coverage drop
    row_kurt = df.loc[df["truth_family"] == "kurtosis"].iloc[0]
    assert row_kurt["D_cov_drop"] > 0.15

    # Convenience wrapper & table serializers
    rep2 = evaluate_misspecification(
        estimators=["segmented"],
        families=("biexp", "triexp"),
        n_voxels=60,
        n_boot=30,
        seed=42,
    )
    assert "| segmented" in rep2.to_markdown()
    assert "\\begin{table}" in rep2.to_latex()
    assert len(json.loads(rep2.to_json())["compartment_summary"]) == 2


def test_scan_rescan_simulation_noise_identities_and_drift() -> None:
    # At n_coils=1 with zero drift, noncentral_chi matches rician bit-for-bit
    p_ric = simulate_scan_rescan(
        protocol="osipi", organ="liver", n_voxels=80, snr=25.0, noise_type="rician", seed=101
    )
    p_ncc1 = simulate_scan_rescan(
        protocol="osipi",
        organ="liver",
        n_voxels=80,
        snr=25.0,
        noise_type="noncentral_chi",
        n_coils=1,
        seed=101,
    )
    np.testing.assert_array_equal(p_ric.scan_A, p_ncc1.scan_A)
    np.testing.assert_array_equal(p_ric.scan_B, p_ncc1.scan_B)
    assert not np.allclose(p_ric.scan_A, p_ric.scan_B)

    # Coil, SNR, and gain drift options
    dro = generate_dro(shape=(12, 12), organ="pancreas", protocol="osipi", snr=30.0, seed=42)
    p_drift = simulate_scan_rescan(
        cohort=dro,
        snr=30.0,
        noise_type="noncentral_chi",
        n_coils=4,
        n_coils_B=8,
        snr_drift=-0.20,
        gain_drift=0.10,
        seed=42,
    )
    assert p_drift.n_coils_A == 4 and p_drift.n_coils_B == 8
    assert p_drift.snr_B == pytest.approx(24.0)
    assert p_drift.s0_B == pytest.approx(1.10)


def test_repeatability_exchange_symmetry_and_spearman_sizing_invariants() -> None:
    pair = simulate_scan_rescan(
        protocol="osipi", organ="liver", n_voxels=120, snr=30.0, noise_type="rician", seed=42
    )
    rep_ab = evaluate_repeatability(pair, estimators=("segmented", "crlb"), alpha=0.10)
    rep_ba = evaluate_repeatability(pair.swapped(), estimators=("segmented", "crlb"), alpha=0.10)
    assert isinstance(rep_ab, RepeatabilityReport)

    df_ab = rep_ab.dataframe.sort_values(["method", "parameter"]).reset_index(drop=True)
    df_ba = rep_ba.dataframe.sort_values(["method", "parameter"]).reset_index(drop=True)

    # Swapping scan_A and scan_B swaps directional coverages and preserves symmetric metrics exactly
    np.testing.assert_allclose(df_ab["retest_cov_A_on_B"], df_ba["retest_cov_B_on_A"], rtol=1e-12)
    np.testing.assert_allclose(df_ab["retest_cov_B_on_A"], df_ba["retest_cov_A_on_B"], rtol=1e-12)
    np.testing.assert_allclose(df_ab["symmetric_retest_cov"], df_ba["symmetric_retest_cov"], rtol=1e-12)
    np.testing.assert_allclose(df_ab["within_voxel_sd"], df_ba["within_voxel_sd"], rtol=1e-12)
    np.testing.assert_allclose(df_ab["repeatability_coeff"], df_ba["repeatability_coeff"], rtol=1e-12)
    np.testing.assert_allclose(df_ab["width_scatter_spearman_r"], df_ba["width_scatter_spearman_r"], rtol=1e-12)

    # Analytical Gaussian retest coverage & heteroscedastic Spearman rank correlation verification:
    # Construct heteroscedastic noisy observations around b=0 where voxel noise scales linearly with index.
    rng = np.random.default_rng(1234)
    n_vox = 2500
    bvals = np.array([0.0, 50.0, 200.0, 800.0])
    het_sigma = np.linspace(0.02, 0.30, n_vox)
    base_val = np.full(n_vox, 1.0)
    # Encode noisy observation in column 0 and known het_sigma in column 1
    scan_a = np.column_stack([base_val + rng.normal(0.0, het_sigma), het_sigma, np.ones(n_vox), np.ones(n_vox)])
    scan_b = np.column_stack([base_val + rng.normal(0.0, het_sigma), het_sigma, np.ones(n_vox), np.ones(n_vox)])

    z90 = float(stats.norm.ppf(0.95))

    @register_estimator("_test_het_exact", overwrite=True)
    def _het_est(signals: np.ndarray, bvalues: np.ndarray) -> dict[str, object]:
        pt = np.column_stack([signals[:, 0], signals[:, 0], signals[:, 0]])
        hw = z90 * signals[:, 1:2] * np.ones((1, 3))
        return {"D": pt[:, 0], "D_star": pt[:, 1], "f": pt[:, 2], "intervals": (pt - hw, pt + hw)}

    @register_estimator("_test_hom_const", overwrite=True)
    def _hom_est(signals: np.ndarray, bvalues: np.ndarray) -> dict[str, object]:
        pt = np.column_stack([signals[:, 0], signals[:, 0], signals[:, 0]])
        hw = np.full_like(pt, z90 * float(np.mean(signals[:, 1])))
        return {"D": pt[:, 0], "D_star": pt[:, 1], "f": pt[:, 2], "intervals": (pt - hw, pt + hw)}

    try:
        from ivimbench.harnesses.repeatability import ScanRescanPair

        syn_pair = ScanRescanPair(
            scan_A=scan_a,
            scan_B=scan_b,
            bvalues=bvals,
            signals_clean=np.ones_like(scan_a),
            params_true=None,
            snr_A=30.0,
            snr_B=30.0,
            n_coils_A=1,
            n_coils_B=1,
            s0_A=1.0,
            s0_B=1.0,
            noise_type="gaussian",
        )
        rep_het = RepeatabilityHarness(alpha=0.10).evaluate(
            syn_pair, estimators=["_test_het_exact", "_test_hom_const"]
        )
        df_h = rep_het.dataframe
        row_het = df_h[(df_h["method"] == "_test_het_exact") & (df_h["parameter"] == "D")].iloc[0]
        row_hom = df_h[(df_h["method"] == "_test_hom_const") & (df_h["parameter"] == "D")].iloc[0]

        expected_retest = float(2.0 * stats.norm.cdf(z90 / np.sqrt(2.0)) - 1.0)  # ~0.755
        assert row_het["symmetric_retest_cov"] == pytest.approx(expected_retest, abs=0.025)
        assert row_het["nominal_retest_prob"] == pytest.approx(expected_retest, abs=1e-6)
        # Heteroscedastic widths correlate strongly with |est_A - est_B|; constant widths give 0.0
        assert row_het["width_scatter_spearman_r"] > 0.45
        assert row_hom["width_scatter_spearman_r"] == pytest.approx(0.0, abs=1e-12)
    finally:
        unregister_estimator("_test_het_exact")
        unregister_estimator("_test_hom_const")

    assert "| segmented |" in rep_ab.to_markdown()
    assert "\\begin{table}" in rep_ab.to_latex()
    assert len(json.loads(rep_ab.to_json())["records"]) == 6
