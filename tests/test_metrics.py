"""Unit tests for UQ coverage, CRLB sharpness, railing metrics, baselines, and BenchmarkSuite."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ivimbench import (
    BenchmarkReport,
    BenchmarkSuite,
    MLPEstimator,
    NLLSEstimator,
    SegmentedEstimator,
    generate_cohort,
    generate_dro,
    plot_calibration_curves,
    plot_tercile_coverage,
)
from ivimbench.metrics.coverage import (
    DEFAULT_QUANTILE_LEVELS,
    MondrianConformalQuantile,
    SplitConformalQuantile,
    SplitConformalResidual,
    central_interval,
    coverage_gap,
    ece_quantile,
    empirical_coverage,
    interval_score,
    pinball_loss,
    snr_stratified_coverage,
    tercile_stratified_coverage,
)
from ivimbench.metrics.railing import (
    DEFAULT_PARAM_BOUNDS,
    bootstrap_railing_ci,
    classify_railing,
    evaluate_parameter_railing,
    railing_fraction,
)
from ivimbench.metrics.sharpness import (
    absolute_width,
    crlb_dstar_wall_ratio,
    crlb_efficiency_ratio,
    crlb_interval_width,
    crlb_std,
    ivim_jacobian,
    normalized_width,
)
from ivimbench.phantoms.synthetic import ivim_biexp


def test_coverage_proper_scoring_and_stratification() -> None:
    rng = np.random.default_rng(42)
    n = 12_000
    mu = rng.uniform(0.5, 2.0, size=n)
    sigma = 0.15
    y = mu + rng.normal(0.0, sigma, size=n)

    z90 = float(stats.norm.ppf(0.95))
    lo, hi = mu - z90 * sigma, mu + z90 * sigma

    cov = empirical_coverage(y, lo, hi)
    assert cov == pytest.approx(0.90, abs=0.012)
    assert coverage_gap(y, lo, hi, nominal=0.90) == pytest.approx(0.0, abs=0.012)

    # Ideal Gaussian quantiles achieve near-zero ECE
    q_grid = mu[:, None] + sigma * stats.norm.ppf(DEFAULT_QUANTILE_LEVELS)[None, :]
    assert ece_quantile(y, q_grid, DEFAULT_QUANTILE_LEVELS) < 0.008
    c_lo, c_hi = central_interval(q_grid, DEFAULT_QUANTILE_LEVELS, alpha=0.10)
    np.testing.assert_allclose(c_lo, lo, rtol=1e-6)
    np.testing.assert_allclose(c_hi, hi, rtol=1e-6)

    # Proper scoring rules: mis-centered intervals incur strictly worse Winkler score & pinball loss
    s_good = float(np.mean(interval_score(lo, hi, y, alpha=0.10)))
    s_shifted = float(np.mean(interval_score(lo + 0.5, hi + 0.5, y, alpha=0.10)))
    assert s_good < s_shifted

    pb_med = float(np.mean(pinball_loss(y, mu, 0.50)))
    pb_bad = float(np.mean(pinball_loss(y, mu + 0.5, 0.50)))
    assert pb_med < pb_bad

    # Tercile-stratified coverage detects high-tercile collapse even when marginal looks reasonable
    cond_dstar = np.linspace(10e-3, 100e-3, n)
    hi_collapsed = np.where(cond_dstar > np.quantile(cond_dstar, 2.0 / 3.0), mu + 0.01 * sigma, hi)
    lo_collapsed = np.where(cond_dstar > np.quantile(cond_dstar, 2.0 / 3.0), mu - 0.01 * sigma, lo)
    terc = tercile_stratified_coverage(y, lo_collapsed, hi_collapsed, conditioning=cond_dstar, nominal=0.90)
    assert terc["low"].coverage == pytest.approx(0.90, abs=0.02)
    assert terc["high"].coverage < 0.10

    # SNR-stratified coverage
    snr_vec = np.repeat([10.0, 25.0, 50.0], n // 3)
    snr_strata = snr_stratified_coverage(y, lo, hi, snr=snr_vec, nominal=0.90)
    assert len(snr_strata) == 3
    for st in snr_strata.values():
        assert st.coverage == pytest.approx(0.90, abs=0.025)


def test_split_and_mondrian_conformal_calibration_restore_coverage() -> None:
    rng = np.random.default_rng(77)
    n_cal, n_test = 1500, 2000
    y_all = rng.normal(1.0, 0.25, size=(n_cal + n_test, 3))
    pt_all = y_all + rng.normal(0.0, 0.20, size=(n_cal + n_test, 3))

    # Deliberately 4x too narrow initial quantiles (severely under-covering)
    z_levels = stats.norm.ppf(DEFAULT_QUANTILE_LEVELS)
    q_all = pt_all[:, :, None] + 0.03 * z_levels[None, None, :]

    y_cal, y_te = y_all[:n_cal], y_all[n_cal:]
    pt_cal, pt_te = pt_all[:n_cal], pt_all[n_cal:]
    q_cal, q_te = q_all[:n_cal], q_all[n_cal:]

    lo_raw, hi_raw = central_interval(q_te, DEFAULT_QUANTILE_LEVELS, alpha=0.10)
    assert empirical_coverage(y_te[:, 0], lo_raw[:, 0], hi_raw[:, 0]) < 0.45

    # 1. SplitConformalResidual
    lo_scr, hi_scr = SplitConformalResidual(alpha=0.10).calibrate_apply(pt_cal, y_cal, pt_te)
    for j in range(3):
        assert empirical_coverage(y_te[:, j], lo_scr[:, j], hi_scr[:, j]) == pytest.approx(0.90, abs=0.03)

    # 2. SplitConformalQuantile (CQR)
    q_cqr = SplitConformalQuantile(DEFAULT_QUANTILE_LEVELS).calibrate_apply(q_cal, y_cal, q_te)
    lo_cqr, hi_cqr = central_interval(q_cqr, DEFAULT_QUANTILE_LEVELS, alpha=0.10)
    for j in range(3):
        assert empirical_coverage(y_te[:, j], lo_cqr[:, j], hi_cqr[:, j]) == pytest.approx(0.90, abs=0.03)

    # 3. MondrianConformalQuantile with heterogeneous noise across groups
    g_cal = np.repeat(["low_snr", "high_snr"], n_cal // 2)
    g_te = np.repeat(["low_snr", "high_snr"], n_test // 2)
    y_het_cal = pt_cal + np.where((g_cal == "low_snr")[:, None], 0.40, 0.08) * rng.normal(size=pt_cal.shape)
    y_het_te = pt_te + np.where((g_te == "low_snr")[:, None], 0.40, 0.08) * rng.normal(size=pt_te.shape)

    q_mond = MondrianConformalQuantile(DEFAULT_QUANTILE_LEVELS).calibrate_apply(
        q_cal, y_het_cal, g_cal, q_te, g_te
    )
    lo_m, hi_m = central_interval(q_mond, DEFAULT_QUANTILE_LEVELS, alpha=0.10)
    for grp in ("low_snr", "high_snr"):
        mask = g_te == grp
        cov_grp = empirical_coverage(y_het_te[mask, 1], lo_m[mask, 1], hi_m[mask, 1])
        assert cov_grp == pytest.approx(0.90, abs=0.04)


def test_sharpness_jacobian_and_crlb_efficiency_identities() -> None:
    b = np.array([0.0, 10.0, 20.0, 40.0, 80.0, 150.0, 300.0, 600.0])
    theta = np.array([[1.2e-3, 35.0e-3, 0.22], [0.9e-3, 60.0e-3, 0.18]])

    # Analytical Jacobian matches central finite differences
    jac = ivim_jacobian(b, theta[0], s0=1.5, include_s0=True)
    eps = 1e-8
    s_plus_d = ivim_biexp(b, theta[0, 0] + eps, theta[0, 1], theta[0, 2], s0=1.5)
    s_minus_d = ivim_biexp(b, theta[0, 0] - eps, theta[0, 1], theta[0, 2], s0=1.5)
    np.testing.assert_allclose(jac[:, 0], (s_plus_d - s_minus_d) / (2.0 * eps), rtol=1e-5)

    # CRLB scales inversely with SNR (doubling SNR halves CRLB SD)
    sd_20 = crlb_std(b, theta, snr=20.0)
    sd_40 = crlb_std(b, theta, snr=40.0)
    np.testing.assert_allclose(sd_40, 0.5 * sd_20, rtol=1e-12)

    # Exact CRLB intervals yield sharpness_to_crlb_ratio == 1.0 and crlb_efficiency == 1.0
    w_crlb = crlb_interval_width(b, theta, snr=30.0, alpha=0.10)
    lo = theta - 0.5 * w_crlb
    hi = theta + 0.5 * w_crlb
    eff = crlb_efficiency_ratio(lo, hi, bvalues=b, y_true=theta, snr=30.0, alpha=0.10)
    np.testing.assert_allclose(eff.sharpness_to_crlb_ratio, np.ones(3), rtol=1e-12)
    np.testing.assert_allclose(eff.crlb_efficiency, np.ones(3), rtol=1e-12)

    # Normalized width equals 0.5 when interval spans half the parameter domain
    bounds = {"D": (0.0, 4e-3), "Dstar": (0.0, 0.20), "f": (0.0, 0.80)}
    lo_half = np.zeros((5, 3))
    hi_half = np.broadcast_to([2e-3, 0.10, 0.40], (5, 3))
    np.testing.assert_allclose(normalized_width(lo_half, hi_half, bounds=bounds), [0.5, 0.5, 0.5])
    assert absolute_width(lo_half[:, 2], hi_half[:, 2]) == pytest.approx(0.40)

    # Sparse clinical schedule inflates high-tercile D* CRLB relative to low-tercile D*
    cohort = generate_cohort(protocol="tcga_lihc", organ="liver", n_voxels=600, snr=30.0, seed=42)
    wall = crlb_dstar_wall_ratio(cohort.bvalues, cohort.params, snr=30.0)
    assert wall > 1.5


def test_boundary_railing_classification_and_bootstrap_ci() -> None:
    # Span of Dstar in DEFAULT_PARAM_BOUNDS is (0.003, 0.20) -> 0.197; 1% tol = 0.00197
    vals = np.array([0.003, 0.004, 0.050, 0.080, 0.1995, 0.200])
    cls = classify_railing(vals, DEFAULT_PARAM_BOUNDS["Dstar"], rel_tol=0.01)
    np.testing.assert_array_equal(cls["lower_mask"], [True, True, False, False, False, False])
    np.testing.assert_array_equal(cls["upper_mask"], [False, False, False, False, True, True])

    frac = railing_fraction(vals, DEFAULT_PARAM_BOUNDS["Dstar"], rel_tol=0.01)
    assert frac["railed_frac"] == pytest.approx(4.0 / 6.0)
    assert frac["railed_pct"] == pytest.approx(100.0 * 4.0 / 6.0)

    ci_a = bootstrap_railing_ci(cls["railed_mask"], n_boot=500, seed=42)
    ci_b = bootstrap_railing_ci(cls["railed_mask"], n_boot=500, seed=42)
    assert ci_a == ci_b
    assert ci_a.lo <= ci_a.point <= ci_a.hi

    est_mat = np.column_stack(
        [
            np.full(100, 1.2e-3),
            np.concatenate([np.full(30, 0.20), np.full(70, 0.04)]),
            np.full(100, 0.20),
        ]
    )
    summary = evaluate_parameter_railing(est_mat, n_boot=300, seed=42)
    assert summary["D"]["railed_pct"] == pytest.approx(0.0)
    assert summary["Dstar"]["railed_pct"] == pytest.approx(30.0)
    assert summary["Dstar"]["upper_pct"] == pytest.approx(30.0)
    assert summary["any"]["railed_pct"] == pytest.approx(30.0)


def test_baselines_and_benchmark_suite_end_to_end(tmp_path: Path) -> None:
    cohort = generate_cohort(protocol="osipi", organ="liver", n_voxels=120, snr=35.0, seed=42)
    suite = BenchmarkSuite(protocol="osipi", snr=35.0, alpha=0.10, n_boot=200, seed=42)

    # 1. Segmented baseline
    seg = SegmentedEstimator(protocol="osipi", uncertainty="asymptotic")
    q_seg = seg.predict_quantiles(cohort.signals)
    assert q_seg.shape == (120, 3, len(DEFAULT_QUANTILE_LEVELS))
    rep_seg = suite.evaluate(cohort, q_pred=q_seg, method="Segmented")
    assert isinstance(rep_seg, BenchmarkReport)
    assert "Dstar" in rep_seg and "marginal_coverage" in rep_seg["Dstar"]
    assert len(rep_seg.to_dataframe()) == 3
    assert "Segmented" in rep_seg.format_table()

    # 2. NLLS baseline (verifying honest vs floored SD conventions)
    nlls_honest = NLLSEstimator(protocol="tcga_lihc", sd_convention="honest", max_nfev=120)
    nlls_floored = NLLSEstimator(protocol="tcga_lihc", sd_convention="floored", max_nfev=120)
    sparse_cohort = generate_cohort(protocol="tcga_lihc", organ="liver", n_voxels=40, snr=20.0, seed=12)
    res_h = nlls_honest.solve(sparse_cohort.signals)
    res_f = nlls_floored.solve(sparse_cohort.signals)
    assert res_h.params.shape == (40, 3)
    assert np.any(res_h.dstar_railed)
    # Honest convention assigns wider D* uncertainty than floored convention on railed fits
    assert float(np.mean(res_h.sigma[:, 1])) >= float(np.mean(res_f.sigma[:, 1]))

    # 3. Standard MLP baseline
    mlp = MLPEstimator(protocol="osipi", hidden_dims=(32, 32), epochs=20, seed=42)
    mlp.fit(organ="liver", snr=35.0, n_synthetic=600)
    pt_mlp = mlp.predict(cohort.signals)
    sd_mlp = mlp.predict_std(cohort.signals)
    assert pt_mlp.shape == (120, 3)
    assert np.all(sd_mlp > 0.0)

    # 4. Side-by-side comparison across input formats & run_baselines
    lo_mlp, hi_mlp = mlp.predict_intervals(cohort.signals, alpha=0.10)
    comp_df = suite.compare(
        cohort,
        {
            "Segmented": {"q_pred": q_seg},
            "MLP_Intervals": {"point_pred": pt_mlp, "intervals": (lo_mlp, hi_mlp)},
            "MLP_Gaussian": {"point_pred": pt_mlp, "std_pred": sd_mlp},
        },
    )
    assert isinstance(comp_df, pd.DataFrame)
    assert set(comp_df["method"]) == {"Segmented", "MLP_Intervals", "MLP_Gaussian"}
    assert len(comp_df) == 9

    base_df = suite.run_baselines(cohort, include=("segmented",))
    assert list(base_df["parameter"]) == ["D", "Dstar", "f"]

    # 5. Evaluate directly on a 2D spatial DRO
    dro = generate_dro(shape=(16, 16), organ="liver", protocol="osipi", snr=35.0, seed=42)
    dro_rep = suite.evaluate(dro, estimator=seg, method="SegmentedDRO")
    assert dro_rep["n_voxels"] == int(np.sum(dro.foreground_mask))

    # 6. Visualization utilities save clean figures
    cal_path = tmp_path / "calibration.png"
    ter_path = tmp_path / "tercile.png"
    fig_c, axes_c = plot_calibration_curves(
        cohort,
        {"Segmented": q_seg, "MLP": (pt_mlp, sd_mlp)},
        save_path=cal_path,
    )
    fig_t, ax_t = plot_tercile_coverage(comp_df, parameter="Dstar", save_path=ter_path)
    assert cal_path.exists() and cal_path.stat().st_size > 1000
    assert ter_path.exists() and ter_path.stat().st_size > 1000
    assert len(axes_c) == 3
    assert ax_t is not None
    plt.close(fig_c)
    plt.close(fig_t)
