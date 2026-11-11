"""Property and unit tests for Fisher Information Matrix, CRLB scaling, and b-schedule design."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ivimbench.acquisition import (
    CRLBResult,
    OptimizedScheduleResult,
    SchemeComparisonReport,
    SchemeEvaluator,
    compute_crlb,
    crlb_optimality_criterion,
    ivim_fisher_matrix,
    ivim_signal_jacobian,
    optimize_b_schedule,
    resolve_schedule_and_nex,
)
from ivimbench.phantoms.synthetic import ivim_biexp, sample_organ_params


def test_fisher_matrix_psd_and_jacobian_finite_difference() -> None:
    b = np.array([0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0])
    theta = np.array([1.2e-3, 35.0e-3, 0.22])
    s0 = 1.4

    jac = ivim_signal_jacobian(b, theta, s0=s0, include_s0=True)
    assert jac.shape == (8, 4)

    # Central finite-difference verification across (D, D*, f, S0)
    eps = 1e-8
    for j in range(3):
        dt = np.zeros(3)
        dt[j] = eps
        sp = ivim_biexp(b, *(theta + dt), s0=s0)
        sm = ivim_biexp(b, *(theta - dt), s0=s0)
        np.testing.assert_allclose(jac[:, j], (sp - sm) / (2.0 * eps), rtol=1e-5, atol=1e-7)

    sp_s0 = ivim_biexp(b, *theta, s0=s0 + eps)
    sm_s0 = ivim_biexp(b, *theta, s0=s0 - eps)
    np.testing.assert_allclose(jac[:, 3], (sp_s0 - sm_s0) / (2.0 * eps), rtol=1e-6)

    # Fisher Information Matrix is symmetric and positive definite on an identifiable 8-b schedule
    fim = ivim_fisher_matrix(b, theta, snr=30.0, nex=2.0, s0=s0, include_s0=True)
    np.testing.assert_allclose(fim, fim.T, rtol=1e-12)
    eigvals = np.linalg.eigvalsh(fim)
    assert np.all(eigvals > 0.0)


def test_crlb_positivity_snr_nex_scaling_and_param_order_invariance() -> None:
    theta = sample_organ_params("liver", n_voxels=50, seed=42)

    res_base = compute_crlb("osipi", theta, snr=20.0, nex=1.0, alpha=0.10)
    assert isinstance(res_base, CRLBResult)
    assert res_base.std.shape == (50, 3)
    assert np.all(res_base.variance > 0.0)
    assert np.all(res_base.std > 0.0)
    assert np.all(res_base.interval_width > 0.0)
    assert np.all(res_base.condition_number >= 1.0)

    # 1. Exact 1 / SNR scaling of CRLB standard deviation (1 / SNR^2 for variance)
    res_snr40 = compute_crlb("osipi", theta, snr=40.0, nex=1.0, alpha=0.10)
    np.testing.assert_allclose(res_snr40.std, 0.5 * res_base.std, rtol=1e-12)
    np.testing.assert_allclose(res_snr40.variance, 0.25 * res_base.variance, rtol=1e-12)

    # 2. Exact 1 / sqrt(NEX) scaling when signal averages are quadrupled
    res_nex4 = compute_crlb("osipi", theta, snr=20.0, nex=4.0, alpha=0.10)
    np.testing.assert_allclose(res_nex4.std, 0.5 * res_base.std, rtol=1e-12)
    assert res_nex4.total_nex == pytest.approx(4.0 * res_base.total_nex)
    assert res_nex4.scan_minutes == pytest.approx(4.0 * res_base.scan_minutes)

    # 3. Equal scan-time / NEX budget normalization
    b_4, nex_4, _, min_4 = resolve_schedule_and_nex("tcga_lihc", scan_time_minutes=2.4)
    b_12, nex_12, _, min_12 = resolve_schedule_and_nex("osipi", scan_time_minutes=2.4)
    assert min_4 == pytest.approx(2.4)
    assert min_12 == pytest.approx(2.4)
    assert np.sum(nex_4) == pytest.approx(np.sum(nex_12))
    assert b_4.size == 4 and b_12.size == 12

    # 4. Permutation invariance for (f, D, D_star) parameter ordering and dict input
    theta_fds = np.column_stack([theta[:, 2], theta[:, 0], theta[:, 1]])
    res_fds = compute_crlb(
        "osipi",
        theta_fds,
        snr=20.0,
        nex=1.0,
        param_order=("f", "D", "D_star"),
    )
    np.testing.assert_allclose(res_fds.std[:, 0], res_base.std[:, 2], rtol=1e-12)
    np.testing.assert_allclose(res_fds.std[:, 1], res_base.std[:, 0], rtol=1e-12)
    np.testing.assert_allclose(res_fds.std[:, 2], res_base.std[:, 1], rtol=1e-12)

    res_dict = compute_crlb(
        "osipi",
        {"f": theta[:, 2], "D": theta[:, 0], "D_star": theta[:, 1]},
        snr=20.0,
    )
    np.testing.assert_allclose(res_dict.std, res_base.std, rtol=1e-12)


def test_optimize_b_schedule_and_scheme_evaluator_comparison() -> None:
    theta_prior = sample_organ_params("liver", n_voxels=120, seed=42)

    # Optimize 8-b calibration-aware and CRLB-optimal schedules
    opt_cal = optimize_b_schedule(
        n_b=8,
        organ="liver",
        snr=30.0,
        objective="calibration_aware",
        total_nex=12,
        n_prior_samples=96,
        n_restarts=2,
        max_iter=45,
        seed=42,
    )
    assert isinstance(opt_cal, OptimizedScheduleResult)
    assert opt_cal.bvalues.shape == (8,)
    assert opt_cal.bvalues[0] == pytest.approx(0.0)
    assert np.all(np.diff(opt_cal.bvalues) >= 0.0)
    assert int(np.sum(opt_cal.nex)) == 12
    assert opt_cal.objective_value <= opt_cal.initial_objective_value + 1e-6

    # Calibration-aware 8-b design dramatically beats sparse clinical 4-b schedule on CRLB criterion
    crit_sparse = crlb_optimality_criterion("acrin_6698", theta_prior, snr=30.0, criterion="calibration_aware")
    crit_opt = crlb_optimality_criterion(opt_cal.protocol, theta_prior, snr=30.0, criterion="calibration_aware")
    assert crit_opt < crit_sparse

    # SchemeEvaluator comparison across fixed clinical and optimized schedules
    evaluator = SchemeEvaluator(organ="liver", snr=30.0, alpha=0.10, n_voxels=80, estimator="crlb", seed=42)
    report = evaluator.compare(["osipi", "acrin_6698", opt_cal])
    assert isinstance(report, SchemeComparisonReport)
    df = report.summary_dataframe
    assert list(df["scheme"]) == ["osipi", "acrin_6698", opt_cal.key]
    for col in ("delta_sharp", "marginal_coverage", "tercile_cov_high", "delta_cond"):
        assert col in df.columns

    # Reference scheme (first entry = osipi) has delta_sharp == 0.0 by definition
    assert df.loc[df["scheme"] == "osipi", "delta_sharp"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    # Sparse 4-b acrin_6698 has strictly degraded sharpness (delta_sharp > 0) and worse high-D* coverage
    row_osipi = df.loc[df["scheme"] == "osipi"].iloc[0]
    row_acrin = df.loc[df["scheme"] == "acrin_6698"].iloc[0]
    assert row_acrin["delta_sharp"] > 0.20
    assert row_acrin["tercile_cov_high"] < row_osipi["tercile_cov_high"]

    assert "| osipi" in report.to_markdown()
    assert "\\begin{table}" in report.to_latex()
    assert len(json.loads(report.to_json())["summary"]) == 3
