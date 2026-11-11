"""Unit and property tests for clinical threshold decision VoI and observable domain-shift checks."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import stats

from ivimbench import generate_cohort
from ivimbench.decision import (
    DecisionVoIReport,
    DomainShiftReport,
    check_domain_shift,
    compare_decision_calibration,
    evaluate_decision_voi,
    infer_exceedance_probability,
)


def test_infer_exceedance_probability_monotonicity_and_asymmetry() -> None:
    lo = np.array([0.8, 0.8])
    pt = np.array([1.0, 1.0])
    hi = np.array([1.2, 1.8])  # Second interval has wider upper tail

    # At tau == point, split-normal CDF equals 0.5 regardless of asymmetry
    p_eq, loc, _ = infer_exceedance_probability(lo, hi, tau=1.0, point=pt, alpha=0.10, direction="less")
    np.testing.assert_allclose(p_eq, [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(loc, pt)

    # Above point (tau = 1.3), wider right tail gives higher probability of exceeding tau (direction='greater')
    p_gt, _, _ = infer_exceedance_probability(lo, hi, tau=1.3, point=pt, alpha=0.10, direction="greater")
    assert p_gt[1] > p_gt[0]

    # Monotonicity in tau for direction='less'
    p_t1, _, _ = infer_exceedance_probability(lo, hi, tau=0.9, point=pt, direction="less")
    p_t2, _, _ = infer_exceedance_probability(lo, hi, tau=1.4, point=pt, direction="less")
    assert np.all(p_t2 > p_t1)


def test_decision_voi_prices_calibration_gain_and_decision_calibration_gap() -> None:
    rng = np.random.default_rng(42)
    n = 4000
    # True predictive model: y | pt ~ N(pt, sigma_true^2)
    pt = rng.uniform(0.6, 1.8, size=n)
    sigma_true = 0.22
    y = pt + rng.normal(0.0, sigma_true, size=n)

    z90 = float(stats.norm.ppf(0.95))
    lo_cal = pt - z90 * sigma_true
    hi_cal = pt + z90 * sigma_true

    # Severely overconfident (10x too narrow) miscalibrated intervals around identical point estimates
    lo_overconf = pt - 0.10 * z90 * sigma_true
    hi_overconf = pt + 0.10 * z90 * sigma_true

    taus = np.linspace(0.8, 1.6, 7)

    rep_overconf = evaluate_decision_voi(
        y,
        lo_overconf,
        hi_overconf,
        point=pt,
        taus=taus,
        direction="less",
        cost_ratio=0.15,
        alpha=0.10,
        calibrated_intervals=(lo_cal, hi_cal),
        seed=42,
    )
    rep_cal = evaluate_decision_voi(
        y,
        lo_cal,
        hi_cal,
        point=pt,
        taus=taus,
        direction="less",
        cost_ratio=0.15,
        alpha=0.10,
        calibrated_intervals=(lo_cal, hi_cal),
        seed=42,
    )
    assert isinstance(rep_overconf, DecisionVoIReport)

    # 1. Well-calibrated intervals achieve near-zero decision-calibration gap and zero VoI shortfall
    assert float(rep_cal.summary["mean_decision_calibration_gap"]) < 0.01
    assert float(rep_cal.summary["mean_second_order_voi"]) == pytest.approx(0.0, abs=1e-12)

    # 2. Overconfident intervals severely underestimate their own realized regret (large decision-cal gap)
    #    and incur strictly higher realized regret under asymmetric false-negative cost (positive 2nd-order VoI)
    assert float(rep_overconf.summary["mean_decision_calibration_gap"]) > 0.04
    assert float(rep_overconf.summary["mean_expected_regret"]) > float(rep_cal.summary["mean_expected_regret"])
    assert float(rep_overconf.summary["mean_second_order_voi"]) > 0.008

    # 3. Side-by-side multi-method comparison & table exports
    comp_df = compare_decision_calibration(
        y,
        {"Calibrated": (lo_cal, hi_cal), "Overconfident": (lo_overconf, hi_overconf)},
        point_preds={"Calibrated": pt, "Overconfident": pt},
        taus=taus,
        cost_ratio=0.15,
        seed=42,
    )
    assert list(comp_df["method"]) == ["Calibrated", "Overconfident"]
    assert comp_df.loc[1, "mean_expected_regret"] > comp_df.loc[0, "mean_expected_regret"]

    assert "| tau" in rep_overconf.to_markdown()
    assert "\\begin{table}" in rep_overconf.to_latex()
    assert len(json.loads(rep_overconf.to_json())["records"]) == 7


def test_observable_domain_shift_check_null_vs_shifted_cohorts() -> None:
    ref_a = generate_cohort(protocol="osipi", organ="liver", n_voxels=300, snr=30.0, seed=10)
    ref_b = generate_cohort(protocol="osipi", organ="liver", n_voxels=300, snr=30.0, seed=20)

    # 1. Exchangeable draws from identical protocol/organ/SNR/family are NOT flagged stale
    null_rep = check_domain_shift(ref_a.signals, ref_b.signals, bvalues=ref_a.bvalues, n_perm=200, seed=42)
    assert isinstance(null_rep, DomainShiftReport)
    assert null_rep.is_stale is False
    assert null_rep.staleness_reasons == ()
    assert null_rep.residual_rms_ratio == pytest.approx(1.0, abs=0.15)

    # Determinism of PCG64 permutation p-values
    null_rep_2 = check_domain_shift(ref_a.signals, ref_b.signals, bvalues=ref_a.bvalues, n_perm=200, seed=42)
    assert null_rep.to_dict() == null_rep_2.to_dict()

    # 2. Organ/gain shift triggers signal_space_shift
    breast_tgt = generate_cohort(protocol="osipi", organ="brain", n_voxels=300, snr=30.0, seed=30)
    sig_shift = check_domain_shift(
        ref_a.signals, 1.35 * breast_tgt.signals, bvalues=ref_a.bvalues, n_perm=150, seed=42
    )
    assert sig_shift.is_stale is True
    assert "signal_space_shift" in sig_shift.staleness_reasons

    # 3. Severe SNR degradation + tri-exponential mismatch triggers residual-space shift
    noisy_tgt = generate_cohort(
        protocol="osipi", organ="liver", n_voxels=300, snr=10.0, family="triexp", seed=40
    )
    res_shift = check_domain_shift(
        ref_a.signals, noisy_tgt.signals, bvalues=ref_a.bvalues, n_perm=150, seed=42
    )
    assert res_shift.is_stale is True
    assert any("residual" in r for r in res_shift.staleness_reasons)
    assert res_shift.residual_rms_ratio > 1.8
