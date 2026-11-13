"""Deterministic unit tests for v1.3.0 anomalous diffusion walls, closed-loop twin RT, neural baselines, and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ivimbench.baselines.neural as neural_mod
from ivimbench import (
    AnomalousCRLBResult,
    ClosedLoopComparisonReport,
    ClosedLoopResult,
    HeteroscedasticQuantileMLP,
    IVIMNetBaseline,
    ParameterizedDoseGeometryTwin,
    RecoveryWallReport,
    SyntheticOrganTwin,
    SyntheticTwinLoop,
    __version__,
    anomalous_signal_jacobian,
    compare_closed_loop_policies,
    compute_anomalous_crlb,
    ctrw_stretched_signal,
    fit_anomalous_nlls,
    fractional_ivim_signal,
    list_estimators,
    map_recovery_collapse_wall,
    run_estimator,
    simulate_anomalous_signals,
)
from ivimbench.cli import main as cli_main
from ivimbench.metrics.coverage import empirical_coverage
from ivimbench.phantoms.synthetic import generate_cohort, ivim_biexp


def test_version_is_current() -> None:
    assert __version__ == "1.6.0"


def test_anomalous_jacobian_central_differences_and_adc_reduction() -> None:
    bvalues = np.array([0.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 800.0, 1200.0])

    # 1. Exact reduction to mono-exponential ADC at alpha = 1.0
    for s0_val, d_val in ((1.0, 0.8e-3), (1.35, 1.4e-3), (0.9, 2.2e-3)):
        s_ctrw = ctrw_stretched_signal(bvalues, D=d_val, alpha=1.0, S0=s0_val)
        s_adc = s0_val * np.exp(-bvalues * d_val)
        np.testing.assert_allclose(s_ctrw, s_adc, rtol=0.0, atol=1.0e-15)

    # 2. Exact reduction of two-compartment fractional IVIM to canonical bi-exponential at alpha=beta=1.0
    s_frac = fractional_ivim_signal(
        bvalues, D=1.1e-3, Dstar=35.0e-3, f=0.22, alpha=1.0, beta=1.0, S0=1.2
    )
    s_biexp = ivim_biexp(bvalues, D=1.1e-3, Dstar=35.0e-3, f=0.22, s0=1.2)
    np.testing.assert_allclose(s_frac, s_biexp, rtol=0.0, atol=1.0e-15)

    # 3. Analytical Jacobian dS/d(S0, D, alpha) vs central finite differences (< 1e-7 max error)
    eps_s0 = 1.0e-7
    eps_d = 1.0e-9
    eps_a = 1.0e-7
    for s0_val, d_val, a_val in (
        (1.0, 1.0e-3, 0.65),
        (1.25, 0.75e-3, 0.45),
        (0.85, 1.85e-3, 0.92),
        (1.0, 1.2e-3, 1.0 - 1.0e-5),
    ):
        jac = anomalous_signal_jacobian(bvalues, D=d_val, alpha=a_val, S0=s0_val)
        assert jac.shape == (bvalues.size, 3)
        # At b = 0: dS/dS0 == 1, dS/dD == 0, dS/dalpha == 0
        assert jac[0, 0] == pytest.approx(1.0, abs=1.0e-15)
        assert jac[0, 1] == 0.0
        assert jac[0, 2] == 0.0

        fd_s0 = (
            ctrw_stretched_signal(bvalues, D=d_val, alpha=a_val, S0=s0_val + eps_s0)
            - ctrw_stretched_signal(bvalues, D=d_val, alpha=a_val, S0=s0_val - eps_s0)
        ) / (2.0 * eps_s0)
        fd_d = (
            ctrw_stretched_signal(bvalues, D=d_val + eps_d, alpha=a_val, S0=s0_val)
            - ctrw_stretched_signal(bvalues, D=d_val - eps_d, alpha=a_val, S0=s0_val)
        ) / (2.0 * eps_d)
        fd_alpha = (
            ctrw_stretched_signal(bvalues, D=d_val, alpha=a_val + eps_a, S0=s0_val)
            - ctrw_stretched_signal(bvalues, D=d_val, alpha=a_val - eps_a, S0=s0_val)
        ) / (2.0 * eps_a)

        assert float(np.max(np.abs(jac[:, 0] - fd_s0))) < 1.0e-7
        assert float(np.max(np.abs(jac[:, 1] - fd_d))) * 1.0e-3 < 1.0e-7
        assert float(np.max(np.abs(jac[:, 2] - fd_alpha))) < 1.0e-7

    # 4. Exact Rician and non-central chi magnitude noise simulation + PCG64 determinism
    sig_ric_1 = simulate_anomalous_signals(
        bvalues, D=1.1e-3, alpha=0.75, snr=25.0, noise_type="rician", seed=42
    )
    sig_ric_2 = simulate_anomalous_signals(
        bvalues, D=1.1e-3, alpha=0.75, snr=25.0, noise_type="rician", seed=42
    )
    sig_ncchi_1coil = simulate_anomalous_signals(
        bvalues, D=1.1e-3, alpha=0.75, snr=25.0, noise_type="noncentral_chi", n_coils=1, seed=42
    )
    sig_ncchi_4coil = simulate_anomalous_signals(
        bvalues,
        D=1.1e-3,
        Dstar=35.0e-3,
        f=0.20,
        alpha=0.75,
        snr=25.0,
        noise_type="noncentral_chi",
        n_coils=4,
        seed=42,
    )
    np.testing.assert_array_equal(sig_ric_1, sig_ric_2)
    np.testing.assert_array_equal(sig_ric_1, sig_ncchi_1coil)
    assert np.all(sig_ncchi_4coil > 0.0)


def test_anomalous_crlb_snr_monotonicity_and_recovery_collapse_wall() -> None:
    snrs = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    crlb_vec = compute_anomalous_crlb("osipi", D=1.1e-3, alpha=0.75, snr=snrs, S0=1.0)
    assert isinstance(crlb_vec, AnomalousCRLBResult)
    assert crlb_vec.fim.shape == (5, 3, 3)
    assert crlb_vec.std.shape == (5, 3)
    assert np.all(crlb_vec.identifiable_mask)

    # Strictly monotonic improvement (decrease) of (S0, D, alpha) CRLB std & CV as SNR increases
    assert np.all(np.diff(crlb_vec.D_std) < 0.0)
    assert np.all(np.diff(crlb_vec.alpha_std) < 0.0)
    assert np.all(np.diff(crlb_vec.D_cv) < 0.0)
    assert np.all(np.diff(crlb_vec.alpha_cv) < 0.0)

    # Doubling SNR halves CRLB standard error identically
    c20 = compute_anomalous_crlb("osipi", D=1.1e-3, alpha=0.75, snr=20.0)
    c40 = compute_anomalous_crlb("osipi", D=1.1e-3, alpha=0.75, snr=40.0)
    np.testing.assert_allclose(c40.std, 0.5 * c20.std, rtol=1.0e-12)

    # Noiseless bounded-NLLS round-trip recovery of (S0, D, alpha)
    clean = ctrw_stretched_signal("lattice_22b", D=1.25e-3, alpha=0.68, S0=1.15)
    fit_clean = fit_anomalous_nlls(clean[None, :], "lattice_22b")
    assert fit_clean.D[0] == pytest.approx(1.25e-3, rel=1.0e-7)
    assert fit_clean.alpha[0] == pytest.approx(0.68, rel=1.0e-7)
    assert not bool(fit_clean.railed_alpha[0])

    # Recovery collapse wall across low-bmax vs wide-bmax schedules and SNR grid
    schedules = {
        "narrow_b250": [0.0, 25.0, 75.0, 150.0, 250.0],
        "osipi": "osipi",
        "wide_b1500": [0.0, 20.0, 50.0, 150.0, 400.0, 800.0, 1500.0],
    }
    wall_1 = map_recovery_collapse_wall(
        b_schedules=schedules,
        snr_grid=[10.0, 30.0, 80.0],
        alpha_grid=[0.65, 0.90],
        seed=42,
        n_replicates=32,
        n_boot=200,
    )
    wall_2 = map_recovery_collapse_wall(
        b_schedules=schedules,
        snr_grid=[10.0, 30.0, 80.0],
        alpha_grid=[0.65, 0.90],
        seed=42,
        n_replicates=32,
        n_boot=200,
    )
    assert isinstance(wall_1, RecoveryWallReport)
    assert len(wall_1.dataframe) == 3 * 3 * 2
    pd.testing.assert_frame_equal(wall_1.dataframe, wall_2.dataframe)

    # Bootstrap 95% CIs properly bracket empirical railing rate & bias
    df = wall_1.dataframe
    assert np.all(df["alpha_railed_ci_lo"] <= df["alpha_railed_rate"] + 1.0e-12)
    assert np.all(df["alpha_railed_rate"] <= df["alpha_railed_ci_hi"] + 1.0e-12)

    # Narrow b_max=250 at SNR=10 collapses, while wide b_max=1500 at SNR=80 recovers cleanly
    low_cell = df[(df["schedule"] == "narrow_b250") & (df["snr"] == 10.0) & (df["alpha"] == 0.65)].iloc[0]
    hi_cell = df[(df["schedule"] == "wide_b1500") & (df["snr"] == 80.0) & (df["alpha"] == 0.65)].iloc[0]
    assert bool(low_cell["is_collapsed"]) is True
    assert bool(hi_cell["is_collapsed"]) is False
    assert float(hi_cell["crlb_cv_alpha"]) < float(low_cell["crlb_cv_alpha"])
    assert "| narrow_b250" in wall_1.to_markdown()


def test_closed_loop_synthetic_organ_and_parameterized_dose_geometry_twins() -> None:
    # 1. SyntheticOrganTwin deterministic multi-fraction comparison (uncalibrated NLLS vs conformal)
    organ_twin = SyntheticOrganTwin(shape=(18, 18), organ="pancreas", seed=42)
    comp_1 = compare_closed_loop_policies(
        substrate=organ_twin,
        protocol="osipi",
        estimator="nlls",
        n_fractions=4,
        snr=25.0,
        seed=42,
    )
    comp_2 = compare_closed_loop_policies(
        substrate=organ_twin,
        protocol="osipi",
        estimator="nlls",
        n_fractions=4,
        snr=25.0,
        seed=42,
    )
    assert isinstance(comp_1, ClosedLoopComparisonReport)
    pd.testing.assert_frame_equal(comp_1.policy_df, comp_2.policy_df)

    uncal = comp_1.runs["uncalibrated_nlls"].summary
    conf = comp_1.runs["conformal_trust_gated"].summary
    # Conformal + trust gate achieves higher f-interval coverage, lower false escalations,
    # lower decision regret, and positive OAR dose sparing with zero prescribed drift on held voxels
    assert float(conf["mean_f_coverage"]) > float(uncal["mean_f_coverage"])
    assert float(conf["mean_false_escalation_rate"]) <= float(uncal["mean_false_escalation_rate"])
    assert float(conf["mean_decision_regret"]) <= float(uncal["mean_decision_regret"])
    assert float(comp_1.summary["conformal_oar_sparing_vs_uncalibrated_gy"]) >= 0.0
    assert float(conf["held_prescribed_drift_gy"]) == 0.0
    assert float(conf["tumor_control_proxy"]) > 0.50

    # 2. User-supplied 3D ParameterizedDoseGeometryTwin (no patient blobs required)
    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, 6),
        np.linspace(-1.0, 1.0, 12),
        np.linspace(-1.0, 1.0, 12),
        indexing="ij",
    )
    r3 = np.sqrt(xx**2 + yy**2 + zz**2)
    organ_mask_3d = r3 <= 0.90
    custom_dose_3d = 2.4 * np.exp(-((r3 / 0.48) ** 3.0)) * organ_mask_3d

    dose_twin = ParameterizedDoseGeometryTwin(
        dose_grid=custom_dose_3d,
        organ_mask=organ_mask_3d,
        seed=17,
    )
    loop_dose = SyntheticTwinLoop(
        substrate=dose_twin,
        protocol="osipi",
        estimator="crlb",
        n_fractions=4,
        snr=30.0,
        interval_mode="conformal",
        seed=17,
    )
    res_dose = loop_dose.run()
    assert isinstance(res_dose, ClosedLoopResult)
    assert len(res_dose.fractions_df) == 4
    shells = loop_dose.substrate.summarize_isodose_perfusion_drop(n_shells=3)  # type: ignore[attr-defined]
    assert len(shells) == 3
    # Higher delivered dose shells exhibit substantial radiation-induced perfusion drop
    assert shells[-1]["mean_delivered_dose_gy"] > shells[0]["mean_delivered_dose_gy"]
    assert shells[-1]["relative_perfusion_drop"] > 0.05

    # 3. Mid-treatment severe SNR drop triggers observable shift monitor in trust gate
    loop_shift = SyntheticTwinLoop(
        substrate="organ",
        protocol="osipi",
        estimator="crlb",
        n_fractions=3,
        snr=35.0,
        snr_schedule=(35.0, 7.0, 7.0),
        interval_mode="conformal",
        enable_trust_gate=True,
        seed=42,
    )
    res_shift = loop_shift.run()
    assert int(res_shift.summary["shift_triggered_fractions"]) >= 1


def test_neural_baselines_missing_torch_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "ivimnet" in list_estimators()
    assert "quantile_mlp" in list_estimators()

    monkeypatch.setattr(neural_mod, "_HAS_TORCH", False)
    with pytest.raises(ImportError, match=r"Install ivimbench\[torch\].*IVIMNetBaseline"):
        IVIMNetBaseline(protocol="osipi")
    with pytest.raises(ImportError, match=r"Install ivimbench\[torch\].*HeteroscedasticQuantileMLP"):
        HeteroscedasticQuantileMLP(protocol="osipi")

    cohort = generate_cohort(protocol="osipi", organ="liver", n_voxels=16, snr=30.0, seed=42)
    with pytest.raises(ImportError, match=r"Install ivimbench\[torch\]"):
        run_estimator("ivimnet", cohort.signals, cohort.bvalues)
    with pytest.raises(ImportError, match=r"Install ivimbench\[torch\]"):
        run_estimator("quantile_mlp", cohort.signals, cohort.bvalues)


def test_neural_baselines_quantile_and_conformal_contracts_when_torch_present() -> None:
    pytest.importorskip("torch")
    cohort = generate_cohort(protocol="osipi", organ="liver", n_voxels=260, snr=30.0, seed=42)
    q_levels = (0.05, 0.25, 0.50, 0.75, 0.95)

    # 1. Self-supervised physics-informed IVIMNetBaseline (Rician & Gaussian modes)
    pinn = IVIMNetBaseline(
        protocol=cohort.bvalues,
        hidden=32,
        loss_type="rician",
        epochs=15,
        batch_size=64,
        n_mc=20,
        seed=42,
    )
    pinn.fit(cohort.signals[:160])
    out_pinn = pinn(cohort.signals[160:200], cohort.bvalues, q_levels, alpha=0.10)
    out_pinn_repeat = pinn(cohort.signals[160:200], cohort.bvalues, q_levels, alpha=0.10)
    np.testing.assert_allclose(out_pinn["quantiles"], out_pinn_repeat["quantiles"])
    assert out_pinn["quantiles"].shape == (40, 3, 5)
    assert np.all(np.diff(out_pinn["quantiles"], axis=-1) >= -1.0e-12)
    assert np.all((out_pinn["D"] >= 1.0e-4) & (out_pinn["D"] <= 4.0e-3))
    assert np.all((out_pinn["f"] >= 0.0) & (out_pinn["f"] <= 0.70))

    pinn_gauss = IVIMNetBaseline(
        protocol=cohort.bvalues, hidden=24, loss_type="gaussian", epochs=8, n_mc=12, seed=42
    )
    pinn_gauss.fit(cohort.signals[:120])
    assert pinn_gauss.predict(cohort.signals[120:136]).shape == (16, 3)

    # 2. Direct pinball-loss HeteroscedasticQuantileMLP + SplitConformalQuantile conformalization
    qmlp = HeteroscedasticQuantileMLP(
        protocol=cohort.bvalues,
        alpha=0.10,
        hidden=32,
        epochs=20,
        batch_size=64,
        seed=42,
    )
    qmlp.fit(cohort.signals[:140], cohort.params[:140])
    q_lo, q_med, q_hi = qmlp.predict_triplet(cohort.signals[140:200])
    assert q_lo.shape == (60, 3)
    # Non-crossing guarantee: q_{alpha/2} < q_{0.5} < q_{1 - alpha/2}
    assert np.all(q_lo < q_med)
    assert np.all(q_med < q_hi)

    out_qmlp = qmlp(cohort.signals[140:200], cohort.bvalues, q_levels, alpha=0.10)
    assert np.all(np.diff(out_qmlp["quantiles"], axis=-1) > 0.0)
    np.testing.assert_allclose(out_qmlp["quantiles"][:, :, 0], q_lo, rtol=1.0e-6)
    np.testing.assert_allclose(out_qmlp["quantiles"][:, :, 2], q_med, rtol=1.0e-6)
    np.testing.assert_allclose(out_qmlp["quantiles"][:, :, 4], q_hi, rtol=1.0e-6)

    # Downstream conformalization achieves healthy marginal coverage on test fold
    cal_lo, cal_hi = qmlp.conformalize(
        cohort.signals[140:200],
        cohort.params[140:200],
        cohort.signals[200:],
        alpha=0.10,
    )
    cov_d = empirical_coverage(cohort.params[200:, 0], cal_lo[:, 0], cal_hi[:, 0])
    assert cov_d >= 0.78

    # Registry execution for both neural baselines
    reg_pinn = run_estimator("ivimnet", cohort.signals[200:220], cohort.bvalues, epochs=6, n_synthetic=128)
    reg_qmlp = run_estimator("quantile_mlp", cohort.signals[200:220], cohort.bvalues, epochs=8, n_synthetic=160)
    assert reg_pinn.point.shape == (20, 3) and reg_pinn.has_intervals
    assert reg_qmlp.point.shape == (20, 3) and reg_qmlp.has_intervals


def test_cli_anomalous_wall_and_closed_loop_subcommands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wall_dir = tmp_path / "wall_out"
    rc_wall = cli_main(
        [
            "anomalous-wall",
            "--protocol",
            "osipi",
            "--snr-min",
            "10",
            "--snr-max",
            "80",
            "--n-snr",
            "3",
            "--alphas",
            "0.6,0.9",
            "--n-replicates",
            "16",
            "--n-boot",
            "100",
            "--out",
            str(wall_dir),
        ]
    )
    out_wall = capsys.readouterr().out
    assert rc_wall == 0
    assert "Anomalous Fractional-Order Diffusion Recovery Wall" in out_wall
    assert "| osipi" in out_wall
    assert (wall_dir / "anomalous_wall.md").exists()
    assert (wall_dir / "anomalous_wall.json").exists()
    assert (wall_dir / "anomalous_wall.tex").exists()

    loop_dir = tmp_path / "loop_out"
    rc_loop = cli_main(
        [
            "closed-loop",
            "--fractions",
            "5",
            "--snr",
            "30",
            "--estimator",
            "crlb",
            "--out",
            str(loop_dir),
        ]
    )
    out_loop = capsys.readouterr().out
    assert rc_loop == 0
    assert "Closed-Loop Synthetic-Twin Adaptive RT" in out_loop
    assert "uncalibrated_nlls" in out_loop and "conformal_trust_gated" in out_loop
    assert (loop_dir / "closed_loop_summary.md").exists()
    assert (loop_dir / "closed_loop_summary.json").exists()
    assert (loop_dir / "closed_loop_summary.tex").exists()

    rc_loop_json = cli_main(
        ["closed-loop", "--fractions", "3", "--snr", "30", "--estimator", "crlb", "--json"]
    )
    loop_json = json.loads(capsys.readouterr().out)
    assert rc_loop_json == 0
    assert loop_json["summary"]["n_fractions"] == 3
