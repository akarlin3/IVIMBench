"""Tests for Lattice continuity/round-trip self-checks, OSIPI fetcher, Caliper dual-SD conventions, CLI selfcheck, and optional torch MAF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import ivimbench.baselines.maf as maf_mod
from ivimbench import (
    BenchmarkSuite,
    Cohort,
    CRLBNormalEstimator,
    MAFPosteriorEstimator,
    NLLSEstimator,
    crlb_normal_intervals,
    fetch_osipi_dro,
    generate_cohort,
    list_estimators,
    load_osipi_dro,
    run_estimator,
    run_nlls_roundtrip_selfcheck,
    to_scorer_inputs,
    verify_continuity_limits,
)
from ivimbench.cli import main as cli_main
from ivimbench.metrics.coverage import (
    DEFAULT_QUANTILE_LEVELS,
    empirical_coverage,
    gaussian_calibration_curve,
    score_quantiles,
    tercile_stratified_coverage,
)
from ivimbench.osipi import OSIPI_DRO_MD5, OSIPI_SOURCE, provenance_record
from ivimbench.phantoms.noise import add_rician_noise
from ivimbench.phantoms.synthetic import ivim_biexp, make_cohort


def test_continuity_limits_exact_and_asymptotic_gamma() -> None:
    res = verify_continuity_limits(organ="generic", protocol="lattice_22b", n_voxels=2000, seed=42)
    assert res["passed"] is True

    # Exact-reduction families vanish to 0.0 within machine epsilon (<= 1e-15)
    for exact_family in ("dispersion_lognormal", "stretched", "triexp"):
        assert res[exact_family] <= 1.0e-15
        assert res["residuals"][exact_family] == 0.0
        assert res["checks"][exact_family]["passed"] is True

    # Asymptotic gamma velocity dispersion at k=1e8 is <= 1e-8 (~3.9e-9)
    gamma_err = res["dispersion_gamma"]
    assert 1.0e-10 < gamma_err <= 1.0e-8
    assert gamma_err == pytest.approx(3.9e-9, abs=2.5e-9)

    # PCG64 determinism across repeated invocations
    res_repeat = verify_continuity_limits(
        organ="generic", protocol="lattice_22b", n_voxels=2000, seed=42
    )
    assert res["residuals"] == res_repeat["residuals"]

    # Tightening gamma tolerance below observed residual triggers AssertionError when check=True
    with pytest.raises(AssertionError, match="dispersion_gamma"):
        verify_continuity_limits(
            organ="generic",
            protocol="lattice_22b",
            n_voxels=200,
            seed=42,
            gamma_k=10.0,
            gamma_tol=1.0e-8,
            check=True,
        )


def test_noiseless_biexp_nlls_roundtrip_recovery() -> None:
    rt = run_nlls_roundtrip_selfcheck(
        organ="generic",
        protocol="lattice_22b",
        n_voxels=60,
        seed=42,
        rel_tol=1.0e-9,
        check=True,
    )
    assert rt["passed"] is True
    assert rt["max_rel_error"] <= 7.99e-10
    assert rt["max_rel_error"] <= 1.0e-9
    for pname in ("D", "Dstar", "f"):
        assert rt["max_rel_per_param"][pname] <= 7.99e-10


def test_cohort_alias_and_to_scorer_inputs_contract() -> None:
    cohort = make_cohort(family="dispersion_gamma", n=80, bvalues="osipi", snr=30.0, seed=17)
    assert isinstance(cohort, Cohort)
    assert cohort.noise == "rician"
    assert cohort.extra_names == ("k",)
    assert cohort.extra is not None and cohort.extra.shape == (80, 1)

    est = CRLBNormalEstimator(protocol=cohort.bvalues, snr=30.0)
    q_pred = est.predict_quantiles(cohort.signals, DEFAULT_QUANTILE_LEVELS)

    packed = cohort.to_scorer_inputs(q_pred, DEFAULT_QUANTILE_LEVELS)
    packed_fn = to_scorer_inputs(cohort, q_pred, DEFAULT_QUANTILE_LEVELS)
    assert set(packed.keys()) == {"y_true", "q_pred", "q_levels", "param_names"}
    np.testing.assert_array_equal(packed["q_pred"], packed_fn["q_pred"])

    scored = score_quantiles(**packed)
    assert set(scored.keys()) == {"D", "Dstar", "f"}
    assert 0.0 <= scored["Dstar"]["marginal_coverage"] <= 1.0

    rep = BenchmarkSuite(protocol=cohort.bvalues, snr=30.0, n_boot=50).evaluate(**packed)
    assert not isinstance(rep, list)
    assert rep["Dstar"]["marginal_coverage"] == pytest.approx(scored["Dstar"]["marginal_coverage"])


def test_fetch_osipi_dro_offline_md5_and_provenance(tmp_path: Path) -> None:
    assert OSIPI_SOURCE["zenodo_doi"] == "10.5281/zenodo.14605039"
    assert len(OSIPI_DRO_MD5) == 32

    payload_bytes = b"synthetic-osipi-archive-payload-for-offline-md5-verification"
    expected_md5 = hashlib.md5(payload_bytes).hexdigest()

    calls: list[str] = []

    def _mock_downloader(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(payload_bytes)

    archive_path = fetch_osipi_dro(
        cache_dir=tmp_path,
        verify_md5=True,
        expected_md5=expected_md5,
        filename="mock_osipi.zip",
        fetched="2026-11-11T00:00:00Z",
        downloader=_mock_downloader,
    )
    assert archive_path.exists()
    assert len(calls) == 1

    # Second call hits local cache without invoking downloader
    cached_path = fetch_osipi_dro(
        cache_dir=tmp_path,
        verify_md5=True,
        expected_md5=expected_md5,
        filename="mock_osipi.zip",
        fetched="2026-11-11T00:00:00Z",
        downloader=_mock_downloader,
    )
    assert cached_path == archive_path
    assert len(calls) == 1

    prov_file = tmp_path / "mock_osipi.zip.provenance.json"
    prov = json.loads(prov_file.read_text(encoding="utf-8"))
    assert prov["md5"] == expected_md5
    assert prov["fetched"] == "2026-11-11T00:00:00Z"
    assert prov["source"]["zenodo_doi"] == "10.5281/zenodo.14605039"
    assert provenance_record(archive_path, md5=expected_md5)["md5"] == expected_md5

    # Corrupted MD5 raises ValueError and removes partial file
    with pytest.raises(ValueError, match="MD5 checksum mismatch"):
        fetch_osipi_dro(
            cache_dir=tmp_path,
            verify_md5=True,
            expected_md5="0" * 32,
            filename="corrupt_osipi.zip",
            downloader=_mock_downloader,
        )
    assert not (tmp_path / "corrupt_osipi.zip").exists()
    assert not (tmp_path / ".corrupt_osipi.zip.part").exists()

    # ExternalDRO .npz round-trip loader
    npz_path = tmp_path / "ext_dro.npz"
    cohort = generate_cohort(protocol="osipi_dro", organ="liver", n_voxels=16, snr=30.0, seed=42)
    np.savez(npz_path, params=cohort.params, signals=cohort.signals, bvalues=cohort.bvalues)
    ext = load_osipi_dro(npz_path)
    assert len(ext) == 16
    q_dummy = CRLBNormalEstimator(protocol=ext.bvalues, snr=30.0).predict_quantiles(ext.signals)
    assert set(ext.to_scorer_inputs(q_dummy).keys()) == {"y_true", "q_pred", "q_levels", "param_names"}


def test_dual_sd_convention_divergence_on_high_dstar_railed_cohort() -> None:
    # Synthesize a high-D* cohort where boundary railing at D*_max (0.20 mm^2/s) is frequent
    rng = np.random.Generator(np.random.PCG64(2026))
    n = 360
    d_true = rng.uniform(0.8e-3, 1.8e-3, size=n)
    dstar_true = rng.uniform(0.06, 0.195, size=n)
    f_true = rng.uniform(0.10, 0.30, size=n)
    y_true = np.column_stack([d_true, dstar_true, f_true])

    b_osipi = generate_cohort(protocol="osipi", n_voxels=4, seed=1).bvalues
    sig_clean = ivim_biexp(b_osipi, d_true, dstar_true, f_true)
    signals = add_rician_noise(sig_clean, snr=25.0, seed=rng)

    est_honest = CRLBNormalEstimator(protocol=b_osipi, snr=25.0, sd_convention="honest_crlb")
    est_floored = CRLBNormalEstimator(protocol=b_osipi, snr=25.0, sd_convention="floored_railed")

    fit_h = est_honest.solve(signals)
    fit_f = est_floored.solve(signals)
    # Identical fitted point estimates and railed flags, divergent SDs on railed voxels
    np.testing.assert_allclose(fit_h.params, fit_f.params)
    assert int(np.sum(fit_h.railed[:, 1])) >= 40
    assert np.all(fit_f.sigma_crlb[fit_f.railed[:, 1], 1] == pytest.approx(3.0e-3))
    assert float(np.mean(fit_h.sigma_crlb[fit_h.railed[:, 1], 1])) > 5.0 * 3.0e-3

    lo_h, hi_h = est_honest.predict_intervals(signals, alpha=0.10)
    lo_f, hi_f = est_floored.predict_intervals(signals, alpha=0.10)

    cov_honest = empirical_coverage(y_true[:, 1], lo_h[:, 1], hi_h[:, 1])
    cov_floored = empirical_coverage(y_true[:, 1], lo_f[:, 1], hi_f[:, 1])

    # Honest CRLB maintains healthy marginal coverage (~0.80-0.92), whereas floored_railed collapses
    assert cov_honest >= 0.78
    assert cov_floored < cov_honest - 0.15

    # Coverage.py sd_convention toggle reproduces the same collapse directly from honest intervals/SDs
    cov_toggle = empirical_coverage(
        y_true[:, 1],
        lo_h[:, 1],
        hi_h[:, 1],
        sd_convention="floored_railed",
        point_pred=fit_h.params[:, 1],
    )
    assert cov_toggle == pytest.approx(cov_floored)

    terc_h = tercile_stratified_coverage(y_true[:, 1], lo_h[:, 1], hi_h[:, 1], nominal=0.90)
    terc_f = tercile_stratified_coverage(
        y_true[:, 1],
        lo_h[:, 1],
        hi_h[:, 1],
        nominal=0.90,
        sd_convention="floored_railed",
        point_pred=fit_h.params[:, 1],
    )
    assert terc_f["high"].coverage < terc_h["high"].coverage - 0.20

    g_h = gaussian_calibration_curve(
        y_true[:, 1], fit_h.params[:, 1], fit_h.sigma_crlb[:, 1], sd_convention="honest_crlb"
    )
    g_f = gaussian_calibration_curve(
        y_true[:, 1], fit_h.params[:, 1], fit_h.sigma_crlb[:, 1], sd_convention="floored_railed"
    )
    assert float(g_f["ece"]) > float(g_h["ece"]) + 0.10

    # Standalone crlb_normal_intervals helper and NLLSEstimator also honor sd_convention
    iv_h = crlb_normal_intervals(b_osipi, fit_h.params, snr=25.0, sd_convention="honest_crlb")
    iv_f = crlb_normal_intervals(b_osipi, fit_h.params, snr=25.0, sd_convention="floored_railed")
    assert empirical_coverage(y_true[:, 1], iv_f[0][:, 1], iv_f[1][:, 1]) < empirical_coverage(
        y_true[:, 1], iv_h[0][:, 1], iv_h[1][:, 1]
    ) - 0.15

    nlls_h = NLLSEstimator(protocol=b_osipi, sd_convention="honest_crlb", max_nfev=120)
    nlls_f = NLLSEstimator(protocol=b_osipi, sd_convention="floored_railed", max_nfev=120)
    sub = signals[:40]
    assert float(np.mean(nlls_f.solve(sub).sigma[:, 1])) < float(np.mean(nlls_h.solve(sub).sigma[:, 1]))

    # BenchmarkSuite & run_estimator forward sd_convention cleanly
    out_reg_h = run_estimator("crlb", sub, b_osipi, snr=25.0, sd_convention="honest_crlb")
    out_reg_f = run_estimator("crlb", sub, b_osipi, snr=25.0, sd_convention="floored_railed")
    assert float(np.mean(out_reg_f.upper[:, 1] - out_reg_f.lower[:, 1])) < float(
        np.mean(out_reg_h.upper[:, 1] - out_reg_h.lower[:, 1])
    )


def test_cli_selfcheck_table_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc_table = cli_main(["selfcheck", "--n-voxels", "500", "--roundtrip-voxels", "25", "--seed", "42"])
    out_table = capsys.readouterr().out
    assert rc_table == 0
    assert "Overall Self-Check Status: PASSED" in out_table
    assert "dispersion_gamma" in out_table
    assert "Noiseless `biexp` NLLS Round-Trip" in out_table

    rc_json = cli_main(
        ["selfcheck", "--n-voxels", "400", "--roundtrip-voxels", "20", "--seed", "42", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc_json == 0
    assert payload["passed"] is True
    assert payload["continuity"]["dispersion_gamma"] <= 1.0e-8
    assert payload["nlls_roundtrip"]["max_rel_error"] <= 1.0e-9


def test_maf_missing_torch_raises_clean_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(maf_mod, "_HAS_TORCH", False)
    with pytest.raises(ImportError, match=r"Install ivimbench\[torch\] to use MAFPosteriorEstimator"):
        MAFPosteriorEstimator(protocol="osipi")


def test_maf_posterior_estimator_quantile_contract_when_torch_available() -> None:
    pytest.importorskip("torch")
    assert "maf" in list_estimators()

    cohort = generate_cohort(protocol="osipi", organ="liver", n_voxels=160, snr=30.0, seed=42)
    est = MAFPosteriorEstimator(
        protocol=cohort.bvalues,
        hidden=32,
        n_layers=3,
        epochs=12,
        batch_size=64,
        n_posterior=150,
        seed=42,
    )
    est.fit(cohort.signals[:120], cohort.params[:120])

    q_levels = (0.05, 0.25, 0.50, 0.75, 0.95)
    out = est(cohort.signals[120:], cohort.bvalues, q_levels)
    assert {"D", "D_star", "f", "intervals"}.issubset(set(out.keys()))
    lo, hi = out["intervals"]
    assert lo.shape == (40, 3) and hi.shape == (40, 3)
    assert np.all(hi >= lo)
    assert out["quantiles"].shape == (40, 3, 5)
    # Monotonic quantiles across q_levels
    assert np.all(np.diff(out["quantiles"], axis=-1) >= -1e-12)
    # Physical support respected: D > 0, D* > 0, 0 < f < 1
    assert np.all(out["D"] > 0.0)
    assert np.all(out["D_star"] > 0.0)
    assert np.all((out["f"] > 0.0) & (out["f"] < 1.0))

    # Registry execution
    reg_out = run_estimator("maf", cohort.signals[120:136], cohort.bvalues, epochs=6, n_synthetic=128)
    assert reg_out.point.shape == (16, 3)
    assert reg_out.has_intervals is True
