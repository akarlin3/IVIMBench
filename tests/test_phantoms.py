"""Numerical validity tests for synthetic IVIM phantoms, spatial DROs, and noise models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ivimbench import DigitalReferenceObject, LesionSpec, SyntheticCohort, generate_cohort, generate_dro
from ivimbench.phantoms.noise import (
    add_gaussian_noise,
    add_ncchi_noise,
    add_noncentral_chi_noise,
    add_rician_noise,
    inject_noise,
    ncchi_analytical_second_moment,
    rician_analytical_mean,
)
from ivimbench.phantoms.synthetic import (
    FAMILIES,
    ORGAN_PRIORS,
    continuity_residual,
    ivim_biexp,
    sample_organ_params,
)


@pytest.mark.parametrize("organ", ["liver", "breast", "brain", "pancreas", "generic"])
@pytest.mark.parametrize("prior_mode", ["realistic", "uniform", "loguniform"])
def test_organ_parameter_samplers_respect_bounds(organ: str, prior_mode: str) -> None:
    spec = ORGAN_PRIORS[organ]
    params = sample_organ_params(organ=organ, n_voxels=4000, prior=prior_mode, seed=42)
    assert params.shape == (4000, 3)
    assert np.all(np.isfinite(params))

    assert np.all((params[:, 0] >= spec.D_range[0]) & (params[:, 0] <= spec.D_range[1]))
    assert np.all((params[:, 1] >= spec.Dstar_range[0]) & (params[:, 1] <= spec.Dstar_range[1]))
    assert np.all((params[:, 2] >= spec.f_range[0]) & (params[:, 2] <= spec.f_range[1]))
    # Pseudo-diffusion strictly exceeds slow tissue diffusion for physiological liver/pancreas/brain
    if organ in ("liver", "brain", "pancreas"):
        assert np.all(params[:, 1] > params[:, 0])


def test_realistic_copula_induces_physiological_correlation() -> None:
    params = sample_organ_params(organ="liver", n_voxels=25000, prior="realistic", seed=7)
    corr_df = float(np.corrcoef(params[:, 0], params[:, 2])[0, 1])
    # Liver copula specifies positive latent correlation (+0.25) between D and f
    assert 0.15 < corr_df < 0.35


def test_generate_cohort_api_reproducibility_and_stream_decoupling(tmp_path: Path) -> None:
    cohort_a = generate_cohort(protocol="osipi", organ="liver", n_voxels=1500, snr=30, seed=42)
    cohort_b = generate_cohort(protocol="osipi", organ="liver", n_voxels=1500, snr=30, seed=42)
    cohort_snr15 = generate_cohort(protocol="osipi", organ="liver", n_voxels=1500, snr=15, seed=42)

    assert isinstance(cohort_a, SyntheticCohort)
    signals, gt = cohort_a
    assert signals.shape == (1500, 12)
    assert gt.shape == (1500, 3)

    # Exact bit-for-bit reproducibility from seed
    np.testing.assert_array_equal(cohort_a.signals, cohort_b.signals)
    np.testing.assert_array_equal(cohort_a.params, cohort_b.params)

    # Changing SNR keeps ground-truth parameters identical due to decoupled RNG stream offsets
    np.testing.assert_array_equal(cohort_a.params, cohort_snr15.params)
    np.testing.assert_array_equal(cohort_a.signals_clean, cohort_snr15.signals_clean)
    assert not np.allclose(cohort_a.signals, cohort_snr15.signals)

    # Clean signal at b = 0 equals s0 = 1.0
    np.testing.assert_allclose(cohort_a.signals_clean[:, 0], 1.0, rtol=0.0, atol=1e-14)
    # Clean bi-exponential signal is strictly decreasing in b
    assert np.all(np.diff(cohort_a.signals_clean, axis=1) < 0.0)

    # Save and verify manifest
    npz_file = cohort_a.save(tmp_path / "cohort_test")
    assert npz_file.exists()
    assert (tmp_path / "cohort_test.json").exists()
    loaded = np.load(npz_file)
    np.testing.assert_array_equal(loaded["params"], cohort_a.params)


def test_multi_snr_schedule_cohort() -> None:
    cohort = generate_cohort(
        protocol="tcga_lihc",
        organ="breast",
        n_voxels=600,
        snr=[15.0, 30.0, 60.0],
        seed=99,
    )
    assert isinstance(cohort.snr, np.ndarray)
    assert cohort.snr.shape == (600,)
    assert set(np.unique(cohort.snr)).issubset({15.0, 30.0, 60.0})


def test_forward_families_continuity_limits() -> None:
    assert set(FAMILIES) == {
        "biexp",
        "dispersion_gamma",
        "dispersion_lognormal",
        "stretched",
        "triexp",
    }
    for exact_family in ("biexp", "dispersion_lognormal", "stretched", "triexp"):
        res = continuity_residual(exact_family, n_voxels=500, seed=42)
        assert res == pytest.approx(0.0, abs=1e-14)

    # Gamma dispersion at k = 1e7 converges asymptotically to biexp within O(1/k)
    res_gamma = continuity_residual("dispersion_gamma", n_voxels=500, seed=42)
    assert res_gamma < 1e-6


def test_rician_and_ncchi_noise_exact_moments_and_continuity() -> None:
    clean = np.full((120_000, 1), 0.35, dtype=np.float64)
    snr = 5.0  # Low SNR regime where Rician floor bias is substantial
    sigma = 1.0 / snr

    rician = add_rician_noise(clean, snr=snr, seed=2026, s0=1.0)
    ncchi_1 = add_noncentral_chi_noise(clean, snr=snr, n_coils=1, seed=2026, s0=1.0)

    # Bit-for-bit equality at n_coils = 1
    np.testing.assert_array_equal(rician, ncchi_1)
    assert np.all(rician >= 0.0)

    # Empirical Rician mean matches analytical Bessel/Laguerre moment (strictly > clean 0.35)
    expected_mean = float(rician_analytical_mean(0.35, sigma))
    assert expected_mean > 0.38
    assert float(np.mean(rician)) == pytest.approx(expected_mean, rel=4e-3)

    # Multi-coil non-central chi second moment E[M^2] = nu^2 + 2 * L * sigma^2
    ncchi_4 = add_ncchi_noise(clean, snr=snr, n_coils=4, seed=2026, s0=1.0)
    expected_m2 = float(ncchi_analytical_second_moment(0.35, sigma, n_coils=4))
    assert float(np.mean(ncchi_4**2)) == pytest.approx(expected_m2, rel=4e-3)

    # Gaussian noise is unbiased around clean signal
    gauss = add_gaussian_noise(clean, snr=snr, seed=2026, s0=1.0)
    assert float(np.mean(gauss)) == pytest.approx(0.35, abs=2e-3)

    # Noise-free passthrough
    clean_copy = inject_noise(clean, snr=None, noise_type="none")
    np.testing.assert_array_equal(clean_copy, clean)

    with pytest.raises(ValueError, match="strictly positive"):
        add_rician_noise(clean, snr=0.0)
    with pytest.raises(ValueError, match="n_coils"):
        add_noncentral_chi_noise(clean, snr=20.0, n_coils=0)


def test_dro_2d_and_3d_spatial_variation_and_lesions() -> None:
    dro2d = generate_dro(shape=(32, 32), organ="liver", protocol="osipi", snr=30.0, seed=42)
    assert isinstance(dro2d, DigitalReferenceObject)
    assert dro2d.signals.shape == (32, 32, 12)
    assert dro2d.D_map.shape == (32, 32)
    assert set(np.unique(dro2d.region_mask)) == {0, 1, 2, 3, 4}

    # Parenchyma has non-trivial smooth spatial heterogeneity
    parenchyma = dro2d.region_mask == 1
    assert float(np.std(dro2d.D_map[parenchyma])) > 1e-5
    # Hypervascular lesion (label 2) has elevated perfusion fraction relative to hypovascular (label 3)
    assert float(np.mean(dro2d.f_map[dro2d.region_mask == 2])) > float(
        np.mean(dro2d.f_map[dro2d.region_mask == 3])
    )

    vox_sig, vox_gt = dro2d.to_voxels(foreground_only=True)
    n_fg = int(np.sum(dro2d.foreground_mask))
    assert vox_sig.shape == (n_fg, 12)
    assert vox_gt.shape == (n_fg, 3)

    reconstructed = dro2d.unflatten(vox_gt[:, 0], foreground_only=True, fill_value=0.0)
    np.testing.assert_allclose(reconstructed[dro2d.foreground_mask], dro2d.D_map[dro2d.foreground_mask])
    assert np.all(reconstructed[~dro2d.foreground_mask] == 0.0)

    # 3D DRO with custom lesion
    custom = [
        LesionSpec(
            name="focal_tumor",
            center=(0.0, 0.0, 0.0),
            radius=0.35,
            D=0.9e-3,
            Dstar=65.0e-3,
            f=0.32,
            S0=1.1,
        )
    ]
    dro3d = generate_dro(
        shape=(16, 16, 12),
        organ="pancreas",
        protocol="acrin_6698",
        snr=25.0,
        noise_type="noncentral_chi",
        n_coils=4,
        lesions=custom,
        seed=11,
    )
    assert dro3d.signals.shape == (16, 16, 12, 4)
    assert 2 in dro3d.lesion_names and dro3d.lesion_names[2] == "focal_tumor"


def test_ivim_biexp_scalar_and_vector_consistency() -> None:
    b = np.array([0.0, 50.0, 400.0, 800.0])
    s_single = ivim_biexp(b, 1.2e-3, 35.0e-3, 0.20, s0=2.0)
    assert s_single.shape == (4,)
    assert s_single[0] == pytest.approx(2.0)
    expected = 2.0 * (0.20 * np.exp(-b * 35.0e-3) + 0.80 * np.exp(-b * 1.2e-3))
    np.testing.assert_allclose(s_single, expected, rtol=1e-14)
