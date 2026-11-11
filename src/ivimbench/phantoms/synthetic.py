"""Organ-specific IVIM parameter samplers, forward physics models, and cohort generation.

Ported and consolidated from Lattice and Datum:
- Organ-specific physiological parameter priors (:data:`ORGAN_PRIORS`) for
  ``"liver"``, ``"breast"``, ``"brain"``, and ``"pancreas"`` (plus ``"generic"``
  matching the Lattice reference distribution), sampled either uniformly or via
  a Gaussian copula capturing physiological ``(D, D*, f)`` couplings.
- Five clean-room IVIM forward signal families (:func:`ivim_biexp`,
  :func:`ivim_dispersion_gamma`, :func:`ivim_dispersion_lognormal`,
  :func:`ivim_stretched`, :func:`ivim_triexp`) sharing exact or asymptotic
  continuity limits at the canonical bi-exponential model.
- Deterministic multi-stream RNG seeding (:func:`generate_cohort`) ensuring
  reproducible ground-truth draws and noise realizations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from scipy import special

from ..protocols import AcquisitionProtocol, resolve_bvalues
from .noise import inject_noise

__all__ = [
    "PARAM_NAMES",
    "DEFAULT_SEED",
    "OrganPrior",
    "ORGAN_PRIORS",
    "DEFAULT_EXTRA",
    "CONTINUITY_EXTRA",
    "FAMILIES",
    "ivim_biexp",
    "ivim_dispersion_gamma",
    "ivim_dispersion_lognormal",
    "ivim_stretched",
    "ivim_triexp",
    "sample_organ_params",
    "SyntheticCohort",
    "generate_cohort",
    "continuity_residual",
]

PARAM_NAMES: tuple[str, str, str] = ("D", "Dstar", "f")
DEFAULT_SEED: int = 42

# Independent deterministic RNG stream offsets (ported from Lattice)
_OFFSET_BASE: int = 0
_OFFSET_EXTRA: int = 4040
_OFFSET_NOISE: int = 7000


@dataclass(frozen=True)
class OrganPrior:
    """Physiological IVIM parameter range and latent copula correlation for an organ.

    All diffusivities (``D`` and ``Dstar``) are in physical ``mm^2/s`` units;
    perfusion fraction ``f`` is dimensionless in ``[0, 1]``.
    """

    organ: str
    D_range: tuple[float, float]
    Dstar_range: tuple[float, float]
    f_range: tuple[float, float]
    corr: tuple[tuple[float, float, float], ...]
    description: str

    @property
    def bounds_dict(self) -> dict[str, tuple[float, float]]:
        """Return ``{"D": ..., "Dstar": ..., "f": ...}`` physiological support."""
        return {
            "D": self.D_range,
            "Dstar": self.Dstar_range,
            "f": self.f_range,
        }

    @property
    def corr_matrix(self) -> np.ndarray:
        """Return the 3x3 symmetric positive-definite copula correlation matrix."""
        return np.asarray(self.corr, dtype=np.float64)


ORGAN_PRIORS: dict[str, OrganPrior] = {
    "liver": OrganPrior(
        organ="liver",
        D_range=(0.6e-3, 1.8e-3),
        Dstar_range=(25.0e-3, 120.0e-3),
        f_range=(0.12, 0.40),
        corr=(
            (1.0, -0.15, 0.25),
            (-0.15, 1.0, -0.30),
            (0.25, -0.30, 1.0),
        ),
        description="Hepatic parenchyma & focal lesion spectrum (high sinusoidal perfusion & fast D*).",
    ),
    "breast": OrganPrior(
        organ="breast",
        D_range=(0.6e-3, 2.4e-3),
        Dstar_range=(10.0e-3, 60.0e-3),
        f_range=(0.03, 0.22),
        corr=(
            (1.0, -0.10, -0.20),
            (-0.10, 1.0, 0.15),
            (-0.20, 0.15, 1.0),
        ),
        description="Fibroglandular breast tissue & neoplastic spectrum (ACRIN 6698 range).",
    ),
    "brain": OrganPrior(
        organ="brain",
        D_range=(0.5e-3, 1.5e-3),
        Dstar_range=(8.0e-3, 45.0e-3),
        f_range=(0.02, 0.12),
        corr=(
            (1.0, -0.05, 0.15),
            (-0.05, 1.0, -0.15),
            (0.15, -0.15, 1.0),
        ),
        description="Cerebral gray/white matter microvascular CBV-consistent IVIM regime (low f).",
    ),
    "pancreas": OrganPrior(
        organ="pancreas",
        D_range=(0.8e-3, 2.2e-3),
        Dstar_range=(15.0e-3, 90.0e-3),
        f_range=(0.10, 0.38),
        corr=(
            (1.0, -0.15, 0.30),
            (-0.15, 1.0, -0.25),
            (0.30, -0.25, 1.0),
        ),
        description="Exocrine/endocrine pancreatic glandular parenchyma (high capillary f).",
    ),
    "generic": OrganPrior(
        organ="generic",
        D_range=(0.5e-3, 3.0e-3),
        Dstar_range=(10.0e-3, 100.0e-3),
        f_range=(0.05, 0.40),
        corr=(
            (1.0, -0.15, 0.25),
            (-0.15, 1.0, -0.30),
            (0.25, -0.30, 1.0),
        ),
        description="Broad abdominal/oncologic IVIM reference distribution from Lattice.",
    ),
}

_ORGAN_ALIASES: dict[str, str] = {
    "hepatic": "liver",
    "abdomen": "generic",
    "lattice": "generic",
    "neuro": "brain",
    "cerebral": "brain",
    "mammary": "breast",
    "pancreatic": "pancreas",
}


def get_organ_prior(organ: str) -> OrganPrior:
    """Resolve an organ name to its :class:`OrganPrior`."""
    key = organ.strip().lower()
    key = _ORGAN_ALIASES.get(key, key)
    if key not in ORGAN_PRIORS:
        raise KeyError(
            f"Unknown organ {organ!r}. Supported organs: {sorted(ORGAN_PRIORS.keys())}"
        )
    return ORGAN_PRIORS[key]


def sample_organ_params(
    organ: str = "liver",
    n_voxels: int = 10000,
    prior: str = "realistic",
    seed: int | np.random.Generator | None = DEFAULT_SEED,
) -> np.ndarray:
    """Sample ``n_voxels`` ground-truth ``(D, Dstar, f)`` tuples for ``organ``.

    Parameters
    ----------
    organ : {"liver", "breast", "brain", "pancreas", "generic"}, default="liver"
        Target anatomical prior distribution.
    n_voxels : int, default=10000
        Number of voxel parameter vectors to draw.
    prior : {"realistic", "uniform", "loguniform"}, default="realistic"
        - ``"realistic"``: Gaussian copula with physiological latent correlation,
          mapping ``D`` and ``f`` linearly and ``Dstar`` log-uniformly across the
          organ's physiological range (reflecting heavy right-skewed microvascular
          flow velocities).
        - ``"uniform"``: Independent uniform marginals over ``D``, ``Dstar``, ``f``.
        - ``"loguniform"``: Independent uniform ``D``, ``f`` and log-uniform ``Dstar``.
    seed : int, np.random.Generator, or None, default=42
        Random seed or NumPy generator.

    Returns
    -------
    np.ndarray
        Ground-truth parameter matrix of shape ``(n_voxels, 3)`` with columns
        ``(D, Dstar, f)`` in physical units (``mm^2/s``, ``mm^2/s``, dimensionless).
    """
    n = int(n_voxels)
    if n < 1:
        raise ValueError(f"`n_voxels` must be >= 1; got {n_voxels}.")
    spec = get_organ_prior(organ)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    mode = prior.strip().lower()
    if mode in ("uniform", "loguniform"):
        u = rng.random((n, 3))
    elif mode == "realistic":
        chol = np.linalg.cholesky(spec.corr_matrix)
        z = rng.standard_normal((n, 3)) @ chol.T
        u = 0.5 * (1.0 + special.erf(z / np.sqrt(2.0)))
    else:
        raise ValueError(
            f"Unknown prior {prior!r}; expected 'realistic', 'uniform', or 'loguniform'."
        )

    d_lo, d_hi = spec.D_range
    ds_lo, ds_hi = spec.Dstar_range
    f_lo, f_hi = spec.f_range

    D = d_lo + (d_hi - d_lo) * u[:, 0]
    if mode in ("realistic", "loguniform"):
        Dstar = np.exp(np.log(ds_lo) + (np.log(ds_hi) - np.log(ds_lo)) * u[:, 1])
    else:
        Dstar = ds_lo + (ds_hi - ds_lo) * u[:, 1]
    f = f_lo + (f_hi - f_lo) * u[:, 2]

    return np.column_stack([D, Dstar, f])


# --------------------------------------------------------------------------- #
# Forward signal models (ported from Lattice)
# --------------------------------------------------------------------------- #
def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def ivim_biexp(
    bvalues: np.ndarray | Sequence[float],
    D: np.ndarray | float,
    Dstar: np.ndarray | float,
    f: np.ndarray | float,
    s0: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Canonical Le Bihan bi-exponential IVIM signal model.

    .. math::

        S(b) = S_0 \\left[ f \\exp(-b D^*) + (1 - f) \\exp(-b D) \\right]
    """
    b = _as_float_array(bvalues)
    d_arr = _as_float_array(D)[..., None] if np.ndim(D) else float(D)
    ds_arr = _as_float_array(Dstar)[..., None] if np.ndim(Dstar) else float(Dstar)
    f_arr = _as_float_array(f)[..., None] if np.ndim(f) else float(f)
    s0_arr = _as_float_array(s0)[..., None] if np.ndim(s0) else float(s0)
    return s0_arr * (f_arr * np.exp(-b * ds_arr) + (1.0 - f_arr) * np.exp(-b * d_arr))


def ivim_dispersion_gamma(
    bvalues: np.ndarray | Sequence[float],
    D: np.ndarray | float,
    mu: np.ndarray | float,
    k: np.ndarray | float,
    f: np.ndarray | float,
    s0: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Gamma velocity-dispersion IVIM forward model ``(1 + b * mu / k)^(-k)``."""
    b = _as_float_array(bvalues)
    d_arr = _as_float_array(D)[..., None] if np.ndim(D) else float(D)
    mu_arr = _as_float_array(mu)[..., None] if np.ndim(mu) else float(mu)
    k_arr = _as_float_array(k)[..., None] if np.ndim(k) else float(k)
    f_arr = _as_float_array(f)[..., None] if np.ndim(f) else float(f)
    s0_arr = _as_float_array(s0)[..., None] if np.ndim(s0) else float(s0)
    perf = np.power(1.0 + b * mu_arr / k_arr, -k_arr)
    return s0_arr * (f_arr * perf + (1.0 - f_arr) * np.exp(-b * d_arr))


def ivim_dispersion_lognormal(
    bvalues: np.ndarray | Sequence[float],
    D: np.ndarray | float,
    mu: np.ndarray | float,
    cv: np.ndarray | float,
    f: np.ndarray | float,
    s0: np.ndarray | float = 1.0,
    n_quad: int = 64,
) -> np.ndarray:
    """Log-normal velocity-dispersion IVIM forward model via Gauss-Hermite quadrature."""
    b = _as_float_array(bvalues)
    d_arr, mu_arr, cv_arr, f_arr = np.broadcast_arrays(
        _as_float_array(D),
        _as_float_array(mu),
        _as_float_array(cv),
        _as_float_array(f),
    )
    scalar = d_arr.ndim == 0
    d_col = np.atleast_1d(d_arr)[:, None]
    mu_col = np.atleast_1d(mu_arr)[:, None]
    cv_col = np.atleast_1d(cv_arr)[:, None]
    f_col = np.atleast_1d(f_arr)[:, None]
    s0_arr = _as_float_array(s0)[..., None] if np.ndim(s0) else float(s0)

    sigma = np.sqrt(np.log1p(cv_col**2))
    m = np.log(np.where(mu_col > 0.0, mu_col, 1.0)) - 0.5 * sigma**2

    nodes, weights = np.polynomial.hermite_e.hermegauss(int(n_quad))
    w = (weights / np.sqrt(2.0 * np.pi))[None, None, :]
    z = nodes[None, None, :]
    dstar = np.exp(m[..., None] + sigma[..., None] * z)
    kernel = np.exp(-b[None, :, None] * dstar)
    perf = np.sum(w * kernel, axis=-1)

    degenerate = cv_col[:, 0] == 0.0
    if np.any(degenerate):
        exact = np.exp(-b[None, :] * mu_col)
        perf = np.where(degenerate[:, None], exact, perf)

    sig = s0_arr * (f_col * perf + (1.0 - f_col) * np.exp(-b[None, :] * d_col))
    return sig[0] if scalar else sig


def ivim_stretched(
    bvalues: np.ndarray | Sequence[float],
    D: np.ndarray | float,
    Dstar: np.ndarray | float,
    f: np.ndarray | float,
    beta: np.ndarray | float,
    s0: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Stretched-exponential anomalous perfusion IVIM forward model."""
    b = _as_float_array(bvalues)
    d_arr = _as_float_array(D)[..., None] if np.ndim(D) else float(D)
    ds_arr = _as_float_array(Dstar)[..., None] if np.ndim(Dstar) else float(Dstar)
    f_arr = _as_float_array(f)[..., None] if np.ndim(f) else float(f)
    beta_arr = _as_float_array(beta)[..., None] if np.ndim(beta) else float(beta)
    s0_arr = _as_float_array(s0)[..., None] if np.ndim(s0) else float(s0)
    perf = np.exp(-np.power(np.clip(b * ds_arr, 0.0, None), beta_arr))
    return s0_arr * (f_arr * perf + (1.0 - f_arr) * np.exp(-b * d_arr))


def ivim_triexp(
    bvalues: np.ndarray | Sequence[float],
    D: np.ndarray | float,
    Dstar: np.ndarray | float,
    f: np.ndarray | float,
    Dstar2: np.ndarray | float,
    g: np.ndarray | float,
    s0: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Tri-exponential three-compartment IVIM forward model."""
    b = _as_float_array(bvalues)
    d_arr = _as_float_array(D)[..., None] if np.ndim(D) else float(D)
    ds_arr = _as_float_array(Dstar)[..., None] if np.ndim(Dstar) else float(Dstar)
    f_arr = _as_float_array(f)[..., None] if np.ndim(f) else float(f)
    ds2_arr = _as_float_array(Dstar2)[..., None] if np.ndim(Dstar2) else float(Dstar2)
    g_arr = _as_float_array(g)[..., None] if np.ndim(g) else float(g)
    s0_arr = _as_float_array(s0)[..., None] if np.ndim(s0) else float(s0)
    return s0_arr * (
        (1.0 - f_arr) * np.exp(-b * d_arr)
        + f_arr * (1.0 - g_arr) * np.exp(-b * ds_arr)
        + f_arr * g_arr * np.exp(-b * ds2_arr)
    )


DEFAULT_EXTRA: dict[str, dict[str, float]] = {
    "biexp": {},
    "dispersion_gamma": {"k": 4.0},
    "dispersion_lognormal": {"cv": 0.5},
    "stretched": {"beta": 0.7},
    "triexp": {"Dstar2_mult": 4.0, "g": 0.3},
}

CONTINUITY_EXTRA: dict[str, dict[str, float]] = {
    "biexp": {},
    "dispersion_gamma": {"k": 1e8},
    "dispersion_lognormal": {"cv": 0.0},
    "stretched": {"beta": 1.0},
    "triexp": {"Dstar2_mult": 4.0, "g": 0.0},
}

FAMILIES: dict[str, dict[str, Any]] = {
    "biexp": {"fn": ivim_biexp, "extra": (), "continuity": "identity"},
    "dispersion_gamma": {
        "fn": ivim_dispersion_gamma,
        "extra": ("k",),
        "continuity": "k -> inf (asymptotic)",
    },
    "dispersion_lognormal": {
        "fn": ivim_dispersion_lognormal,
        "extra": ("cv",),
        "continuity": "cv = 0 (exact)",
    },
    "stretched": {
        "fn": ivim_stretched,
        "extra": ("beta",),
        "continuity": "beta = 1 (exact)",
    },
    "triexp": {
        "fn": ivim_triexp,
        "extra": ("Dstar2_mult", "g"),
        "continuity": "g = 0 (exact)",
    },
}


def _evaluate_family(
    family: str,
    bvalues: np.ndarray,
    params: np.ndarray,
    extra_kw: dict[str, float],
    s0: float = 1.0,
) -> np.ndarray:
    D, Dstar, f = params[:, 0], params[:, 1], params[:, 2]
    if family == "biexp":
        return ivim_biexp(bvalues, D, Dstar, f, s0=s0)
    if family == "dispersion_gamma":
        return ivim_dispersion_gamma(bvalues, D, Dstar, extra_kw["k"], f, s0=s0)
    if family == "dispersion_lognormal":
        return ivim_dispersion_lognormal(bvalues, D, Dstar, extra_kw["cv"], f, s0=s0)
    if family == "stretched":
        return ivim_stretched(bvalues, D, Dstar, f, extra_kw["beta"], s0=s0)
    if family == "triexp":
        dstar2 = extra_kw["Dstar2_mult"] * Dstar
        return ivim_triexp(bvalues, D, Dstar, f, dstar2, extra_kw["g"], s0=s0)
    raise ValueError(f"Unknown forward family {family!r}; choose from {list(FAMILIES)}.")


@dataclass
class SyntheticCohort:
    """Container for synthetic IVIM multi-b signals and ground-truth parameters.

    Supports:
    - Attribute access: ``cohort.signals``, ``cohort.params``, ``cohort.D``,
      ``cohort.Dstar``, ``cohort.f``, ``cohort.bvalues``, ``cohort.ground_truth``.
    - 2-tuple unpacking: ``signals, ground_truth = generate_cohort(...)``, where
      ``signals`` has shape ``(n_voxels, n_b)`` and ``ground_truth`` is the
      ``(n_voxels, 3)`` array ordered ``(D, Dstar, f)``.
    - Dictionary-style lookup: ``cohort["signals"]``, ``cohort["Dstar"]``, etc.
    """

    signals: np.ndarray
    params: np.ndarray
    bvalues: np.ndarray
    signals_clean: np.ndarray
    snr: float | np.ndarray
    organ: str = "liver"
    protocol: str = "osipi"
    family: str = "biexp"
    noise_type: str = "rician"
    n_coils: int = 1
    prior: str = "realistic"
    seed: int = DEFAULT_SEED
    param_names: tuple[str, ...] = PARAM_NAMES

    def __len__(self) -> int:
        return int(self.params.shape[0])

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield ``(signals, params)`` for convenient one-line unpacking."""
        yield self.signals
        yield self.params

    @property
    def D(self) -> np.ndarray:
        """Ground-truth slow tissue diffusivity ``D`` (``mm^2/s``)."""
        return self.params[:, 0]

    @property
    def Dstar(self) -> np.ndarray:
        """Ground-truth pseudo-diffusion coefficient ``D*`` (``mm^2/s``)."""
        return self.params[:, 1]

    @property
    def D_star(self) -> np.ndarray:
        """Alias for :attr:`Dstar`."""
        return self.params[:, 1]

    @property
    def f(self) -> np.ndarray:
        """Ground-truth microvascular perfusion fraction ``f`` (dimensionless)."""
        return self.params[:, 2]

    @property
    def ground_truth(self) -> dict[str, np.ndarray]:
        """Dictionary mapping ``"D"``, ``"Dstar"``, ``"f"`` to 1D ground-truth arrays."""
        return {
            "D": self.D,
            "Dstar": self.Dstar,
            "f": self.f,
        }

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return (self.signals, self.params)[key]
        mapping = {
            "signals": self.signals,
            "signal": self.signals,
            "signals_clean": self.signals_clean,
            "clean_signal": self.signals_clean,
            "params": self.params,
            "ground_truth": self.ground_truth,
            "bvalues": self.bvalues,
            "bvals": self.bvalues,
            "D": self.D,
            "Dstar": self.Dstar,
            "D_star": self.Dstar,
            "f": self.f,
            "snr": self.snr,
            "organ": self.organ,
            "protocol": self.protocol,
            "seed": self.seed,
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(f"Unknown key {key!r} in SyntheticCohort.")

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable reproducibility manifest."""
        snr_meta = (
            float(self.snr)
            if np.ndim(self.snr) == 0
            else np.asarray(self.snr, dtype=float).tolist()
        )
        return {
            "n_voxels": len(self),
            "organ": self.organ,
            "protocol": self.protocol,
            "bvalues": self.bvalues.tolist(),
            "family": self.family,
            "prior": self.prior,
            "noise_type": self.noise_type,
            "n_coils": self.n_coils,
            "snr": snr_meta,
            "seed": int(self.seed),
            "param_names": list(self.param_names),
        }

    def save(self, path: str | Path) -> Path:
        """Save arrays to ``<path>.npz`` and provenance manifest to ``<path>.json``."""
        target = Path(path)
        npz_path = target.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            signals=self.signals,
            signals_clean=self.signals_clean,
            params=self.params,
            bvalues=self.bvalues,
            snr=np.asarray(self.snr, dtype=np.float64),
        )
        target.with_suffix(".json").write_text(json.dumps(self.manifest(), indent=2))
        return npz_path


def generate_cohort(
    protocol: str | AcquisitionProtocol | Sequence[float] = "osipi",
    organ: str = "liver",
    n_voxels: int = 10000,
    snr: float | Sequence[float] | np.ndarray = 30.0,
    seed: int = DEFAULT_SEED,
    *,
    noise_type: str = "rician",
    n_coils: int = 1,
    prior: str = "realistic",
    family: str = "biexp",
    extra: dict[str, float] | None = None,
    s0: float = 1.0,
) -> SyntheticCohort:
    """One-line reproducible synthetic IVIM cohort generator.

    Draws physiologically grounded ``(D, Dstar, f)`` parameters for ``organ``,
    evaluates the clean forward signal on ``protocol``, and injects exact Rician,
    non-central chi, or Gaussian noise using offset RNG streams so ground-truth
    draws and noise realizations are deterministically decoupled.

    Parameters
    ----------
    protocol : str, AcquisitionProtocol, or sequence of float, default="osipi"
        Clinical acquisition schedule (e.g., ``"osipi"``, ``"tcga_lihc"``,
        ``"acrin_6698"``, ``"lattice_22b"``) or custom 1D array of b-values.
    organ : {"liver", "breast", "brain", "pancreas", "generic"}, default="liver"
        Anatomical parameter prior distribution.
    n_voxels : int, default=10000
        Number of synthetic voxels to generate.
    snr : float or sequence of float, default=30.0
        Signal-to-noise ratio at ``b = 0``. If a sequence/grid of SNRs is
        supplied (and its length differs from ``n_voxels``), voxels are assigned
        across the SNR grid deterministically; if length equals ``n_voxels``,
        each voxel uses its corresponding SNR.
    seed : int, default=42
        Master integer seed controlling parameter and noise RNG streams.
    noise_type : {"rician", "noncentral_chi", "gaussian", "none"}, default="rician"
        Noise model applied to the clean signal.
    n_coils : int, default=1
        Number of receiver coils when ``noise_type="noncentral_chi"``.
    prior : {"realistic", "uniform", "loguniform"}, default="realistic"
        Parameter prior sampling scheme.
    family : str, default="biexp"
        Forward model family (``"biexp"``, ``"dispersion_gamma"``,
        ``"dispersion_lognormal"``, ``"stretched"``, ``"triexp"``).
    extra : dict or None, optional
        Optional overrides for non-bi-exponential family hyperparameters.
    s0 : float, default=1.0
        Baseline ``b = 0`` signal scale.

    Returns
    -------
    SyntheticCohort
        Cohort object supporting both attribute access (``cohort.signals``,
        ``cohort.params``) and 2-tuple unpacking
        (``signals, ground_truth = generate_cohort(...)``).
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}; choose from {list(FAMILIES)}.")

    bvalues, protocol_name = resolve_bvalues(protocol)
    organ_spec = get_organ_prior(organ)

    rng_base = np.random.default_rng(int(seed) + _OFFSET_BASE)
    params = sample_organ_params(
        organ=organ_spec.organ,
        n_voxels=int(n_voxels),
        prior=prior,
        seed=rng_base,
    )

    extra_kw = dict(DEFAULT_EXTRA[family])
    if extra:
        extra_kw.update(extra)

    signals_clean = _evaluate_family(family, bvalues, params, extra_kw, s0=float(s0))

    snr_in = np.asarray(snr, dtype=np.float64)
    if snr_in.ndim == 0:
        snr_effective: float | np.ndarray = float(snr_in)
    elif snr_in.ndim == 1 and snr_in.size == int(n_voxels):
        snr_effective = snr_in
    elif snr_in.ndim == 1 and snr_in.size > 0:
        rng_extra = np.random.default_rng(int(seed) + _OFFSET_EXTRA)
        snr_effective = snr_in[rng_extra.integers(0, snr_in.size, size=int(n_voxels))]
    else:
        raise ValueError(f"Invalid `snr` shape {snr_in.shape}.")

    rng_noise = np.random.default_rng(int(seed) + _OFFSET_NOISE)
    signals_noisy = inject_noise(
        signals_clean,
        snr=snr_effective,
        noise_type=noise_type,
        n_coils=n_coils,
        seed=rng_noise,
        s0=float(s0),
    )

    return SyntheticCohort(
        signals=signals_noisy,
        params=params,
        bvalues=bvalues,
        signals_clean=signals_clean,
        snr=snr_effective,
        organ=organ_spec.organ,
        protocol=protocol_name,
        family=family,
        noise_type=noise_type,
        n_coils=int(n_coils),
        prior=prior,
        seed=int(seed),
    )


def continuity_residual(
    family: str,
    organ: str = "generic",
    protocol: str | AcquisitionProtocol | Sequence[float] = "lattice_22b",
    n_voxels: int = 2000,
    seed: int = DEFAULT_SEED,
) -> float:
    """Maximum absolute signal discrepancy between ``family`` at its continuity limit and ``biexp``."""
    bvalues, _ = resolve_bvalues(protocol)
    rng_base = np.random.default_rng(int(seed) + _OFFSET_BASE)
    params = sample_organ_params(organ=organ, n_voxels=n_voxels, prior="realistic", seed=rng_base)
    ref = _evaluate_family("biexp", bvalues, params, {})
    lim = _evaluate_family(family, bvalues, params, CONTINUITY_EXTRA[family])
    return float(np.max(np.abs(lim - ref)))
