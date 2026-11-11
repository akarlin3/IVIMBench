"""Exact MRI magnitude noise injectors (Rician, non-central chi, and Gaussian).

Ported and unified from Lattice and Gauge:
- **Rician noise** (:func:`add_rician_noise`): Single-coil complex quadrature
  magnitude detection ``sqrt((S + n_re)^2 + n_im^2)`` with
  ``n_re, n_im ~ N(0, sigma^2)``, ``sigma = S0 / SNR``.
- **Non-central chi noise** (:func:`add_noncentral_chi_noise`): Multi-coil
  sum-of-squares (SoS) magnitude reconstruction across ``n_coils = L`` receiver
  channels (``2L`` degrees of freedom). At ``n_coils = 1``, draws are consumed
  in the identical order as :func:`add_rician_noise` so it reduces to Rician
  noise byte-for-bit.
- **Gaussian noise** (:func:`add_gaussian_noise`): Symmetric high-SNR reference
  ``S + N(0, sigma^2)`` without a magnitude floor.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import special

__all__ = [
    "NoiseModel",
    "resolve_rng",
    "add_rician_noise",
    "add_noncentral_chi_noise",
    "add_ncchi_noise",
    "add_gaussian_noise",
    "inject_noise",
    "rician_analytical_mean",
    "ncchi_analytical_second_moment",
]

NoiseModel = Literal["rician", "noncentral_chi", "ncchi", "gaussian", "none"]


def resolve_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """Return a deterministic NumPy :class:`~numpy.random.Generator`."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _compute_sigma(
    signal_shape: tuple[int, ...],
    snr: float | np.ndarray,
    s0: float | np.ndarray = 1.0,
) -> np.ndarray:
    snr_arr = np.asarray(snr, dtype=np.float64)
    if np.any(snr_arr <= 0.0):
        raise ValueError(f"`snr` must be strictly positive; got min={np.min(snr_arr)}.")
    s0_arr = np.asarray(s0, dtype=np.float64)
    sigma = s0_arr / snr_arr
    if sigma.ndim > 0 and sigma.ndim < len(signal_shape):
        sigma = sigma.reshape(sigma.shape + (1,) * (len(signal_shape) - sigma.ndim))
    return np.broadcast_to(sigma, signal_shape)


def add_rician_noise(
    signal: np.ndarray,
    snr: float | np.ndarray,
    seed: int | np.random.Generator | None = None,
    s0: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Inject exact single-coil Rician magnitude noise at ``sigma = s0 / snr``.

    Parameters
    ----------
    signal : np.ndarray
        Clean non-negative signal magnitudes of arbitrary shape ``(..., n_b)``.
    snr : float or np.ndarray
        Signal-to-noise ratio defined at ``b = 0`` (scalar or broadcastable across
        leading voxel dimensions).
    seed : int, np.random.Generator, or None, optional
        Random seed or generator for deterministic realizations.
    s0 : float or np.ndarray, default=1.0
        Baseline ``b = 0`` reference amplitude defining ``sigma = s0 / snr``.

    Returns
    -------
    np.ndarray
        Rician-corrupted magnitude signal with the same shape as ``signal``.
    """
    clean = np.asarray(signal, dtype=np.float64)
    sigma = _compute_sigma(clean.shape, snr=snr, s0=s0)
    rng = resolve_rng(seed)
    n_re = rng.normal(0.0, sigma, size=clean.shape)
    n_im = rng.normal(0.0, sigma, size=clean.shape)
    return np.sqrt((clean + n_re) ** 2 + n_im**2)


def add_noncentral_chi_noise(
    signal: np.ndarray,
    snr: float | np.ndarray,
    n_coils: int = 4,
    seed: int | np.random.Generator | None = None,
    s0: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Inject exact multi-coil non-central chi (sum-of-squares) magnitude noise.

    Simulates an ``L = n_coils`` parallel receiver array under root-sum-of-squares
    magnitude combination (``2L`` degrees of freedom) with per-channel thermal
    noise ``sigma = s0 / snr``:

    .. math::

        S_{\\text{obs}} = \\sqrt{(S_{\\text{clean}} + \\varepsilon_{\\text{re},1})^2
        + \\varepsilon_{\\text{im},1}^2
        + \\sum_{\\ell=2}^{L} \\left(\\varepsilon_{\\text{re},\\ell}^2
        + \\varepsilon_{\\text{im},\\ell}^2\\right)}

    When ``n_coils == 1``, the Gaussian draws are identical in order and scale to
    :func:`add_rician_noise`, guaranteeing bit-for-bit Rician continuity.

    Parameters
    ----------
    signal : np.ndarray
        Clean signal magnitudes of shape ``(..., n_b)``.
    snr : float or np.ndarray
        Per-channel ``b = 0`` signal-to-noise ratio.
    n_coils : int, default=4
        Number of effective uncorrelated receiver coils ``L >= 1``.
    seed : int, np.random.Generator, or None, optional
        Random seed or generator.
    s0 : float or np.ndarray, default=1.0
        Baseline ``b = 0`` reference signal intensity.

    Returns
    -------
    np.ndarray
        Non-central chi magnitude signal with the same shape as ``signal``.
    """
    coils = int(n_coils)
    if coils < 1:
        raise ValueError(f"`n_coils` must be an integer >= 1; got {n_coils}.")
    clean = np.asarray(signal, dtype=np.float64)
    sigma = _compute_sigma(clean.shape, snr=snr, s0=s0)
    rng = resolve_rng(seed)

    n_re = rng.normal(0.0, sigma, size=clean.shape)
    n_im = rng.normal(0.0, sigma, size=clean.shape)
    sq_sum = (clean + n_re) ** 2 + n_im**2
    for _ in range(coils - 1):
        c_re = rng.normal(0.0, sigma, size=clean.shape)
        c_im = rng.normal(0.0, sigma, size=clean.shape)
        sq_sum = sq_sum + c_re**2 + c_im**2
    return np.sqrt(sq_sum)


add_ncchi_noise = add_noncentral_chi_noise


def add_gaussian_noise(
    signal: np.ndarray,
    snr: float | np.ndarray,
    seed: int | np.random.Generator | None = None,
    s0: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Inject additive zero-mean Gaussian noise ``S + N(0, (s0 / snr)^2)``."""
    clean = np.asarray(signal, dtype=np.float64)
    sigma = _compute_sigma(clean.shape, snr=snr, s0=s0)
    rng = resolve_rng(seed)
    return clean + rng.normal(0.0, sigma, size=clean.shape)


def inject_noise(
    signal: np.ndarray,
    snr: float | np.ndarray | None,
    noise_type: str = "rician",
    n_coils: int = 1,
    seed: int | np.random.Generator | None = None,
    s0: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Unified dispatcher for Rician, non-central chi, Gaussian, or noise-free signals."""
    clean = np.asarray(signal, dtype=np.float64)
    key = noise_type.strip().lower().replace("-", "_")
    if key == "none" or snr is None or (np.ndim(snr) == 0 and np.isinf(snr)):
        return clean.copy()
    if key == "rician":
        return add_rician_noise(clean, snr=snr, seed=seed, s0=s0)
    if key in ("noncentral_chi", "ncchi", "non_central_chi"):
        return add_noncentral_chi_noise(clean, snr=snr, n_coils=n_coils, seed=seed, s0=s0)
    if key == "gaussian":
        return add_gaussian_noise(clean, snr=snr, seed=seed, s0=s0)
    raise ValueError(
        f"Unknown noise_type {noise_type!r}; choose from "
        f"'rician', 'noncentral_chi' ('ncchi'), 'gaussian', or 'none'."
    )


def rician_analytical_mean(signal: np.ndarray | float, sigma: np.ndarray | float) -> np.ndarray:
    """Exact analytical first moment of a Rician(nu=signal, sigma) random variable.

    Uses exponentially scaled modified Bessel functions ``i0e`` and ``i1e`` for
    numerical stability across all SNR regimes:

    .. math::

        \\mathbb{E}[M] = \\sigma \\sqrt{\\frac{\\pi}{2}} \\, L_{1/2}\\!\\left(-\\frac{\\nu^2}{2\\sigma^2}\\right)
    """
    nu = np.asarray(signal, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    x = (nu**2) / (4.0 * sig**2)
    # L_{1/2}(-2x) = exp(-x) * [(1 + 2x) I_0(x) + 2x I_1(x)]
    # special.i0e(x) = exp(-|x|) I_0(x)
    laguerre_half = (1.0 + 2.0 * x) * special.i0e(x) + 2.0 * x * special.i1e(x)
    return sig * np.sqrt(np.pi / 2.0) * laguerre_half


def ncchi_analytical_second_moment(
    signal: np.ndarray | float,
    sigma: np.ndarray | float,
    n_coils: int = 4,
) -> np.ndarray:
    """Exact analytical second raw moment ``E[M^2] = nu^2 + 2 * L * sigma^2``."""
    nu = np.asarray(signal, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    return nu**2 + 2.0 * int(n_coils) * (sig**2)
