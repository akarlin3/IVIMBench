"""Spatial Digital Reference Objects (DROs) with smooth parenchymal variation and focal lesions.

Constructs 2D ``(H, W, n_b)`` and 3D ``(H, W, Z, n_b)`` structured anatomical
phantoms with:
- Elliptical organ foreground masks and smooth spatially correlated parenchymal
  parameter fields grounded in :data:`~ivimbench.phantoms.synthetic.ORGAN_PRIORS`.
- Embedded focal lesions (hypervascular, hypovascular/fibrotic, necrotic/cystic,
  or custom ``LesionSpec`` inclusions) with distinct diffusion/perfusion signatures.
- Seamless voxel flattening (:meth:`DigitalReferenceObject.to_voxels`) and spatial
  map reconstruction (:meth:`DigitalReferenceObject.unflatten`) for evaluating
  1D or spatial estimators under :class:`~ivimbench.evaluate.BenchmarkSuite`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from ..protocols import AcquisitionProtocol, resolve_bvalues
from .noise import inject_noise
from .synthetic import DEFAULT_SEED, PARAM_NAMES, get_organ_prior, ivim_biexp

__all__ = [
    "LesionSpec",
    "DigitalReferenceObject",
    "default_lesions_for_organ",
    "generate_dro",
]


@dataclass(frozen=True)
class LesionSpec:
    """Specification of a focal spherical/circular lesion in normalized ``[-1, 1]`` coordinates.

    Parameters
    ----------
    name : str
        Identifier for the lesion (e.g., ``"hypervascular_tumor"``).
    center : tuple of float
        Normalized ``(y, x)`` or ``(y, x, z)`` coordinates in ``[-1, 1]``.
    radius : float
        Normalized lesion radius (e.g., ``0.18``).
    D : float
        Lesion slow diffusivity ``D`` in ``mm^2/s``.
    Dstar : float
        Lesion pseudo-diffusion coefficient ``D*`` in ``mm^2/s``.
    f : float
        Lesion microvascular perfusion fraction ``f`` in ``[0, 1]``.
    S0 : float, default=1.0
        Baseline ``b = 0`` signal intensity within the lesion.
    """

    name: str
    center: tuple[float, ...]
    radius: float
    D: float
    Dstar: float
    f: float
    S0: float = 1.0


def default_lesions_for_organ(organ: str = "liver", ndim: int = 2) -> list[LesionSpec]:
    """Return physiologically representative focal lesions for a 2D or 3D organ phantom."""
    prior = get_organ_prior(organ)
    d_mid = 0.5 * (prior.D_range[0] + prior.D_range[1])
    ds_mid = np.sqrt(prior.Dstar_range[0] * prior.Dstar_range[1])
    f_mid = 0.5 * (prior.f_range[0] + prior.f_range[1])

    z_tail = (0.0,) if ndim == 3 else ()
    return [
        LesionSpec(
            name="hypervascular_lesion",
            center=(-0.28, -0.25) + z_tail,
            radius=0.20,
            D=float(np.clip(0.75 * d_mid, 0.4e-3, 3.0e-3)),
            Dstar=float(np.clip(1.45 * ds_mid, 8.0e-3, 150.0e-3)),
            f=float(np.clip(1.65 * f_mid, 0.05, 0.50)),
            S0=1.05,
        ),
        LesionSpec(
            name="hypovascular_lesion",
            center=(0.28, 0.24) + z_tail,
            radius=0.18,
            D=float(np.clip(0.65 * d_mid, 0.35e-3, 3.0e-3)),
            Dstar=float(np.clip(0.55 * ds_mid, 5.0e-3, 120.0e-3)),
            f=float(np.clip(0.40 * f_mid, 0.015, 0.35)),
            S0=0.95,
        ),
        LesionSpec(
            name="necrotic_core",
            center=(-0.22, 0.30) + z_tail,
            radius=0.12,
            D=2.6e-3,
            Dstar=12.0e-3,
            f=0.02,
            S0=1.20,
        ),
    ]


@dataclass
class DigitalReferenceObject:
    """Structured 2D or 3D spatial IVIM Digital Reference Object (DRO).

    Attributes
    ----------
    signals : np.ndarray
        Simulated noisy DWI volume of shape ``(*spatial_shape, n_b)``.
    signals_clean : np.ndarray
        Noise-free bi-exponential signal volume of shape ``(*spatial_shape, n_b)``.
    D_map, Dstar_map, f_map, S0_map : np.ndarray
        Ground-truth spatial parameter maps of shape ``spatial_shape``.
    region_mask : np.ndarray
        Integer tissue label map (``0`` = background, ``1`` = parenchyma,
        ``2..K`` = focal lesions).
    lesion_names : dict of int to str
        Mapping from integer lesion label ``>= 2`` to lesion name.
    bvalues : np.ndarray
        1D array of diffusion b-values in ``s/mm^2``.
    """

    signals: np.ndarray
    signals_clean: np.ndarray
    D_map: np.ndarray
    Dstar_map: np.ndarray
    f_map: np.ndarray
    S0_map: np.ndarray
    region_mask: np.ndarray
    bvalues: np.ndarray
    organ: str = "liver"
    protocol: str = "osipi"
    snr: float | None = 30.0
    noise_type: str = "rician"
    n_coils: int = 1
    seed: int = DEFAULT_SEED
    lesion_names: dict[int, str] = field(default_factory=dict)
    param_names: tuple[str, ...] = PARAM_NAMES

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Spatial dimensions ``(H, W)`` or ``(H, W, Z)`` of the DRO."""
        return self.D_map.shape

    @property
    def foreground_mask(self) -> np.ndarray:
        """Boolean mask of all organ/lesion voxels (``region_mask > 0``)."""
        return self.region_mask > 0

    @property
    def params_map(self) -> np.ndarray:
        """Stacked ground-truth parameter volume of shape ``(*spatial_shape, 3)``."""
        return np.stack([self.D_map, self.Dstar_map, self.f_map], axis=-1)

    @property
    def params(self) -> np.ndarray:
        """Foreground ground-truth parameters ``(N_fg, 3)`` ordered ``(D, Dstar, f)``."""
        return self.params_map[self.foreground_mask]

    def to_voxels(self, foreground_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Extract 2D ``(N, n_b)`` signals and ``(N, 3)`` ground-truth ``(D, Dstar, f)``."""
        if foreground_only:
            mask = self.foreground_mask
            return self.signals[mask], self.params_map[mask]
        n_b = self.bvalues.size
        return self.signals.reshape(-1, n_b), self.params_map.reshape(-1, 3)

    def unflatten(self, voxel_values: np.ndarray, foreground_only: bool = True, fill_value: float = np.nan) -> np.ndarray:
        """Project 1D/2D foreground predictions back onto ``spatial_shape``."""
        vals = np.asarray(voxel_values, dtype=np.float64)
        extra_shape = vals.shape[1:]
        if not foreground_only:
            return vals.reshape(self.spatial_shape + extra_shape)
        out = np.full(self.spatial_shape + extra_shape, fill_value, dtype=np.float64)
        out[self.foreground_mask] = vals
        return out


def generate_dro(
    shape: tuple[int, ...] = (64, 64),
    organ: str = "liver",
    protocol: str | AcquisitionProtocol | Sequence[float] = "osipi",
    snr: float | None = 30.0,
    *,
    noise_type: str = "rician",
    n_coils: int = 1,
    lesions: Sequence[LesionSpec] | str | None = "default",
    spatial_variation: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> DigitalReferenceObject:
    """Generate a 2D or 3D spatial IVIM Digital Reference Object with lesions.

    Parameters
    ----------
    shape : tuple of int, default=(64, 64)
        2D ``(H, W)`` or 3D ``(H, W, Z)`` spatial grid dimensions.
    organ : {"liver", "breast", "brain", "pancreas", "generic"}, default="liver"
        Anatomical baseline parameter regime.
    protocol : str, AcquisitionProtocol, or sequence of float, default="osipi"
        Clinical b-value schedule.
    snr : float or None, default=30.0
        Signal-to-noise ratio relative to baseline ``S0 = 1.0``.
    noise_type : {"rician", "noncentral_chi", "gaussian", "none"}, default="rician"
        MRI magnitude noise distribution.
    n_coils : int, default=1
        Number of receiver channels for non-central chi noise.
    lesions : "default", sequence of LesionSpec, or None, default="default"
        Focal lesions embedded inside the organ parenchyma. Pass ``None`` or
        ``()`` for a lesion-free smooth parenchymal phantom.
    spatial_variation : float, default=0.15
        Relative amplitude of smooth spatially correlated parenchymal variation.
    seed : int, default=42
        Deterministic RNG seed.

    Returns
    -------
    DigitalReferenceObject
        Spatial phantom with clean/noisy 4D signals, ground-truth maps, and
        anatomical region masks.
    """
    spatial_shape = tuple(int(s) for s in shape)
    if len(spatial_shape) not in (2, 3) or any(s < 8 for s in spatial_shape):
        raise ValueError(
            f"`shape` must be a 2D or 3D tuple with dimensions >= 8; got {shape!r}."
        )

    bvalues, protocol_name = resolve_bvalues(protocol)
    prior = get_organ_prior(organ)
    rng_spatial = np.random.default_rng(int(seed) + 101)
    rng_noise = np.random.default_rng(int(seed) + 7000)

    axes = [np.linspace(-1.0, 1.0, s, dtype=np.float64) for s in spatial_shape]
    grids = np.meshgrid(*axes, indexing="ij")

    # Elliptical anatomical organ contour
    aspect = (0.86, 0.78) if len(spatial_shape) == 2 else (0.86, 0.78, 0.85)
    rad_sq = sum((g / a) ** 2 for g, a in zip(grids, aspect))
    foreground = rad_sq <= 1.0

    region_mask = np.zeros(spatial_shape, dtype=int)
    region_mask[foreground] = 1

    # Smooth spatially correlated Gaussian random fields for parenchymal heterogeneity
    sigma_px = max(min(spatial_shape) / 10.0, 1.5)
    fields = []
    for _ in range(3):
        raw = rng_spatial.standard_normal(spatial_shape)
        smooth = gaussian_filter(raw, sigma=sigma_px, mode="nearest")
        std = float(np.std(smooth[foreground])) or 1.0
        fields.append(np.clip(smooth / std, -2.5, 2.5))

    # Combine mild deterministic anatomical gradient with smooth random field
    grad_field = 0.5 * grids[0] - 0.35 * grids[1]
    d_lo, d_hi = prior.D_range
    ds_lo, ds_hi = prior.Dstar_range
    f_lo, f_hi = prior.f_range

    d_base = 0.5 * (d_lo + d_hi)
    ds_base = float(np.sqrt(ds_lo * ds_hi))
    f_base = 0.5 * (f_lo + f_hi)

    amp = float(np.clip(spatial_variation, 0.0, 1.0))
    D_map = np.where(
        foreground,
        np.clip(d_base * (1.0 + amp * (0.6 * fields[0] + 0.4 * grad_field)), d_lo, d_hi),
        0.0,
    )
    Dstar_map = np.where(
        foreground,
        np.clip(ds_base * np.exp(amp * (0.7 * fields[1] - 0.3 * grad_field)), ds_lo, ds_hi),
        0.0,
    )
    f_map = np.where(
        foreground,
        np.clip(f_base * (1.0 + amp * (0.6 * fields[2] + 0.4 * grad_field)), f_lo, f_hi),
        0.0,
    )
    S0_map = np.where(
        foreground,
        np.clip(1.0 + 0.05 * amp * grad_field, 0.5, 1.5),
        0.0,
    )

    if lesions == "default":
        lesion_list = default_lesions_for_organ(prior.organ, ndim=len(spatial_shape))
    elif lesions is None:
        lesion_list = []
    else:
        lesion_list = list(lesions)

    lesion_names: dict[int, str] = {}
    for idx, les in enumerate(lesion_list, start=2):
        if len(les.center) != len(spatial_shape):
            raise ValueError(
                f"Lesion {les.name!r} center dimensionality ({len(les.center)}) "
                f"does not match DRO spatial dimensions ({len(spatial_shape)})."
            )
        dist_sq = sum((g - c) ** 2 for g, c in zip(grids, les.center))
        lmask = foreground & (dist_sq <= float(les.radius) ** 2)
        if np.any(lmask):
            region_mask[lmask] = idx
            lesion_names[idx] = les.name
            # Smooth partial-volume transition near lesion boundary
            D_map[lmask] = float(les.D)
            Dstar_map[lmask] = float(les.Dstar)
            f_map[lmask] = float(les.f)
            S0_map[lmask] = float(les.S0)

    signals_clean = ivim_biexp(bvalues, D_map, Dstar_map, f_map, s0=S0_map)
    signals_clean[~foreground] = 0.0

    signals_noisy = inject_noise(
        signals_clean,
        snr=snr,
        noise_type=noise_type,
        n_coils=n_coils,
        seed=rng_noise,
        s0=1.0,
    )

    return DigitalReferenceObject(
        signals=signals_noisy,
        signals_clean=signals_clean,
        D_map=D_map,
        Dstar_map=Dstar_map,
        f_map=f_map,
        S0_map=S0_map,
        region_mask=region_mask,
        bvalues=bvalues,
        organ=prior.organ,
        protocol=protocol_name,
        snr=(None if (snr is None or np.isinf(snr)) else float(snr)),
        noise_type=noise_type,
        n_coils=int(n_coils),
        seed=int(seed),
        lesion_names=lesion_names,
    )
