"""Spatial Digital Reference Objects (DROs) with smooth parenchymal variation and focal lesions.

Constructs 2D ``(H, W, n_b)`` and 3D ``(H, W, Z, n_b)`` structured anatomical
phantoms with:
- Organ-specific elliptical/lobulated contours and smooth spatially correlated
  parenchymal parameter fields grounded in :data:`~ivimbench.phantoms.synthetic.ORGAN_PRIORS`,
  including specialized presets for ``"liver"``, ``"pancreas"`` (fibrotic/hypovascular
  core surrounded by a steep hypervascular perfusion rim), and ``"breast"``
  (ACRIN-6698 multi-center breast DWI aligned).
- Optional deterministic spatial RF transmit ``B1`` bias maps (``b1_inhomogeneity``)
  and surface-coil ``SNR`` sensitivity maps (``snr_inhomogeneity``) seeded strictly
  with ``PCG64`` and zero patient/PHI data.
- Seamless voxel flattening (:meth:`DigitalReferenceObject.to_voxels`) and spatial
  map reconstruction (:meth:`DigitalReferenceObject.unflatten`) for evaluating
  1D or spatial estimators under :class:`~ivimbench.evaluate.BenchmarkSuite`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from ..protocols import AcquisitionProtocol, resolve_bvalues
from .noise import inject_noise
from .synthetic import (
    DEFAULT_EXTRA,
    DEFAULT_SEED,
    FAMILIES,
    PARAM_NAMES,
    _evaluate_family,
    get_organ_prior,
    ivim_biexp,
    ivim_kurtosis,
)

__all__ = [
    "LesionSpec",
    "OrganDROPreset",
    "ORGAN_DRO_PRESETS",
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
        Normalized lesion outer radius (e.g., ``0.18``).
    D : float
        Lesion slow diffusivity ``D`` in ``mm^2/s``.
    Dstar : float
        Lesion pseudo-diffusion coefficient ``D*`` in ``mm^2/s``.
    f : float
        Lesion microvascular perfusion fraction ``f`` in ``[0, 1]``.
    S0 : float, default=1.0
        Baseline ``b = 0`` signal intensity within the lesion.
    inner_radius : float, default=0.0
        Optional inner radius ``>= 0`` defining a hollow/annular shell (used for
        steep perilesional perfusion rims).
    edge_steepness : float or None, optional
        Optional radial logistic steepness controlling smooth rim transitions.
    """

    name: str
    center: tuple[float, ...]
    radius: float
    D: float
    Dstar: float
    f: float
    S0: float = 1.0
    inner_radius: float = 0.0
    edge_steepness: float | None = None


@dataclass(frozen=True)
class OrganDROPreset:
    """Anatomical geometry and default clinical protocol metadata for an organ DRO."""

    organ: str
    aspect_2d: tuple[float, float]
    aspect_3d: tuple[float, float, float]
    recommended_protocol: str
    description: str


ORGAN_DRO_PRESETS: dict[str, OrganDROPreset] = {
    "liver": OrganDROPreset(
        organ="liver",
        aspect_2d=(0.86, 0.78),
        aspect_3d=(0.86, 0.78, 0.85),
        recommended_protocol="osipi",
        description="Hepatic lobe contour with hypervascular, hypovascular, and necrotic focal lesions.",
    ),
    "pancreas": OrganDROPreset(
        organ="pancreas",
        aspect_2d=(0.68, 0.88),
        aspect_3d=(0.68, 0.88, 0.80),
        recommended_protocol="osipi",
        description="Elongated pancreatic parenchyma with desmoplastic fibrotic/hypovascular core and steep hypervascular perfusion rim.",
    ),
    "breast": OrganDROPreset(
        organ="breast",
        aspect_2d=(0.82, 0.84),
        aspect_3d=(0.82, 0.84, 0.84),
        recommended_protocol="acrin_6698",
        description="ACRIN-6698 aligned fibroglandular breast phantom with invasive ductal carcinoma, benign fibroadenoma, and necrotic core.",
    ),
    "brain": OrganDROPreset(
        organ="brain",
        aspect_2d=(0.88, 0.76),
        aspect_3d=(0.88, 0.76, 0.82),
        recommended_protocol="osipi",
        description="Cerebral parenchymal contour with microvascular CBV-consistent lesions.",
    ),
    "generic": OrganDROPreset(
        organ="generic",
        aspect_2d=(0.86, 0.78),
        aspect_3d=(0.86, 0.78, 0.85),
        recommended_protocol="osipi",
        description="Generic abdominal reference phantom.",
    ),
}


def default_lesions_for_organ(organ: str = "liver", ndim: int = 2) -> list[LesionSpec]:
    """Return physiologically representative focal lesions for a 2D or 3D organ phantom."""
    prior = get_organ_prior(organ)
    z_tail = (0.0,) if ndim == 3 else ()

    if prior.organ == "pancreas":
        # Concentric pancreatic ductal adenocarcinoma / chronic pancreatitis signature:
        # Outer steep hypervascular perfusion rim (label 2) + inner fibrotic/hypovascular core (label 3)
        core_center = (-0.08, -0.10) + z_tail
        return [
            LesionSpec(
                name="steep_perfusion_rim",
                center=core_center,
                radius=0.34,
                inner_radius=0.19,
                D=1.42e-3,
                Dstar=68.0e-3,
                f=0.34,
                S0=1.08,
                edge_steepness=35.0,
            ),
            LesionSpec(
                name="fibrotic_hypovascular_core",
                center=core_center,
                radius=0.19,
                D=0.98e-3,
                Dstar=17.5e-3,
                f=0.05,
                S0=0.88,
            ),
            LesionSpec(
                name="pancreatic_pseudocyst",
                center=(0.24, 0.36) + z_tail,
                radius=0.12,
                D=2.55e-3,
                Dstar=11.5e-3,
                f=0.02,
                S0=1.22,
            ),
        ]

    if prior.organ == "breast":
        # ACRIN-6698 / I-SPY2 aligned breast lesion spectrum
        return [
            LesionSpec(
                name="invasive_ductal_carcinoma",
                center=(-0.24, -0.22) + z_tail,
                radius=0.20,
                D=0.85e-3,
                Dstar=42.0e-3,
                f=0.18,
                S0=1.12,
            ),
            LesionSpec(
                name="benign_fibroadenoma",
                center=(0.26, 0.24) + z_tail,
                radius=0.17,
                D=1.68e-3,
                Dstar=21.0e-3,
                f=0.06,
                S0=0.98,
            ),
            LesionSpec(
                name="necrotic_tumor_core",
                center=(-0.20, 0.28) + z_tail,
                radius=0.11,
                D=2.32e-3,
                Dstar=11.0e-3,
                f=0.02,
                S0=1.18,
            ),
        ]

    d_mid = 0.5 * (prior.D_range[0] + prior.D_range[1])
    ds_mid = np.sqrt(prior.Dstar_range[0] * prior.Dstar_range[1])
    f_mid = 0.5 * (prior.f_range[0] + prior.f_range[1])

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
        Noise-free signal volume of shape ``(*spatial_shape, n_b)``.
    D_map, Dstar_map, f_map, S0_map : np.ndarray
        Ground-truth spatial parameter maps of shape ``spatial_shape``.
    region_mask : np.ndarray
        Integer tissue label map (``0`` = background, ``1`` = parenchyma,
        ``2..K`` = focal lesions).
    lesion_names : dict of int to str
        Mapping from integer lesion label ``>= 2`` to lesion name.
    bvalues : np.ndarray
        1D array of diffusion b-values in ``s/mm^2``.
    b1_map : np.ndarray or None
        Spatial RF transmit ``B1+`` shading multiplier map of shape ``spatial_shape``.
    snr_map : np.ndarray or None
        Spatial effective ``b = 0`` SNR map of shape ``spatial_shape``.
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
    b1_map: np.ndarray | None = None
    snr_map: np.ndarray | None = None
    family: str = "biexp"

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

    def unflatten(
        self,
        voxel_values: np.ndarray,
        foreground_only: bool = True,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """Project 1D/2D foreground predictions back onto ``spatial_shape``."""
        vals = np.asarray(voxel_values, dtype=np.float64)
        extra_shape = vals.shape[1:]
        if not foreground_only:
            return vals.reshape(self.spatial_shape + extra_shape)
        out = np.full(self.spatial_shape + extra_shape, fill_value, dtype=np.float64)
        out[self.foreground_mask] = vals
        return out


def _pcg64_rng(seed: int) -> np.random.Generator:
    """Create a strictly deterministic PCG64 NumPy random generator."""
    return np.random.Generator(np.random.PCG64(int(seed)))


def _build_inhomogeneity_fields(
    grids: list[np.ndarray],
    foreground: np.ndarray,
    *,
    b1_inhomogeneity: float,
    snr_inhomogeneity: float,
    nominal_snr: float | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Synthesize smooth RF transmit B1+ shading and surface-coil SNR maps."""
    spatial_shape = foreground.shape
    b1_amp = float(np.clip(b1_inhomogeneity, 0.0, 0.8))
    snr_amp = float(np.clip(snr_inhomogeneity, 0.0, 0.85))
    rng_field = _pcg64_rng(int(seed) + 303)
    sigma_px = max(min(spatial_shape) / 4.0, 2.0)

    if b1_amp > 0.0:
        # Standing-wave central dielectric brightening + smooth random RF phase shading
        r2 = sum(g**2 for g in grids)
        raw_b1 = rng_field.standard_normal(spatial_shape)
        smooth_b1 = gaussian_filter(raw_b1, sigma=sigma_px, mode="nearest")
        sd_b1 = float(np.std(smooth_b1[foreground])) or 1.0
        shading = 0.65 * (1.0 - 1.35 * r2) + 0.35 * (smooth_b1 / sd_b1)
        shading = np.clip(shading, -1.5, 1.5) / 1.5
        b1_fg = np.clip(1.0 + b1_amp * shading, 0.25, 2.0)
        b1_map = np.where(foreground, b1_fg, 0.0)
    else:
        b1_map = np.where(foreground, 1.0, 0.0)

    if nominal_snr is None or (np.ndim(nominal_snr) == 0 and np.isinf(nominal_snr)):
        return b1_map, None

    base_snr = float(nominal_snr)
    if snr_amp > 0.0:
        # Anterior/lateral receiver array proximity gradient + smooth g-factor modulation
        raw_snr = rng_field.standard_normal(spatial_shape)
        smooth_snr = gaussian_filter(raw_snr, sigma=sigma_px, mode="nearest")
        sd_snr = float(np.std(smooth_snr[foreground])) or 1.0
        coil_grad = -0.60 * grids[0] + 0.25 * (grids[1] ** 2 - 0.35) + 0.30 * (smooth_snr / sd_snr)
        coil_grad = np.clip(coil_grad, -1.5, 1.5) / 1.5
        snr_fg = np.clip(base_snr * (1.0 + snr_amp * coil_grad), 0.15 * base_snr, 2.5 * base_snr)
        snr_map = np.where(foreground, snr_fg, 0.0)
    else:
        snr_map = np.where(foreground, base_snr, 0.0)

    return b1_map, snr_map


def generate_dro(
    shape: tuple[int, ...] = (64, 64),
    organ: str = "liver",
    protocol: str | AcquisitionProtocol | Sequence[float] | None = "osipi",
    snr: float | None = 30.0,
    *,
    noise_type: str = "rician",
    n_coils: int = 1,
    lesions: Sequence[LesionSpec] | str | None = "default",
    spatial_variation: float = 0.15,
    b1_inhomogeneity: float = 0.0,
    snr_inhomogeneity: float = 0.0,
    family: str = "biexp",
    extra: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
) -> DigitalReferenceObject:
    """Generate a 2D or 3D spatial IVIM Digital Reference Object with lesions.

    Parameters
    ----------
    shape : tuple of int, default=(64, 64)
        2D ``(H, W)`` or 3D ``(H, W, Z)`` spatial grid dimensions.
    organ : {"liver", "pancreas", "breast", "brain", "generic"}, default="liver"
        Anatomical baseline parameter and geometry preset.
    protocol : str, AcquisitionProtocol, sequence of float, or "organ_default", default="osipi"
        Clinical b-value schedule. Pass ``"organ_default"`` (or ``None``) to select
        the organ preset's canonical schedule (e.g. ``"acrin_6698"`` for ``"breast"``).
    snr : float or None, default=30.0
        Nominal signal-to-noise ratio relative to baseline ``S0 = 1.0``.
    noise_type : {"rician", "noncentral_chi", "gaussian", "none"}, default="rician"
        MRI magnitude noise distribution.
    n_coils : int, default=1
        Number of receiver channels for non-central chi noise.
    lesions : "default", sequence of LesionSpec, or None, default="default"
        Focal lesions embedded inside the organ parenchyma. Pass ``None`` or
        ``()`` for a lesion-free smooth parenchymal phantom.
    spatial_variation : float, default=0.15
        Relative amplitude of smooth spatially correlated parenchymal variation.
    b1_inhomogeneity : float, default=0.0
        Amplitude in ``[0, 0.8]`` of smooth RF transmit ``B1+`` shading applied
        multiplicatively to ``S0_map``.
    snr_inhomogeneity : float, default=0.0
        Amplitude in ``[0, 0.85]`` of spatial receiver-coil SNR variation.
    family : str, default="biexp"
        Forward IVIM physics family (``"biexp"``, ``"dispersion_gamma"``,
        ``"dispersion_lognormal"``, ``"stretched"``, ``"triexp"``, or ``"kurtosis"``).
    extra : dict of str to float or None, optional
        Optional extra parameters for non-bi-exponential forward physics.
    seed : int, default=42
        Deterministic ``PCG64`` RNG seed.

    Returns
    -------
    DigitalReferenceObject
        Spatial phantom with clean/noisy signals, ground-truth maps, optional
        ``B1``/``SNR`` inhomogeneity maps, and anatomical region masks.
    """
    spatial_shape = tuple(int(s) for s in shape)
    if len(spatial_shape) not in (2, 3) or any(s < 8 for s in spatial_shape):
        raise ValueError(
            f"`shape` must be a 2D or 3D tuple with dimensions >= 8; got {shape!r}."
        )
    if b1_inhomogeneity < 0.0 or snr_inhomogeneity < 0.0:
        raise ValueError("`b1_inhomogeneity` and `snr_inhomogeneity` must be >= 0.")

    prior = get_organ_prior(organ)
    preset = ORGAN_DRO_PRESETS.get(prior.organ, ORGAN_DRO_PRESETS["generic"])

    if protocol is None or (isinstance(protocol, str) and protocol.strip().lower() == "organ_default"):
        protocol_eff: str | AcquisitionProtocol | Sequence[float] = preset.recommended_protocol
    else:
        protocol_eff = protocol

    bvalues, protocol_name = resolve_bvalues(protocol_eff)
    rng_spatial = _pcg64_rng(int(seed) + 101)
    rng_noise = _pcg64_rng(int(seed) + 7000)

    axes = [np.linspace(-1.0, 1.0, s, dtype=np.float64) for s in spatial_shape]
    grids = list(np.meshgrid(*axes, indexing="ij"))

    # Organ-specific elliptical anatomical contour
    aspect: tuple[float, ...] = (
        preset.aspect_2d if len(spatial_shape) == 2 else preset.aspect_3d
    )
    rad_sq = sum((g / a) ** 2 for g, a in zip(grids, aspect, strict=True))
    if prior.organ == "pancreas":
        # Subtle curved pancreatic body/tail bow while preserving central core containment
        bow_y = grids[0] - 0.10 * (grids[1] ** 2 - 0.25)
        rad_sq = (bow_y / aspect[0]) ** 2 + (grids[1] / aspect[1]) ** 2
        if len(spatial_shape) == 3:
            rad_sq = rad_sq + (grids[2] / aspect[2]) ** 2
    foreground: np.ndarray = np.asarray(rad_sq <= 1.0, dtype=bool)

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
    elif isinstance(lesions, str):
        raise ValueError(f"Unsupported lesions preset {lesions!r}; expected 'default', None, or sequence.")
    else:
        lesion_list = list(lesions)

    lesion_names: dict[int, str] = {}
    for idx, les in enumerate(lesion_list, start=2):
        if len(les.center) != len(spatial_shape):
            raise ValueError(
                f"Lesion {les.name!r} center dimensionality ({len(les.center)}) "
                f"does not match DRO spatial dimensions ({len(spatial_shape)})."
            )
        dist_sq = sum((g - c) ** 2 for g, c in zip(grids, les.center, strict=True))
        dist = np.sqrt(dist_sq)
        in_outer = dist_sq <= float(les.radius) ** 2
        if float(les.inner_radius) > 0.0:
            in_shell = in_outer & (dist_sq > float(les.inner_radius) ** 2)
        else:
            in_shell = in_outer
        lmask = foreground & in_shell
        if np.any(lmask):
            region_mask[lmask] = idx
            lesion_names[idx] = les.name
            if les.edge_steepness is not None and float(les.inner_radius) > 0.0:
                # Steep radial perfusion crest peaking near the core-rim interface
                r_norm = (dist[lmask] - float(les.inner_radius)) / max(
                    float(les.radius - les.inner_radius), 1e-6
                )
                crest = 1.0 / (1.0 + np.exp(float(les.edge_steepness) * (r_norm - 0.75) / 10.0))
                D_map[lmask] = crest * float(les.D) + (1.0 - crest) * D_map[lmask]
                Dstar_map[lmask] = crest * float(les.Dstar) + (1.0 - crest) * Dstar_map[lmask]
                f_map[lmask] = crest * float(les.f) + (1.0 - crest) * f_map[lmask]
                S0_map[lmask] = crest * float(les.S0) + (1.0 - crest) * S0_map[lmask]
            else:
                D_map[lmask] = float(les.D)
                Dstar_map[lmask] = float(les.Dstar)
                f_map[lmask] = float(les.f)
                S0_map[lmask] = float(les.S0)

    b1_map, snr_map = _build_inhomogeneity_fields(
        grids,
        foreground,
        b1_inhomogeneity=b1_inhomogeneity,
        snr_inhomogeneity=snr_inhomogeneity,
        nominal_snr=snr,
        seed=int(seed),
    )
    if b1_inhomogeneity > 0.0:
        S0_map = np.where(foreground, S0_map * b1_map, 0.0)

    if family == "biexp":
        signals_clean = ivim_biexp(bvalues, D_map, Dstar_map, f_map, s0=S0_map)
    elif family == "kurtosis":
        k_val = float((extra or {}).get("K", 0.75))
        signals_clean = ivim_kurtosis(bvalues, D_map, Dstar_map, f_map, K=k_val, s0=S0_map)
    elif family in FAMILIES:
        extra_kw = dict(DEFAULT_EXTRA[family])
        if extra:
            extra_kw.update(extra)
        flat_params = np.column_stack([D_map.ravel(), Dstar_map.ravel(), f_map.ravel()])
        flat_sig = _evaluate_family(family, bvalues, flat_params, extra_kw, s0=1.0)
        signals_clean = flat_sig.reshape(spatial_shape + (bvalues.size,)) * S0_map[..., None]
    else:
        raise ValueError(f"Unknown DRO forward family {family!r}.")

    signals_clean[~foreground] = 0.0

    if snr_inhomogeneity > 0.0 and snr_map is not None:
        # Ensure positive SNR everywhere before noise draw (background has clean signal 0)
        snr_for_noise: float | np.ndarray | None = np.where(
            foreground, np.maximum(snr_map, 1e-3), float(snr or 30.0)
        )
    else:
        snr_for_noise = snr

    signals_noisy = inject_noise(
        signals_clean,
        snr=snr_for_noise,
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
        b1_map=b1_map,
        snr_map=snr_map,
        family=family,
    )
