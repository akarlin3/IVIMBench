"""Zero-scanner, 100% synthetic PHI-free geometry substrates for closed-loop adaptive radiotherapy.

Provides two interchangeable longitudinal digital-twin substrates:
1. :class:`SyntheticOrganTwin`: 2D or 3D procedural tumor + adjacent organ-at-risk
   (OAR) digital reference object featuring continuous radial hypoxia/perfusion
   gradients (hypovascular radioresistant core surrounded by a normoxic rim
   abutting the OAR).
2. :class:`ParameterizedDoseGeometryTwin`: Accepts any user-supplied or idealized
   2D/3D dose-volume grid ``D_dose(x)`` and ``organ_mask`` (zero patient blobs
   required) to evaluate delivered-dose microvascular perfusion drop and
   action-suppression vs. outcome-protection dynamics across isodose shells.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from ..phantoms.noise import inject_noise
from ..phantoms.synthetic import (
    DEFAULT_EXTRA,
    DEFAULT_SEED,
    FAMILIES,
    _evaluate_family,
    get_organ_prior,
    ivim_biexp,
)
from ..protocols import AcquisitionProtocol, resolve_bvalues

__all__ = [
    "TwinScan",
    "BaseTwinSubstrate",
    "SyntheticOrganTwin",
    "ParameterizedDoseGeometryTwin",
]


@dataclass
class TwinScan:
    """Single-fraction synthetic multi-b DWI scan acquired on a twin substrate."""

    fraction: int
    bvalues: np.ndarray
    signals: np.ndarray
    spatial_signals: np.ndarray
    params_true: np.ndarray
    s0_true: np.ndarray
    tumor_fg_mask: np.ndarray
    oar_fg_mask: np.ndarray
    hypoxic_fg_mask: np.ndarray
    nominal_dose_fg: np.ndarray
    snr_effective: float


def _normalized_grid(shape: tuple[int, ...]) -> list[np.ndarray]:
    axes = [np.linspace(-1.0, 1.0, int(s), dtype=np.float64) for s in shape]
    return list(np.meshgrid(*axes, indexing="ij"))


class BaseTwinSubstrate:
    """Shared longitudinal radiobiology, synthetic DWI scanning, and dose-replan mechanics."""

    shape: tuple[int, ...]
    foreground_mask: np.ndarray
    tumor_mask: np.ndarray
    oar_mask: np.ndarray
    parenchyma_mask: np.ndarray
    nominal_fraction_dose_map: np.ndarray
    D_map: np.ndarray
    Dstar_map: np.ndarray
    f_map: np.ndarray
    S0_map: np.ndarray
    viable_cell_fraction: np.ndarray
    cumulative_dose_map: np.ndarray

    _init_D_map: np.ndarray
    _init_Dstar_map: np.ndarray
    _init_f_map: np.ndarray
    _init_S0_map: np.ndarray

    alpha_lq: float = 0.26
    beta_lq: float = 0.030
    vascular_dose_sensitivity: float = 0.045

    def _finalize_initial_state(self) -> None:
        self._init_D_map = self.D_map.copy()
        self._init_Dstar_map = self.Dstar_map.copy()
        self._init_f_map = self.f_map.copy()
        self._init_S0_map = self.S0_map.copy()
        self.viable_cell_fraction = np.where(self.foreground_mask, 1.0, 0.0).astype(np.float64)
        self.cumulative_dose_map = np.zeros(self.shape, dtype=np.float64)

    def reset(self) -> BaseTwinSubstrate:
        """Reset longitudinal tissue physiology, clonogenic survival, and delivered dose to t=0."""
        self.D_map = self._init_D_map.copy()
        self.Dstar_map = self._init_Dstar_map.copy()
        self.f_map = self._init_f_map.copy()
        self.S0_map = self._init_S0_map.copy()
        self.viable_cell_fraction = np.where(self.foreground_mask, 1.0, 0.0).astype(np.float64)
        self.cumulative_dose_map = np.zeros(self.shape, dtype=np.float64)
        return self

    @property
    def n_foreground(self) -> int:
        """Number of active foreground tissue voxels."""
        return int(np.sum(self.foreground_mask))

    def foreground_params(self) -> np.ndarray:
        """Return ``(n_fg, 3)`` ground-truth ``(D, Dstar, f)`` on foreground voxels."""
        fg = self.foreground_mask
        return np.column_stack([self.D_map[fg], self.Dstar_map[fg], self.f_map[fg]])

    def tumor_control_proxy(self) -> float:
        """Return clonogenic eradication proxy ``1 - mean(viable_cell_fraction[tumor])`` in ``[0, 1]``."""
        if not np.any(self.tumor_mask):
            return 0.0
        mean_viable = float(np.mean(self.viable_cell_fraction[self.tumor_mask]))
        return float(np.clip(1.0 - mean_viable, 0.0, 1.0))

    def oar_perfusion_drop_fraction(self) -> float:
        """Relative drop in mean OAR perfusion ``f`` compared to pre-treatment baseline."""
        if not np.any(self.oar_mask):
            return 0.0
        f0 = max(float(np.mean(self._init_f_map[self.oar_mask])), 1.0e-8)
        ft = float(np.mean(self.f_map[self.oar_mask]))
        return float(np.clip((f0 - ft) / f0, 0.0, 1.0))

    def scan(
        self,
        bvalues: str | AcquisitionProtocol | Sequence[float] | np.ndarray = "osipi",
        snr: float = 30.0,
        *,
        fraction: int = 1,
        hypoxia_f_threshold: float = 0.12,
        noise_type: str = "rician",
        n_coils: int = 4,
        family: str = "biexp",
        snr_scale: float = 1.0,
        seed: int | np.random.Generator = DEFAULT_SEED,
    ) -> TwinScan:
        """Simulate a zero-scanner multi-b DWI scan at the current longitudinal fraction state."""
        b_arr, _ = resolve_bvalues(bvalues)
        eff_snr = float(snr) * float(snr_scale)
        if eff_snr <= 0.0:
            raise ValueError(f"Effective scan SNR must be > 0; got {eff_snr}.")

        fg = self.foreground_mask
        params_fg = self.foreground_params()
        s0_fg = self.S0_map[fg]

        fam_key = family.strip().lower()
        if fam_key == "biexp":
            clean_fg = ivim_biexp(
                b_arr, params_fg[:, 0], params_fg[:, 1], params_fg[:, 2], s0=s0_fg
            )
        elif fam_key in FAMILIES:
            extra_kw = dict(DEFAULT_EXTRA.get(fam_key, {}))
            clean_fg = s0_fg[:, None] * _evaluate_family(fam_key, b_arr, params_fg, extra_kw)
        else:
            raise ValueError(f"Unknown forward signal family {family!r}.")

        rng = (
            seed
            if isinstance(seed, np.random.Generator)
            else np.random.Generator(np.random.PCG64(int(seed)))
        )
        noisy_fg = inject_noise(
            clean_fg,
            snr=eff_snr,
            noise_type=noise_type,
            n_coils=int(n_coils),
            seed=rng,
            s0=s0_fg,
        )

        spatial = np.zeros(self.shape + (b_arr.size,), dtype=np.float64)
        spatial[fg] = noisy_fg

        tumor_fg = self.tumor_mask[fg]
        oar_fg = self.oar_mask[fg]
        hypoxic_fg = tumor_fg & (params_fg[:, 2] <= float(hypoxia_f_threshold))

        return TwinScan(
            fraction=int(fraction),
            bvalues=b_arr,
            signals=noisy_fg,
            spatial_signals=spatial,
            params_true=params_fg,
            s0_true=s0_fg,
            tumor_fg_mask=tumor_fg,
            oar_fg_mask=oar_fg,
            hypoxic_fg_mask=hypoxic_fg,
            nominal_dose_fg=self.nominal_fraction_dose_map[fg].copy(),
            snr_effective=eff_snr,
        )

    def replan_dose(
        self,
        escalate_fg_mask: np.ndarray,
        *,
        boost_factor: float = 1.35,
        deescalate_fg_mask: np.ndarray | None = None,
        deescalate_factor: float = 0.85,
        dose_spill_sigma: float = 1.15,
        oar_max_fraction_gy: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute spatial delivered fraction dose with realistic penumbra spill into adjacent OAR/tissue."""
        fg = self.foreground_mask
        esc = np.asarray(escalate_fg_mask, dtype=bool).ravel()
        if esc.size != int(np.sum(fg)):
            raise ValueError(
                f"`escalate_fg_mask` length {esc.size} must match n_foreground {int(np.sum(fg))}."
            )

        nom = self.nominal_fraction_dose_map
        delta_fg = np.zeros(esc.size, dtype=np.float64)
        nom_fg = nom[fg]
        delta_fg[esc] = (float(boost_factor) - 1.0) * nom_fg[esc]

        if deescalate_fg_mask is not None:
            deesc = np.asarray(deescalate_fg_mask, dtype=bool).ravel() & (~esc)
            delta_fg[deesc] = (float(deescalate_factor) - 1.0) * nom_fg[deesc]

        delta_spatial = np.zeros(self.shape, dtype=np.float64)
        delta_spatial[fg] = delta_fg
        if float(dose_spill_sigma) > 0.0 and np.any(delta_fg != 0.0):
            # Physical beam penumbra / scatter spreads adaptive dose modifications to neighboring voxels
            delta_spatial = gaussian_filter(delta_spatial, sigma=float(dose_spill_sigma), mode="nearest")

        delivered = np.maximum(nom + delta_spatial, 0.0) * fg.astype(np.float64)
        if oar_max_fraction_gy is not None and np.any(self.oar_mask):
            oar_excess = np.maximum(delivered[self.oar_mask] - float(oar_max_fraction_gy), 0.0)
            if float(np.max(oar_excess)) > 0.0:
                delivered[self.oar_mask] = np.minimum(
                    delivered[self.oar_mask], float(oar_max_fraction_gy)
                )

        return delivered, delivered[fg]

    def apply_fraction_response(
        self,
        delivered_dose_map: np.ndarray,
        *,
        seed: int | np.random.Generator = DEFAULT_SEED,
    ) -> dict[str, float]:
        """Advance longitudinal tissue cellularity, microvascular perfusion, and survival by one fraction."""
        dose = np.asarray(delivered_dose_map, dtype=np.float64)
        if dose.shape != self.shape:
            raise ValueError(f"`delivered_dose_map` shape {dose.shape} != substrate shape {self.shape}.")

        rng = (
            seed
            if isinstance(seed, np.random.Generator)
            else np.random.Generator(np.random.PCG64(int(seed)))
        )
        fg = self.foreground_mask

        # Oxygen Enhancement Ratio (OER) modulation: hypoxic low-f subvolumes are more radioresistant
        oer_factor = 0.52 + 0.48 * (self.f_map / (self.f_map + 0.10))
        lq_exponent = oer_factor * (self.alpha_lq * dose + self.beta_lq * (dose**2))
        survival_step = np.where(fg, np.exp(-lq_exponent), 1.0)

        prev_viable = self.viable_cell_fraction.copy()
        self.viable_cell_fraction = np.where(fg, prev_viable * survival_step, 0.0)
        self.cumulative_dose_map = self.cumulative_dose_map + np.where(fg, dose, 0.0)

        # Smooth inter-fraction biological variation
        bio_noise_f = gaussian_filter(rng.normal(0.0, 0.003, size=self.shape), sigma=0.8)
        bio_noise_d = gaussian_filter(rng.normal(0.0, 0.015e-3, size=self.shape), sigma=0.8)
        bio_noise_ds = gaussian_filter(rng.normal(0.0, 0.35e-3, size=self.shape), sigma=0.8)

        # Delivered-dose endothelial perfusion drop + mild tumor decompression reoxygenation
        vasc_atten = np.exp(-self.vascular_dose_sensitivity * dose)
        reox_boost = np.where(self.tumor_mask, 0.010 * (prev_viable - self.viable_cell_fraction), 0.0)
        self.f_map = np.where(
            fg,
            np.clip(self.f_map * vasc_atten + reox_boost + bio_noise_f, 0.015, 0.55),
            0.0,
        )

        # Tumor cellularity reduction elevates slow tissue diffusivity D
        d_lysis = np.where(
            self.tumor_mask, 0.22e-3 * (prev_viable - self.viable_cell_fraction), 0.02e-3 * (1.0 - vasc_atten)
        )
        self.D_map = np.where(
            fg,
            np.clip(self.D_map + d_lysis + bio_noise_d, 0.35e-3, 3.2e-3),
            0.0,
        )
        self.Dstar_map = np.where(
            fg,
            np.clip(self.Dstar_map * np.exp(-0.018 * dose) + bio_noise_ds, 5.0e-3, 0.16),
            0.0,
        )

        return {
            "tumor_control_proxy": self.tumor_control_proxy(),
            "mean_tumor_viability": (
                float(np.mean(self.viable_cell_fraction[self.tumor_mask]))
                if np.any(self.tumor_mask)
                else 0.0
            ),
            "mean_tumor_f": float(np.mean(self.f_map[self.tumor_mask])) if np.any(self.tumor_mask) else 0.0,
            "mean_oar_f": float(np.mean(self.f_map[self.oar_mask])) if np.any(self.oar_mask) else 0.0,
            "oar_perfusion_drop_fraction": self.oar_perfusion_drop_fraction(),
        }


class SyntheticOrganTwin(BaseTwinSubstrate):
    """Procedural 2D/3D tumor + organ-at-risk (OAR) DRO with radial hypoxia/perfusion gradients.

    Constructs a 100% synthetic anatomical contour containing:
    - A target tumor with a radial oxygen/perfusion gradient: a radioresistant
      hypoxic core (low ``f ~ 0.045``, restricted ``D ~ 0.82e-3 mm^2/s``)
      transitioning smoothly to a well-perfused normoxic rim (``f ~ 0.26``).
    - An adjacent critical Organ-at-Risk (OAR) abutting the normoxic tumor rim so
      unnecessary dose escalations on noisy rim voxels directly spill penumbra
      dose into the OAR.
    """

    def __init__(
        self,
        shape: Sequence[int] = (28, 28),
        *,
        organ: str = "pancreas",
        nominal_tumor_dose_gy: float = 2.0,
        oar_falloff_sigma: float = 2.0,
        seed: int = DEFAULT_SEED,
    ) -> None:
        shp = tuple(int(s) for s in shape)
        if len(shp) not in (2, 3) or any(s < 10 for s in shp):
            raise ValueError(f"`shape` must be 2D or 3D with each side >= 10; got {shape!r}.")
        self.shape = shp
        self.organ = organ
        self.seed = int(seed)

        rng = np.random.Generator(np.random.PCG64(self.seed))
        prior = get_organ_prior(organ)
        coords = _normalized_grid(shp)

        # Elliptical anatomical organ contour
        if len(shp) == 2:
            yy, xx = coords
            r_organ = (yy / 0.86) ** 2 + (xx / 0.88) ** 2
            tumor_center = (-0.16, -0.14)
            oar_center = (0.20, 0.22)
            dist_tumor = np.sqrt((yy - tumor_center[0]) ** 2 + (xx - tumor_center[1]) ** 2)
            dist_oar = np.sqrt((yy - oar_center[0]) ** 2 + (xx - oar_center[1]) ** 2)
        else:
            yy, xx, zz = coords
            r_organ = (yy / 0.86) ** 2 + (xx / 0.88) ** 2 + (zz / 0.86) ** 2
            tumor_center_3d = (-0.16, -0.14, 0.0)
            oar_center_3d = (0.20, 0.22, 0.0)
            dist_tumor = np.sqrt(
                (yy - tumor_center_3d[0]) ** 2
                + (xx - tumor_center_3d[1]) ** 2
                + (zz - tumor_center_3d[2]) ** 2
            )
            dist_oar = np.sqrt(
                (yy - oar_center_3d[0]) ** 2
                + (xx - oar_center_3d[1]) ** 2
                + (zz - oar_center_3d[2]) ** 2
            )

        tumor_radius = 0.36
        oar_radius = 0.27

        fg = r_organ <= 1.0
        tumor = fg & (dist_tumor <= tumor_radius)
        oar = fg & (~tumor) & (dist_oar <= oar_radius)
        parenchyma = fg & (~tumor) & (~oar)

        self.foreground_mask = fg
        self.tumor_mask = tumor
        self.oar_mask = oar
        self.parenchyma_mask = parenchyma

        # Smooth background parenchymal & OAR physiology
        smooth_tex = gaussian_filter(rng.standard_normal(shp), sigma=1.4)
        smooth_tex = smooth_tex / max(float(np.std(smooth_tex)), 1.0e-6)

        d_mid = 0.5 * (prior.D_range[0] + prior.D_range[1])
        ds_mid = float(np.sqrt(prior.Dstar_range[0] * prior.Dstar_range[1]))
        f_mid = 0.5 * (prior.f_range[0] + prior.f_range[1])

        d_map = np.clip(d_mid * (1.0 + 0.08 * smooth_tex), prior.D_range[0], prior.D_range[1])
        ds_map = np.clip(ds_mid * (1.0 + 0.12 * smooth_tex), prior.Dstar_range[0], prior.Dstar_range[1])
        f_map = np.clip(f_mid * (1.0 + 0.10 * smooth_tex), prior.f_range[0], prior.f_range[1])
        s0_map = np.clip(1.0 + 0.04 * smooth_tex, 0.85, 1.15)

        # Radial hypoxia/perfusion gradient inside tumor:
        # r_norm = 0 at hypoxic core (f ~ 0.045), r_norm = 1 at normoxic hypervascular rim (f ~ 0.27)
        r_norm = np.clip(dist_tumor / tumor_radius, 0.0, 1.0)
        radial_prof = np.power(r_norm, 1.45)
        f_tumor = np.clip(0.045 + 0.225 * radial_prof + 0.008 * smooth_tex, 0.025, 0.36)
        d_tumor = np.clip((0.80e-3 + 0.45e-3 * radial_prof) * (1.0 + 0.04 * smooth_tex), 0.55e-3, 2.0e-3)
        ds_tumor = np.clip((15.0e-3 + 45.0e-3 * radial_prof) * (1.0 + 0.06 * smooth_tex), 8.0e-3, 0.12)

        d_map = np.where(tumor, d_tumor, d_map)
        ds_map = np.where(tumor, ds_tumor, ds_map)
        f_map = np.where(tumor, f_tumor, f_map)
        s0_map = np.where(tumor, 1.06, s0_map)

        # Richly perfused critical OAR (e.g., duodenum / bowel / healthy hepatic portal zone)
        f_map = np.where(oar, np.clip(0.24 + 0.02 * smooth_tex, 0.16, 0.35), f_map)
        d_map = np.where(oar, np.clip(1.45e-3 + 0.06e-3 * smooth_tex, 1.0e-3, 2.2e-3), d_map)
        ds_map = np.where(oar, np.clip(48.0e-3 + 4.0e-3 * smooth_tex, 20.0e-3, 95.0e-3), ds_map)

        self.D_map = np.where(fg, d_map, 0.0)
        self.Dstar_map = np.where(fg, ds_map, 0.0)
        self.f_map = np.where(fg, f_map, 0.0)
        self.S0_map = np.where(fg, s0_map, 0.0)

        # Nominal baseline conformal radiotherapy dose plan (prescribed to tumor with penumbra falloff)
        rx_target = np.where(tumor, float(nominal_tumor_dose_gy), 0.0)
        falloff = gaussian_filter(rx_target, sigma=float(oar_falloff_sigma), mode="constant")
        nom_plan = np.where(tumor, float(nominal_tumor_dose_gy), falloff) * fg.astype(np.float64)
        self.nominal_fraction_dose_map = nom_plan

        self._finalize_initial_state()


class ParameterizedDoseGeometryTwin(BaseTwinSubstrate):
    """Idealized or user-supplied 2D/3D dose-volume grid ``D_dose(x)`` + organ mask substrate.

    Requires zero patient anatomy or lesion blobs: accepts any ``dose_grid``
    ``D_dose(x)`` (in Gy/fraction) and boolean ``organ_mask`` (synthesizing a
    clean analytical 3D super-Gaussian dose-volume grid when ``dose_grid=None``)
    to test delivered-dose perfusion drop and action-suppression vs.
    outcome-protection dynamics across high-, mid-, and low-isodose shells.
    """

    def __init__(
        self,
        dose_grid: np.ndarray | None = None,
        organ_mask: np.ndarray | None = None,
        *,
        shape: Sequence[int] = (16, 16, 8),
        prescription_gy: float = 2.5,
        tumor_mask: np.ndarray | None = None,
        oar_mask: np.ndarray | None = None,
        target_isodose_frac: float = 0.72,
        oar_isodose_band: tuple[float, float] = (0.20, 0.72),
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.seed = int(seed)
        rng = np.random.Generator(np.random.PCG64(self.seed))

        if dose_grid is None:
            shp = tuple(int(s) for s in shape)
            if len(shp) not in (2, 3) or any(s < 6 for s in shp):
                raise ValueError(f"`shape` must be 2D or 3D with sides >= 6; got {shape!r}.")
            coords = _normalized_grid(shp)
            r_sq = np.zeros(shp, dtype=np.float64)
            for c in coords:
                r_sq = r_sq + c**2
            r = np.sqrt(r_sq)
            fg = (r <= 0.92) if organ_mask is None else np.asarray(organ_mask, dtype=bool)
            # Analytical super-Gaussian conformal dose falloff D_dose(x)
            d_grid = float(prescription_gy) * np.exp(-((r / 0.46) ** 3.2)) * fg.astype(np.float64)
        else:
            d_grid = np.asarray(dose_grid, dtype=np.float64)
            if d_grid.ndim not in (2, 3):
                raise ValueError(f"`dose_grid` must be 2D or 3D; got shape {d_grid.shape}.")
            if np.any(d_grid < 0.0) or float(np.max(d_grid)) <= 0.0:
                raise ValueError("`dose_grid` must be non-negative with positive maximum dose.")
            shp = d_grid.shape
            fg = (
                np.ones(shp, dtype=bool)
                if organ_mask is None
                else np.asarray(organ_mask, dtype=bool)
            )
            if fg.shape != shp:
                raise ValueError(f"`organ_mask` shape {fg.shape} must match `dose_grid` shape {shp}.")
            d_grid = d_grid * fg.astype(np.float64)

        if not np.any(fg):
            raise ValueError("`organ_mask` must contain at least one foreground voxel.")

        self.shape = shp
        self.foreground_mask = fg
        max_dose = float(np.max(d_grid[fg]))

        if tumor_mask is not None:
            t_mask = np.asarray(tumor_mask, dtype=bool) & fg
        else:
            t_mask = fg & (d_grid >= float(target_isodose_frac) * max_dose)

        if oar_mask is not None:
            o_mask = np.asarray(oar_mask, dtype=bool) & fg & (~t_mask)
        else:
            lo_iso, hi_iso = float(oar_isodose_band[0]), float(oar_isodose_band[1])
            o_mask = fg & (~t_mask) & (d_grid >= lo_iso * max_dose) & (d_grid < hi_iso * max_dose)

        self.tumor_mask = t_mask
        self.oar_mask = o_mask
        self.parenchyma_mask = fg & (~t_mask) & (~o_mask)
        self.nominal_fraction_dose_map = d_grid.copy()

        # Parameterize baseline physiology directly along normalized isodose shells (no anatomical blobs):
        # Inner peak-isodose core contains hypoxic subvolume (low f ~ 0.055), transitioning across
        # target boundary to normoxic perfusion (f ~ 0.24) in the intermediate-isodose OAR shell.
        iso_norm = np.clip(d_grid / max(max_dose, 1.0e-8), 0.0, 1.0)
        jitter = gaussian_filter(rng.normal(0.0, 1.0, size=shp), sigma=0.9)
        jitter = jitter / max(float(np.std(jitter)), 1.0e-6)

        f_base = np.clip(0.26 - 0.205 * (iso_norm**2.2) + 0.012 * jitter, 0.03, 0.38)
        d_base = np.clip((1.48e-3 - 0.58e-3 * (iso_norm**1.8)) * (1.0 + 0.04 * jitter), 0.55e-3, 2.4e-3)
        ds_base = np.clip((52.0e-3 - 32.0e-3 * (iso_norm**2.0)) * (1.0 + 0.06 * jitter), 10.0e-3, 0.11)
        s0_base = np.clip(1.0 + 0.03 * jitter, 0.90, 1.10)

        self.D_map = np.where(fg, d_base, 0.0)
        self.Dstar_map = np.where(fg, ds_base, 0.0)
        self.f_map = np.where(fg, f_base, 0.0)
        self.S0_map = np.where(fg, s0_base, 0.0)

        self._finalize_initial_state()

    def summarize_isodose_perfusion_drop(self, n_shells: int = 3) -> list[dict[str, float]]:
        """Quantify delivered-dose perfusion drop ``Delta f`` stratified across isodose shells."""
        fg = self.foreground_mask
        cum_d = self.cumulative_dose_map[fg]
        f0 = self._init_f_map[fg]
        ft = self.f_map[fg]
        edges = np.quantile(cum_d, np.linspace(0.0, 1.0, int(n_shells) + 1))
        shells: list[dict[str, float]] = []
        for k in range(int(n_shells)):
            lo_e, hi_e = float(edges[k]), float(edges[k + 1])
            if k == int(n_shells) - 1:
                sel = (cum_d >= lo_e) & (cum_d <= hi_e)
            else:
                sel = (cum_d >= lo_e) & (cum_d < hi_e)
            if not np.any(sel):
                continue
            f0_m = float(np.mean(f0[sel]))
            ft_m = float(np.mean(ft[sel]))
            shells.append(
                {
                    "shell": float(k + 1),
                    "dose_min_gy": lo_e,
                    "dose_max_gy": hi_e,
                    "mean_delivered_dose_gy": float(np.mean(cum_d[sel])),
                    "baseline_f": f0_m,
                    "final_f": ft_m,
                    "relative_perfusion_drop": float((f0_m - ft_m) / max(f0_m, 1.0e-8)),
                }
            )
        return shells
