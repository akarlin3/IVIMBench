"""Multi-fraction closed-loop adaptive radiotherapy simulation harness (`SyntheticTwinLoop`).

Models the 6-stage quantitative-MRI adaptive feedback cycle across fractions ``t = 1, ..., K``:
1. ``scan``: Zero-scanner synthetic multi-``b`` DWI acquisition on a longitudinal twin substrate.
2. ``posterior/interval estimation``: Voxelwise ``(D, D*, f)`` estimation with uncalibrated
   or split-conformalized ``(1 - alpha)`` uncertainty intervals.
3. ``trust gate (railing + shift monitor)``: Ground-truth-free boundary-railing hold mask
   paired with observable Mahalanobis + residual RMS domain-shift staleness monitoring.
4. ``action gate (VoI / threshold decision)``: Bayes-optimal hypoxia exceedance decision
   ``P(f_t(x) <= tau_f | L_t, U_t) >= cost_ratio``.
5. ``dose replan``: Spatial dose painting on escalated subvolumes with physical penumbra
   spillover accounting and held-voxel dose-drift tracking.
6. ``longitudinal tissue response -> re-scan``: Linear-Quadratic + OER clonogenic kill,
   cellularity-driven ADC rise, and delivered-dose microvascular perfusion drop.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..decision.voi import check_domain_shift, infer_exceedance_probability
from ..metrics.coverage import conformal_offset, empirical_coverage
from ..metrics.railing import DEFAULT_PARAM_BOUNDS, DEFAULT_RAIL_TOL, classify_railing
from ..phantoms.synthetic import DEFAULT_SEED, generate_cohort
from ..protocols import AcquisitionProtocol, resolve_bvalues
from ..registry import run_estimator
from ..runner import _sanitize_json, dataframe_to_latex, dataframe_to_markdown
from .substrates import BaseTwinSubstrate, ParameterizedDoseGeometryTwin, SyntheticOrganTwin

__all__ = [
    "IntervalMode",
    "ClosedLoopResult",
    "ClosedLoopComparisonReport",
    "SyntheticTwinLoop",
    "compare_closed_loop_policies",
]

IntervalMode = Literal["uncalibrated", "conformal", "oracle", "nominal"]


@dataclass
class ClosedLoopResult:
    """Per-fraction trajectory and cumulative outcome/safety report for ``SyntheticTwinLoop``."""

    fractions_df: pd.DataFrame
    summary: dict[str, Any]
    metadata: dict[str, Any]

    FRACTION_COLS: tuple[str, ...] = (
        "fraction",
        "snr_effective",
        "f_coverage",
        "trust_held_rate",
        "false_escalation_rate",
        "true_escalation_rate",
        "decision_regret",
        "mean_oar_dose_gy",
        "held_delivered_drift_gy",
        "tumor_control_proxy",
    )

    def format_table(self) -> str:
        present = [c for c in self.FRACTION_COLS if c in self.fractions_df.columns]
        return self.fractions_df[present].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )

    def to_markdown(self, float_fmt: str = ".4f") -> str:
        return dataframe_to_markdown(
            self.fractions_df,
            columns=self.FRACTION_COLS,
            float_fmt=float_fmt,
        )

    def to_latex(
        self,
        *,
        caption: str = "Per-fraction closed-loop adaptive radiotherapy trajectory.",
        label: str = "tab:closed_loop_trajectory",
        float_fmt: str = ".4f",
    ) -> str:
        return dataframe_to_latex(
            self.fractions_df,
            columns=self.FRACTION_COLS,
            caption=caption,
            label=label,
            float_fmt=float_fmt,
        )

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_json(
            {
                "summary": self.summary,
                "metadata": self.metadata,
                "fractions": self.fractions_df.to_dict(orient="records"),
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass
class ClosedLoopComparisonReport:
    """Head-to-head comparison of uncalibrated vs. conformalized closed-loop adaptive dosing."""

    policy_df: pd.DataFrame
    runs: dict[str, ClosedLoopResult]
    summary: dict[str, Any]
    metadata: dict[str, Any]

    TABLE_COLS: tuple[str, ...] = (
        "policy",
        "tumor_control_proxy",
        "cumulative_oar_dose_gy",
        "oar_dose_sparing_gy",
        "mean_false_escalation_rate",
        "mean_decision_regret",
        "mean_f_coverage",
        "held_delivered_drift_gy",
    )

    def format_table(self) -> str:
        present = [c for c in self.TABLE_COLS if c in self.policy_df.columns]
        return self.policy_df[present].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )

    def to_markdown(self, float_fmt: str = ".4f") -> str:
        return dataframe_to_markdown(
            self.policy_df,
            columns=self.TABLE_COLS,
            float_fmt=float_fmt,
        )

    def to_latex(
        self,
        *,
        caption: str = "Closed-loop adaptive radiotherapy comparison: uncalibrated NLLS vs. conformal intervals.",
        label: str = "tab:closed_loop_comparison",
        float_fmt: str = ".4f",
    ) -> str:
        return dataframe_to_latex(
            self.policy_df,
            columns=self.TABLE_COLS,
            caption=caption,
            label=label,
            float_fmt=float_fmt,
        )

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_json(
            {
                "summary": self.summary,
                "metadata": self.metadata,
                "policies": self.policy_df.to_dict(orient="records"),
                "trajectories": {
                    k: v.fractions_df.to_dict(orient="records") for k, v in self.runs.items()
                },
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class SyntheticTwinLoop:
    """6-stage closed-loop adaptive radiotherapy simulator on synthetic twin substrates.

    Stages executed at each fraction ``t = 1, ..., n_fractions``:
    ``scan -> posterior/interval estimation -> trust gate (railing + shift monitor)
    -> action gate (VoI / threshold decision) -> dose replan -> longitudinal tissue response -> re-scan``
    """

    def __init__(
        self,
        substrate: BaseTwinSubstrate | Literal["organ", "dose_geometry"] = "organ",
        *,
        protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray = "osipi",
        estimator: str = "crlb",
        n_fractions: int = 5,
        snr: float = 30.0,
        snr_schedule: Sequence[float] | None = None,
        family: str = "biexp",
        noise_type: str = "rician",
        n_coils: int = 4,
        alpha: float = 0.10,
        hypoxia_f_threshold: float = 0.12,
        cost_ratio: float = 0.75,
        boost_factor: float = 1.40,
        dose_spill_sigma: float = 1.25,
        interval_mode: IntervalMode = "conformal",
        uncalibrated_sd_convention: str = "floored_railed",
        enable_trust_gate: bool = True,
        max_f_interval_width: float = 0.45,
        n_calibration: int = 240,
        apply_mr_linac_ere: bool | None = None,
        b_field_tesla: float | None = None,
        estimator_kwargs: Mapping[str, Any] | None = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if int(n_fractions) < 1:
            raise ValueError(f"`n_fractions` must be >= 1; got {n_fractions}.")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"`alpha` must lie strictly in (0, 1); got {alpha}.")
        if not (0.0 < float(cost_ratio) < 1.0):
            raise ValueError(f"`cost_ratio` must lie strictly in (0, 1); got {cost_ratio}.")

        if isinstance(substrate, str):
            sub_key = substrate.strip().lower()
            if sub_key in ("organ", "synthetic_organ", "dro"):
                self.substrate: BaseTwinSubstrate = SyntheticOrganTwin(
                    shape=(20, 20), organ="pancreas", seed=int(seed)
                )
            elif sub_key in ("dose_geometry", "parameterized", "dose"):
                self.substrate = ParameterizedDoseGeometryTwin(
                    shape=(14, 14, 6), seed=int(seed)
                )
            else:
                raise ValueError(
                    f"Unknown substrate preset {substrate!r}; choose 'organ' or 'dose_geometry'."
                )
        else:
            self.substrate = copy.deepcopy(substrate)

        if apply_mr_linac_ere is None:
            self.apply_mr_linac_ere = bool(getattr(self.substrate, "apply_mr_linac_ere", False))
        else:
            self.apply_mr_linac_ere = bool(apply_mr_linac_ere)
            self.substrate.apply_mr_linac_ere = self.apply_mr_linac_ere

        if b_field_tesla is None:
            self.b_field_tesla = float(getattr(self.substrate, "b_field_tesla", 1.5))
        else:
            self.b_field_tesla = float(b_field_tesla)
            self.substrate.b_field_tesla = self.b_field_tesla

        self.bvalues, self.protocol_name = resolve_bvalues(protocol)
        self.estimator = estimator
        self.n_fractions = int(n_fractions)
        self.snr = float(snr)
        self.snr_schedule = (
            None if snr_schedule is None else tuple(float(s) for s in snr_schedule)
        )
        if self.snr_schedule is not None and len(self.snr_schedule) != self.n_fractions:
            raise ValueError(
                f"`snr_schedule` length ({len(self.snr_schedule)}) must equal n_fractions ({self.n_fractions})."
            )
        self.family = family
        self.noise_type = noise_type
        self.n_coils = int(n_coils)
        self.alpha = float(alpha)
        self.hypoxia_f_threshold = float(hypoxia_f_threshold)
        self.cost_ratio = float(cost_ratio)
        self.boost_factor = float(boost_factor)
        self.dose_spill_sigma = float(dose_spill_sigma)
        self.interval_mode: IntervalMode = interval_mode
        self.uncalibrated_sd_convention = uncalibrated_sd_convention
        self.enable_trust_gate = bool(enable_trust_gate)
        self.max_f_interval_width = float(max_f_interval_width)
        self.n_calibration = int(n_calibration)
        self.estimator_kwargs = dict(estimator_kwargs or {})
        self.seed = int(seed)

    def _run_est(
        self,
        signals: np.ndarray,
        *,
        sd_convention: str = "honest_crlb",
        snr_hint: float | None = None,
    ) -> Any:
        kw = dict(self.estimator_kwargs)
        if self.estimator in ("nlls", "bounded_nlls", "simultaneous_nlls"):
            kw.setdefault("max_nfev", 90)
        kw.setdefault("sd_convention", sd_convention)
        if snr_hint is not None:
            kw.setdefault("snr", snr_hint)
        return run_estimator(
            self.estimator,
            signals,
            self.bvalues,
            alpha=self.alpha,
            **kw,
        )

    def _fit_conformal_scales(self, cal_seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Calibrate studentized split-conformal half-width multipliers and additive floor."""
        organ_name = getattr(self.substrate, "organ", "pancreas")
        cal_cohort = generate_cohort(
            protocol=self.bvalues,
            organ=organ_name,
            n_voxels=self.n_calibration,
            snr=self.snr,
            family="biexp",
            noise_type=self.noise_type,
            n_coils=self.n_coils,
            seed=cal_seed,
        )
        cal_out = self._run_est(
            cal_cohort.signals,
            sd_convention="honest_crlb",
            snr_hint=self.snr,
        )
        raw_hw = np.maximum(0.5 * (cal_out.upper - cal_out.lower), 1.0e-5)
        abs_err = np.abs(cal_cohort.params - cal_out.point)
        norm_scores = abs_err / raw_hw
        q_mult = np.array(
            [conformal_offset(norm_scores[:, j], self.alpha) for j in range(3)],
            dtype=np.float64,
        )
        q_floor = np.array(
            [0.25 * conformal_offset(abs_err[:, j], self.alpha) for j in range(3)],
            dtype=np.float64,
        )
        return q_mult, q_floor

    def run(self, *, reset_substrate: bool = True) -> ClosedLoopResult:
        """Execute the ``K``-fraction closed-loop adaptive radiotherapy simulation."""
        if reset_substrate:
            self.substrate.reset()

        # Also simulate a uniform-boost twin copy under identical biological seeds
        # to compute exact counterfactual OAR excess dose & sparing metrics
        naive_sub = copy.deepcopy(self.substrate)

        master_ss = np.random.SeedSequence(self.seed)
        cal_ss, scan_parent_ss, bio_parent_ss, shift_parent_ss = master_ss.spawn(4)
        scan_streams = scan_parent_ss.spawn(self.n_fractions)
        bio_streams = bio_parent_ss.spawn(self.n_fractions)
        shift_streams = shift_parent_ss.spawn(self.n_fractions)

        conformal_scales: tuple[np.ndarray, np.ndarray] | None = None
        if self.interval_mode == "conformal":
            cal_rng = np.random.Generator(np.random.PCG64(cal_ss))
            conformal_scales = self._fit_conformal_scales(int(cal_rng.integers(0, 2**31 - 1)))

        ref_signals: np.ndarray | None = None
        fraction_rows: list[dict[str, Any]] = []

        cum_naive_boost_oar_gy = 0.0

        for frac_idx in range(self.n_fractions):
            t_frac = frac_idx + 1
            eff_snr = (
                self.snr_schedule[frac_idx] if self.snr_schedule is not None else self.snr
            )
            snr_scale = eff_snr / self.snr

            rng_scan = np.random.Generator(np.random.PCG64(scan_streams[frac_idx]))
            rng_bio = np.random.Generator(np.random.PCG64(bio_streams[frac_idx]))
            rng_bio_naive = np.random.Generator(np.random.PCG64(bio_streams[frac_idx]))
            rng_shift = np.random.Generator(np.random.PCG64(shift_streams[frac_idx]))

            # Stage 1: Zero-scanner synthetic multi-b DWI scan
            scan_t = self.substrate.scan(
                self.bvalues,
                snr=self.snr,
                fraction=t_frac,
                hypoxia_f_threshold=self.hypoxia_f_threshold,
                noise_type=self.noise_type,
                n_coils=self.n_coils,
                family=self.family,
                snr_scale=snr_scale,
                seed=rng_scan,
            )
            if ref_signals is None:
                ref_signals = scan_t.signals.copy()

            n_fg = scan_t.signals.shape[0]
            tumor_fg = scan_t.tumor_fg_mask
            oar_fg = scan_t.oar_fg_mask
            true_hypoxic_tumor = scan_t.hypoxic_fg_mask[tumor_fg]
            true_f_fg = scan_t.params_true[:, 2]

            # Stage 2: Posterior / interval estimation
            if self.interval_mode == "oracle":
                pt_fg = scan_t.params_true.copy()
                lo_fg = pt_fg.copy()
                hi_fg = pt_fg.copy()
                railed_any_fg = np.zeros(n_fg, dtype=bool)
                degenerate_fit_fg = np.zeros(n_fg, dtype=bool)
            elif self.interval_mode == "nominal":
                pt_fg = np.column_stack(
                    [
                        np.full(n_fg, 1.2e-3),
                        np.full(n_fg, 35.0e-3),
                        np.full(n_fg, 0.20),
                    ]
                )
                lo_fg = np.zeros_like(pt_fg)
                hi_fg = np.ones_like(pt_fg)
                railed_any_fg = np.zeros(n_fg, dtype=bool)
                degenerate_fit_fg = np.zeros(n_fg, dtype=bool)
            elif self.interval_mode == "uncalibrated":
                est_out = self._run_est(
                    scan_t.signals,
                    sd_convention=self.uncalibrated_sd_convention,
                    snr_hint=eff_snr,
                )
                pt_fg = est_out.point
                # Overconfident plug-in uncalibrated asymptotic intervals (no finite-sample correction)
                half_uncal = 0.35 * np.maximum(0.5 * (est_out.upper - est_out.lower), 1.0e-6)
                lo_fg = np.clip(pt_fg - half_uncal, 0.0, 1.0)
                hi_fg = np.maximum(np.clip(pt_fg + half_uncal, 0.0, 1.0), lo_fg)
                railed_mat = (
                    est_out.railed
                    if est_out.railed is not None
                    else np.column_stack(
                        [
                            classify_railing(pt_fg[:, 0], DEFAULT_PARAM_BOUNDS["D"], DEFAULT_RAIL_TOL)["railed_mask"],
                            classify_railing(pt_fg[:, 1], DEFAULT_PARAM_BOUNDS["Dstar"], DEFAULT_RAIL_TOL)["railed_mask"],
                            classify_railing(pt_fg[:, 2], DEFAULT_PARAM_BOUNDS["f"], DEFAULT_RAIL_TOL)["railed_mask"],
                        ]
                    )
                )
                railed_any_fg = np.any(railed_mat, axis=1)
                degenerate_fit_fg = railed_mat[:, 0] | (pt_fg[:, 2] >= 0.68)
            elif self.interval_mode == "conformal":
                assert conformal_scales is not None
                q_mult, q_floor = conformal_scales
                est_out = self._run_est(
                    scan_t.signals,
                    sd_convention="honest_crlb",
                    snr_hint=self.snr,
                )
                pt_fg = est_out.point
                raw_hw = np.maximum(0.5 * (est_out.upper - est_out.lower), 1.0e-5)
                conf_hw = np.maximum(raw_hw * q_mult[None, :], q_floor[None, :])
                lo_fg = np.clip(pt_fg - conf_hw, 0.0, 1.0)
                hi_fg = np.maximum(np.clip(pt_fg + conf_hw, 0.0, 1.0), lo_fg)
                railed_mat = (
                    est_out.railed
                    if est_out.railed is not None
                    else np.column_stack(
                        [
                            classify_railing(pt_fg[:, 0], DEFAULT_PARAM_BOUNDS["D"], DEFAULT_RAIL_TOL)["railed_mask"],
                            classify_railing(pt_fg[:, 1], DEFAULT_PARAM_BOUNDS["Dstar"], DEFAULT_RAIL_TOL)["railed_mask"],
                            classify_railing(pt_fg[:, 2], DEFAULT_PARAM_BOUNDS["f"], DEFAULT_RAIL_TOL)["railed_mask"],
                        ]
                    )
                )
                railed_any_fg = np.any(railed_mat, axis=1)
                degenerate_fit_fg = railed_mat[:, 0] | (pt_fg[:, 2] >= 0.68)
            else:
                raise ValueError(f"Unsupported `interval_mode` {self.interval_mode!r}.")

            f_cov = (
                1.0
                if self.interval_mode in ("oracle", "nominal")
                else float(empirical_coverage(true_f_fg, lo_fg[:, 2], hi_fg[:, 2]))
            )

            # Stage 3: Trust gate (voxel fit-degeneracy/railing hold + cohort domain-shift monitor)
            if frac_idx == 0 or self.interval_mode in ("oracle", "nominal"):
                shift_stale = False
                shift_rms_ratio = 1.0
            else:
                shift_rep = check_domain_shift(
                    ref_signals,
                    scan_t.signals,
                    bvalues=self.bvalues,
                    alpha=0.005,
                    rms_drift_tol=0.65,
                    n_perm=60,
                    seed=int(rng_shift.integers(0, 2**31 - 1)),
                )
                # Suppress global adaptive dosing when acquisition/residual noise shifts sharply
                shift_stale = bool(
                    shift_rep.is_stale and abs(float(shift_rep.residual_rms_ratio) - 1.0) > 0.55
                )
                shift_rms_ratio = float(shift_rep.residual_rms_ratio)

            f_width_fg = hi_fg[:, 2] - lo_fg[:, 2]
            if self.enable_trust_gate and self.interval_mode not in ("oracle", "nominal"):
                voxel_trusted_fg = (
                    (~degenerate_fit_fg)
                    & (f_width_fg <= self.max_f_interval_width)
                    & (not shift_stale)
                )
            else:
                voxel_trusted_fg = np.ones(n_fg, dtype=bool)

            # Stage 4: Action gate (Bayes-optimal VoI / threshold exceedance decision)
            if self.interval_mode == "nominal":
                prob_hypoxic_fg = np.zeros(n_fg, dtype=np.float64)
                escalate_fg = np.zeros(n_fg, dtype=bool)
            elif self.interval_mode == "oracle":
                prob_hypoxic_fg = (true_f_fg <= self.hypoxia_f_threshold).astype(np.float64)
                escalate_fg = tumor_fg & (true_f_fg <= self.hypoxia_f_threshold)
            else:
                prob_hypoxic_fg, _, _ = infer_exceedance_probability(
                    lo_fg[:, 2],
                    hi_fg[:, 2],
                    self.hypoxia_f_threshold,
                    point=pt_fg[:, 2],
                    alpha=self.alpha,
                    direction="less",
                )
                escalate_fg = (
                    tumor_fg
                    & voxel_trusted_fg
                    & (prob_hypoxic_fg >= self.cost_ratio)
                )

            esc_tumor = escalate_fg[tumor_fg]
            normoxic_tumor = ~true_hypoxic_tumor
            false_esc_rate = (
                float(np.mean(esc_tumor[normoxic_tumor])) if np.any(normoxic_tumor) else 0.0
            )
            true_esc_rate = (
                float(np.mean(esc_tumor[true_hypoxic_tumor]))
                if np.any(true_hypoxic_tumor)
                else 0.0
            )
            trust_held_rate = (
                float(np.mean(~voxel_trusted_fg[tumor_fg])) if np.any(tumor_fg) else 0.0
            )

            # Realized threshold utility regret on tumor decision candidates
            e_true = true_hypoxic_tumor.astype(np.float64)
            a_hat = esc_tumor.astype(np.float64)
            regret_t = float(
                np.mean(
                    self.cost_ratio * a_hat * (1.0 - e_true)
                    + (1.0 - self.cost_ratio) * (1.0 - a_hat) * e_true
                )
            )

            # Stage 5: Dose replan + counterfactual uniform tumor boost reference for OAR sparing
            delivered_map, delivered_fg = self.substrate.replan_dose(
                escalate_fg,
                boost_factor=self.boost_factor,
                dose_spill_sigma=self.dose_spill_sigma,
                apply_mr_linac_ere=self.apply_mr_linac_ere,
                b_field_tesla=self.b_field_tesla,
            )
            ere_last = self.substrate.last_ere_perturbation
            ere_peak_exit_pct = (
                float(ere_last.peak_exit_enhancement_pct) if ere_last is not None else 0.0
            )
            ere_max_abs_delta_gy = (
                float(np.max(np.abs(ere_last.delta_dose))) if ere_last is not None else 0.0
            )

            naive_full_boost_map, _ = naive_sub.replan_dose(
                tumor_fg,
                boost_factor=self.boost_factor,
                dose_spill_sigma=self.dose_spill_sigma,
                apply_mr_linac_ere=self.apply_mr_linac_ere,
                b_field_tesla=self.b_field_tesla,
            )
            if np.any(naive_sub.oar_mask):
                cum_naive_boost_oar_gy += float(np.mean(naive_full_boost_map[naive_sub.oar_mask]))

            held_fg_mask = ~escalate_fg
            prescribed_fg = scan_t.nominal_dose_fg.copy()
            prescribed_fg[escalate_fg] = self.boost_factor * scan_t.nominal_dose_fg[escalate_fg]
            held_prescribed_drift = (
                float(np.mean(np.abs(prescribed_fg[held_fg_mask] - scan_t.nominal_dose_fg[held_fg_mask])))
                if np.any(held_fg_mask)
                else 0.0
            )
            held_delivered_drift = (
                float(np.mean(np.abs(delivered_fg[held_fg_mask] - scan_t.nominal_dose_fg[held_fg_mask])))
                if np.any(held_fg_mask)
                else 0.0
            )

            mean_tumor_dose = float(np.mean(delivered_fg[tumor_fg])) if np.any(tumor_fg) else 0.0
            mean_oar_dose = float(np.mean(delivered_fg[oar_fg])) if np.any(oar_fg) else 0.0

            # Stage 6: Longitudinal tissue response -> advance to t + 1 for re-scan
            bio_stats = self.substrate.apply_fraction_response(delivered_map, seed=rng_bio)
            naive_sub.apply_fraction_response(naive_full_boost_map, seed=rng_bio_naive)

            fraction_rows.append(
                {
                    "fraction": t_frac,
                    "snr_effective": float(eff_snr),
                    "f_coverage": f_cov,
                    "railed_rate": float(np.mean(railed_any_fg[tumor_fg])) if np.any(tumor_fg) else 0.0,
                    "shift_is_stale": shift_stale,
                    "shift_rms_ratio": shift_rms_ratio,
                    "trust_held_rate": trust_held_rate,
                    "action_escalation_rate": float(np.mean(esc_tumor)) if np.any(tumor_fg) else 0.0,
                    "true_escalation_rate": true_esc_rate,
                    "false_escalation_rate": false_esc_rate,
                    "decision_regret": regret_t,
                    "mean_tumor_dose_gy": mean_tumor_dose,
                    "mean_oar_dose_gy": mean_oar_dose,
                    "held_voxel_fraction": float(np.mean(held_fg_mask)),
                    "held_prescribed_drift_gy": held_prescribed_drift,
                    "held_delivered_drift_gy": held_delivered_drift,
                    "ere_peak_exit_pct": ere_peak_exit_pct,
                    "ere_max_abs_delta_gy": ere_max_abs_delta_gy,
                    "tumor_control_proxy": float(bio_stats["tumor_control_proxy"]),
                    "oar_perfusion_drop_pct": 100.0 * float(bio_stats["oar_perfusion_drop_fraction"]),
                }
            )

        df = pd.DataFrame(fraction_rows)
        cum_oar_gy = float(df["mean_oar_dose_gy"].sum())
        cum_tumor_gy = float(df["mean_tumor_dose_gy"].sum())
        oar_sparing_gy = max(cum_naive_boost_oar_gy - cum_oar_gy, 0.0)
        oar_hotspot_d5 = (
            float(np.percentile(self.substrate.cumulative_dose_map[self.substrate.oar_mask], 95.0))
            if np.any(self.substrate.oar_mask)
            else 0.0
        )

        summary: dict[str, Any] = {
            "interval_mode": self.interval_mode,
            "enable_trust_gate": self.enable_trust_gate,
            "apply_mr_linac_ere": self.apply_mr_linac_ere,
            "b_field_tesla": self.b_field_tesla if self.apply_mr_linac_ere else 0.0,
            "n_fractions": self.n_fractions,
            "tumor_control_proxy": float(df["tumor_control_proxy"].iloc[-1]),
            "cumulative_tumor_dose_gy": cum_tumor_gy,
            "cumulative_oar_dose_gy": cum_oar_gy,
            "oar_d5_hotspot_gy": oar_hotspot_d5,
            "oar_dose_sparing_gy": float(oar_sparing_gy),
            "mean_false_escalation_rate": float(df["false_escalation_rate"].mean()),
            "mean_true_escalation_rate": float(df["true_escalation_rate"].mean()),
            "mean_decision_regret": float(df["decision_regret"].mean()),
            "cumulative_decision_regret": float(df["decision_regret"].sum()),
            "mean_f_coverage": float(df["f_coverage"].mean()),
            "mean_trust_held_rate": float(df["trust_held_rate"].mean()),
            "shift_triggered_fractions": int(df["shift_is_stale"].sum()),
            "held_prescribed_drift_gy": float(df["held_prescribed_drift_gy"].sum()),
            "held_delivered_drift_gy": float(df["held_delivered_drift_gy"].sum()),
            "mean_ere_peak_exit_pct": float(df["ere_peak_exit_pct"].mean()),
            "max_ere_abs_delta_gy": float(df["ere_max_abs_delta_gy"].max()),
            "oar_perfusion_drop_pct": float(df["oar_perfusion_drop_pct"].iloc[-1]),
        }
        metadata: dict[str, Any] = {
            "estimator": self.estimator,
            "protocol": self.protocol_name,
            "snr": self.snr,
            "alpha": self.alpha,
            "hypoxia_f_threshold": self.hypoxia_f_threshold,
            "cost_ratio": self.cost_ratio,
            "boost_factor": self.boost_factor,
            "apply_mr_linac_ere": self.apply_mr_linac_ere,
            "b_field_tesla": self.b_field_tesla if self.apply_mr_linac_ere else 0.0,
            "substrate_class": self.substrate.__class__.__name__,
            "substrate_shape": list(self.substrate.shape),
            "seed": self.seed,
        }
        return ClosedLoopResult(fractions_df=df, summary=summary, metadata=metadata)


def compare_closed_loop_policies(
    substrate: BaseTwinSubstrate | Literal["organ", "dose_geometry"] = "organ",
    *,
    protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray = "osipi",
    estimator: str = "crlb",
    n_fractions: int = 5,
    snr: float = 30.0,
    snr_schedule: Sequence[float] | None = None,
    alpha: float = 0.10,
    hypoxia_f_threshold: float = 0.12,
    cost_ratio: float = 0.75,
    boost_factor: float = 1.40,
    include_oracle: bool = True,
    apply_mr_linac_ere: bool | None = None,
    b_field_tesla: float | None = None,
    seed: int = DEFAULT_SEED,
) -> ClosedLoopComparisonReport:
    """Compare uncalibrated NLLS vs. conformalized + trust-gated intervals on identical twin trajectories."""
    configs: list[tuple[str, IntervalMode, bool, str]] = [
        ("uncalibrated_nlls", "uncalibrated", False, "floored_railed"),
        ("conformal_trust_gated", "conformal", True, "honest_crlb"),
    ]
    if include_oracle:
        configs.append(("oracle_reference", "oracle", True, "honest_crlb"))

    runs: dict[str, ClosedLoopResult] = {}
    for label, mode, trust_flag, sd_conv in configs:
        loop = SyntheticTwinLoop(
            substrate=substrate,
            protocol=protocol,
            estimator=estimator,
            n_fractions=n_fractions,
            snr=snr,
            snr_schedule=snr_schedule,
            alpha=alpha,
            hypoxia_f_threshold=hypoxia_f_threshold,
            cost_ratio=cost_ratio,
            boost_factor=boost_factor,
            interval_mode=mode,
            uncalibrated_sd_convention=sd_conv,
            enable_trust_gate=trust_flag,
            apply_mr_linac_ere=apply_mr_linac_ere,
            b_field_tesla=b_field_tesla,
            seed=seed,
        )
        runs[label] = loop.run()

    uncal_oar = float(runs["uncalibrated_nlls"].summary["cumulative_oar_dose_gy"])
    rows: list[dict[str, Any]] = []
    for label, res in runs.items():
        s = res.summary
        sparing_vs_uncal = uncal_oar - float(s["cumulative_oar_dose_gy"])
        rows.append(
            {
                "policy": label,
                "tumor_control_proxy": float(s["tumor_control_proxy"]),
                "cumulative_tumor_dose_gy": float(s["cumulative_tumor_dose_gy"]),
                "cumulative_oar_dose_gy": float(s["cumulative_oar_dose_gy"]),
                "oar_dose_sparing_gy": float(sparing_vs_uncal),
                "mean_false_escalation_rate": float(s["mean_false_escalation_rate"]),
                "mean_true_escalation_rate": float(s["mean_true_escalation_rate"]),
                "mean_decision_regret": float(s["mean_decision_regret"]),
                "mean_f_coverage": float(s["mean_f_coverage"]),
                "mean_trust_held_rate": float(s["mean_trust_held_rate"]),
                "held_delivered_drift_gy": float(s["held_delivered_drift_gy"]),
                "oar_perfusion_drop_pct": float(s["oar_perfusion_drop_pct"]),
            }
        )

    policy_df = pd.DataFrame(rows)
    uncal_s = runs["uncalibrated_nlls"].summary
    conf_s = runs["conformal_trust_gated"].summary

    summary: dict[str, Any] = {
        "n_fractions": int(n_fractions),
        "snr": float(snr),
        "apply_mr_linac_ere": bool(conf_s["apply_mr_linac_ere"]),
        "b_field_tesla": float(conf_s["b_field_tesla"]),
        "false_escalation_reduction": float(
            uncal_s["mean_false_escalation_rate"] - conf_s["mean_false_escalation_rate"]
        ),
        "regret_reduction": float(
            uncal_s["mean_decision_regret"] - conf_s["mean_decision_regret"]
        ),
        "conformal_oar_sparing_vs_uncalibrated_gy": float(
            uncal_oar - float(conf_s["cumulative_oar_dose_gy"])
        ),
        "conformal_tumor_control_proxy": float(conf_s["tumor_control_proxy"]),
        "uncalibrated_tumor_control_proxy": float(uncal_s["tumor_control_proxy"]),
    }
    metadata: dict[str, Any] = {
        "estimator": estimator,
        "protocol": str(protocol),
        "alpha": float(alpha),
        "hypoxia_f_threshold": float(hypoxia_f_threshold),
        "cost_ratio": float(cost_ratio),
        "apply_mr_linac_ere": bool(conf_s["apply_mr_linac_ere"]),
        "b_field_tesla": float(conf_s["b_field_tesla"]),
        "seed": int(seed),
    }
    return ClosedLoopComparisonReport(
        policy_df=policy_df,
        runs=runs,
        summary=summary,
        metadata=metadata,
    )
