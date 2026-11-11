"""Model-misspecification stress harness for bi-exponential IVIM estimators.

Evaluates bi-exponential estimators under controlled non-bi-exponential forward
truths (``"dispersion_gamma"``, ``"dispersion_lognormal"``, ``"stretched"``,
``"triexp"``, ``"kurtosis"``, and ``"heavy_tail"``), explicitly separating
global marginal coverage from conditional coverage degradation on well-identified
slow tissue diffusion ``D`` versus microvascular pseudo-diffusion ``D*``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..evaluate import BenchmarkSuite
from ..phantoms.noise import inject_noise
from ..phantoms.synthetic import (
    CONTINUITY_EXTRA,
    DEFAULT_EXTRA,
    DEFAULT_SEED,
    FAMILIES,
    _evaluate_family,
    ivim_biexp,
    ivim_kurtosis,
    sample_organ_params,
)
from ..protocols import AcquisitionProtocol, resolve_bvalues
from ..registry import DEFAULT_REGISTERED_ESTIMATORS, RegisteredEstimator, run_estimator
from ..runner import _resolve_estimator_map, _sanitize_json, dataframe_to_latex, dataframe_to_markdown

__all__ = [
    "MISSPEC_FAMILIES",
    "DEFAULT_MISSPEC_EXTRA",
    "CONTINUITY_MISSPEC_EXTRA",
    "MisspecificationReport",
    "MisspecificationHarness",
    "evaluate_misspecification",
    "misspecification_continuity_residual",
]

MISSPEC_FAMILIES: tuple[str, ...] = (
    "biexp",
    "dispersion_gamma",
    "dispersion_lognormal",
    "stretched",
    "triexp",
    "kurtosis",
    "heavy_tail",
)

DEFAULT_MISSPEC_EXTRA: dict[str, dict[str, float]] = {
    **DEFAULT_EXTRA,
    "kurtosis": {"K": 0.85},
    "heavy_tail": {"df": 3.5, "outlier_scale": 1.35},
}

CONTINUITY_MISSPEC_EXTRA: dict[str, dict[str, float]] = {
    **CONTINUITY_EXTRA,
    "kurtosis": {"K": 0.0},
    "heavy_tail": {"df": 1e9, "outlier_scale": 0.0},
}


def _simulate_truth_signals(
    family: str,
    bvalues: np.ndarray,
    params: np.ndarray,
    *,
    snr: float,
    noise_type: str,
    n_coils: int,
    extra: Mapping[str, float] | None,
    seed: int,
    s0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate clean and noisy multi-b signals for any forward truth regime."""
    if family not in MISSPEC_FAMILIES:
        raise ValueError(
            f"Unknown misspecification truth {family!r}; choose from {list(MISSPEC_FAMILIES)}."
        )
    kw = dict(DEFAULT_MISSPEC_EXTRA[family])
    if extra is not None:
        kw.update(extra)

    if family in FAMILIES:
        clean = _evaluate_family(family, bvalues, params, kw, s0=s0)
    elif family == "kurtosis":
        clean = ivim_kurtosis(
            bvalues,
            params[:, 0],
            params[:, 1],
            params[:, 2],
            K=kw["K"],
            s0=s0,
        )
    else:
        # "heavy_tail": Student-t physiological/pulsatile shot-to-shot perturbations
        clean = ivim_biexp(bvalues, params[:, 0], params[:, 1], params[:, 2], s0=s0)

    rng_noise = np.random.Generator(np.random.PCG64(int(seed) + 7000))
    if family == "heavy_tail":
        df_val = max(float(kw["df"]), 2.05)
        out_scale = float(kw["outlier_scale"])
        if out_scale == 0.0 or np.isinf(snr):
            noisy = inject_noise(
                clean,
                snr=snr,
                noise_type=noise_type,
                n_coils=n_coils,
                seed=rng_noise,
                s0=s0,
            )
        else:
            sigma = float(s0) / float(snr)
            # Scale Student-t innovations by sqrt((df - 2) / df) * out_scale so df->inf matches Gaussian
            t_scale = sigma * out_scale * np.sqrt((df_val - 2.0) / df_val)
            eps_re = rng_noise.standard_t(df_val, size=clean.shape) * t_scale
            eps_im = rng_noise.standard_t(df_val, size=clean.shape) * t_scale
            if noise_type.strip().lower() == "gaussian":
                noisy = clean + eps_re
            else:
                noisy = np.sqrt((clean + eps_re) ** 2 + eps_im**2)
    else:
        noisy = inject_noise(
            clean,
            snr=snr,
            noise_type=noise_type,
            n_coils=n_coils,
            seed=rng_noise,
            s0=s0,
        )
    return clean, noisy


def misspecification_continuity_residual(
    family: str,
    *,
    organ: str = "liver",
    protocol: str | AcquisitionProtocol | Sequence[float] = "lattice_22b",
    n_voxels: int = 1000,
    seed: int = DEFAULT_SEED,
) -> float:
    """Maximum clean-signal discrepancy between ``family`` at its continuity limit and ``biexp``."""
    bvalues, _ = resolve_bvalues(protocol)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    params = sample_organ_params(organ=organ, n_voxels=n_voxels, prior="realistic", seed=rng)
    ref_clean, _ = _simulate_truth_signals(
        "biexp",
        bvalues,
        params,
        snr=float("inf"),
        noise_type="none",
        n_coils=1,
        extra={},
        seed=seed,
    )
    lim_clean, _ = _simulate_truth_signals(
        family,
        bvalues,
        params,
        snr=float("inf"),
        noise_type="none",
        n_coils=1,
        extra=CONTINUITY_MISSPEC_EXTRA[family],
        seed=seed,
    )
    return float(np.max(np.abs(lim_clean - ref_clean)))


@dataclass
class MisspecificationReport:
    """Structured report separating global vs. ``D``/``D*`` conditional misspecification degradation."""

    dataframe: pd.DataFrame
    compartment_summary: pd.DataFrame
    metadata: dict[str, Any]

    SUMMARY_COLS: tuple[str, ...] = (
        "method",
        "truth_family",
        "D_marginal_cov",
        "D_worst_cond_cov",
        "Dstar_marginal_cov",
        "Dstar_high_tercile_cov",
        "D_cov_drop",
        "Dstar_cov_drop",
        "d_vs_dstar_cond_gap",
    )

    def format_table(self) -> str:
        present = [c for c in self.SUMMARY_COLS if c in self.compartment_summary.columns]
        return self.compartment_summary[present].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )

    def to_markdown(self, float_fmt: str = ".3f") -> str:
        return dataframe_to_markdown(
            self.compartment_summary,
            columns=self.SUMMARY_COLS,
            float_fmt=float_fmt,
        )

    def to_latex(
        self,
        *,
        caption: str = "Misspecification stress benchmark separating tissue D vs. pseudo-diffusion D* coverage.",
        label: str = "tab:misspecification",
        float_fmt: str = ".3f",
    ) -> str:
        return dataframe_to_latex(
            self.compartment_summary,
            columns=self.SUMMARY_COLS,
            caption=caption,
            label=label,
            float_fmt=float_fmt,
        )

    @property
    def summary_df(self) -> pd.DataFrame:
        """Alias for :attr:`compartment_summary`."""
        return self.compartment_summary

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_json(
            {
                "metadata": self.metadata,
                "compartment_summary": self.compartment_summary.to_dict(orient="records"),
                "records": self.dataframe.to_dict(orient="records"),
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class MisspecificationHarness:
    """Stress harness evaluating bi-exponential IVIM estimators under forward-model mismatch."""

    def __init__(
        self,
        protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray = "osipi",
        *,
        organ: str = "liver",
        snr: float = 30.0,
        alpha: float = 0.10,
        n_voxels: int = 300,
        noise_type: str = "rician",
        n_coils: int = 1,
        n_boot: int = 200,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.bvalues, self.protocol_name = resolve_bvalues(protocol)
        self.organ = organ
        self.snr = float(snr)
        self.alpha = float(alpha)
        self.nominal = 1.0 - float(alpha)
        self.n_voxels = int(n_voxels)
        self.noise_type = noise_type
        self.n_coils = int(n_coils)
        self.n_boot = int(n_boot)
        self.seed = int(seed)

    def evaluate(
        self,
        estimators: Sequence[str | RegisteredEstimator | Callable[..., Any] | Any]
        | Mapping[str, Any]
        | None = None,
        *,
        truth_families: Sequence[str] = MISSPEC_FAMILIES,
        families: Sequence[str] | None = None,
        extra_overrides: Mapping[str, Mapping[str, float]] | None = None,
    ) -> MisspecificationReport:
        """Run ``estimators`` across matched ``biexp`` and non-bi-exponential ``truth_families``."""
        est_map = _resolve_estimator_map(estimators or DEFAULT_REGISTERED_ESTIMATORS)
        fams = list(families if families is not None else truth_families)
        if "biexp" not in fams:
            fams = ["biexp", *fams]

        rng_params = np.random.Generator(np.random.PCG64(self.seed))
        truth = sample_organ_params(
            organ=self.organ,
            n_voxels=self.n_voxels,
            prior="realistic",
            seed=rng_params,
        )

        suite = BenchmarkSuite(
            protocol=self.bvalues,
            snr=self.snr,
            alpha=self.alpha,
            n_boot=self.n_boot,
            seed=self.seed,
        )

        frames: list[pd.DataFrame] = []
        for fam in fams:
            ov = (extra_overrides or {}).get(fam)
            _, noisy = _simulate_truth_signals(
                fam,
                self.bvalues,
                truth,
                snr=self.snr,
                noise_type=self.noise_type,
                n_coils=self.n_coils,
                extra=ov,
                seed=self.seed,
            )
            for est_name, est_obj in est_map.items():
                norm_out = run_estimator(
                    est_obj,
                    noisy,
                    self.bvalues,
                    alpha=self.alpha,
                    snr=self.snr,
                )
                df_f = suite.evaluate(
                    truth,
                    point_pred=norm_out.point,
                    intervals=(norm_out.lower, norm_out.upper),
                    std_pred=norm_out.std,
                    method=est_name,
                    bvalues=self.bvalues,
                    snr=self.snr,
                    as_dataframe=True,
                )
                assert isinstance(df_f, pd.DataFrame)
                df_f = df_f.copy()
                df_f.insert(1, "truth_family", fam)
                # Worst stratum conditional coverage across low/mid/high terciles
                df_f["worst_tercile_cov"] = df_f[
                    ["tercile_cov_low", "tercile_cov_mid", "tercile_cov_high"]
                ].min(axis=1)
                df_f["conditional_gap"] = df_f["worst_tercile_cov"] - self.nominal
                mean_abs_true = [float(np.mean(np.abs(truth[:, j]))) for j in range(3)]
                df_f["rel_bias"] = df_f["bias"] / np.maximum(mean_abs_true, 1e-12)
                frames.append(df_f)

        full_df = pd.concat(frames, ignore_index=True)

        # Compute coverage degradation relative to matched biexp baseline per (method, parameter)
        biexp_ref = (
            full_df[full_df["truth_family"] == "biexp"]
            .set_index(["method", "parameter"])[["marginal_coverage", "worst_tercile_cov", "rmse"]]
        )
        marg_drops = []
        cond_drops = []
        rmse_ratios = []
        for _, row in full_df.iterrows():
            key = (str(row["method"]), str(row["parameter"]))
            ref_m = float(biexp_ref.loc[key, "marginal_coverage"])
            ref_c = float(biexp_ref.loc[key, "worst_tercile_cov"])
            ref_r = max(float(biexp_ref.loc[key, "rmse"]), 1e-12)
            marg_drops.append(ref_m - float(row["marginal_coverage"]))
            cond_drops.append(ref_c - float(row["worst_tercile_cov"]))
            rmse_ratios.append(float(row["rmse"]) / ref_r)

        full_df["marginal_drop_vs_biexp"] = marg_drops
        full_df["conditional_drop_vs_biexp"] = cond_drops
        full_df["rmse_inflation_vs_biexp"] = rmse_ratios

        # Build D vs D* compartment comparison table
        comp_rows: list[dict[str, Any]] = []
        for (m_name, fam), grp in full_df.groupby(["method", "truth_family"], sort=False):
            d_row = grp[grp["parameter"] == "D"].iloc[0]
            ds_row = grp[grp["parameter"] == "Dstar"].iloc[0]
            f_row = grp[grp["parameter"] == "f"].iloc[0]
            comp_rows.append(
                {
                    "method": m_name,
                    "truth_family": fam,
                    "global_marginal_cov": float(grp["marginal_coverage"].mean()),
                    "D_marginal_cov": float(d_row["marginal_coverage"]),
                    "D_worst_cond_cov": float(d_row["worst_tercile_cov"]),
                    "D_cov_drop": float(d_row["marginal_drop_vs_biexp"]),
                    "D_rel_bias": float(d_row["rel_bias"]),
                    "Dstar_marginal_cov": float(ds_row["marginal_coverage"]),
                    "Dstar_high_tercile_cov": float(ds_row["tercile_cov_high"]),
                    "Dstar_worst_cond_cov": float(ds_row["worst_tercile_cov"]),
                    "Dstar_cov_drop": float(ds_row["marginal_drop_vs_biexp"]),
                    "Dstar_rel_bias": float(ds_row["rel_bias"]),
                    "f_marginal_cov": float(f_row["marginal_coverage"]),
                    "d_vs_dstar_cond_gap": float(
                        d_row["worst_tercile_cov"] - ds_row["worst_tercile_cov"]
                    ),
                }
            )
        comp_df = pd.DataFrame.from_records(comp_rows)

        return MisspecificationReport(
            dataframe=full_df,
            compartment_summary=comp_df,
            metadata={
                "protocol": self.protocol_name,
                "organ": self.organ,
                "snr": self.snr,
                "alpha": self.alpha,
                "nominal_coverage": self.nominal,
                "n_voxels": self.n_voxels,
                "noise_type": self.noise_type,
                "truth_families": fams,
                "seed": self.seed,
            },
        )


def evaluate_misspecification(
    estimators: Sequence[str | RegisteredEstimator | Callable[..., Any] | Any]
    | Mapping[str, Any]
    | None = None,
    *,
    protocol: str | AcquisitionProtocol | Sequence[float] = "osipi",
    organ: str = "liver",
    snr: float = 30.0,
    alpha: float = 0.10,
    n_voxels: int = 300,
    n_boot: int = 200,
    truth_families: Sequence[str] = MISSPEC_FAMILIES,
    families: Sequence[str] | None = None,
    noise_type: str = "rician",
    seed: int = DEFAULT_SEED,
) -> MisspecificationReport:
    """Convenience functional entry point for :class:`MisspecificationHarness`."""
    harness = MisspecificationHarness(
        protocol=protocol,
        organ=organ,
        snr=snr,
        alpha=alpha,
        n_voxels=n_voxels,
        n_boot=n_boot,
        noise_type=noise_type,
        seed=seed,
    )
    return harness.evaluate(
        estimators=estimators,
        truth_families=families if families is not None else truth_families,
    )
