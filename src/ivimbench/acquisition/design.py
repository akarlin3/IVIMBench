"""Acquisition-design evaluation and CRLB / calibration-aware b-value schedule optimization.

Compares fixed clinical diffusion schedules (e.g., ``"osipi"``, ``"tcga_lihc"``,
``"acrin_6698"``) against analytical CRLB-optimal and calibration-aware b-value
designs, reporting interval sharpness (``delta_sharp``), marginal coverage, and
high-``D*`` tercile conditional coverage (``delta_cond``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..evaluate import BenchmarkSuite
from ..metrics.railing import DEFAULT_PARAM_BOUNDS
from ..phantoms.noise import inject_noise
from ..phantoms.synthetic import DEFAULT_SEED, PARAM_NAMES, ivim_biexp, sample_organ_params
from ..protocols import DEFAULT_SECONDS_PER_VOLUME, AcquisitionProtocol
from ..registry import run_estimator
from ..runner import _sanitize_json, dataframe_to_latex, dataframe_to_markdown
from .crlb import compute_crlb, crlb_optimality_criterion, resolve_schedule_and_nex

__all__ = [
    "OptimizedScheduleResult",
    "SchemeComparisonReport",
    "SchemeEvaluator",
    "optimize_b_schedule",
]


def _logits_to_bvalues(
    u: np.ndarray,
    b_min: float,
    b_max: float,
    pin_b0: bool,
) -> np.ndarray:
    """Map unconstrained logits ``u`` onto non-decreasing b-values in ``[b_min, b_max]``."""
    # Log-compressed spacing concentrates resolution in the sensitive low-b IVIM regime
    u_clipped = np.clip(u, -20.0, 20.0)
    weights = np.exp(u_clipped)
    fracs = np.cumsum(weights) / np.sum(weights)  # len = len(u), ends at 1.0
    scale = 40.0  # characteristic perceptual b-scale in s/mm^2
    log_lo = np.log1p(b_min / scale)
    log_hi = np.log1p(b_max / scale)
    if pin_b0:
        # u has length n_b (last weight represents slack above b_{n_b-1} if desired;
        # here we pin last b to b_max only if slack is small, so use n_b weights where first is pinned to b_min)
        pts_log = log_lo + (log_hi - log_lo) * fracs[:-1]
        interior = scale * np.expm1(pts_log)
        b_full = np.concatenate([[b_min], interior])
    else:
        pts_log = log_lo + (log_hi - log_lo) * fracs[:-1]
        b_full = scale * np.expm1(pts_log)
    return np.sort(np.clip(b_full, b_min, b_max))


@dataclass(frozen=True)
class OptimizedScheduleResult:
    """Container for an optimized IVIM b-value schedule and NEX allocation."""

    key: str
    protocol: AcquisitionProtocol
    bvalues: np.ndarray
    nex: np.ndarray
    objective: str
    objective_value: float
    initial_objective_value: float
    organ: str
    snr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "bvalues": self.bvalues.tolist(),
            "nex": self.nex.astype(int).tolist(),
            "total_nex": int(np.sum(self.nex)),
            "scan_minutes": float(self.protocol.scan_minutes()),
            "objective": self.objective,
            "objective_value": float(self.objective_value),
            "initial_objective_value": float(self.initial_objective_value),
            "organ": self.organ,
            "snr": float(self.snr),
        }


def optimize_b_schedule(
    n_b: int = 8,
    *,
    b_bounds: tuple[float, float] = (0.0, 800.0),
    organ: str = "liver",
    snr: float = 30.0,
    objective: str = "calibration_aware",
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    total_nex: int | None = None,
    pin_b0: bool = True,
    round_step: float | None = 5.0,
    n_prior_samples: int = 256,
    n_restarts: int = 4,
    max_iter: int = 90,
    seed: int = DEFAULT_SEED,
    key: str | None = None,
) -> OptimizedScheduleResult:
    """Optimize an ``n_b``-point IVIM b-value schedule (and optional NEX allocation).

    Parameters
    ----------
    n_b : int, default=8
        Number of distinct b-value entries in the designed schedule (``>= 4``).
    b_bounds : (float, float), default=(0.0, 800.0)
        Minimum and maximum allowable b-values in ``s/mm^2``.
    organ : str, default="liver"
        Target anatomical parameter prior (:data:`~ivimbench.phantoms.ORGAN_PRIORS`).
    snr : float, default=30.0
        Nominal ``b = 0`` signal-to-noise ratio.
    objective : {"calibration_aware", "crlb", "d_optimal"}, default="calibration_aware"
        - ``"crlb"``: Minimizes prior-expected squared relative CRLB error across
          ``(D, D*, f)``.
        - ``"calibration_aware"``: Augments CRLB minimization with a top-``D*``
          tercile variance and Fisher condition-number barrier to preserve low-``b``
          leverage and reduce high-``D*`` conditional under-coverage.
        - ``"d_optimal"``: Minimizes expected ``-log det(FIM)``.
    total_nex : int or None, optional
        Total integer signal excitations budget ``>= n_b``. When ``> n_b``, extra
        excitations are allocated greedily to maximize objective reduction.
    pin_b0 : bool, default=True
        Whether to pin the lowest b-value to ``b_bounds[0]`` (typically ``0.0``).
    round_step : float or None, default=5.0
        Scanner b-value quantization step in ``s/mm^2`` (``None`` keeps continuous floats).
    n_prior_samples : int, default=256
        Number of prior parameter vectors drawn via ``PCG64`` for expectation.
    n_restarts : int, default=4
        Number of deterministic multi-start initializations.
    max_iter : int, default=90
        Maximum Nelder–Mead iterations per restart.
    seed : int, default=42
        Deterministic ``PCG64`` seed.
    """
    nb = int(n_b)
    if nb < 4:
        raise ValueError(f"`n_b` must be >= 4 to identify (S0, D, D*, f); got {n_b}.")
    b_min, b_max = float(b_bounds[0]), float(b_bounds[1])
    if b_min < 0.0 or b_max <= b_min:
        raise ValueError(f"Invalid `b_bounds` {b_bounds!r}.")
    if total_nex is not None and int(total_nex) < nb:
        raise ValueError(f"`total_nex` ({total_nex}) must be >= `n_b` ({nb}).")

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    theta_prior = sample_organ_params(
        organ=organ,
        n_voxels=int(n_prior_samples),
        prior="realistic",
        seed=rng,
    )

    dim_u = nb if pin_b0 else (nb + 1)

    def _loss(u: np.ndarray) -> float:
        b_cand = _logits_to_bvalues(u, b_min=b_min, b_max=b_max, pin_b0=pin_b0)
        return crlb_optimality_criterion(
            b_cand,
            theta_prior,
            criterion=objective,
            weights=weights,
            snr=snr,
            include_s0=True,
        )

    # Baseline log-uniform initialization
    u0_base = np.zeros(dim_u, dtype=np.float64)
    u0_base[-1] = -2.5  # push highest b near b_max
    init_loss = _loss(u0_base)

    best_u = u0_base.copy()
    best_val = init_loss

    for r in range(max(int(n_restarts), 1)):
        if r == 0:
            u_start = u0_base.copy()
        elif r == 1:
            # Low-b enriched initialization
            u_start = np.linspace(-0.8, 0.8, dim_u, dtype=np.float64)
            u_start[-1] = -3.0
        else:
            u_start = u0_base + rng.normal(0.0, 0.75, size=dim_u)
        opt = minimize(
            _loss,
            u_start,
            method="Nelder-Mead",
            options={"maxiter": int(max_iter), "xatol": 1e-2, "fatol": 1e-4},
        )
        if float(opt.fun) < best_val:
            best_val = float(opt.fun)
            best_u = np.asarray(opt.x, dtype=np.float64)

    b_opt = _logits_to_bvalues(best_u, b_min=b_min, b_max=b_max, pin_b0=pin_b0)

    if round_step is not None and float(round_step) > 0.0:
        step = float(round_step)
        b_rounded = np.sort(np.clip(np.round(b_opt / step) * step, b_min, b_max))
        if pin_b0:
            b_rounded[0] = b_min
        # Ensure at least 4 distinct b-values after rounding
        if np.unique(b_rounded).size >= 4:
            rounded_val = crlb_optimality_criterion(
                b_rounded,
                theta_prior,
                criterion=objective,
                weights=weights,
                snr=snr,
                include_s0=True,
            )
            if np.isfinite(rounded_val):
                b_opt = b_rounded
                best_val = rounded_val

    nex_arr = np.ones(nb, dtype=int)
    if total_nex is not None and int(total_nex) > nb:
        rem = int(total_nex) - nb
        for _ in range(rem):
            best_k = 0
            best_k_loss = best_val
            for k in range(nb):
                cand_nex = nex_arr.copy()
                cand_nex[k] += 1
                val_k = crlb_optimality_criterion(
                    b_opt,
                    theta_prior,
                    criterion=objective,
                    weights=weights,
                    snr=snr,
                    nex=cand_nex,
                    include_s0=True,
                )
                if val_k < best_k_loss:
                    best_k_loss = val_k
                    best_k = k
            nex_arr[best_k] += 1
            best_val = best_k_loss

    sched_key = key or f"opt_{objective}_{nb}b"
    proto = AcquisitionProtocol(
        key=sched_key,
        title=f"Optimized {objective} {nb}-b IVIM Schedule ({organ})",
        bvalues=tuple(float(v) for v in b_opt),
        averages=tuple(int(v) for v in nex_arr),
        organ=organ,
        description=f"Optimized {nb}-b schedule under {objective!r} criterion at SNR={snr:g}.",
    )
    return OptimizedScheduleResult(
        key=sched_key,
        protocol=proto,
        bvalues=b_opt,
        nex=nex_arr,
        objective=objective,
        objective_value=float(best_val),
        initial_objective_value=float(init_loss),
        organ=organ,
        snr=float(snr),
    )


@dataclass
class SchemeComparisonReport:
    """Comparison report across clinical and optimized b-value acquisition schemes."""

    dataframe: pd.DataFrame
    summary_dataframe: pd.DataFrame
    reference_scheme: str
    metadata: dict[str, Any]

    SUMMARY_COLS: tuple[str, ...] = (
        "scheme",
        "n_b",
        "total_nex",
        "scan_minutes",
        "marginal_coverage",
        "tercile_cov_high",
        "delta_cond",
        "norm_width",
        "delta_sharp",
        "dstar_railing_pct",
    )

    def format_table(self) -> str:
        """Render a clean terminal table of scheme comparison metrics."""
        present = [c for c in self.SUMMARY_COLS if c in self.summary_dataframe.columns]
        return self.summary_dataframe[present].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )

    def to_markdown(self, float_fmt: str = ".3f") -> str:
        return dataframe_to_markdown(
            self.summary_dataframe,
            columns=self.SUMMARY_COLS,
            float_fmt=float_fmt,
        )

    def to_latex(
        self,
        *,
        caption: str = "IVIM acquisition scheme comparison across sharpness and conditional coverage.",
        label: str = "tab:scheme_comparison",
        float_fmt: str = ".3f",
    ) -> str:
        return dataframe_to_latex(
            self.summary_dataframe,
            columns=self.SUMMARY_COLS,
            caption=caption,
            label=label,
            float_fmt=float_fmt,
        )

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_json(
            {
                "reference_scheme": self.reference_scheme,
                "metadata": self.metadata,
                "summary": self.summary_dataframe.to_dict(orient="records"),
                "records": self.dataframe.to_dict(orient="records"),
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class SchemeEvaluator:
    """Evaluator comparing fixed clinical schedules against CRLB- and calibration-aware designs.

    Reports per-scheme and per-parameter:
    - ``marginal_coverage``: Empirical ``(1 - alpha)`` interval coverage.
    - ``tercile_cov_high``: Empirical coverage in the top ground-truth ``D*`` tercile.
    - ``delta_cond``: High-``D*`` tercile conditional coverage gap
      ``tercile_cov_high - (1 - alpha)`` (``0.0`` = nominal conditional coverage;
      negative values expose high-``D*`` under-coverage).
    - ``norm_width``: Domain-normalized interval sharpness.
    - ``delta_sharp``: Relative change in normalized interval width vs. the
      reference schedule ``(norm_width - ref_norm_width) / ref_norm_width``
      (negative values indicate sharper intervals than the reference schedule).
    """

    def __init__(
        self,
        *,
        organ: str = "liver",
        snr: float = 30.0,
        alpha: float = 0.10,
        n_voxels: int = 300,
        noise_type: str = "rician",
        n_coils: int = 1,
        estimator: Any = "crlb",
        equal_scan_time_minutes: float | None = None,
        seconds_per_volume: float = DEFAULT_SECONDS_PER_VOLUME,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        n_boot: int = 200,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.organ = organ
        self.snr = float(snr)
        self.alpha = float(alpha)
        self.nominal = 1.0 - float(alpha)
        self.n_voxels = int(n_voxels)
        self.noise_type = noise_type
        self.n_coils = int(n_coils)
        self.estimator = estimator
        self.equal_scan_time_minutes = equal_scan_time_minutes
        self.seconds_per_volume = float(seconds_per_volume)
        active = dict(DEFAULT_PARAM_BOUNDS)
        if bounds is not None:
            active.update(bounds)
        self.bounds = active
        self.n_boot = int(n_boot)
        self.seed = int(seed)

    def _unpack_scheme(
        self,
        spec: str | AcquisitionProtocol | OptimizedScheduleResult | Sequence[float] | np.ndarray,
        fallback_label: str | None = None,
    ) -> tuple[str, np.ndarray, np.ndarray, float]:
        if isinstance(spec, OptimizedScheduleResult):
            b_arr, w_nex, label, scan_min = resolve_schedule_and_nex(
                spec.protocol,
                scan_time_minutes=self.equal_scan_time_minutes,
                seconds_per_volume=self.seconds_per_volume,
            )
            return fallback_label or spec.key, b_arr, w_nex, scan_min
        b_arr, w_nex, proto_name, scan_min = resolve_schedule_and_nex(
            spec,
            scan_time_minutes=self.equal_scan_time_minutes,
            seconds_per_volume=self.seconds_per_volume,
        )
        label = fallback_label or proto_name
        return label, b_arr, w_nex, scan_min

    def compare(
        self,
        schemes: Sequence[str | AcquisitionProtocol | OptimizedScheduleResult | Sequence[float]]
        | Mapping[str, Any],
        *,
        reference: str | None = None,
        include_optimized: bool = False,
        optimized_n_b: int = 8,
    ) -> SchemeComparisonReport:
        """Evaluate and compare multiple b-value schedules on a matched organ cohort."""
        scheme_items: list[tuple[str | None, Any]]
        if isinstance(schemes, Mapping):
            scheme_items = [(str(k), v) for k, v in schemes.items()]
        else:
            scheme_items = [(None, s) for s in schemes]

        if include_optimized:
            opt_crlb = optimize_b_schedule(
                n_b=optimized_n_b,
                organ=self.organ,
                snr=self.snr,
                objective="crlb",
                seed=self.seed,
                key=f"opt_crlb_{optimized_n_b}b",
            )
            opt_cal = optimize_b_schedule(
                n_b=optimized_n_b,
                organ=self.organ,
                snr=self.snr,
                objective="calibration_aware",
                seed=self.seed,
                key=f"opt_cal_{optimized_n_b}b",
            )
            scheme_items.extend([(opt_crlb.key, opt_crlb), (opt_cal.key, opt_cal)])

        if not scheme_items:
            raise ValueError("Provide at least one acquisition scheme to `SchemeEvaluator.compare`.")

        # Shared underlying physiological ground-truth parameters across all schedules
        rng_params = np.random.Generator(np.random.PCG64(self.seed))
        truth = sample_organ_params(
            organ=self.organ,
            n_voxels=self.n_voxels,
            prior="realistic",
            seed=rng_params,
        )
        spans = np.array(
            [self.bounds[p][1] - self.bounds[p][0] for p in PARAM_NAMES],
            dtype=np.float64,
        )

        frames: list[pd.DataFrame] = []
        scheme_meta: list[dict[str, Any]] = []

        for idx, (explicit_name, spec) in enumerate(scheme_items):
            label, b_arr, w_nex, scan_min = self._unpack_scheme(spec, fallback_label=explicit_name)
            clean = ivim_biexp(b_arr, truth[:, 0], truth[:, 1], truth[:, 2], s0=1.0)
            # NEX averaging scales per-b effective SNR by sqrt(w_nex)
            snr_b = self.snr * np.sqrt(np.maximum(w_nex, 1e-6))[None, :]
            snr_grid = np.broadcast_to(snr_b, clean.shape)
            rng_noise = np.random.Generator(np.random.PCG64(self.seed + 7000 + idx))
            noisy = inject_noise(
                clean,
                snr=snr_grid,
                noise_type=self.noise_type,
                n_coils=self.n_coils,
                seed=rng_noise,
                s0=1.0,
            )

            norm_out = run_estimator(
                self.estimator,
                noisy,
                b_arr,
                alpha=self.alpha,
                snr=self.snr,
                nex=w_nex,
                bounds=self.bounds,
            )

            suite = BenchmarkSuite(
                protocol=b_arr,
                snr=self.snr,
                alpha=self.alpha,
                bounds=self.bounds,
                n_boot=self.n_boot,
                seed=self.seed + idx,
            )
            df_s = suite.evaluate(
                truth,
                point_pred=norm_out.point,
                intervals=(norm_out.lower, norm_out.upper),
                std_pred=norm_out.std,
                method=label,
                bvalues=b_arr,
                snr=self.snr,
                as_dataframe=True,
            )
            assert isinstance(df_s, pd.DataFrame)
            df_s = df_s.rename(columns={"method": "scheme"})

            crlb_res = compute_crlb(
                b_arr,
                truth,
                snr=self.snr,
                nex=w_nex,
                alpha=self.alpha,
                include_s0=(b_arr.size >= 4),
            )
            crlb_w = np.atleast_2d(crlb_res.interval_width)
            crlb_norm_w = []
            for j in range(3):
                fin = np.isfinite(crlb_w[:, j])
                crlb_norm_w.append(
                    float(np.median(crlb_w[fin, j]) / spans[j]) if np.any(fin) else float("inf")
                )

            df_s["n_b"] = int(b_arr.size)
            df_s["total_nex"] = float(np.sum(w_nex))
            df_s["scan_minutes"] = float(scan_min)
            df_s["crlb_norm_width"] = crlb_norm_w
            df_s["delta_cond"] = df_s["tercile_cov_high"] - self.nominal
            frames.append(df_s)

            dstar_row = df_s[df_s["parameter"] == "Dstar"].iloc[0]
            scheme_meta.append(
                {
                    "scheme": label,
                    "n_b": int(b_arr.size),
                    "total_nex": float(np.sum(w_nex)),
                    "scan_minutes": float(scan_min),
                    "marginal_coverage": float(df_s["marginal_coverage"].mean()),
                    "marginal_coverage_gap": float(df_s["marginal_coverage_gap"].mean()),
                    "tercile_cov_high": float(dstar_row["tercile_cov_high"]),
                    "delta_cond": float(dstar_row["delta_cond"]),
                    "norm_width": float(df_s["norm_width"].mean()),
                    "crlb_norm_width": float(np.mean(crlb_norm_w)),
                    "dstar_railing_pct": float(dstar_row["railing_pct"]),
                    "crlb_objective": crlb_optimality_criterion(
                        b_arr,
                        truth,
                        criterion="calibration_aware",
                        snr=self.snr,
                        nex=w_nex,
                        include_s0=(b_arr.size >= 4),
                    ),
                }
            )

        full_df = pd.concat(frames, ignore_index=True)
        summary_df = pd.DataFrame.from_records(scheme_meta)

        ref_name = reference if reference is not None else str(summary_df["scheme"].iloc[0])
        if ref_name not in set(summary_df["scheme"]):
            raise KeyError(f"Reference scheme {ref_name!r} not in evaluated schemes.")

        ref_summary = summary_df[summary_df["scheme"] == ref_name].iloc[0]
        ref_nw = max(float(ref_summary["norm_width"]), 1e-12)
        summary_df["delta_sharp"] = (summary_df["norm_width"] - ref_nw) / ref_nw
        summary_df["delta_cond_vs_ref"] = (
            summary_df["tercile_cov_high"] - float(ref_summary["tercile_cov_high"])
        )

        ref_by_param = (
            full_df[full_df["scheme"] == ref_name]
            .set_index("parameter")["norm_width"]
            .to_dict()
        )
        full_df["delta_sharp"] = [
            (float(row["norm_width"]) - max(float(ref_by_param[str(row["parameter"])]), 1e-12))
            / max(float(ref_by_param[str(row["parameter"])]), 1e-12)
            for _, row in full_df.iterrows()
        ]

        return SchemeComparisonReport(
            dataframe=full_df,
            summary_dataframe=summary_df,
            reference_scheme=ref_name,
            metadata={
                "organ": self.organ,
                "snr": self.snr,
                "alpha": self.alpha,
                "nominal_coverage": self.nominal,
                "n_voxels": self.n_voxels,
                "noise_type": self.noise_type,
                "seed": self.seed,
            },
        )
