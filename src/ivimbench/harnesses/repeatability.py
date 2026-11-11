"""Synthetic scan–rescan repeatability harness under Rician and non-central chi noise.

Simulates paired same-session scan–rescan acquisitions (``scan_A``, ``scan_B``)
under exact Rician or multi-coil non-central chi magnitude noise with optional
coil/SNR/gain drift. Evaluates ground-truth-free test–retest interval coverage
(whether ``est_B`` falls inside ``scan_A``'s interval and vice versa) alongside
Spearman width-vs-scatter rank correlation to separate uncertainty precision
sizing from systematic bias.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..phantoms.dro import DigitalReferenceObject
from ..phantoms.noise import inject_noise
from ..phantoms.synthetic import (
    DEFAULT_EXTRA,
    DEFAULT_SEED,
    FAMILIES,
    PARAM_NAMES,
    SyntheticCohort,
    _evaluate_family,
    sample_organ_params,
)
from ..protocols import AcquisitionProtocol, resolve_bvalues
from ..registry import DEFAULT_REGISTERED_ESTIMATORS, RegisteredEstimator, run_estimator
from ..runner import _resolve_estimator_map, _sanitize_json, dataframe_to_latex, dataframe_to_markdown

__all__ = [
    "ScanRescanPair",
    "RepeatabilityReport",
    "RepeatabilityHarness",
    "simulate_scan_rescan",
    "evaluate_repeatability",
]


@dataclass
class ScanRescanPair:
    """Paired synthetic same-session scan–rescan DWI acquisition."""

    scan_A: np.ndarray
    scan_B: np.ndarray
    bvalues: np.ndarray
    signals_clean: np.ndarray
    params_true: np.ndarray | None
    snr_A: float | np.ndarray
    snr_B: float | np.ndarray
    n_coils_A: int
    n_coils_B: int
    s0_A: float
    s0_B: float
    noise_type: str = "rician"
    organ: str = "liver"
    protocol: str = "osipi"
    seed: int = DEFAULT_SEED

    def swapped(self) -> ScanRescanPair:
        """Return the scan–rescan pair with ``scan_A`` and ``scan_B`` exchanged."""
        return ScanRescanPair(
            scan_A=self.scan_B,
            scan_B=self.scan_A,
            bvalues=self.bvalues,
            signals_clean=self.signals_clean,
            params_true=self.params_true,
            snr_A=self.snr_B,
            snr_B=self.snr_A,
            n_coils_A=self.n_coils_B,
            n_coils_B=self.n_coils_A,
            s0_A=self.s0_B,
            s0_B=self.s0_A,
            noise_type=self.noise_type,
            organ=self.organ,
            protocol=self.protocol,
            seed=self.seed,
        )


def simulate_scan_rescan(
    protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray = "osipi",
    *,
    organ: str = "liver",
    n_voxels: int = 500,
    snr: float | np.ndarray = 30.0,
    noise_type: str = "rician",
    n_coils: int = 1,
    snr_drift: float = 0.0,
    gain_drift: float = 0.0,
    n_coils_B: int | None = None,
    family: str = "biexp",
    cohort: SyntheticCohort | DigitalReferenceObject | None = None,
    seed: int = DEFAULT_SEED,
    seed_A: int | None = None,
    seed_B: int | None = None,
) -> ScanRescanPair:
    """Simulate paired same-session scan–rescan DWI acquisitions ``(scan_A, scan_B)``.

    Parameters
    ----------
    protocol : str, AcquisitionProtocol, or sequence of float, default="osipi"
        Diffusion b-value schedule in ``s/mm^2`` (ignored when ``cohort`` is given).
    organ : str, default="liver"
        Anatomical parameter prior.
    n_voxels : int, default=500
        Number of voxels when synthesizing a new cohort.
    snr : float or ndarray, default=30.0
        Baseline ``b = 0`` signal-to-noise ratio for ``scan_A``.
    noise_type : {"rician", "noncentral_chi", "gaussian"}, default="rician"
        Magnitude noise distribution.
    n_coils : int, default=1
        Number of receiver array channels for ``scan_A``.
    snr_drift : float, default=0.0
        Fractional SNR drift on ``scan_B``: ``snr_B = snr_A * (1 + snr_drift)``
        (must satisfy ``1 + snr_drift > 0``).
    gain_drift : float, default=0.0
        Multiplicative receiver gain / ``S0`` drift on ``scan_B``:
        ``s0_B = 1.0 * (1 + gain_drift)``.
    n_coils_B : int or None, optional
        Optional receiver channel count for ``scan_B`` (e.g., simulating coil
        element dropout or array reconfiguration). Defaults to ``n_coils``.
    family : str, default="biexp"
        Forward signal family used when generating a fresh cohort.
    cohort : SyntheticCohort, DigitalReferenceObject, or None, optional
        Existing clean cohort/DRO to draw ``scan_A`` and ``scan_B`` from.
    seed : int, default=42
        Master ``PCG64`` seed.
    seed_A, seed_B : int or None, optional
        Explicit per-scan noise seeds (default ``seed + 7001`` and ``seed + 7002``).
    """
    if 1.0 + float(snr_drift) <= 0.0:
        raise ValueError(f"`snr_drift` must exceed -1.0; got {snr_drift}.")
    if 1.0 + float(gain_drift) <= 0.0:
        raise ValueError(f"`gain_drift` must exceed -1.0; got {gain_drift}.")

    coils_a = int(n_coils)
    coils_b = coils_a if n_coils_B is None else int(n_coils_B)
    if coils_a < 1 or coils_b < 1:
        raise ValueError("`n_coils` and `n_coils_B` must be >= 1.")

    if cohort is not None:
        if isinstance(cohort, DigitalReferenceObject):
            mask = cohort.foreground_mask
            clean_a = cohort.signals_clean[mask]
            params_true = cohort.params_map[mask]
            b_arr = cohort.bvalues
            proto_name = cohort.protocol
            organ_name = cohort.organ
        else:
            clean_a = cohort.signals_clean
            params_true = cohort.params
            b_arr = cohort.bvalues
            proto_name = cohort.protocol
            organ_name = cohort.organ
    else:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family {family!r}; choose from {list(FAMILIES)}.")
        b_arr, proto_name = resolve_bvalues(protocol)
        rng_p = np.random.Generator(np.random.PCG64(int(seed)))
        params_true = sample_organ_params(
            organ=organ,
            n_voxels=int(n_voxels),
            prior="realistic",
            seed=rng_p,
        )
        clean_a = _evaluate_family(family, b_arr, params_true, dict(DEFAULT_EXTRA[family]), s0=1.0)
        organ_name = organ

    s0_a = 1.0
    s0_b = 1.0 * (1.0 + float(gain_drift))
    clean_b = clean_a * (1.0 + float(gain_drift))

    snr_a_eff = float(snr) if np.ndim(snr) == 0 else np.asarray(snr, dtype=np.float64)
    snr_b_eff = snr_a_eff * (1.0 + float(snr_drift))

    rng_a = np.random.Generator(
        np.random.PCG64(int(seed) + 7001 if seed_A is None else int(seed_A))
    )
    rng_b = np.random.Generator(
        np.random.PCG64(int(seed) + 7002 if seed_B is None else int(seed_B))
    )

    scan_a = inject_noise(
        clean_a,
        snr=snr_a_eff,
        noise_type=noise_type,
        n_coils=coils_a,
        seed=rng_a,
        s0=s0_a,
    )
    scan_b = inject_noise(
        clean_b,
        snr=snr_b_eff,
        noise_type=noise_type,
        n_coils=coils_b,
        seed=rng_b,
        s0=s0_b,
    )

    return ScanRescanPair(
        scan_A=scan_a,
        scan_B=scan_b,
        bvalues=b_arr,
        signals_clean=clean_a,
        params_true=params_true,
        snr_A=snr_a_eff,
        snr_B=snr_b_eff,
        n_coils_A=coils_a,
        n_coils_B=coils_b,
        s0_A=s0_a,
        s0_B=s0_b,
        noise_type=noise_type,
        organ=organ_name,
        protocol=proto_name,
        seed=int(seed),
    )


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Deterministic Spearman rank correlation returning ``(0.0, 1.0)`` on constant vectors."""
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(np.sum(valid)) < 3:
        return 0.0, 1.0
    xv = x_arr[valid]
    yv = y_arr[valid]
    if float(np.std(xv)) <= 1e-15 or float(np.std(yv)) <= 1e-15:
        return 0.0, 1.0
    res = stats.spearmanr(xv, yv)
    r_val = float(res.statistic)
    p_val = float(res.pvalue)
    if np.isnan(r_val):
        return 0.0, 1.0
    return r_val, p_val


@dataclass
class RepeatabilityReport:
    """Ground-truth-free scan–rescan repeatability and width-vs-scatter report."""

    dataframe: pd.DataFrame
    summary_dataframe: pd.DataFrame
    metadata: dict[str, Any]

    SUMMARY_COLS: tuple[str, ...] = (
        "method",
        "parameter",
        "retest_cov_A_on_B",
        "retest_cov_B_on_A",
        "symmetric_retest_cov",
        "nominal_retest_prob",
        "width_scatter_spearman_r",
        "within_voxel_sd",
        "repeatability_coeff",
    )

    def format_table(self) -> str:
        present = [c for c in self.SUMMARY_COLS if c in self.dataframe.columns]
        return self.dataframe[present].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )

    def to_markdown(self, float_fmt: str = ".3f") -> str:
        return dataframe_to_markdown(
            self.dataframe,
            columns=self.SUMMARY_COLS,
            float_fmt=float_fmt,
        )

    def to_latex(
        self,
        *,
        caption: str = "Ground-truth-free scan--rescan interval coverage and Spearman width-vs-scatter correlation.",
        label: str = "tab:repeatability",
        float_fmt: str = ".3f",
    ) -> str:
        return dataframe_to_latex(
            self.dataframe,
            columns=self.SUMMARY_COLS,
            caption=caption,
            label=label,
            float_fmt=float_fmt,
        )

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_json(
            {
                "metadata": self.metadata,
                "summary": self.summary_dataframe.to_dict(orient="records"),
                "records": self.dataframe.to_dict(orient="records"),
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class RepeatabilityHarness:
    """Ground-truth-free scan–rescan interval calibration and repeatability evaluator."""

    def __init__(self, *, alpha: float = 0.10) -> None:
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"`alpha` must lie strictly in (0, 1); got {alpha}.")
        self.alpha = float(alpha)
        self.nominal = 1.0 - float(alpha)
        z_crit = float(stats.norm.ppf(1.0 - 0.5 * self.alpha))
        # Exact analytical probability P(|theta_hat_B - theta_hat_A| <= z * sigma) for iid N(theta, sigma^2)
        self.nominal_retest_prob = float(2.0 * stats.norm.cdf(z_crit / np.sqrt(2.0)) - 1.0)

    def evaluate_pair(
        self,
        scan_A: np.ndarray | ScanRescanPair,
        scan_B: np.ndarray | None = None,
        bvalues: Sequence[float] | np.ndarray | str | None = None,
        *,
        estimators: Sequence[str | RegisteredEstimator | Callable[..., Any] | Any]
        | Mapping[str, Any]
        | None = None,
        audit_truth: np.ndarray | None = None,
    ) -> RepeatabilityReport:
        """Evaluate ground-truth-free scan–rescan interval capture and Spearman sizing."""
        if isinstance(scan_A, ScanRescanPair):
            sig_a = scan_A.scan_A
            sig_b = scan_A.scan_B
            b_arr = scan_A.bvalues
            truth_opt = audit_truth if audit_truth is not None else scan_A.params_true
            pair_meta = {
                "noise_type": scan_A.noise_type,
                "n_coils_A": scan_A.n_coils_A,
                "n_coils_B": scan_A.n_coils_B,
                "s0_A": scan_A.s0_A,
                "s0_B": scan_A.s0_B,
                "organ": scan_A.organ,
                "protocol": scan_A.protocol,
            }
        else:
            if scan_B is None or bvalues is None:
                raise ValueError("Provide `scan_B` and `bvalues` when passing raw `scan_A` arrays.")
            sig_a = np.atleast_2d(np.asarray(scan_A, dtype=np.float64))
            sig_b = np.atleast_2d(np.asarray(scan_B, dtype=np.float64))
            b_arr, proto_label = resolve_bvalues(bvalues)
            truth_opt = audit_truth
            pair_meta = {"protocol": proto_label}

        if sig_a.shape != sig_b.shape:
            raise ValueError(f"`scan_A` {sig_a.shape} and `scan_B` {sig_b.shape} shapes must match.")

        est_map = _resolve_estimator_map(estimators or DEFAULT_REGISTERED_ESTIMATORS)
        records: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []

        for est_name, est_obj in est_map.items():
            out_a = run_estimator(est_obj, sig_a, b_arr, alpha=self.alpha)
            out_b = run_estimator(est_obj, sig_b, b_arr, alpha=self.alpha)

            method_records: list[dict[str, Any]] = []
            for j, pname in enumerate(PARAM_NAMES):
                pt_a, lo_a, hi_a = out_a.point[:, j], out_a.lower[:, j], out_a.upper[:, j]
                pt_b, lo_b, hi_b = out_b.point[:, j], out_b.lower[:, j], out_b.upper[:, j]

                width_a = np.maximum(hi_a - lo_a, 0.0)
                width_b = np.maximum(hi_b - lo_b, 0.0)
                sym_width = 0.5 * (width_a + width_b)

                diff = pt_b - pt_a
                abs_scatter = np.abs(diff)
                wsd = float(np.sqrt(0.5 * np.mean(diff * diff)))
                rc = float(1.96 * np.sqrt(2.0) * wsd)

                # Ground-truth-free test-retest interval capture
                cov_a_on_b = float(np.mean((pt_b >= lo_a) & (pt_b <= hi_a)))
                cov_b_on_a = float(np.mean((pt_a >= lo_b) & (pt_a <= hi_b)))
                sym_cov = 0.5 * (cov_a_on_b + cov_b_on_a)

                # Replicate-adjusted sqrt(2) prediction interval capture (nominal = 1 - alpha)
                mid_a = 0.5 * (lo_a + hi_a)
                half_a = 0.5 * width_a * np.sqrt(2.0)
                mid_b = 0.5 * (lo_b + hi_b)
                half_b = 0.5 * width_b * np.sqrt(2.0)
                rep_pred_a = float(np.mean((pt_b >= mid_a - half_a) & (pt_b <= mid_a + half_a)))
                rep_pred_b = float(np.mean((pt_a >= mid_b - half_b) & (pt_a <= mid_b + half_b)))
                sym_rep_pred_cov = 0.5 * (rep_pred_a + rep_pred_b)

                overlap_rate = float(np.mean((lo_a <= hi_b) & (lo_b <= hi_a)))

                r_a, p_a = _safe_spearman(width_a, abs_scatter)
                r_sym, p_sym = _safe_spearman(sym_width, abs_scatter)

                row: dict[str, Any] = {
                    "method": est_name,
                    "parameter": pname,
                    "n_voxels": int(sig_a.shape[0]),
                    "alpha": self.alpha,
                    "nominal_coverage": self.nominal,
                    "nominal_retest_prob": self.nominal_retest_prob,
                    "retest_cov_A_on_B": cov_a_on_b,
                    "retest_cov_B_on_A": cov_b_on_a,
                    "symmetric_retest_cov": sym_cov,
                    "retest_cov_gap": sym_cov - self.nominal_retest_prob,
                    "replicate_prediction_cov": sym_rep_pred_cov,
                    "interval_overlap_rate": overlap_rate,
                    "width_scatter_spearman_r": r_sym,
                    "width_scatter_pvalue": p_sym,
                    "width_A_scatter_spearman_r": r_a,
                    "width_A_scatter_pvalue": p_a,
                    "mean_width_A": float(np.mean(width_a)),
                    "mean_width_B": float(np.mean(width_b)),
                    "within_voxel_sd": wsd,
                    "repeatability_coeff": rc,
                    "mean_retest_drift": float(np.mean(diff)),
                }

                # Optional ground-truth audit separating precision sizing from systematic bias
                if truth_opt is not None:
                    yj = np.asarray(truth_opt, dtype=np.float64)[:, j]
                    bias_a = float(np.mean(pt_a - yj))
                    rmse_a = float(np.sqrt(np.mean((pt_a - yj) ** 2)))
                    truth_cov_a = float(np.mean((yj >= lo_a) & (yj <= hi_a)))
                    row["audit_truth_cov_A"] = truth_cov_a
                    row["audit_bias_A"] = bias_a
                    row["audit_rmse_A"] = rmse_a
                    row["bias_to_scatter_ratio"] = abs(bias_a) / max(wsd, 1e-12)

                records.append(row)
                method_records.append(row)

            summary_rows.append(
                {
                    "method": est_name,
                    "mean_retest_cov_A_on_B": float(
                        np.mean([r["retest_cov_A_on_B"] for r in method_records])
                    ),
                    "mean_symmetric_retest_cov": float(
                        np.mean([r["symmetric_retest_cov"] for r in method_records])
                    ),
                    "mean_replicate_prediction_cov": float(
                        np.mean([r["replicate_prediction_cov"] for r in method_records])
                    ),
                    "mean_width_scatter_spearman_r": float(
                        np.mean([r["width_scatter_spearman_r"] for r in method_records])
                    ),
                    "dstar_retest_cov": float(method_records[1]["symmetric_retest_cov"]),
                    "dstar_spearman_r": float(method_records[1]["width_scatter_spearman_r"]),
                }
            )

        return RepeatabilityReport(
            dataframe=pd.DataFrame.from_records(records),
            summary_dataframe=pd.DataFrame.from_records(summary_rows),
            metadata={
                "alpha": self.alpha,
                "nominal_coverage": self.nominal,
                "nominal_retest_prob": self.nominal_retest_prob,
                **pair_meta,
            },
        )

    evaluate = evaluate_pair


def evaluate_repeatability(
    pair_or_scan_A: ScanRescanPair | np.ndarray | None = None,
    scan_B: np.ndarray | None = None,
    *,
    bvalues: Sequence[float] | np.ndarray | str = "osipi",
    estimators: Sequence[str | RegisteredEstimator | Callable[..., Any] | Any]
    | Mapping[str, Any]
    | None = None,
    alpha: float = 0.10,
    organ: str = "liver",
    n_voxels: int = 400,
    snr: float = 30.0,
    noise_type: str = "rician",
    n_coils: int = 1,
    snr_drift: float = 0.0,
    gain_drift: float = 0.0,
    seed: int = DEFAULT_SEED,
) -> RepeatabilityReport:
    """Simulate (if needed) and evaluate scan–rescan interval repeatability."""
    harness = RepeatabilityHarness(alpha=alpha)
    if pair_or_scan_A is None:
        pair = simulate_scan_rescan(
            protocol=bvalues,
            organ=organ,
            n_voxels=n_voxels,
            snr=snr,
            noise_type=noise_type,
            n_coils=n_coils,
            snr_drift=snr_drift,
            gain_drift=gain_drift,
            seed=seed,
        )
        return harness.evaluate_pair(pair, estimators=estimators)
    if isinstance(pair_or_scan_A, ScanRescanPair):
        return harness.evaluate_pair(pair_or_scan_A, estimators=estimators)
    return harness.evaluate_pair(
        pair_or_scan_A,
        scan_B=scan_B,
        bvalues=bvalues,
        estimators=estimators,
    )
