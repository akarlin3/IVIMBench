"""License-safe cross-package `Trust -> Value -> Action` adapter (`avery_stack`).

Bridges the three-layer quantitative IVIM ecosystem while preserving strict MIT
licensing and a NumPy/SciPy-light core install:
- ``ivimbench`` (MIT): Forward physics, DRO substrates, metrics, VoI pricing, and
  closed-loop adaptive radiotherapy simulation.
- ``conformal-ivim`` (MIT, optional ``ivimbench[ecosystem]``): Distribution-free
  Split / CQR / Mondrian conformal prediction (`ConformalIVIM`) and label-free
  two-sample deployment shift monitoring (`DeploymentValidityMonitor`).
- ``sextant-core`` / ``sextant-ivim`` (AGPL-3.0, strictly optional runtime-detected
  via :func:`importlib.util.find_spec` — **never** a static dependency): Bounded
  NLLS fitting (`fit_ivim`), boundary-railing classification (`classify_railing`),
  and bootstrap railing CIs (`bootstrap_railing_ci`).

When optional sibling packages are absent, both :class:`ConformalIVIMEstimator`
and :class:`SextantNLLSEstimator` fall back seamlessly to deterministic (`PCG64`)
MIT-licensed internal implementations inside ``ivimbench``.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.stats import norm

from ..baselines.crlb_estimator import crlb_normal_intervals
from ..baselines.mlp import MLPEstimator
from ..baselines.nlls import NLLSEstimator
from ..metrics.coverage import (
    conformal_offset,
    empirical_coverage,
)
from ..metrics.railing import (
    DEFAULT_PARAM_BOUNDS,
    DEFAULT_RAIL_TOL,
    bootstrap_railing_ci,
    classify_railing,
)
from ..phantoms.dro import generate_dro
from ..phantoms.synthetic import DEFAULT_SEED, PARAM_NAMES, generate_cohort
from ..protocols import AcquisitionProtocol, resolve_bvalues
from ..registry import register_estimator, run_estimator

__all__ = [
    "ConformalMethod",
    "has_conformal_ivim",
    "has_sextant_ivim",
    "ecosystem_availability",
    "ConformalIVIMEstimator",
    "SextantNLLSEstimator",
    "PipelineExecutionReport",
    "run_trust_value_action_pipeline",
]

ConformalMethod = Literal["split", "cqr", "mondrian"]
_VALID_CONFORMAL_METHODS: tuple[str, ...] = ("split", "cqr", "mondrian")


def has_conformal_ivim() -> bool:
    """Return ``True`` if ``conformal_ivim`` is installed in the current runtime."""
    return importlib.util.find_spec("conformal_ivim") is not None


def has_sextant_ivim() -> bool:
    """Return ``True`` if ``sextant_core`` or ``sextant_ivim`` is dynamically discoverable."""
    return (
        importlib.util.find_spec("sextant_core") is not None
        or importlib.util.find_spec("sextant_ivim") is not None
    )


def _load_sextant_module() -> Any | None:
    if importlib.util.find_spec("sextant_core") is not None:
        return importlib.import_module("sextant_core")
    if importlib.util.find_spec("sextant_ivim") is not None:
        return importlib.import_module("sextant_ivim")
    return None


def ecosystem_availability() -> dict[str, Any]:
    """Inspect runtime availability and license boundary status across the IVIM stack."""
    conf_avail = has_conformal_ivim()
    sext_avail = has_sextant_ivim()
    return {
        "ivimbench": {"installed": True, "license": "MIT"},
        "conformal_ivim": {
            "installed": conf_avail,
            "license": "MIT",
            "active_backend": "conformal_ivim" if conf_avail else "ivimbench_internal_conformal",
        },
        "sextant_core": {
            "installed": sext_avail,
            "license": "AGPL-3.0 (dynamic runtime plugin only; not linked in core)",
            "active_backend": "sextant_core" if sext_avail else "ivimbench_internal_nlls",
        },
    }


def _compute_railed_matrix(
    point: np.ndarray,
    bounds: Mapping[str, tuple[float, float]],
    rail_tol: float = DEFAULT_RAIL_TOL,
) -> np.ndarray:
    cols = [
        classify_railing(point[:, j], bounds[name], rel_tol=rail_tol)["railed_mask"]
        for j, name in enumerate(PARAM_NAMES)
    ]
    return np.column_stack(cols)


def _clip_intervals_to_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    bounds: Mapping[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lower, dtype=np.float64).copy()
    hi = np.asarray(upper, dtype=np.float64).copy()
    for j, name in enumerate(PARAM_NAMES):
        b_lo, b_hi = bounds[name]
        lo[:, j] = np.clip(lo[:, j], b_lo, b_hi)
        hi[:, j] = np.clip(hi[:, j], b_lo, b_hi)
    hi = np.maximum(hi, lo)
    return lo, hi


@register_estimator(
    "sextant_nlls",
    description=(
        "License-safe bounded NLLS + boundary-railing adapter dynamically dispatching to "
        "`sextant_core.fit_ivim` when present or `ivimbench` internal bounded NLLS."
    ),
    tags=("bridge", "deterministic", "nlls"),
)
class SextantNLLSEstimator:
    """Dynamic adapter for ``sextant_core`` bounded NLLS and railing diagnostics."""

    def __init__(
        self,
        *,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        b_thresh: float = 200.0,
        rail_tol: float = DEFAULT_RAIL_TOL,
        snr: float = 30.0,
        max_nfev: int = 120,
        n_boot_railing: int = 1000,
        use_external: bool = True,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.bounds: dict[str, tuple[float, float]] = dict(DEFAULT_PARAM_BOUNDS)
        if bounds is not None:
            for k, v in bounds.items():
                key = "Dstar" if k == "D_star" else k
                self.bounds[key] = (float(v[0]), float(v[1]))
        self.b_thresh = float(b_thresh)
        self.rail_tol = float(rail_tol)
        self.snr = float(snr)
        self.max_nfev = int(max_nfev)
        self.n_boot_railing = int(n_boot_railing)
        self.use_external = bool(use_external)
        self.seed = int(seed)

    def fit_predict(
        self,
        signals: np.ndarray,
        bvalues: Sequence[float] | np.ndarray | str = "osipi",
        *,
        alpha: float = 0.10,
        **_extra: Any,
    ) -> dict[str, Any]:
        sig = np.atleast_2d(np.asarray(signals, dtype=np.float64))
        b_arr, _ = resolve_bvalues(bvalues)
        z_crit = float(norm.ppf(1.0 - 0.5 * float(alpha)))

        sextant_mod = _load_sextant_module() if self.use_external else None
        if sextant_mod is not None and hasattr(sextant_mod, "fit_ivim"):
            fit_res = sextant_mod.fit_ivim(
                b_arr,
                sig,
                method="nlls",
                bounds_D=self.bounds["D"],
                bounds_D_star=self.bounds["Dstar"],
                bounds_f=self.bounds["f"],
                b_thresh=self.b_thresh,
                rail_tol=self.rail_tol,
            )
            d_hat = np.atleast_1d(np.asarray(fit_res.D, dtype=np.float64)).ravel()
            ds_hat = np.atleast_1d(np.asarray(fit_res.D_star, dtype=np.float64)).ravel()
            f_hat = np.atleast_1d(np.asarray(fit_res.f, dtype=np.float64)).ravel()
            s0_hat = np.atleast_1d(
                np.asarray(getattr(fit_res, "S0", np.ones_like(d_hat)), dtype=np.float64)
            ).ravel()
            point = np.column_stack([d_hat, ds_hat, f_hat])

            if hasattr(sextant_mod, "classify_railing"):
                rail_ds_info = sextant_mod.classify_railing(
                    ds_hat,
                    bounds_D_star=self.bounds["Dstar"],
                    rel_tol=self.rail_tol,
                )
                dstar_railed = np.atleast_1d(
                    np.asarray(rail_ds_info["railed_mask"], dtype=bool)
                ).ravel()
            else:
                dstar_railed = classify_railing(
                    ds_hat, self.bounds["Dstar"], rel_tol=self.rail_tol
                )["railed_mask"]

            railed_mat = _compute_railed_matrix(point, self.bounds, rail_tol=self.rail_tol)
            railed_mat[:, 1] = dstar_railed

            if hasattr(sextant_mod, "bootstrap_railing_ci"):
                raw_ci = sextant_mod.bootstrap_railing_ci(
                    dstar_railed, n_boot=self.n_boot_railing, seed=self.seed
                )
                if isinstance(raw_ci, tuple) and len(raw_ci) == 3:
                    scale = 0.01 if float(raw_ci[0]) > 1.0001 or float(raw_ci[2]) > 1.0001 else 1.0
                    railing_ci_dict = {
                        "estimate": float(raw_ci[0]) * scale,
                        "ci_low": float(raw_ci[1]) * scale,
                        "ci_high": float(raw_ci[2]) * scale,
                    }
                else:
                    railing_ci_dict = {
                        "estimate": float(getattr(raw_ci, "estimate", np.mean(dstar_railed))),
                        "ci_low": float(getattr(raw_ci, "ci_low", np.mean(dstar_railed))),
                        "ci_high": float(getattr(raw_ci, "ci_high", np.mean(dstar_railed))),
                    }
            else:
                bci = bootstrap_railing_ci(
                    dstar_railed, n_boot=self.n_boot_railing, seed=self.seed
                )
                railing_ci_dict = {
                    "estimate": float(bci.point),
                    "ci_low": float(bci.lo),
                    "ci_high": float(bci.hi),
                }

            lo_crlb, hi_crlb = crlb_normal_intervals(
                b_arr,
                point,
                snr=self.snr,
                alpha=float(alpha),
                sd_convention="honest_crlb",
                bounds=self.bounds,
                s0=s0_hat,
            )
            std = 0.5 * (hi_crlb - lo_crlb) / max(z_crit, 1.0e-8)
            lower, upper = _clip_intervals_to_bounds(lo_crlb, hi_crlb, self.bounds)
            return {
                "point": point,
                "lower": lower,
                "upper": upper,
                "std": std,
                "s0": s0_hat,
                "railed": railed_mat,
                "backend": "sextant_core",
                "dstar_railing_ci": railing_ci_dict,
            }

        # License-safe internal fallback: ivimbench bounded NLLSEstimator + railing metrics
        fallback = NLLSEstimator(
            protocol=b_arr,
            bounds=self.bounds,
            rail_tol=self.rail_tol,
            sd_convention="honest_crlb",
            max_nfev=self.max_nfev,
        )
        solved = fallback.solve(sig)
        point = np.asarray(solved.params, dtype=np.float64)
        std = np.asarray(solved.sigma, dtype=np.float64)
        railed_mat = np.asarray(solved.railed, dtype=bool)
        lower, upper = _clip_intervals_to_bounds(
            point - z_crit * std, point + z_crit * std, self.bounds
        )
        bci = bootstrap_railing_ci(
            railed_mat[:, 1], n_boot=self.n_boot_railing, seed=self.seed
        )
        return {
            "point": point,
            "lower": lower,
            "upper": upper,
            "std": std,
            "s0": np.asarray(solved.s0, dtype=np.float64),
            "railed": railed_mat,
            "backend": "ivimbench_internal_nlls",
            "dstar_railing_ci": {
                "estimate": float(bci.point),
                "ci_low": float(bci.lo),
                "ci_high": float(bci.hi),
            },
        }


@register_estimator(
    "conformal_ivim",
    description=(
        "Distribution-free Split, CQR, and railed/interior Mondrian conformal IVIM adapter "
        "wrapping `conformal_ivim.ConformalIVIM` + `DeploymentValidityMonitor` with internal fallback."
    ),
    tags=("bridge", "conformal", "shift_monitor"),
)
class ConformalIVIMEstimator:
    """Dynamic adapter for ``conformal_ivim`` across ``split``, ``cqr``, and ``mondrian`` modes."""

    def __init__(
        self,
        *,
        method: ConformalMethod = "cqr",
        alpha: float = 0.10,
        organ: str = "pancreas",
        snr: float = 30.0,
        n_train: int = 360,
        n_calibration: int = 320,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        rail_tol: float = DEFAULT_RAIL_TOL,
        monitor_fpr: float = 0.05,
        monitor_n_null: int = 200,
        use_external: bool = True,
        seed: int = DEFAULT_SEED,
    ) -> None:
        m_clean = str(method).strip().lower()
        if m_clean not in _VALID_CONFORMAL_METHODS:
            raise ValueError(
                f"`method` must be one of {_VALID_CONFORMAL_METHODS}; got {method!r}."
            )
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"`alpha` must lie strictly in (0, 1); got {alpha}.")

        self.method: ConformalMethod = m_clean  # type: ignore[assignment]
        self.alpha = float(alpha)
        self.organ = str(organ)
        self.snr = float(snr)
        self.n_train = int(n_train)
        self.n_calibration = int(n_calibration)
        self.bounds: dict[str, tuple[float, float]] = dict(DEFAULT_PARAM_BOUNDS)
        if bounds is not None:
            for k, v in bounds.items():
                key = "Dstar" if k == "D_star" else k
                self.bounds[key] = (float(v[0]), float(v[1]))
        self.rail_tol = float(rail_tol)
        self.monitor_fpr = float(monitor_fpr)
        self.monitor_n_null = int(monitor_n_null)
        self.use_external = bool(use_external)
        self.seed = int(seed)

        self._calibrated_key: tuple[Any, ...] | None = None
        self._cal_signals: np.ndarray | None = None
        self._external_model: Any | None = None
        self._external_monitor: Any | None = None
        self._mlp: MLPEstimator | None = None
        self._split_offsets: np.ndarray | None = None
        self._cqr_offsets: np.ndarray | None = None
        self._mondrian_offsets: dict[bool, np.ndarray] | None = None

    def _railing_strata_labels(self, point_est: np.ndarray) -> np.ndarray:
        railed_ds = classify_railing(
            point_est[:, 1], self.bounds["Dstar"], rel_tol=self.rail_tol
        )["railed_mask"]
        return np.where(railed_ds, "railed", "interior")

    def calibrate(
        self,
        cal_signals: np.ndarray,
        cal_params: np.ndarray,
        bvalues: Sequence[float] | np.ndarray | str = "osipi",
        *,
        train_signals: np.ndarray | None = None,
        train_params: np.ndarray | None = None,
        alpha: float | None = None,
    ) -> ConformalIVIMEstimator:
        """Fit base heteroscedastic quantile/point model and calibrate conformal scores."""
        eff_alpha = self.alpha if alpha is None else float(alpha)
        b_arr, _ = resolve_bvalues(bvalues)
        x_cal = np.atleast_2d(np.asarray(cal_signals, dtype=np.float64))
        y_cal = np.atleast_2d(np.asarray(cal_params, dtype=np.float64))

        ss = np.random.SeedSequence(self.seed)
        tr_ss, mlp_ss = ss.spawn(2)
        rng_tr = np.random.Generator(np.random.PCG64(tr_ss))

        if train_signals is None or train_params is None:
            tr_cohort = generate_cohort(
                protocol=b_arr,
                organ=self.organ,
                n_voxels=self.n_train,
                snr=self.snr,
                seed=int(rng_tr.integers(0, 2**31 - 1)),
            )
            x_tr = tr_cohort.signals
            y_tr = tr_cohort.params
        else:
            x_tr = np.atleast_2d(np.asarray(train_signals, dtype=np.float64))
            y_tr = np.atleast_2d(np.asarray(train_params, dtype=np.float64))

        self._cal_signals = x_cal.copy()

        if self.use_external and has_conformal_ivim():
            conf_pkg = importlib.import_module("conformal_ivim")
            base_method = "cqr" if self.method == "mondrian" else self.method
            model = conf_pkg.ConformalIVIM(
                alpha=eff_alpha,
                method=base_method,
                b=b_arr,
                clip_bounds=tuple(
                    (float(self.bounds[n][0]), float(self.bounds[n][1]))
                    for n in PARAM_NAMES
                ),
                random_state=self.seed,
            )
            model.fit(x_tr, y_tr)
            if self.method == "mondrian":
                cal_nlls = NLLSEstimator(
                    protocol=b_arr, bounds=self.bounds, rail_tol=self.rail_tol, max_nfev=80
                ).solve(x_cal)
                strata_cal = self._railing_strata_labels(cal_nlls.params)
                model.calibrate(x_cal, y_cal, strata=strata_cal)
            else:
                model.calibrate(x_cal, y_cal)

            monitor = conf_pkg.DeploymentValidityMonitor(
                fpr=self.monitor_fpr,
                n_null=self.monitor_n_null,
                seed=self.seed,
            )
            monitor.fit_signals(x_cal, b=b_arr)
            self._external_model = model
            self._external_monitor = monitor
            self._calibrated_key = (tuple(np.round(b_arr, 6).tolist()), eff_alpha, self.method, "ext")
            return self

        # Pure-NumPy MIT fallback across split, cqr, and railed/interior mondrian
        self._external_model = None
        self._external_monitor = None
        mlp_seed = int(np.random.Generator(np.random.PCG64(mlp_ss)).integers(0, 2**31 - 1))
        mlp = MLPEstimator(
            protocol=b_arr,
            hidden_dims=(48, 48),
            epochs=55,
            bounds=self.bounds,
            seed=mlp_seed,
        )
        mlp.fit(x_tr, y_tr)
        self._mlp = mlp

        pt_cal = mlp.predict(x_cal)
        q_lo_cal, q_hi_cal = mlp.predict_intervals(x_cal, alpha=eff_alpha)

        if self.method == "split":
            abs_res = np.abs(y_cal - pt_cal)
            self._split_offsets = np.array(
                [conformal_offset(abs_res[:, j], eff_alpha) for j in range(3)],
                dtype=np.float64,
            )
        elif self.method == "cqr":
            cqr_scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
            self._cqr_offsets = np.array(
                [conformal_offset(cqr_scores[:, j], eff_alpha) for j in range(3)],
                dtype=np.float64,
            )
        elif self.method == "mondrian":
            cal_nlls = NLLSEstimator(
                protocol=b_arr, bounds=self.bounds, rail_tol=self.rail_tol, max_nfev=75
            ).solve(x_cal)
            cal_railed_ds = np.asarray(cal_nlls.railed[:, 1], dtype=bool)
            cqr_scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
            global_off = np.array(
                [conformal_offset(cqr_scores[:, j], eff_alpha) for j in range(3)],
                dtype=np.float64,
            )
            mond_off: dict[bool, np.ndarray] = {}
            for is_rail in (False, True):
                sel = cal_railed_ds == is_rail
                if int(np.sum(sel)) >= 8:
                    mond_off[is_rail] = np.array(
                        [conformal_offset(cqr_scores[sel, j], eff_alpha) for j in range(3)],
                        dtype=np.float64,
                    )
                else:
                    mond_off[is_rail] = global_off.copy()
            self._mondrian_offsets = mond_off

        self._calibrated_key = (tuple(np.round(b_arr, 6).tolist()), eff_alpha, self.method, "int")
        return self

    def _ensure_calibrated(self, b_arr: np.ndarray, eff_alpha: float) -> None:
        expected_mode = "ext" if (self.use_external and has_conformal_ivim()) else "int"
        target_key = (tuple(np.round(b_arr, 6).tolist()), eff_alpha, self.method, expected_mode)
        if self._calibrated_key == target_key and self._cal_signals is not None:
            return

        ss = np.random.SeedSequence(self.seed)
        tr_ss, cal_ss = ss.spawn(2)
        rng_tr = np.random.Generator(np.random.PCG64(tr_ss))
        rng_cal = np.random.Generator(np.random.PCG64(cal_ss))

        tr_cohort = generate_cohort(
            protocol=b_arr,
            organ=self.organ,
            n_voxels=self.n_train,
            snr=self.snr,
            seed=int(rng_tr.integers(0, 2**31 - 1)),
        )
        cal_cohort = generate_cohort(
            protocol=b_arr,
            organ=self.organ,
            n_voxels=self.n_calibration,
            snr=self.snr,
            seed=int(rng_cal.integers(0, 2**31 - 1)),
        )
        self.calibrate(
            cal_cohort.signals,
            cal_cohort.params,
            b_arr,
            train_signals=tr_cohort.signals,
            train_params=tr_cohort.params,
            alpha=eff_alpha,
        )

    def evaluate_shift(
        self,
        test_signals: np.ndarray,
        bvalues: Sequence[float] | np.ndarray | str = "osipi",
    ) -> dict[str, Any]:
        """Run label-free two-sample signal + residual deployment shift monitoring."""
        b_arr, _ = resolve_bvalues(bvalues)
        self._ensure_calibrated(b_arr, self.alpha)
        x_test = np.atleast_2d(np.asarray(test_signals, dtype=np.float64))

        if self._external_monitor is not None:
            verdict = self._external_monitor.evaluate_signals(x_test, b=b_arr)
            maha_info = verdict.get("maha", {})
            resid_info = verdict.get("resid", {})
            return {
                "fires": bool(verdict.get("fires", False)),
                "is_stale": bool(verdict.get("fires", False)),
                "auc": float(verdict.get("auc", 0.5)),
                "mahalanobis_stat": float(maha_info.get("stat", 0.0)),
                "p_value": float(maha_info.get("p_value", 1.0)),
                "residual_rms_ratio": float(resid_info.get("ratio", 1.0)),
                "backend": "conformal_ivim.DeploymentValidityMonitor",
            }

        from ..decision.voi import check_domain_shift

        assert self._cal_signals is not None
        rep = check_domain_shift(
            self._cal_signals,
            x_test,
            bvalues=b_arr,
            alpha=self.monitor_fpr,
            n_perm=self.monitor_n_null,
            seed=self.seed,
        )
        return {
            "fires": bool(rep.is_stale),
            "is_stale": bool(rep.is_stale),
            "mahalanobis_stat": float(rep.signal_mahalanobis_sq),
            "residual_mahalanobis_stat": float(rep.residual_mahalanobis_sq),
            "p_value": float(min(rep.signal_pvalue, rep.residual_pvalue)),
            "signal_pvalue": float(rep.signal_pvalue),
            "residual_pvalue": float(rep.residual_pvalue),
            "residual_rms_ratio": float(rep.residual_rms_ratio),
            "signal_outlier_rate": float(rep.signal_outlier_rate),
            "staleness_reasons": list(rep.staleness_reasons),
            "backend": "ivimbench.decision.voi.check_domain_shift",
        }

    def fit_predict(
        self,
        signals: np.ndarray,
        bvalues: Sequence[float] | np.ndarray | str = "osipi",
        *,
        alpha: float | None = None,
        **_extra: Any,
    ) -> dict[str, Any]:
        eff_alpha = self.alpha if alpha is None else float(alpha)
        sig = np.atleast_2d(np.asarray(signals, dtype=np.float64))
        b_arr, _ = resolve_bvalues(bvalues)
        self._ensure_calibrated(b_arr, eff_alpha)

        shift_verdict = self.evaluate_shift(sig, b_arr)

        if self._external_model is not None:
            nlls_ref = NLLSEstimator(
                protocol=b_arr, bounds=self.bounds, rail_tol=self.rail_tol, max_nfev=80
            ).solve(sig)
            railed_mat = np.asarray(nlls_ref.railed, dtype=bool)
            if self.method == "mondrian":
                strata_test = np.where(railed_mat[:, 1], "railed", "interior")
                lo_ext, hi_ext = self._external_model.predict_interval(
                    sig, alpha=eff_alpha, strata=strata_test
                )
            else:
                lo_ext, hi_ext = self._external_model.predict_interval(sig, alpha=eff_alpha)
            pt_ext = self._external_model.predict(sig)
            lo_ext, hi_ext = _clip_intervals_to_bounds(lo_ext, hi_ext, self.bounds)
            pt_ext = np.clip(np.asarray(pt_ext, dtype=np.float64), lo_ext, hi_ext)
            return {
                "point": pt_ext,
                "lower": lo_ext,
                "upper": hi_ext,
                "railed": railed_mat,
                "backend": "conformal_ivim",
                "conformal_method": self.method,
                "shift_monitor": shift_verdict,
                "strata_counts": {
                    "railed": int(np.sum(railed_mat[:, 1])),
                    "interior": int(np.sum(~railed_mat[:, 1])),
                },
            }

        assert self._mlp is not None
        pt = self._mlp.predict(sig)
        q_lo, q_hi = self._mlp.predict_intervals(sig, alpha=eff_alpha)

        if self.method == "split":
            assert self._split_offsets is not None
            railed_mat = _compute_railed_matrix(pt, self.bounds, rail_tol=self.rail_tol)
            lo_raw = pt - self._split_offsets[None, :]
            hi_raw = pt + self._split_offsets[None, :]
        elif self.method == "cqr":
            assert self._cqr_offsets is not None
            railed_mat = _compute_railed_matrix(pt, self.bounds, rail_tol=self.rail_tol)
            lo_raw = q_lo - self._cqr_offsets[None, :]
            hi_raw = q_hi + self._cqr_offsets[None, :]
        else:
            assert self._mondrian_offsets is not None
            nlls_ref = NLLSEstimator(
                protocol=b_arr, bounds=self.bounds, rail_tol=self.rail_tol, max_nfev=75
            ).solve(sig)
            railed_mat = np.asarray(nlls_ref.railed, dtype=bool)
            is_rail = railed_mat[:, 1]
            off_per_vox = np.where(
                is_rail[:, None],
                self._mondrian_offsets[True][None, :],
                self._mondrian_offsets[False][None, :],
            )
            lo_raw = q_lo - off_per_vox
            hi_raw = q_hi + off_per_vox

        lo_clip, hi_clip = _clip_intervals_to_bounds(lo_raw, hi_raw, self.bounds)
        pt_clip = np.clip(pt, lo_clip, hi_clip)
        return {
            "point": pt_clip,
            "lower": lo_clip,
            "upper": hi_clip,
            "railed": railed_mat,
            "backend": "ivimbench_internal_conformal",
            "conformal_method": self.method,
            "shift_monitor": shift_verdict,
            "strata_counts": {
                "railed": int(np.sum(railed_mat[:, 1])),
                "interior": int(np.sum(~railed_mat[:, 1])),
            },
        }


@dataclass(frozen=True)
class PipelineExecutionReport:
    """Structured 4-stage `Trust -> Value -> Action` execution report."""

    config: dict[str, Any]
    ecosystem: dict[str, Any]
    stage1_substrate_fit: dict[str, Any]
    stage2_trust_shift: dict[str, Any]
    stage3_value_of_information: dict[str, Any]
    stage4_closed_loop_action: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        from ..runner import _sanitize_json

        return _sanitize_json(
            {
                "config": self.config,
                "ecosystem": self.ecosystem,
                "stage1_substrate_fit": self.stage1_substrate_fit,
                "stage2_trust_shift": self.stage2_trust_shift,
                "stage3_value_of_information": self.stage3_value_of_information,
                "stage4_closed_loop_action": self.stage4_closed_loop_action,
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def format_summary_table(self) -> str:
        cfg = self.config
        s1 = self.stage1_substrate_fit
        s2 = self.stage2_trust_shift
        s3 = self.stage3_value_of_information
        s4 = self.stage4_closed_loop_action

        lines = [
            "IVIMBench Unified 4-Stage Pipeline (`Trust -> Value -> Action`)",
            "=" * 84,
            (
                f"Organ={cfg['organ']} | Protocol={cfg['protocol']} | SNR={cfg['snr']} | "
                f"Conformal={cfg['conformal_method'].upper()} (1-alpha={1.0 - cfg['alpha']:.2f}) | "
                f"Fractions={cfg['fractions']} | Seed={cfg['seed']}"
            ),
            (
                f"Backends: Fit={s1['backend']} | Conformal={s2['backend']} | "
                f"ShiftMonitor={s2['in_distribution_shift']['backend']}"
            ),
            "-" * 84,
            f"{'Stage':<30} {'Key Quantitative Indicators':<52}",
            "-" * 84,
            (
                f"{'1. Substrate & NLLS Fit':<30} "
                f"n_fg={s1['n_foreground_voxels']} | D* railed={100.0 * s1['dstar_railing_rate']:.1f}% "
                f"[{100.0 * s1['dstar_railing_ci'][0]:.1f}%, {100.0 * s1['dstar_railing_ci'][1]:.1f}%]"
            ),
            (
                f"{'2. Trust & Shift Gate':<30} "
                f"f cov: uncal={s2['uncalibrated_coverage']['f']:.3f} -> "
                f"conf={s2['conformal_coverage']['f']:.3f} | "
                f"ID shift={s2['in_distribution_shift']['fires']} / OOD={s2['ood_stress_shift']['fires']}"
            ),
            (
                f"{'3. Value-of-Information Gate':<30} "
                f"Regret: {s3['uncalibrated_regret']:.4f} -> {s3['conformal_regret']:.4f} | "
                f"CalGap={s3['uncalibrated_decision_cal_gap']:.4f} -> {s3['conformal_decision_cal_gap']:.4f} | "
                f"VoI2={s3['second_order_voi']:+.4f}"
            ),
            (
                f"{'4. Closed-Loop Adaptive RT':<30} "
                f"TCP={s4['conformal_tumor_control_proxy']:.3f} | "
                f"OAR sparing={s4['conformal_oar_sparing_vs_uncalibrated_gy']:+.3f} Gy | "
                f"False-esc delta={-s4['false_escalation_reduction']:+.3f}"
            ),
            "=" * 84,
        ]
        return "\n".join(lines)


def run_trust_value_action_pipeline(
    protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray = "osipi",
    organ: str = "pancreas",
    snr: float = 30.0,
    conformal_method: ConformalMethod = "cqr",
    alpha: float = 0.10,
    fractions: int = 5,
    seed: int = 42,
    *,
    dro_shape: Sequence[int] = (20, 20),
    hypoxia_f_threshold: float = 0.12,
    cost_ratio: float = 0.75,
    apply_mr_linac_ere: bool = False,
) -> PipelineExecutionReport:
    """Execute the unified 4-stage `Trust -> Value -> Action` quantitative IVIM spine."""
    from ..closed_loop.loop import compare_closed_loop_policies
    from ..closed_loop.substrates import SyntheticOrganTwin
    from ..decision.voi import evaluate_decision_voi

    m_clean = str(conformal_method).strip().lower()
    if m_clean not in _VALID_CONFORMAL_METHODS:
        raise ValueError(
            f"`conformal_method` must be one of {_VALID_CONFORMAL_METHODS}; got {conformal_method!r}."
        )
    if int(fractions) < 1:
        raise ValueError(f"`fractions` must be >= 1; got {fractions}.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"`alpha` must lie strictly in (0, 1); got {alpha}.")

    b_arr, proto_name = resolve_bvalues(protocol)
    ss = np.random.SeedSequence(int(seed))
    dro_ss, fit_ss, tr_ss, cal_ss, ood_ss, voi_ss, loop_ss = ss.spawn(7)

    def _subseed(child_ss: np.random.SeedSequence) -> int:
        return int(np.random.Generator(np.random.PCG64(child_ss)).integers(0, 2**31 - 1))

    # Stage 1: Spatial DRO Substrate & Per-Voxel Bounded NLLS + Railing
    dro = generate_dro(
        shape=tuple(int(s) for s in dro_shape),
        organ=organ,
        protocol=b_arr.tolist(),
        snr=float(snr),
        seed=_subseed(dro_ss),
    )
    signals_fg_raw, params_true_raw = dro.to_voxels(foreground_only=True)
    signals_fg = np.ascontiguousarray(signals_fg_raw, dtype=np.float64)
    params_true_fg = np.ascontiguousarray(params_true_raw, dtype=np.float64)
    n_fg = int(signals_fg.shape[0])

    fit_out = run_estimator(
        "sextant_nlls",
        signals_fg,
        b_arr,
        alpha=float(alpha),
        snr=float(snr),
        max_nfev=95,
        n_boot_railing=500,
        seed=_subseed(fit_ss),
    )
    pt_nlls = fit_out.point
    railed_mat = (
        fit_out.railed
        if fit_out.railed is not None
        else _compute_railed_matrix(pt_nlls, DEFAULT_PARAM_BOUNDS)
    )
    dstar_railed = railed_mat[:, 1]
    rail_ci_info = fit_out.extras.get("dstar_railing_ci")
    if isinstance(rail_ci_info, Mapping):
        rail_ci_tuple = (float(rail_ci_info["ci_low"]), float(rail_ci_info["ci_high"]))
    else:
        bci = bootstrap_railing_ci(dstar_railed, n_boot=500, seed=_subseed(fit_ss))
        rail_ci_tuple = (float(bci.lo), float(bci.hi))

    rmse_per_param = {
        name: float(np.sqrt(np.mean((pt_nlls[:, j] - params_true_fg[:, j]) ** 2)))
        for j, name in enumerate(PARAM_NAMES)
    }
    stage1_dict: dict[str, Any] = {
        "dro_shape": list(dro.spatial_shape),
        "n_foreground_voxels": n_fg,
        "backend": str(fit_out.extras.get("backend", "ivimbench_internal_nlls")),
        "dstar_railing_rate": float(np.mean(dstar_railed)),
        "dstar_railing_ci": [rail_ci_tuple[0], rail_ci_tuple[1]],
        "railing_rates": {
            name: float(np.mean(railed_mat[:, j])) for j, name in enumerate(PARAM_NAMES)
        },
        "rmse": rmse_per_param,
    }

    # Stage 2: Conformal Trust Intervals (Split/CQR/Mondrian) & Shift Gate
    half_uncal = 0.35 * np.maximum(0.5 * (fit_out.upper - fit_out.lower), 1.0e-6)
    half_uncal = np.where(railed_mat, 1.0e-5, half_uncal)
    lo_uncal, hi_uncal = _clip_intervals_to_bounds(
        pt_nlls - half_uncal, pt_nlls + half_uncal, DEFAULT_PARAM_BOUNDS
    )

    tr_cohort = generate_cohort(
        protocol=b_arr,
        organ=organ,
        n_voxels=360,
        snr=float(snr),
        seed=_subseed(tr_ss),
    )
    cal_cohort = generate_cohort(
        protocol=b_arr,
        organ=organ,
        n_voxels=320,
        snr=float(snr),
        seed=_subseed(cal_ss),
    )
    conf_est = ConformalIVIMEstimator(
        method=m_clean,  # type: ignore[arg-type]
        alpha=float(alpha),
        organ=organ,
        snr=float(snr),
        seed=_subseed(cal_ss),
    )
    conf_est.calibrate(
        cal_cohort.signals,
        cal_cohort.params,
        b_arr,
        train_signals=tr_cohort.signals,
        train_params=tr_cohort.params,
        alpha=float(alpha),
    )
    conf_res = conf_est.fit_predict(signals_fg, b_arr, alpha=float(alpha))
    lo_conf = np.asarray(conf_res["lower"], dtype=np.float64)
    hi_conf = np.asarray(conf_res["upper"], dtype=np.float64)
    pt_conf = np.asarray(conf_res["point"], dtype=np.float64)

    id_shift = conf_est.evaluate_shift(cal_cohort.signals[: min(160, n_fg)], b_arr)
    ood_cohort = generate_cohort(
        protocol=b_arr,
        organ=organ,
        n_voxels=160,
        snr=max(float(snr) * 0.22, 4.0),
        seed=_subseed(ood_ss),
    )
    ood_shift = conf_est.evaluate_shift(ood_cohort.signals * 1.18, b_arr)

    uncal_cov = {
        name: float(empirical_coverage(params_true_fg[:, j], lo_uncal[:, j], hi_uncal[:, j]))
        for j, name in enumerate(PARAM_NAMES)
    }
    conf_cov = {
        name: float(empirical_coverage(params_true_fg[:, j], lo_conf[:, j], hi_conf[:, j]))
        for j, name in enumerate(PARAM_NAMES)
    }
    railed_sel = dstar_railed
    interior_sel = ~dstar_railed
    cond_dstar_cov = {
        "railed_voxels": (
            float(
                empirical_coverage(
                    params_true_fg[railed_sel, 1],
                    lo_conf[railed_sel, 1],
                    hi_conf[railed_sel, 1],
                )
            )
            if np.any(railed_sel)
            else float("nan")
        ),
        "interior_voxels": (
            float(
                empirical_coverage(
                    params_true_fg[interior_sel, 1],
                    lo_conf[interior_sel, 1],
                    hi_conf[interior_sel, 1],
                )
            )
            if np.any(interior_sel)
            else float("nan")
        ),
    }

    stage2_dict: dict[str, Any] = {
        "conformal_method": m_clean,
        "nominal_coverage": 1.0 - float(alpha),
        "backend": str(conf_res["backend"]),
        "uncalibrated_coverage": uncal_cov,
        "conformal_coverage": conf_cov,
        "conditional_dstar_coverage": cond_dstar_cov,
        "strata_counts": conf_res["strata_counts"],
        "dro_shift_check": conf_res["shift_monitor"],
        "in_distribution_shift": id_shift,
        "ood_stress_shift": ood_shift,
    }

    # Stage 3: Value-of-Information (VoI) & Threshold Decision Pricing
    tau_grid = np.unique(
        np.concatenate(
            [
                np.array([float(hypoxia_f_threshold)], dtype=np.float64),
                np.quantile(params_true_fg[:, 2], [0.25, 0.50, 0.75]),
            ]
        )
    )
    voi_uncal = evaluate_decision_voi(
        params_true_fg[:, 2],
        lo_uncal[:, 2],
        hi_uncal[:, 2],
        point=pt_nlls[:, 2],
        taus=tau_grid,
        parameter="f",
        direction="less",
        cost_ratio=float(cost_ratio),
        alpha=float(alpha),
        calibrated_intervals=(lo_conf[:, 2], hi_conf[:, 2]),
        seed=_subseed(voi_ss),
    )
    voi_conf = evaluate_decision_voi(
        params_true_fg[:, 2],
        lo_conf[:, 2],
        hi_conf[:, 2],
        point=pt_conf[:, 2],
        taus=tau_grid,
        parameter="f",
        direction="less",
        cost_ratio=float(cost_ratio),
        alpha=float(alpha),
        calibrated_intervals=(lo_conf[:, 2], hi_conf[:, 2]),
        seed=_subseed(voi_ss),
    )

    stage3_dict: dict[str, Any] = {
        "parameter": "f",
        "hypoxia_f_threshold": float(hypoxia_f_threshold),
        "cost_ratio": float(cost_ratio),
        "uncalibrated_regret": float(voi_uncal.summary["mean_expected_regret"]),
        "conformal_regret": float(voi_conf.summary["mean_expected_regret"]),
        "uncalibrated_decision_cal_gap": float(
            voi_uncal.summary["mean_decision_calibration_gap"]
        ),
        "conformal_decision_cal_gap": float(voi_conf.summary["mean_decision_calibration_gap"]),
        "evpi": float(voi_conf.summary["mean_evpi"]),
        "second_order_voi": float(voi_uncal.summary["mean_second_order_voi"]),
        "conformal_net_benefit": float(voi_conf.summary["mean_net_benefit"]),
        "thresholds_evaluated": tau_grid.tolist(),
    }

    # Stage 4: Closed-Loop Multi-Fraction Adaptive Radiotherapy Action
    twin_sub = SyntheticOrganTwin(
        shape=dro_shape,
        organ=organ,
        apply_mr_linac_ere=bool(apply_mr_linac_ere),
        seed=_subseed(loop_ss),
    )
    loop_cmp = compare_closed_loop_policies(
        substrate=twin_sub,
        protocol=b_arr,
        estimator="crlb",
        n_fractions=int(fractions),
        snr=float(snr),
        alpha=float(alpha),
        hypoxia_f_threshold=float(hypoxia_f_threshold),
        cost_ratio=float(cost_ratio),
        include_oracle=True,
        apply_mr_linac_ere=bool(apply_mr_linac_ere),
        seed=_subseed(loop_ss),
    )
    stage4_dict: dict[str, Any] = {
        **loop_cmp.summary,
        "policies": loop_cmp.policy_df.to_dict(orient="records"),
        "conformal_trajectory": loop_cmp.runs["conformal_trust_gated"].fractions_df.to_dict(
            orient="records"
        ),
    }

    config_dict: dict[str, Any] = {
        "protocol": proto_name,
        "bvalues": b_arr.tolist(),
        "organ": organ,
        "snr": float(snr),
        "conformal_method": m_clean,
        "alpha": float(alpha),
        "fractions": int(fractions),
        "dro_shape": [int(s) for s in dro_shape],
        "hypoxia_f_threshold": float(hypoxia_f_threshold),
        "cost_ratio": float(cost_ratio),
        "apply_mr_linac_ere": bool(apply_mr_linac_ere),
        "seed": int(seed),
    }

    return PipelineExecutionReport(
        config=config_dict,
        ecosystem=ecosystem_availability(),
        stage1_substrate_fit=stage1_dict,
        stage2_trust_shift=stage2_dict,
        stage3_value_of_information=stage3_dict,
        stage4_closed_loop_action=stage4_dict,
    )
