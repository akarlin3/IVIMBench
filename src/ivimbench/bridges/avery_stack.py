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
from scipy.optimize import least_squares
from scipy.stats import norm

from ..baselines.crlb_estimator import crlb_normal_intervals
from ..baselines.mlp import MLPEstimator
from ..baselines.nlls import NLLSEstimator
from ..baselines.segmented import SegmentedEstimator
from ..metrics.coverage import (
    SequentialConformalCalibrator,
    SplitConformalResidual,
    conformal_offset,
    empirical_coverage,
)
from ..metrics.railing import (
    DEFAULT_PARAM_BOUNDS,
    DEFAULT_RAIL_TOL,
    bootstrap_railing_ci,
    classify_railing,
)
from ..metrics.sharpness import ivim_jacobian
from ..phantoms.dro import generate_dro
from ..phantoms.synthetic import DEFAULT_SEED, PARAM_NAMES, generate_cohort
from ..protocols import AcquisitionProtocol, resolve_bvalues
from ..registry import register_estimator, run_estimator

__all__ = [
    "ConformalMethod",
    "has_conformal_ivim",
    "has_sextant_ivim",
    "ecosystem_availability",
    "CanonicalFitter",
    "get_canonical_fitter",
    "evaluate_deployment_shift",
    "detect_voxel_ood",
    "ConformalIVIMEstimator",
    "SextantNLLSEstimator",
    "PipelineExecutionReport",
    "run_trust_value_action_pipeline",
]

ConformalMethod = Literal["split", "cqr", "mondrian", "aci", "conformal_pid"]
_VALID_CONFORMAL_METHODS: tuple[str, ...] = ("split", "cqr", "mondrian", "aci", "conformal_pid")


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


class CanonicalFitter:
    """License-safe forward fitter, 2D (b, TE) solver, CRLB, and railing facade.

    Dynamically delegates to ``sextant_core`` when installed and ``prefer_sextant=True``,
    falling back seamlessly to MIT-licensed ``ivimbench`` builtin solvers otherwise.
    """

    def __init__(
        self,
        *,
        prefer_sextant: bool = True,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        b_thresh: float = 200.0,
        rail_tol: float = DEFAULT_RAIL_TOL,
        snr: float = 30.0,
        max_nfev: int = 120,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.prefer_sextant = bool(prefer_sextant)
        self.bounds: dict[str, tuple[float, float]] = dict(DEFAULT_PARAM_BOUNDS)
        if bounds is not None:
            for k, v in bounds.items():
                key = "Dstar" if k == "D_star" else k
                self.bounds[key] = (float(v[0]), float(v[1]))
        self.b_thresh = float(b_thresh)
        self.rail_tol = float(rail_tol)
        self.snr = float(snr)
        self.max_nfev = int(max_nfev)
        self.seed = int(seed)

    @property
    def backend(self) -> str:
        """Active forward-fitting backend identifier (`'sextant_core'` or `'ivimbench_builtin'`)."""
        sextant_mod = _load_sextant_module() if self.prefer_sextant else None
        return "sextant_core" if sextant_mod is not None else "ivimbench_builtin"

    def classify_railing(
        self,
        values: np.ndarray | Sequence[float],
        param: str = "Dstar",
        *,
        bounds: tuple[float, float] | None = None,
        rel_tol: float | None = None,
    ) -> dict[str, Any]:
        """Classify boundary-railed parameter estimates via ``sextant_core`` or builtin fallback."""
        key = "Dstar" if param == "D_star" else param
        b_pair = bounds if bounds is not None else self.bounds.get(key, self.bounds["Dstar"])
        tol = self.rail_tol if rel_tol is None else float(rel_tol)
        arr = np.asarray(values, dtype=np.float64)
        sextant_mod = _load_sextant_module() if self.prefer_sextant else None
        if sextant_mod is not None and hasattr(sextant_mod, "classify_railing"):
            raw = sextant_mod.classify_railing(arr, bounds_D_star=b_pair, rel_tol=tol)
            r_mask = np.asarray(raw["railed_mask"], dtype=bool)
            lo_mask = np.asarray(raw.get("lower_railed_mask", raw.get("lower_mask", False)), dtype=bool)
            hi_mask = np.asarray(raw.get("upper_railed_mask", raw.get("upper_mask", False)), dtype=bool)
            return {
                "railed_mask": r_mask,
                "lower_railed_mask": lo_mask,
                "upper_railed_mask": hi_mask,
                "lower_mask": lo_mask,
                "upper_mask": hi_mask,
                "railed_fraction": float(np.mean(r_mask)) if r_mask.size else 0.0,
                "fitter_backend": "sextant_core",
            }
        fb = classify_railing(arr, b_pair, rel_tol=tol)
        r_mask = np.asarray(fb["railed_mask"], dtype=bool)
        lo_mask = np.asarray(fb.get("lower_railed_mask", fb.get("lower_mask", False)), dtype=bool)
        hi_mask = np.asarray(fb.get("upper_railed_mask", fb.get("upper_mask", False)), dtype=bool)
        return {
            "railed_mask": r_mask,
            "lower_railed_mask": lo_mask,
            "upper_railed_mask": hi_mask,
            "lower_mask": lo_mask,
            "upper_mask": hi_mask,
            "railed_fraction": float(
                fb.get("railed_fraction", float(np.mean(r_mask)) if r_mask.size else 0.0)
            ),
            "fitter_backend": "ivimbench_builtin",
        }

    def fit(
        self,
        signals: np.ndarray,
        bvalues: Sequence[float] | np.ndarray | str = "osipi",
        *,
        method: Literal["segmented", "nlls"] = "nlls",
        b_thresh: float | None = None,
    ) -> dict[str, Any]:
        """Fit 1D IVIM curves (`'segmented'` or `'nlls'`) via ``sextant_core`` or builtin fallback."""
        m_clean = str(method).strip().lower()
        if m_clean not in ("segmented", "nlls"):
            raise ValueError(f"`method` must be 'segmented' or 'nlls'; got {method!r}.")
        sig = np.atleast_2d(np.asarray(signals, dtype=np.float64))
        b_arr, _ = resolve_bvalues(bvalues)
        eff_thresh = self.b_thresh if b_thresh is None else float(b_thresh)

        sextant_mod = _load_sextant_module() if self.prefer_sextant else None
        if sextant_mod is not None and hasattr(sextant_mod, "fit_ivim"):
            res = sextant_mod.fit_ivim(
                b_arr,
                sig,
                method=m_clean,
                b_thresh=eff_thresh,
                bounds_D=self.bounds["D"],
                bounds_D_star=self.bounds["Dstar"],
                bounds_f=self.bounds["f"],
                rail_tol=self.rail_tol,
            )
            d_hat = np.atleast_1d(np.asarray(res.D, dtype=np.float64)).ravel()
            ds_hat = np.atleast_1d(np.asarray(res.D_star, dtype=np.float64)).ravel()
            f_hat = np.atleast_1d(np.asarray(res.f, dtype=np.float64)).ravel()
            s0_hat = np.atleast_1d(
                np.asarray(getattr(res, "S0", np.ones_like(d_hat)), dtype=np.float64)
            ).ravel()
            point = np.column_stack([d_hat, ds_hat, f_hat])
            rail_ds = self.classify_railing(ds_hat, "Dstar")["railed_mask"]
            railed_mat = _compute_railed_matrix(point, self.bounds, rail_tol=self.rail_tol)
            railed_mat[:, 1] = np.atleast_1d(rail_ds).ravel()
            return {
                "point": point,
                "s0": s0_hat,
                "railed": railed_mat,
                "dstar_railed": railed_mat[:, 1],
                "fitter_backend": "sextant_core",
            }

        if m_clean == "segmented":
            seg = SegmentedEstimator(
                protocol=b_arr, b_thresh=eff_thresh, bounds=self.bounds
            ).solve(sig)
            point = np.asarray(seg.params, dtype=np.float64)
            s0_hat = np.asarray(seg.s0, dtype=np.float64)
            railed_mat = _compute_railed_matrix(point, self.bounds, rail_tol=self.rail_tol)
        else:
            nlls = NLLSEstimator(
                protocol=b_arr,
                bounds=self.bounds,
                rail_tol=self.rail_tol,
                max_nfev=self.max_nfev,
            ).solve(sig)
            point = np.asarray(nlls.params, dtype=np.float64)
            s0_hat = np.asarray(nlls.s0, dtype=np.float64)
            railed_mat = np.asarray(nlls.railed, dtype=bool)

        return {
            "point": point,
            "s0": s0_hat,
            "railed": railed_mat,
            "dstar_railed": railed_mat[:, 1],
            "fitter_backend": "ivimbench_builtin",
        }

    def fit_segmented(
        self,
        signals: np.ndarray,
        bvalues: Sequence[float] | np.ndarray | str = "osipi",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.fit(signals, bvalues=bvalues, method="segmented", **kwargs)

    def fit_nlls(
        self,
        signals: np.ndarray,
        bvalues: Sequence[float] | np.ndarray | str = "osipi",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.fit(signals, bvalues=bvalues, method="nlls", **kwargs)

    def fit_t2_ivim(
        self,
        signals: np.ndarray,
        bvalues: Sequence[float] | np.ndarray,
        te_values: Sequence[float] | np.ndarray,
        *,
        bounds_T2_p: tuple[float, float] = (20.0, 400.0),
        bounds_T2_t: tuple[float, float] = (20.0, 250.0),
        grid: bool | None = None,
    ) -> dict[str, Any]:
        """Fit relaxation-compensated 2D ``(b, TE)`` T2-IVIM via ``sextant_core`` or builtin NLLS."""
        b_arr = np.asarray(bvalues, dtype=np.float64)
        te_arr = np.asarray(te_values, dtype=np.float64)
        sig = np.asarray(signals, dtype=np.float64)

        sextant_mod = _load_sextant_module() if self.prefer_sextant else None
        if sextant_mod is not None and hasattr(sextant_mod, "fit_t2_ivim"):
            res = sextant_mod.fit_t2_ivim(
                sig,
                b_arr,
                te_arr,
                bounds_D=self.bounds["D"],
                bounds_D_star=self.bounds["Dstar"],
                bounds_f=self.bounds["f"],
                bounds_T2_p=bounds_T2_p,
                bounds_T2_t=bounds_T2_t,
                grid=grid,
                rail_tol=self.rail_tol,
            )
            return {
                "D": np.asarray(res.D, dtype=np.float64),
                "Dstar": np.asarray(res.D_star, dtype=np.float64),
                "f": np.asarray(res.f, dtype=np.float64),
                "T2_p": np.asarray(res.T2_p, dtype=np.float64),
                "T2_t": np.asarray(res.T2_t, dtype=np.float64),
                "S0": np.asarray(res.S0, dtype=np.float64),
                "railed_mask": np.asarray(res.railed_mask, dtype=bool),
                "fitter_backend": "sextant_core",
            }

        # Builtin bounded least-squares 2D (b, TE) fallback
        is_grid = (
            bool(grid)
            if grid is not None
            else (
                b_arr.ndim == 1
                and te_arr.ndim == 1
                and sig.ndim >= 2
                and sig.shape[-2:] == (b_arr.size, te_arr.size)
            )
        )
        if is_grid:
            bb, tt = np.meshgrid(b_arr, te_arr, indexing="ij")
            b_flat, te_flat = bb.ravel(), tt.ravel()
            spatial_shape = sig.shape[:-2]
            flat_sig = sig.reshape(-1, b_flat.size)
        else:
            b_flat, te_flat = np.broadcast_arrays(b_arr.ravel(), te_arr.ravel())
            b_flat = np.asarray(b_flat, dtype=np.float64)
            te_flat = np.asarray(te_flat, dtype=np.float64)
            spatial_shape = sig.shape[:-1]
            flat_sig = np.atleast_2d(sig).reshape(-1, b_flat.size)

        n_vox = flat_sig.shape[0]
        lo = np.array(
            [
                self.bounds["D"][0],
                self.bounds["Dstar"][0],
                self.bounds["f"][0],
                bounds_T2_p[0],
                bounds_T2_t[0],
                1e-6,
            ],
            dtype=np.float64,
        )
        hi = np.array(
            [
                self.bounds["D"][1],
                self.bounds["Dstar"][1],
                self.bounds["f"][1],
                bounds_T2_p[1],
                bounds_T2_t[1],
                max(float(np.max(flat_sig)) * 5.0, 10.0),
            ],
            dtype=np.float64,
        )
        out_params = np.empty((n_vox, 6), dtype=np.float64)
        for i in range(n_vox):
            y_obs = flat_sig[i]
            s0_init = float(np.clip(np.max(y_obs) * 1.4, lo[5], hi[5]))
            x0 = np.clip(
                np.array([1.2e-3, 20.0e-3, 0.15, 160.0, 55.0, s0_init], dtype=np.float64),
                lo + 1e-8,
                hi - 1e-8,
            )

            def _res_fn(theta: np.ndarray, _y: np.ndarray = y_obs) -> np.ndarray:
                d_v, ds_v, f_v, t2p_v, t2t_v, s0_v = theta
                pred = s0_v * (
                    f_v * np.exp(-b_flat * ds_v - te_flat / t2p_v)
                    + (1.0 - f_v) * np.exp(-b_flat * d_v - te_flat / t2t_v)
                )
                return pred - _y

            sol = least_squares(_res_fn, x0, bounds=(lo, hi), max_nfev=150)
            out_params[i] = sol.x

        def _reshape_param(col: np.ndarray) -> np.ndarray | float:
            return float(col[0]) if spatial_shape == () else col.reshape(spatial_shape)

        ds_est = _reshape_param(out_params[:, 1])
        rail_mask = self.classify_railing(np.atleast_1d(ds_est), "Dstar")["railed_mask"]
        rail_out = bool(rail_mask.ravel()[0]) if spatial_shape == () else rail_mask.reshape(spatial_shape)
        return {
            "D": _reshape_param(out_params[:, 0]),
            "Dstar": ds_est,
            "f": _reshape_param(out_params[:, 2]),
            "T2_p": _reshape_param(out_params[:, 3]),
            "T2_t": _reshape_param(out_params[:, 4]),
            "S0": _reshape_param(out_params[:, 5]),
            "railed_mask": rail_out,
            "fitter_backend": "ivimbench_builtin",
        }

    def compute_crlb(
        self,
        bvalues: Sequence[float] | np.ndarray | str = "osipi",
        params: np.ndarray | Sequence[float] | None = None,
        *,
        D: float | None = None,
        Dstar: float | None = None,
        f: float | None = None,
        snr: float | None = None,
        S0: float = 1.0,
        reg: float = 1e-12,
    ) -> dict[str, Any]:
        """Compute Fisher Information Matrix & CRLB covariance via ``sextant_core`` or builtin."""
        b_arr, _ = resolve_bvalues(bvalues)
        eff_snr = self.snr if snr is None else float(snr)
        if params is None:
            if D is None or Dstar is None or f is None:
                raise ValueError("Provide either `params` or explicit `D`, `Dstar`, `f`.")
            p_arr = np.array([[float(D), float(Dstar), float(f)]], dtype=np.float64)
            single = True
        else:
            raw_p = np.asarray(params, dtype=np.float64)
            single = raw_p.ndim == 1
            p_arr = np.atleast_2d(raw_p)

        sextant_mod = _load_sextant_module() if self.prefer_sextant else None
        if sextant_mod is not None and hasattr(sextant_mod, "compute_crlb"):
            stds, covs, fims, cvs = [], [], [], []
            for row in p_arr:
                res = sextant_mod.compute_crlb(
                    b_arr,
                    D=float(row[0]),
                    D_star=float(row[1]),
                    f=float(row[2]),
                    snr=eff_snr,
                    S0=float(S0),
                    reg=float(reg),
                )
                stds.append(np.asarray(res["std"], dtype=np.float64))
                covs.append(np.asarray(res["cov"], dtype=np.float64))
                fims.append(np.asarray(res["fim"], dtype=np.float64))
                cvs.append(float(res["cv_D_star"]))
            if single:
                return {
                    "std": stds[0],
                    "cov": covs[0],
                    "fim": fims[0],
                    "cv_Dstar": cvs[0],
                    "fitter_backend": "sextant_core",
                }
            return {
                "std": np.stack(stds, axis=0),
                "cov": np.stack(covs, axis=0),
                "fim": np.stack(fims, axis=0),
                "cv_Dstar": np.asarray(cvs, dtype=np.float64),
                "fitter_backend": "sextant_core",
            }

        sigma = float(S0) / max(eff_snr, 1e-12)
        jac = ivim_jacobian(b_arr, p_arr, s0=float(S0))
        fim_batch = np.einsum("nbi,nbj->nij", jac, jac) / (sigma**2)
        eye3 = np.eye(3, dtype=np.float64)[None, :, :]
        cov_batch = np.linalg.inv(fim_batch + float(reg) * eye3)
        std_batch = np.sqrt(np.maximum(np.diagonal(cov_batch, axis1=-2, axis2=-1), 0.0))
        cv_batch = std_batch[:, 1] / np.maximum(p_arr[:, 1], 1e-12)
        if single:
            return {
                "std": std_batch[0],
                "cov": cov_batch[0],
                "fim": fim_batch[0],
                "cv_Dstar": float(cv_batch[0]),
                "fitter_backend": "ivimbench_builtin",
            }
        return {
            "std": std_batch,
            "cov": cov_batch,
            "fim": fim_batch,
            "cv_Dstar": cv_batch,
            "fitter_backend": "ivimbench_builtin",
        }


def get_canonical_fitter(prefer_sextant: bool = True, **kwargs: Any) -> CanonicalFitter:
    """Return a license-safe canonical forward fitter delegating to ``sextant_core`` when available."""
    return CanonicalFitter(prefer_sextant=prefer_sextant, **kwargs)


def evaluate_deployment_shift(
    cal_signals: np.ndarray,
    test_signals: np.ndarray,
    bvalues: Sequence[float] | np.ndarray | str = "osipi",
    *,
    fpr: float = 0.05,
    n_null: int = 200,
    use_external: bool = True,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Delegate label-free deployment shift detection to ``conformal_ivim.DeploymentValidityMonitor``."""
    b_arr, _ = resolve_bvalues(bvalues)
    x_cal = np.atleast_2d(np.asarray(cal_signals, dtype=np.float64))
    x_test = np.atleast_2d(np.asarray(test_signals, dtype=np.float64))

    if use_external and has_conformal_ivim():
        conf_pkg = importlib.import_module("conformal_ivim")
        monitor = conf_pkg.DeploymentValidityMonitor(
            fpr=float(fpr),
            n_null=int(n_null),
            seed=int(seed),
        )
        monitor.fit_signals(x_cal, b=b_arr)
        verdict = monitor.evaluate_signals(x_test, b=b_arr)
        maha_info = verdict.get("maha", {})
        resid_info = verdict.get("resid", {})
        return {
            "fires": bool(verdict.get("fires", False)),
            "is_stale": bool(verdict.get("fires", False)),
            "auc": float(verdict.get("auc", 0.5)),
            "mahalanobis_stat": float(maha_info.get("stat", 0.0)),
            "p_value": float(maha_info.get("p_value", 1.0)),
            "residual_rms_ratio": float(resid_info.get("ratio", 1.0)),
            "backend": "conformal_ivim",
        }

    from ..decision.voi import check_domain_shift

    rep = check_domain_shift(
        x_cal,
        x_test,
        bvalues=b_arr,
        alpha=float(fpr),
        n_perm=int(n_null),
        seed=int(seed),
    )
    return {
        "fires": bool(rep.is_stale),
        "is_stale": bool(rep.is_stale),
        "auc": float(np.clip(0.5 + 0.12 * np.sqrt(max(rep.signal_mahalanobis_sq, 0.0)), 0.5, 0.99)),
        "mahalanobis_stat": float(rep.signal_mahalanobis_sq),
        "p_value": float(min(rep.signal_pvalue, rep.residual_pvalue)),
        "residual_rms_ratio": float(rep.residual_rms_ratio),
        "backend": "ivimbench_builtin",
    }


def detect_voxel_ood(
    cal_signals: np.ndarray,
    test_signals: np.ndarray,
    bvalues: Sequence[float] | np.ndarray | str = "osipi",
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    alpha_ood: float = 0.05,
    nominal_alpha: float = 0.10,
    use_external: bool = True,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Delegate per-voxel conformal OOD screening to ``conformal_ivim.VoxelOODDetector``."""
    b_arr, _ = resolve_bvalues(bvalues)
    x_cal = np.atleast_2d(np.asarray(cal_signals, dtype=np.float64))
    x_test = np.atleast_2d(np.asarray(test_signals, dtype=np.float64))

    if use_external and has_conformal_ivim():
        conf_pkg = importlib.import_module("conformal_ivim")
        detector = conf_pkg.VoxelOODDetector(
            alpha_ood=float(alpha_ood),
            b=b_arr,
            seed=int(seed),
        )
        detector.fit(x_cal, b=b_arr)
        pred = detector.predict_ood(x_test, b=b_arr)
        ood_mask = np.asarray(pred["ood_mask"], dtype=bool)
        out: dict[str, Any] = {
            "ood_mask": ood_mask,
            "p_values": np.asarray(pred["p_values"], dtype=np.float64),
            "ood_scores": np.asarray(pred["ood_scores"], dtype=np.float64),
            "flagged_fraction": float(pred["flagged_fraction"]),
            "backend": "conformal_ivim",
        }
        if lower is not None and upper is not None:
            adj = detector.adjust_intervals(
                np.asarray(lower, dtype=np.float64),
                np.asarray(upper, dtype=np.float64),
                ood_mask,
                nominal_alpha=float(nominal_alpha),
            )
            out["adjusted_lower"] = np.asarray(adj["lower"], dtype=np.float64)
            out["adjusted_upper"] = np.asarray(adj["upper"], dtype=np.float64)
        return out

    # Standalone NumPy conformal Mahalanobis + physical monotonicity OOD fallback
    s0_cal = np.maximum(x_cal[:, 0:1], 1e-6)
    s0_test = np.maximum(x_test[:, 0:1], 1e-6)
    zn_cal = x_cal / s0_cal
    zn_test = x_test / s0_test
    mu = np.mean(zn_cal, axis=0)
    cov = np.atleast_2d(np.cov(zn_cal, rowvar=False)) + 1e-4 * np.eye(zn_cal.shape[1])
    inv_cov = np.linalg.inv(cov)

    def _score_batch(arr: np.ndarray, raw: np.ndarray) -> np.ndarray:
        diff = arr - mu[None, :]
        maha = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", diff, inv_cov, diff), 0.0))
        order = np.argsort(b_arr)
        s_ord = arr[:, order]
        rebound = np.sum(np.maximum(0.0, s_ord[:, 1:] - s_ord[:, :-1] - 0.08), axis=1) * 8.0
        neg_pen = np.maximum(0.0, -np.min(raw, axis=1)) * 10.0
        return maha + rebound + neg_pen

    cal_scores = np.sort(_score_batch(zn_cal, x_cal))
    test_scores = _score_batch(zn_test, x_test)
    n_cal = cal_scores.size
    idx = np.searchsorted(cal_scores, test_scores, side="left")
    p_vals = (1.0 + (n_cal - idx).astype(np.float64)) / float(n_cal + 1)
    ood_mask = p_vals <= float(alpha_ood)
    out = {
        "ood_mask": ood_mask,
        "p_values": p_vals,
        "ood_scores": test_scores,
        "flagged_fraction": float(np.mean(ood_mask)) if ood_mask.size else 0.0,
        "backend": "ivimbench_builtin",
    }
    if lower is not None and upper is not None:
        lo_adj = np.asarray(lower, dtype=np.float64).copy()
        hi_adj = np.asarray(upper, dtype=np.float64).copy()
        if np.any(ood_mask):
            if lo_adj.ndim == 1:
                b_lo, b_hi = DEFAULT_PARAM_BOUNDS["f"]
                lo_adj[ood_mask] = np.minimum(lo_adj[ood_mask], b_lo)
                hi_adj[ood_mask] = np.maximum(hi_adj[ood_mask], b_hi)
            else:
                for j, pname in enumerate(PARAM_NAMES[: lo_adj.shape[1]]):
                    b_lo, b_hi = DEFAULT_PARAM_BOUNDS[pname]
                    lo_adj[ood_mask, j] = np.minimum(lo_adj[ood_mask, j], b_lo)
                    hi_adj[ood_mask, j] = np.maximum(hi_adj[ood_mask, j], b_hi)
        out["adjusted_lower"] = lo_adj
        out["adjusted_upper"] = hi_adj
    return out


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
            clip_b = tuple(
                (float(self.bounds[n][0]), float(self.bounds[n][1]))
                for n in PARAM_NAMES
            )
            if self.method in ("aci", "conformal_pid"):
                model = conf_pkg.AdaptiveConformalIVIM(
                    alpha=eff_alpha,
                    method=self.method,
                    b=b_arr,
                    clip_bounds=clip_b,
                    random_state=self.seed,
                )
                model.fit_calibrate(x_tr, y_tr, x_cal, y_cal, fit_monitor=False)
            else:
                base_method = "cqr" if self.method == "mondrian" else self.method
                model = conf_pkg.ConformalIVIM(
                    alpha=eff_alpha,
                    method=base_method,
                    b=b_arr,
                    clip_bounds=clip_b,
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

        # Pure-NumPy MIT fallback across split, cqr, mondrian, aci, and conformal_pid
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

        if self.method in ("split", "aci", "conformal_pid"):
            split_cal = SplitConformalResidual(alpha=eff_alpha, use_external=False).calibrate(
                pt_cal, y_cal
            )
            self._split_offsets = split_cal.offsets_
        elif self.method == "cqr":
            cqr_scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
            self._cqr_offsets = np.array(
                [conformal_offset(cqr_scores[:, j], eff_alpha, use_external=False) for j in range(3)],
                dtype=np.float64,
            )
        elif self.method == "mondrian":
            cal_nlls = NLLSEstimator(
                protocol=b_arr, bounds=self.bounds, rail_tol=self.rail_tol, max_nfev=75
            ).solve(x_cal)
            cal_railed_ds = np.asarray(cal_nlls.railed[:, 1], dtype=bool)
            cqr_scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
            global_off = np.array(
                [conformal_offset(cqr_scores[:, j], eff_alpha, use_external=False) for j in range(3)],
                dtype=np.float64,
            )
            mond_off: dict[bool, np.ndarray] = {}
            for is_rail in (False, True):
                sel = cal_railed_ds == is_rail
                if int(np.sum(sel)) >= 8:
                    mond_off[is_rail] = np.array(
                        [
                            conformal_offset(cqr_scores[sel, j], eff_alpha, use_external=False)
                            for j in range(3)
                        ],
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

    def detect_ood(
        self,
        test_signals: np.ndarray,
        bvalues: Sequence[float] | np.ndarray | str = "osipi",
        *,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        alpha_ood: float = 0.05,
    ) -> dict[str, Any]:
        """Run per-voxel conformal OOD screening against the fitted calibration signals."""
        b_arr, _ = resolve_bvalues(bvalues)
        self._ensure_calibrated(b_arr, self.alpha)
        assert self._cal_signals is not None
        return detect_voxel_ood(
            self._cal_signals,
            test_signals,
            bvalues=b_arr,
            lower=lower,
            upper=upper,
            alpha_ood=alpha_ood,
            nominal_alpha=self.alpha,
            use_external=self.use_external,
            seed=self.seed,
        )

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
            elif self.method in ("aci", "conformal_pid"):
                lo_ext, hi_ext = self._external_model.predict_interval(sig)
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

        if self.method in ("split", "aci", "conformal_pid"):
            assert self._split_offsets is not None
            railed_mat = _compute_railed_matrix(pt, self.bounds, rail_tol=self.rail_tol)
            if self.method in ("aci", "conformal_pid") and "stream_true" in _extra:
                seq_cal = SequentialConformalCalibrator(
                    method=self.method,  # type: ignore[arg-type]
                    alpha=eff_alpha,
                    use_external=False,
                )
                assert self._cal_signals is not None
                pt_cal = self._mlp.predict(self._cal_signals)
                lo_raw, hi_raw, _ = seq_cal.calibrate_stream(
                    pt_cal, np.asarray(_extra.get("cal_true", pt_cal)), pt, np.asarray(_extra["stream_true"])
                )
            else:
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
