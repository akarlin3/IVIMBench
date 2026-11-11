"""Box-constrained non-linear least squares (NLLS) IVIM baseline estimator.

Ported and unified from Caliper (``caliper.baselines``) and Sextant
(``sextant_core.fitting``):
- Fits the four-parameter bi-exponential model ``(S0, D, Dstar, f)`` per voxel
  via ``scipy.optimize.least_squares`` (Trust-Region Reflective, ``method="trf"``)
  with parameter-scaled trust regions.
- Computes textbook asymptotic Gaussian standard errors ``sigma^2 (J^T J)^+``
  under either the ``"honest"`` span-clipped convention (default) or the
  ``"floored"`` over-confident convention on railed ``D*`` estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy import stats

from ..metrics.coverage import DEFAULT_QUANTILE_LEVELS
from ..metrics.railing import DEFAULT_PARAM_BOUNDS, DEFAULT_RAIL_TOL
from ..protocols import AcquisitionProtocol, resolve_bvalues

__all__ = ["NLLSFitResult", "NLLSEstimator", "NLLSIVIMEstimator"]


def _nlls_residuals(theta: np.ndarray, b: np.ndarray, y: np.ndarray) -> np.ndarray:
    s0, d_val, ds_val, f_val = theta
    pred = s0 * (f_val * np.exp(-b * ds_val) + (1.0 - f_val) * np.exp(-b * d_val))
    return pred - y


def _nlls_jacobian(theta: np.ndarray, b: np.ndarray, y: np.ndarray) -> np.ndarray:
    _ = y
    s0, d_val, ds_val, f_val = theta
    e_d = np.exp(-b * d_val)
    e_ds = np.exp(-b * ds_val)
    j_s0 = f_val * e_ds + (1.0 - f_val) * e_d
    j_d = -s0 * (1.0 - f_val) * b * e_d
    j_ds = -s0 * f_val * b * e_ds
    j_f = s0 * (e_ds - e_d)
    return np.column_stack([j_s0, j_d, j_ds, j_f])


@dataclass
class NLLSFitResult:
    """Result container for :class:`NLLSEstimator`."""

    params: np.ndarray
    s0: np.ndarray
    sigma: np.ndarray
    railed: np.ndarray
    success: np.ndarray

    @property
    def dstar_railed(self) -> np.ndarray:
        """Boolean mask indicating voxels where ``D*`` pinned to a box bound."""
        return self.railed[:, 1]


@dataclass
class NLLSEstimator:
    """Simultaneous box-constrained NLLS bi-exponential IVIM estimator.

    Parameters
    ----------
    protocol : str, AcquisitionProtocol, or sequence of float, default="osipi"
        Acquisition b-value schedule in ``s/mm^2``.
    bounds : mapping of str to (float, float) or None, optional
        Box bounds for ``"D"``, ``"Dstar"``, and ``"f"`` in physical units.
    s0_bounds : tuple of (float, float), default=(0.1, 2.5)
        Multiplicative bounds relative to observed ``S(b_min)`` for ``S0``.
    init : tuple of (float, float, float), default=(1.1e-3, 25.0e-3, 0.15)
        Shared neutral initialization ``(D0, Dstar0, f0)``.
    sd_convention : {"honest", "floored"}, default="honest"
        Convention for asymptotic SD on railed ``D*`` voxels.
    railed_sd_floor : float, default=3.0e-3
        Narrow floored SD (``mm^2/s``) applied when ``sd_convention="floored"``.
    rail_tol : float, default=1e-2
        Relative tolerance for flagging boundary-railed parameters.
    max_nfev : int, default=300
        Maximum function evaluations per voxel.
    """

    protocol: str | AcquisitionProtocol | Sequence[float] = "osipi"
    bounds: Mapping[str, tuple[float, float]] | None = None
    s0_bounds: tuple[float, float] = (0.1, 2.5)
    init: tuple[float, float, float] = (1.1e-3, 25.0e-3, 0.15)
    sd_convention: str = "honest"
    railed_sd_floor: float = 3.0e-3
    rail_tol: float = DEFAULT_RAIL_TOL
    sigma_floor: float = 1e-8
    max_nfev: int = 300
    bvalues: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.bvalues, _ = resolve_bvalues(self.protocol)
        active = dict(DEFAULT_PARAM_BOUNDS)
        if self.bounds is not None:
            active.update(self.bounds)
        self._bounds = active
        if self.sd_convention not in ("honest", "floored"):
            raise ValueError("`sd_convention` must be 'honest' or 'floored'.")

    def fit(self, signals: np.ndarray | None = None, params: np.ndarray | None = None) -> "NLLSEstimator":
        """No-op fit method for API parity with learned estimators."""
        _ = (signals, params)
        return self

    def solve(self, signals: np.ndarray) -> NLLSFitResult:
        """Fit ``(S0, D, Dstar, f)`` and asymptotic SDs across ``(n, n_b)`` signals."""
        sig = np.atleast_2d(np.asarray(signals, dtype=np.float64))
        b = self.bvalues
        if sig.shape[1] != b.size:
            raise ValueError(
                f"Signal b-axis ({sig.shape[1]}) does not match protocol length ({b.size})."
            )
        n = sig.shape[0]
        b0_idx = int(np.argmin(b))

        d_lo, d_hi = self._bounds["D"]
        ds_lo, ds_hi = self._bounds["Dstar"]
        f_lo, f_hi = self._bounds["f"]

        params_out = np.empty((n, 3), dtype=np.float64)
        s0_out = np.empty(n, dtype=np.float64)
        sigma_out = np.empty((n, 3), dtype=np.float64)
        success_out = np.empty(n, dtype=bool)

        x_scale = np.array([1.0, 1e-3, 3e-2, 0.2], dtype=np.float64)

        for i in range(n):
            y = sig[i]
            s0_obs = max(float(y[b0_idx]), 1e-4)
            lo4 = np.array(
                [self.s0_bounds[0] * s0_obs, d_lo, ds_lo, f_lo], dtype=np.float64
            )
            hi4 = np.array(
                [self.s0_bounds[1] * s0_obs, d_hi, ds_hi, f_hi], dtype=np.float64
            )
            span4 = hi4 - lo4
            x0 = np.clip(
                np.array([s0_obs, self.init[0], self.init[1], self.init[2]], dtype=np.float64),
                lo4 + 1e-9,
                hi4 - 1e-9,
            )
            res = least_squares(
                _nlls_residuals,
                x0,
                jac=_nlls_jacobian,
                bounds=(lo4, hi4),
                args=(b, y),
                method="trf",
                x_scale=x_scale,
                max_nfev=int(self.max_nfev),
            )
            s0_out[i] = float(res.x[0])
            params_out[i] = res.x[1:]
            success_out[i] = bool(res.success)

            dof = max(b.size - 4, 1)
            resid_var = float(np.sum(res.fun**2)) / dof
            jtj = res.jac.T @ res.jac
            cov = resid_var * np.linalg.pinv(jtj)
            var3 = np.clip(np.diag(cov)[1:], 0.0, span4[1:] ** 2)
            sigma_out[i] = np.sqrt(var3) + float(self.sigma_floor)

        spans3 = np.array([d_hi - d_lo, ds_hi - ds_lo, f_hi - f_lo], dtype=np.float64)
        lo3 = np.array([d_lo, ds_lo, f_lo], dtype=np.float64)
        hi3 = np.array([d_hi, ds_hi, f_hi], dtype=np.float64)
        railed = ((params_out - lo3[None, :]) <= self.rail_tol * spans3[None, :]) | (
            (hi3[None, :] - params_out) <= self.rail_tol * spans3[None, :]
        )

        if self.sd_convention == "floored":
            sigma_out[railed[:, 1], 1] = float(self.railed_sd_floor)

        return NLLSFitResult(
            params=params_out,
            s0=s0_out,
            sigma=sigma_out,
            railed=railed,
            success=success_out,
        )

    def predict(self, signals: np.ndarray) -> np.ndarray:
        """Return ``(n, 3)`` point estimates ordered ``(D, Dstar, f)``."""
        return self.solve(signals).params

    predict_point = predict

    def predict_intervals(
        self,
        signals: np.ndarray,
        alpha: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return central ``(1 - alpha)`` asymptotic Gaussian intervals ``(lower, upper)``."""
        fit = self.solve(signals)
        z = float(stats.norm.ppf(1.0 - 0.5 * float(alpha)))
        return fit.params - z * fit.sigma, fit.params + z * fit.sigma

    def predict_quantiles(
        self,
        signals: np.ndarray,
        q_levels: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS,
    ) -> np.ndarray:
        """Return ``(n, 3, n_levels)`` asymptotic Gaussian quantile predictions."""
        levels = np.asarray(q_levels, dtype=np.float64)
        if np.any((levels <= 0.0) | (levels >= 1.0)):
            raise ValueError("`q_levels` must lie strictly in (0, 1).")
        fit = self.solve(signals)
        z = stats.norm.ppf(levels)
        return fit.params[:, :, None] + fit.sigma[:, :, None] * z[None, None, :]


NLLSIVIMEstimator = NLLSEstimator
