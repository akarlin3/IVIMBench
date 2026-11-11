"""Vectorized two-step segmented least-squares IVIM baseline estimator.

Implements the OSIPI consensus two-step segmented IVIM algorithm entirely in
vectorized NumPy:
1. **High-b tissue step**: Log-linear ordinary least squares on ``b >= b_thresh``
   recovers slow diffusivity ``D`` and extrapolated tissue intercept, yielding
   perfusion fraction ``f``.
2. **All-b pseudo-diffusion step**: Vectorized multi-start log-grid search
   followed by damped Gauss-Newton / Levenberg-Marquardt refinement recovers
   ``D*`` within ``bounds["Dstar"]`` while holding ``(D, f)`` fixed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..metrics.coverage import DEFAULT_QUANTILE_LEVELS
from ..metrics.railing import DEFAULT_PARAM_BOUNDS
from ..protocols import AcquisitionProtocol, resolve_bvalues

__all__ = ["SegmentedFitResult", "SegmentedEstimator", "SegmentedIVIMEstimator"]


def _vectorized_ols_1d(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form OLS slope and intercept of ``y`` ``(n, k)`` on ``x`` ``(k,)``."""
    x_mean = float(np.mean(x))
    xc = x - x_mean
    denom = max(float(np.sum(xc * xc)), 1e-16)
    y_mean = np.mean(y, axis=1)
    slope = ((y - y_mean[:, None]) @ xc) / denom
    intercept = y_mean - slope * x_mean
    return slope, intercept


@dataclass
class SegmentedFitResult:
    """Result container for :class:`SegmentedEstimator`."""

    params: np.ndarray
    s0: np.ndarray
    sigma: np.ndarray


@dataclass
class SegmentedEstimator:
    """Two-step segmented IVIM estimator with asymptotic or fixed Gaussian uncertainty.

    Parameters
    ----------
    protocol : str, AcquisitionProtocol, or sequence of float, default="osipi"
        Acquisition b-value schedule in ``s/mm^2``.
    b_thresh : float, default=200.0
        High-b threshold separating tissue diffusion from microvascular perfusion.
    bounds : mapping of str to (float, float) or None, optional
        Physiological box bounds for ``"D"``, ``"Dstar"``, and ``"f"``.
    uncertainty : {"asymptotic", "homoscedastic"}, default="homoscedastic"
        - ``"homoscedastic"``: Constant nominal standard errors matching Caliper's
          deliberately over-confident segmented reference baseline.
        - ``"asymptotic"``: Per-voxel residual-scaled local Jacobian standard errors.
    """

    protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray = "osipi"
    b_thresh: float = 200.0
    bounds: Mapping[str, tuple[float, float]] | None = None
    uncertainty: str = "homoscedastic"
    homoscedastic_sigma: tuple[float, float, float] = (0.11e-3, 6.0e-3, 0.022)
    n_grid: int = 32
    n_gn_iter: int = 18
    eps: float = 1e-6
    bvalues: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.bvalues, _ = resolve_bvalues(self.protocol)
        active = dict(DEFAULT_PARAM_BOUNDS)
        if self.bounds is not None:
            active.update(self.bounds)
        self._bounds = active
        if self.uncertainty not in ("asymptotic", "homoscedastic"):
            raise ValueError("`uncertainty` must be 'asymptotic' or 'homoscedastic'.")

    def fit(self, signals: np.ndarray | None = None, params: np.ndarray | None = None) -> SegmentedEstimator:
        """No-op fit method for API parity with learned estimators."""
        _ = (signals, params)
        return self

    def solve(self, signals: np.ndarray) -> SegmentedFitResult:
        """Fit ``(D, Dstar, f)`` and uncertainty ``sigma`` across ``(n, n_b)`` signals."""
        sig = np.atleast_2d(np.asarray(signals, dtype=np.float64))
        b = self.bvalues
        if sig.shape[1] != b.size:
            raise ValueError(
                f"Signal b-axis ({sig.shape[1]}) does not match protocol length ({b.size})."
            )
        n = sig.shape[0]

        b_min = float(np.min(b))
        min_mask = np.isclose(b, b_min, atol=1e-6)
        s0 = np.maximum(np.mean(sig[:, min_mask], axis=1), self.eps)
        y = np.clip(sig / s0[:, None], self.eps, None)

        # Step 1: High-b log-linear fit for D and f
        high_mask = b >= float(self.b_thresh)
        if int(np.sum(high_mask)) < 2:
            order = np.argsort(b)
            high_mask = np.zeros_like(b, dtype=bool)
            high_mask[order[-2:]] = True

        slope_hi, intercept_hi = _vectorized_ols_1d(b[high_mask], np.log(y[:, high_mask]))
        d_lo, d_hi = self._bounds["D"]
        f_lo, f_hi = self._bounds["f"]
        ds_lo, ds_hi = self._bounds["Dstar"]

        d_hat = np.clip(-slope_hi, d_lo, d_hi)
        s_int = np.exp(intercept_hi)
        f_hat = np.clip(1.0 - s_int * np.exp(-b_min * d_hat), f_lo, f_hi)

        # Step 2: Vectorized multi-start log-grid + damped Gauss-Newton for D*
        eff_lo = np.maximum(ds_lo, d_hat)
        tail = (1.0 - f_hat[:, None]) * np.exp(-d_hat[:, None] * b[None, :])  # (n, n_b)

        grid = np.geomspace(max(ds_lo, 1e-5), ds_hi, int(self.n_grid))  # (G,)
        pred_grid = tail[:, None, :] + f_hat[:, None, None] * np.exp(
            -grid[None, :, None] * b[None, None, :]
        )
        sse_grid = np.sum((pred_grid - y[:, None, :]) ** 2, axis=2)  # (n, G)
        ds = grid[np.argmin(sse_grid, axis=1)].copy()
        ds = np.clip(ds, eff_lo, ds_hi)

        lam = np.full(n, 1e-4, dtype=np.float64)
        for _ in range(int(self.n_gn_iter)):
            exp_ds = np.exp(-ds[:, None] * b[None, :])
            r = (tail + f_hat[:, None] * exp_ds) - y
            cost = 0.5 * np.sum(r * r, axis=1)
            jac = -b[None, :] * f_hat[:, None] * exp_ds
            grad = np.sum(jac * r, axis=1)
            hess = np.sum(jac * jac, axis=1)
            step = -grad / (hess + lam + 1e-16)
            ds_cand = np.clip(ds + step, eff_lo, ds_hi)
            r_cand = (tail + f_hat[:, None] * np.exp(-ds_cand[:, None] * b[None, :])) - y
            cost_cand = 0.5 * np.sum(r_cand * r_cand, axis=1)
            improved = cost_cand < cost
            ds = np.where(improved, ds_cand, ds)
            lam = np.where(improved, np.maximum(lam * 0.3, 1e-12), np.minimum(lam * 5.0, 1e8))

        params = np.column_stack([d_hat, ds, f_hat])

        if self.uncertainty == "homoscedastic":
            sigma = np.broadcast_to(
                np.asarray(self.homoscedastic_sigma, dtype=np.float64), (n, 3)
            ).copy()
        else:
            e_d = np.exp(-d_hat[:, None] * b[None, :])
            e_ds = np.exp(-ds[:, None] * b[None, :])
            res = (f_hat[:, None] * e_ds + (1.0 - f_hat[:, None]) * e_d) - y
            dof = max(b.size - 3, 1)
            s2 = np.sum(res * res, axis=1) / dof
            j_d = -(1.0 - f_hat[:, None]) * b[None, :] * e_d
            j_ds = -f_hat[:, None] * b[None, :] * e_ds
            j_f = e_ds - e_d
            jac3 = np.stack([j_d, j_ds, j_f], axis=-1)
            spans = np.array([d_hi - d_lo, ds_hi - ds_lo, f_hi - f_lo], dtype=np.float64)
            sigma = np.empty((n, 3), dtype=np.float64)
            for i in range(n):
                jtj = jac3[i].T @ jac3[i]
                cov = s2[i] * np.linalg.pinv(jtj)
                var = np.clip(np.diag(cov), 0.0, spans**2)
                sigma[i] = np.sqrt(var) + 1e-8

        return SegmentedFitResult(params=params, s0=s0, sigma=sigma)

    def predict(self, signals: np.ndarray) -> np.ndarray:
        """Return ``(n, 3)`` point estimates ordered ``(D, Dstar, f)``."""
        return self.solve(signals).params

    predict_point = predict

    def predict_intervals(
        self,
        signals: np.ndarray,
        alpha: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return central ``(1 - alpha)`` intervals ``(lower, upper)`` of shape ``(n, 3)``."""
        res = self.solve(signals)
        z = float(stats.norm.ppf(1.0 - 0.5 * float(alpha)))
        return res.params - z * res.sigma, res.params + z * res.sigma

    def predict_quantiles(
        self,
        signals: np.ndarray,
        q_levels: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS,
    ) -> np.ndarray:
        """Return Gaussian predictive quantiles of shape ``(n, 3, n_levels)``."""
        levels = np.asarray(q_levels, dtype=np.float64)
        if np.any((levels <= 0.0) | (levels >= 1.0)):
            raise ValueError("`q_levels` must lie strictly in (0, 1).")
        res = self.solve(signals)
        z = stats.norm.ppf(levels)
        return res.params[:, :, None] + res.sigma[:, :, None] * z[None, None, :]


SegmentedIVIMEstimator = SegmentedEstimator
