"""Analytical Cramér–Rao Lower Bound (CRLB) normal-approximation IVIM baseline estimator.

Computes deterministic bi-exponential parameter estimates via vectorized
two-stage segmented initialization + simultaneous damped Gauss–Newton
refinement, estimates per-voxel thermal noise scale from residual variance (or
explicit SNR), and forms analytical Fisher-information CRLB normal-approximation
prediction intervals ``theta_hat +/- z_{1 - alpha/2} * sigma_CRLB(theta_hat)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats

from ..acquisition.crlb import compute_crlb
from ..metrics.coverage import (
    DEFAULT_QUANTILE_LEVELS,
    DEFAULT_RAILED_SD_FLOOR,
    SDConvention,
    apply_sd_convention,
    normalize_sd_convention,
)
from ..metrics.railing import DEFAULT_PARAM_BOUNDS, DEFAULT_RAIL_TOL
from ..protocols import AcquisitionProtocol, resolve_bvalues
from .segmented import SegmentedEstimator

__all__ = [
    "CRLBFitResult",
    "CRLBNormalEstimator",
    "CRLBEstimator",
    "crlb_normal_intervals",
]


def crlb_normal_intervals(
    bvalues: str | AcquisitionProtocol | Sequence[float] | np.ndarray,
    params_fit: np.ndarray,
    *,
    snr: float | Sequence[float] | np.ndarray = 30.0,
    alpha: float = 0.10,
    sd_convention: Literal["honest_crlb", "floored_railed"] | str = "honest_crlb",
    bounds: Mapping[str, tuple[float, float]] | None = None,
    rail_tol: float = DEFAULT_RAIL_TOL,
    railed_sd_floor: float = DEFAULT_RAILED_SD_FLOOR,
    nex: Sequence[float] | np.ndarray | None = None,
    s0: float | np.ndarray = 1.0,
    include_s0: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build central ``(1 - alpha)`` CRLB normal-approximation intervals at ``params_fit``.

    Parameters
    ----------
    bvalues : str, AcquisitionProtocol, or sequence of float
        Diffusion weighting schedule in ``s/mm^2``.
    params_fit : (n, 3) ndarray
        Fitted IVIM parameters ``(D, Dstar, f)`` in physical units.
    snr : float or sequence of float, default=30.0
        Baseline ``b = 0`` signal-to-noise ratio.
    alpha : float, default=0.10
        Central interval miscoverage rate (``0.10`` -> 90% nominal coverage).
    sd_convention : {"honest_crlb", "floored_railed"}, default="honest_crlb"
        - ``"honest_crlb"`` (default): Full Fisher/CRLB standard deviations at
          ``params_fit``, yielding honest marginal coverage with under-coverage
          localized conditionally to the high-``D*`` tercile.
        - ``"floored_railed"``: Collapses boundary-railed voxel SDs to
          ``railed_sd_floor`` to reproduce legacy railed-voxel under-coverage.
    """
    conv = normalize_sd_convention(sd_convention)
    b_arr, _ = resolve_bvalues(bvalues)
    pt = np.atleast_2d(np.asarray(params_fit, dtype=np.float64))
    active = dict(DEFAULT_PARAM_BOUNDS)
    if bounds is not None:
        active.update(bounds)

    use_s0 = bool(include_s0) and (b_arr.size >= 4)
    snr_val: float | np.ndarray = (
        float(snr) if isinstance(snr, (int, float)) else np.asarray(snr, dtype=np.float64)
    )
    crlb_res = compute_crlb(b_arr, pt, snr=snr_val, nex=nex, s0=s0, include_s0=use_s0)
    spans = np.array(
        [
            active["D"][1] - active["D"][0],
            active["Dstar"][1] - active["Dstar"][0],
            active["f"][1] - active["f"][0],
        ],
        dtype=np.float64,
    )
    raw_std = np.atleast_2d(crlb_res.std)
    sigma = np.where(np.isfinite(raw_std), raw_std, spans[None, :])
    sigma = np.maximum(sigma, 1e-8)
    sigma = apply_sd_convention(
        sigma,
        sd_convention=conv,
        point_pred=pt,
        bounds=active,
        rail_tol=rail_tol,
        railed_sd_floor=railed_sd_floor,
    )
    z = float(stats.norm.ppf(1.0 - 0.5 * float(alpha)))
    return pt - z * sigma, pt + z * sigma


@dataclass
class CRLBFitResult:
    """Result container for :class:`CRLBNormalEstimator`."""

    params: np.ndarray
    s0: np.ndarray
    sigma_crlb: np.ndarray
    snr_hat: np.ndarray
    railed: np.ndarray


@dataclass
class CRLBNormalEstimator:
    """Plug-in analytical CRLB normal-approximation IVIM baseline estimator.

    Parameters
    ----------
    protocol : str, AcquisitionProtocol, or sequence of float, default="osipi"
        Acquisition b-value schedule in ``s/mm^2``.
    bounds : mapping of str to (float, float) or None, optional
        Physiological box bounds for ``"D"``, ``"Dstar"``, and ``"f"``.
    snr : float or None, default=None
        Optional known ``b = 0`` SNR. When ``None``, per-voxel noise SD is
        estimated directly from residual sum of squares ``sqrt(RSS / dof)``.
    nex : sequence of float or None, optional
        Optional per-b signal averages (NEX) passed into the Fisher matrix.
    include_s0 : bool, default=True
        Whether to marginalize over ``S0`` uncertainty in the 4x4 Fisher matrix
        when ``n_b >= 4``.
    n_lm_iter : int, default=20
        Number of vectorized simultaneous Levenberg–Marquardt refinement steps.
    sd_convention : {"honest_crlb", "floored_railed"}, default="honest_crlb"
        Dual-SD convention on boundary-railed fits:
        - ``"honest_crlb"`` (default): Unclipped Fisher/CRLB standard errors at
          fitted parameters (domain-bounded only on singular entries).
        - ``"floored_railed"``: Assigns collapsed SD floor (``railed_sd_floor``)
          to boundary-railed voxels, reproducing legacy marginal under-coverage.
    railed_sd_floor : float, default=3.0e-3
        Collapsed SD floor (``mm^2/s``) applied to railed ``D*`` voxels when
        ``sd_convention="floored_railed"``.
    rail_tol : float, default=1e-2
        Relative tolerance for boundary-railing diagnostics.
    """

    protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray = "osipi"
    bounds: Mapping[str, tuple[float, float]] | None = None
    snr: float | None = None
    nex: Sequence[float] | np.ndarray | None = None
    include_s0: bool = True
    n_lm_iter: int = 20
    sd_convention: SDConvention | str = "honest_crlb"
    railed_sd_floor: float = DEFAULT_RAILED_SD_FLOOR
    rail_tol: float = DEFAULT_RAIL_TOL
    bvalues: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.bvalues, _ = resolve_bvalues(self.protocol)
        active = dict(DEFAULT_PARAM_BOUNDS)
        if self.bounds is not None:
            active.update(self.bounds)
        self._bounds = active
        self.sd_convention = normalize_sd_convention(self.sd_convention)

    def fit(
        self,
        signals: np.ndarray | None = None,
        params: np.ndarray | None = None,
    ) -> CRLBNormalEstimator:
        """No-op fit method for API parity with learned estimators."""
        _ = (signals, params)
        return self

    def solve(
        self,
        signals: np.ndarray,
        *,
        sd_convention: SDConvention | str | None = None,
    ) -> CRLBFitResult:
        """Fit ``(S0, D, Dstar, f)`` and plug-in analytical CRLB standard errors."""
        conv = (
            normalize_sd_convention(self.sd_convention)
            if sd_convention is None
            else normalize_sd_convention(sd_convention)
        )
        sig = np.atleast_2d(np.asarray(signals, dtype=np.float64))
        b = self.bvalues
        if sig.shape[1] != b.size:
            raise ValueError(
                f"Signal b-axis ({sig.shape[1]}) does not match protocol length ({b.size})."
            )
        n = sig.shape[0]

        # Warm-start from vectorized two-step segmented estimator
        seg = SegmentedEstimator(protocol=b, bounds=self._bounds, uncertainty="homoscedastic")
        seg_res = seg.solve(sig)
        theta = seg_res.params.copy()  # (n, 3): [D, Dstar, f]
        s0_hat = seg_res.s0.copy()

        d_lo, d_hi = self._bounds["D"]
        ds_lo, ds_hi = self._bounds["Dstar"]
        f_lo, f_hi = self._bounds["f"]
        s0_lo = np.maximum(0.1 * s0_hat, 1e-4)
        s0_hi = 2.5 * np.maximum(s0_hat, 1e-4)

        x = np.column_stack([s0_hat, theta])  # (n, 4): [s0, D, Dstar, f]
        lo4 = np.column_stack(
            [s0_lo, np.full(n, d_lo), np.full(n, ds_lo), np.full(n, f_lo)]
        )
        hi4 = np.column_stack(
            [s0_hi, np.full(n, d_hi), np.full(n, ds_hi), np.full(n, f_hi)]
        )
        scale4 = np.array([1.0, 1e-3, 3e-2, 0.2], dtype=np.float64)
        lam: np.ndarray = np.full(n, 1e-3, dtype=np.float64)
        eye4 = np.eye(4, dtype=np.float64)[None, :, :]

        # Vectorized damped Levenberg-Marquardt refinement across all voxels
        for _ in range(int(self.n_lm_iter)):
            s0_c, d_c, ds_c, f_c = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
            e_d = np.exp(-b[None, :] * d_c)
            e_ds = np.exp(-b[None, :] * ds_c)
            pred = s0_c * (f_c * e_ds + (1.0 - f_c) * e_d)
            r = pred - sig
            cost = 0.5 * np.sum(r * r, axis=1)

            j_s0 = f_c * e_ds + (1.0 - f_c) * e_d
            j_d = -s0_c * (1.0 - f_c) * b[None, :] * e_d
            j_ds = -s0_c * f_c * b[None, :] * e_ds
            j_f = s0_c * (e_ds - e_d)
            jac = np.stack([j_s0, j_d, j_ds, j_f], axis=-1) * scale4[None, None, :]

            jtj = np.einsum("nbi,nbj->nij", jac, jac)
            jtr = np.einsum("nbi,nb->ni", jac, r)
            lhs = jtj + (lam[:, None, None] + 1e-12) * eye4
            try:
                delta_sc = -np.linalg.solve(lhs, jtr[..., None])[..., 0]
            except np.linalg.LinAlgError:
                break

            x_cand = np.clip(x + delta_sc * scale4[None, :], lo4, hi4)
            # Enforce D* >= D separation
            x_cand[:, 2] = np.maximum(x_cand[:, 2], np.minimum(x_cand[:, 1] + 5e-4, ds_hi))

            s0_n, d_n, ds_n, f_n = (
                x_cand[:, 0:1],
                x_cand[:, 1:2],
                x_cand[:, 2:3],
                x_cand[:, 3:4],
            )
            pred_n = s0_n * (
                f_n * np.exp(-b[None, :] * ds_n) + (1.0 - f_n) * np.exp(-b[None, :] * d_n)
            )
            cost_n = 0.5 * np.sum((pred_n - sig) ** 2, axis=1)
            improved = cost_n < cost
            x = np.where(improved[:, None], x_cand, x)
            lam = np.where(improved, np.maximum(lam * 0.4, 1e-10), np.minimum(lam * 5.0, 1e6))

        s0_fit = x[:, 0]
        params_fit = x[:, 1:4]

        if self.snr is not None:
            snr_eff = np.full(n, float(self.snr), dtype=np.float64)
        else:
            e_d = np.exp(-b[None, :] * params_fit[:, 0:1])
            e_ds = np.exp(-b[None, :] * params_fit[:, 1:2])
            pred_f = s0_fit[:, None] * (
                params_fit[:, 2:3] * e_ds + (1.0 - params_fit[:, 2:3]) * e_d
            )
            dof = max(b.size - 4, 1)
            sigma_noise = np.sqrt(np.sum((pred_f - sig) ** 2, axis=1) / dof)
            sigma_noise = np.maximum(sigma_noise, 1e-4 * np.maximum(s0_fit, 1e-4))
            snr_eff = np.clip(s0_fit / sigma_noise, 1.0, 1.0e4)

        use_s0 = bool(self.include_s0) and (b.size >= 4)
        crlb_res = compute_crlb(
            b,
            params_fit,
            snr=snr_eff,
            nex=self.nex,
            s0=s0_fit,
            include_s0=use_s0,
        )

        spans = np.array([d_hi - d_lo, ds_hi - ds_lo, f_hi - f_lo], dtype=np.float64)
        raw_std = np.atleast_2d(crlb_res.std)
        # Replace non-identifiable / singular entries with honest domain-span uncertainty
        sigma_crlb = np.where(
            np.isfinite(raw_std),
            np.minimum(raw_std, spans[None, :]),
            0.5 * spans[None, :],
        )
        sigma_crlb = np.maximum(sigma_crlb, 1e-8)

        lo3 = np.array([d_lo, ds_lo, f_lo], dtype=np.float64)
        hi3 = np.array([d_hi, ds_hi, f_hi], dtype=np.float64)
        railed = ((params_fit - lo3[None, :]) <= self.rail_tol * spans[None, :]) | (
            (hi3[None, :] - params_fit) <= self.rail_tol * spans[None, :]
        )

        if conv == "floored_railed":
            sigma_crlb = apply_sd_convention(
                sigma_crlb,
                sd_convention="floored_railed",
                railed_mask=railed,
                bounds=self._bounds,
                rail_tol=self.rail_tol,
                railed_sd_floor=float(self.railed_sd_floor),
            )

        return CRLBFitResult(
            params=params_fit,
            s0=s0_fit,
            sigma_crlb=sigma_crlb,
            snr_hat=snr_eff,
            railed=railed,
        )

    def predict(self, signals: np.ndarray) -> np.ndarray:
        """Return ``(n, 3)`` point estimates ordered ``(D, Dstar, f)``."""
        return self.solve(signals).params

    predict_point = predict

    def predict_intervals(
        self,
        signals: np.ndarray,
        alpha: float = 0.10,
        *,
        sd_convention: SDConvention | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return central ``(1 - alpha)`` analytical CRLB normal-approximation intervals."""
        fit = self.solve(signals, sd_convention=sd_convention)
        z = float(stats.norm.ppf(1.0 - 0.5 * float(alpha)))
        return fit.params - z * fit.sigma_crlb, fit.params + z * fit.sigma_crlb

    def predict_quantiles(
        self,
        signals: np.ndarray,
        q_levels: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS,
        *,
        sd_convention: SDConvention | str | None = None,
    ) -> np.ndarray:
        """Return ``(n, 3, n_levels)`` analytical CRLB normal-approximation quantiles."""
        levels = np.asarray(q_levels, dtype=np.float64)
        if np.any((levels <= 0.0) | (levels >= 1.0)):
            raise ValueError("`q_levels` must lie strictly in (0, 1).")
        fit = self.solve(signals, sd_convention=sd_convention)
        z = stats.norm.ppf(levels)
        return fit.params[:, :, None] + fit.sigma_crlb[:, :, None] * z[None, None, :]


CRLBEstimator = CRLBNormalEstimator
