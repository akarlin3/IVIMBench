"""Analytical IVIM Fisher Information Matrix and Cramér–Rao Lower Bound (CRLB) calculator.

Provides closed-form derivatives, Fisher Information Matrices (FIM), and
Cramér–Rao lower bounds across tissue diffusion ``D``, pseudo-diffusion ``D*``
(``D_star``), microvascular perfusion fraction ``f``, and optional nuisance
baseline intensity ``S0`` with explicit scan-time and NEX (number of
excitations / signal averages) weighting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from ..phantoms.synthetic import PARAM_NAMES
from ..protocols import DEFAULT_SECONDS_PER_VOLUME, AcquisitionProtocol, resolve_bvalues

__all__ = [
    "CRLBResult",
    "resolve_schedule_and_nex",
    "ivim_signal_jacobian",
    "ivim_fisher_matrix",
    "compute_crlb",
    "crlb_optimality_criterion",
]

_CANONICAL_ALIAS: dict[str, str] = {
    "D": "D",
    "d": "D",
    "Dstar": "Dstar",
    "D_star": "Dstar",
    "dstar": "Dstar",
    "d_star": "Dstar",
    "f": "f",
}


def _coerce_theta(
    params: np.ndarray | Sequence[float] | Mapping[str, Any],
    param_order: Sequence[str] = PARAM_NAMES,
) -> tuple[np.ndarray, bool]:
    """Coerce parameters into canonical ``(n, 3)`` ``[D, Dstar, f]`` array."""
    if isinstance(params, Mapping):
        cols: list[np.ndarray] = []
        scalar = True
        for key in ("D", "Dstar", "f"):
            if key in params:
                raw = params[key]
            elif key == "Dstar" and "D_star" in params:
                raw = params["D_star"]
            else:
                raise KeyError(f"Missing parameter {key!r} in `params` mapping.")
            arr = np.asarray(raw, dtype=np.float64)
            if arr.ndim > 0:
                scalar = False
            cols.append(np.atleast_1d(arr).ravel())
        bcast = np.broadcast_arrays(*cols)
        return np.column_stack(bcast), scalar

    arr = np.asarray(params, dtype=np.float64)
    single = arr.ndim == 1
    p2d = np.atleast_2d(arr)
    if p2d.shape[1] != 3:
        raise ValueError(f"Expected 3 IVIM parameters (D, D*, f), got shape {arr.shape}.")

    canonical_order = [_CANONICAL_ALIAS.get(k, k) for k in param_order]
    if tuple(canonical_order) == PARAM_NAMES:
        return p2d, single

    if set(canonical_order) != set(PARAM_NAMES) or len(canonical_order) != 3:
        raise ValueError(
            f"`param_order` must be a permutation of ('D', 'Dstar'/'D_star', 'f'); got {param_order!r}."
        )
    idx_map = {name: i for i, name in enumerate(canonical_order)}
    reordered = np.column_stack([p2d[:, idx_map["D"]], p2d[:, idx_map["Dstar"]], p2d[:, idx_map["f"]]])
    return reordered, single


def _reorder_from_canonical(
    arr: np.ndarray,
    param_order: Sequence[str],
    *,
    is_matrix: bool = False,
) -> np.ndarray:
    canonical_order = [_CANONICAL_ALIAS.get(k, k) for k in param_order]
    if tuple(canonical_order) == PARAM_NAMES:
        return arr
    perm = [PARAM_NAMES.index(k) for k in canonical_order]
    if arr.shape[-1] == 4:
        perm = perm + [3]
    if is_matrix:
        return arr[..., perm, :][..., :, perm]
    return arr[..., perm]


def resolve_schedule_and_nex(
    protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray,
    nex: Sequence[float] | np.ndarray | float | None = None,
    *,
    nex_budget: float | None = None,
    scan_time_minutes: float | None = None,
    seconds_per_volume: float | Sequence[float] = DEFAULT_SECONDS_PER_VOLUME,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Resolve b-values, NEX weights, protocol key, and nominal scan duration (minutes).

    Parameters
    ----------
    protocol : str, AcquisitionProtocol, or 1D sequence of float
        Diffusion b-value schedule in ``s/mm^2``.
    nex : float, sequence of float, or None, optional
        Number of signal averages (NEX) per b-value entry. Defaults to
        ``protocol.averages_arr`` if ``protocol`` is an :class:`AcquisitionProtocol`,
        else ``1.0`` per b-value.
    nex_budget : float or None, optional
        If provided, rescales ``nex`` proportionally so ``sum(nex) == nex_budget``
        for equal-excitation efficiency comparisons.
    scan_time_minutes : float or None, optional
        If provided, rescales ``nex`` proportionally so total acquisition time
        equals ``scan_time_minutes``.
    seconds_per_volume : float or sequence of float, default=12.0
        Repetition/acquisition duration in seconds per single-NEX volume.
    """
    if isinstance(protocol, (str, AcquisitionProtocol)):
        spec = protocol if isinstance(protocol, AcquisitionProtocol) else resolve_bvalues(protocol)[1]
        if isinstance(spec, str):
            from ..protocols import get_protocol

            proto_obj = get_protocol(spec)
        else:
            proto_obj = spec
        b_arr = proto_obj.b
        proto_name = proto_obj.key
        default_nex = proto_obj.averages_arr.astype(np.float64)
    else:
        b_arr, proto_name = resolve_bvalues(protocol)
        default_nex = np.ones(b_arr.size, dtype=np.float64)

    if nex is None:
        w = default_nex
    else:
        nex_arr = np.asarray(nex, dtype=np.float64)
        if nex_arr.ndim == 0:
            w = np.full(b_arr.size, float(nex_arr.item()), dtype=np.float64)
        else:
            w = nex_arr
            if w.shape != b_arr.shape:
                raise ValueError(f"`nex` shape {w.shape} must match bvalues shape {b_arr.shape}.")

    if np.any(w < 0.0) or float(np.sum(w)) <= 0.0:
        raise ValueError("`nex` weights must be non-negative with strictly positive sum.")

    spv = np.broadcast_to(np.asarray(seconds_per_volume, dtype=np.float64), b_arr.shape)
    if np.any(spv <= 0.0):
        raise ValueError("`seconds_per_volume` must be strictly positive.")

    if nex_budget is not None and scan_time_minutes is not None:
        raise ValueError("Specify at most one of `nex_budget` or `scan_time_minutes`.")

    if nex_budget is not None:
        if float(nex_budget) <= 0.0:
            raise ValueError("`nex_budget` must be > 0.")
        w = w * (float(nex_budget) / float(np.sum(w)))

    raw_seconds = float(np.sum(w * spv))
    if scan_time_minutes is not None:
        if float(scan_time_minutes) <= 0.0:
            raise ValueError("`scan_time_minutes` must be > 0.")
        target_sec = 60.0 * float(scan_time_minutes)
        w = w * (target_sec / raw_seconds)
        raw_seconds = target_sec

    return b_arr, w, proto_name, raw_seconds / 60.0


@dataclass(frozen=True)
class CRLBResult:
    """Analytical Fisher Information and Cramér–Rao Lower Bound result container."""

    fim: np.ndarray
    cov: np.ndarray
    std: np.ndarray
    cv: np.ndarray
    interval_width: np.ndarray
    condition_number: np.ndarray | float
    identifiable_mask: np.ndarray | bool
    bvalues: np.ndarray
    nex: np.ndarray
    total_nex: float
    scan_minutes: float
    alpha: float
    param_order: tuple[str, ...] = PARAM_NAMES

    def _param_index(self, target: str) -> int:
        canon = _CANONICAL_ALIAS.get(target, target)
        canon_order = [_CANONICAL_ALIAS.get(k, k) for k in self.param_order]
        return canon_order.index(canon)

    @property
    def variance(self) -> np.ndarray:
        """Parameter CRLB variances (diagonal of inverse Fisher Information Matrix)."""
        return self.std**2

    @property
    def D_std(self) -> np.ndarray | float:
        """CRLB standard error for slow tissue diffusion ``D`` (``mm^2/s``)."""
        idx = self._param_index("D")
        return float(self.std[idx]) if self.std.ndim == 1 else self.std[:, idx]

    @property
    def Dstar_std(self) -> np.ndarray | float:
        """CRLB standard error for pseudo-diffusion ``D*`` (``mm^2/s``)."""
        idx = self._param_index("Dstar")
        return float(self.std[idx]) if self.std.ndim == 1 else self.std[:, idx]

    @property
    def D_star_std(self) -> np.ndarray | float:
        """Alias for :attr:`Dstar_std`."""
        return self.Dstar_std

    @property
    def f_std(self) -> np.ndarray | float:
        """CRLB standard error for microvascular perfusion fraction ``f``."""
        idx = self._param_index("f")
        return float(self.std[idx]) if self.std.ndim == 1 else self.std[:, idx]

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary statistics into a JSON-ready dictionary."""
        std_2d = np.atleast_2d(self.std)
        cv_2d = np.atleast_2d(self.cv)
        width_2d = np.atleast_2d(self.interval_width)
        id_1d = np.atleast_1d(np.asarray(self.identifiable_mask, dtype=bool))
        summary: dict[str, Any] = {
            "total_nex": float(self.total_nex),
            "scan_minutes": float(self.scan_minutes),
            "alpha": float(self.alpha),
            "identifiable_fraction": float(np.mean(id_1d)),
            "parameters": {},
        }
        for j, name in enumerate(self.param_order):
            finite = np.isfinite(std_2d[:, j])
            summary["parameters"][name] = {
                "median_std": float(np.median(std_2d[finite, j])) if np.any(finite) else float("inf"),
                "mean_std": float(np.mean(std_2d[finite, j])) if np.any(finite) else float("inf"),
                "median_cv": float(np.median(cv_2d[finite, j])) if np.any(finite) else float("inf"),
                "median_interval_width": (
                    float(np.median(width_2d[finite, j])) if np.any(finite) else float("inf")
                ),
            }
        return summary


def ivim_signal_jacobian(
    bvalues: np.ndarray | Sequence[float],
    params: np.ndarray | Sequence[float] | Mapping[str, Any],
    *,
    s0: float | np.ndarray = 1.0,
    include_s0: bool = False,
    param_order: Sequence[str] = PARAM_NAMES,
) -> np.ndarray:
    """Analytical Jacobian ``dS/dtheta`` of the bi-exponential IVIM forward model."""
    b = np.asarray(bvalues, dtype=np.float64)
    p2d, single = _coerce_theta(params, param_order=param_order)
    n = p2d.shape[0]

    d_col = p2d[:, 0:1]
    ds_col = p2d[:, 1:2]
    f_col = p2d[:, 2:3]
    s0_col = np.broadcast_to(np.asarray(s0, dtype=np.float64), (n,))[:, None]
    bb = b[None, :]

    e_d = np.exp(-bb * d_col)
    e_ds = np.exp(-bb * ds_col)

    j_d = -s0_col * (1.0 - f_col) * bb * e_d
    j_ds = -s0_col * f_col * bb * e_ds
    j_f = s0_col * (e_ds - e_d)

    jac3 = np.stack([j_d, j_ds, j_f], axis=-1)
    jac3 = _reorder_from_canonical(jac3, param_order=param_order, is_matrix=False)

    if include_s0:
        j_s0 = (f_col * e_ds + (1.0 - f_col) * e_d)[..., None]
        jac = np.concatenate([jac3, j_s0], axis=-1)
    else:
        jac = jac3

    return jac[0] if single else jac


def ivim_fisher_matrix(
    protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray,
    params: np.ndarray | Sequence[float] | Mapping[str, Any],
    *,
    snr: float | np.ndarray = 30.0,
    nex: Sequence[float] | np.ndarray | float | None = None,
    s0: float | np.ndarray = 1.0,
    include_s0: bool = False,
    nex_budget: float | None = None,
    scan_time_minutes: float | None = None,
    seconds_per_volume: float | Sequence[float] = DEFAULT_SECONDS_PER_VOLUME,
    param_order: Sequence[str] = PARAM_NAMES,
) -> np.ndarray:
    """Analytical NEX- and scan-time-weighted Gaussian Fisher Information Matrix.

    .. math::

        I(\\theta) = \\frac{1}{\\sigma_0^2} \\sum_{k=1}^{N_b} n_k \\,
        \\nabla_\\theta S(b_k; \\theta) \\nabla_\\theta S(b_k; \\theta)^\\top,
        \\qquad \\sigma_0 = \\frac{S_0}{\\mathrm{SNR}}
    """
    b_arr, w_nex, _, _ = resolve_schedule_and_nex(
        protocol,
        nex=nex,
        nex_budget=nex_budget,
        scan_time_minutes=scan_time_minutes,
        seconds_per_volume=seconds_per_volume,
    )
    p2d, single = _coerce_theta(params, param_order=param_order)
    n = p2d.shape[0]

    snr_arr = np.broadcast_to(np.asarray(snr, dtype=np.float64), (n,))
    if np.any(snr_arr <= 0.0):
        raise ValueError("`snr` must be strictly positive.")
    s0_arr = np.broadcast_to(np.asarray(s0, dtype=np.float64), (n,))
    sigma0 = s0_arr / snr_arr

    jac = ivim_signal_jacobian(
        b_arr,
        p2d,
        s0=s0_arr,
        include_s0=include_s0,
        param_order=PARAM_NAMES,
    )
    jac = _reorder_from_canonical(jac, param_order=param_order, is_matrix=False)
    jac_w = jac * w_nex[None, :, None]
    fim = np.einsum("nbi,nbj->nij", jac_w, jac) / (sigma0**2)[:, None, None]
    # Symmetrize numerical rounding
    fim = 0.5 * (fim + np.swapaxes(fim, -1, -2))
    return fim[0] if single else fim


def compute_crlb(
    protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray,
    params: np.ndarray | Sequence[float] | Mapping[str, Any],
    *,
    snr: float | np.ndarray = 30.0,
    nex: Sequence[float] | np.ndarray | float | None = None,
    alpha: float = 0.10,
    s0: float | np.ndarray = 1.0,
    include_s0: bool = False,
    nex_budget: float | None = None,
    scan_time_minutes: float | None = None,
    seconds_per_volume: float | Sequence[float] = DEFAULT_SECONDS_PER_VOLUME,
    param_order: Sequence[str] = PARAM_NAMES,
    max_cond: float = 1.0e14,
) -> CRLBResult:
    """Compute NEX-weighted Fisher Information and Cramér–Rao Lower Bounds across ``(D, D*, f)``."""
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"`alpha` must lie strictly in (0, 1); got {alpha}.")

    b_arr, w_nex, _, scan_min = resolve_schedule_and_nex(
        protocol,
        nex=nex,
        nex_budget=nex_budget,
        scan_time_minutes=scan_time_minutes,
        seconds_per_volume=seconds_per_volume,
    )
    p2d_canon, single = _coerce_theta(params, param_order=param_order)
    p2d_ordered = _reorder_from_canonical(p2d_canon, param_order=param_order, is_matrix=False)
    n = p2d_canon.shape[0]
    p_dim = 4 if include_s0 else 3

    fim_3d = ivim_fisher_matrix(
        b_arr,
        p2d_ordered,
        snr=snr,
        nex=w_nex,
        s0=s0,
        include_s0=include_s0,
        param_order=param_order,
    )

    cov = np.full((n, p_dim, p_dim), np.inf, dtype=np.float64)
    std = np.full((n, 3), np.inf, dtype=np.float64)
    cond_arr = np.full(n, np.inf, dtype=np.float64)
    identifiable = np.zeros(n, dtype=bool)

    # Pre-scale Fisher matrix by parameter magnitudes to avoid artificial ill-conditioning
    # purely from physical unit differences (D ~ 1e-3 mm^2/s vs f ~ 0.2 vs S0 ~ 1.0).
    scales = np.ones((n, p_dim), dtype=np.float64)
    scales[:, :3] = np.maximum(np.abs(p2d_ordered), 1.0e-5)
    if include_s0:
        scales[:, 3] = np.maximum(np.abs(np.broadcast_to(s0, (n,))), 1.0e-4)

    for i in range(n):
        mat = fim_3d[i]
        if not np.all(np.isfinite(mat)):
            continue
        sc = scales[i]
        # Dimensionless scaled FIM: F_scaled = diag(sc) @ mat @ diag(sc)
        mat_scaled = (mat * sc[:, None]) * sc[None, :]
        cond_val = float(np.linalg.cond(mat_scaled))
        cond_arr[i] = cond_val
        if not np.isfinite(cond_val) or cond_val > float(max_cond):
            continue
        try:
            inv_scaled = np.linalg.inv(mat_scaled)
            inv_mat = (inv_scaled * sc[:, None]) * sc[None, :]
            diag3 = np.diag(inv_mat)[:3]
            if np.all(np.isfinite(diag3)) and np.all(diag3 > 0.0):
                cov[i] = inv_mat
                std[i] = np.sqrt(diag3)
                identifiable[i] = True
        except np.linalg.LinAlgError:
            continue

    denom = np.maximum(np.abs(p2d_ordered), 1.0e-12)
    cv = std / denom
    z_crit = float(stats.norm.ppf(1.0 - 0.5 * float(alpha)))
    widths = 2.0 * z_crit * std

    order_tuple = tuple(param_order)
    if single:
        return CRLBResult(
            fim=fim_3d[0],
            cov=cov[0],
            std=std[0],
            cv=cv[0],
            interval_width=widths[0],
            condition_number=float(cond_arr[0]),
            identifiable_mask=bool(identifiable[0]),
            bvalues=b_arr,
            nex=w_nex,
            total_nex=float(np.sum(w_nex)),
            scan_minutes=scan_min,
            alpha=float(alpha),
            param_order=order_tuple,
        )

    return CRLBResult(
        fim=fim_3d,
        cov=cov,
        std=std,
        cv=cv,
        interval_width=widths,
        condition_number=cond_arr,
        identifiable_mask=identifiable,
        bvalues=b_arr,
        nex=w_nex,
        total_nex=float(np.sum(w_nex)),
        scan_minutes=scan_min,
        alpha=float(alpha),
        param_order=order_tuple,
    )


def crlb_optimality_criterion(
    protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray,
    params: np.ndarray,
    *,
    criterion: str = "calibration_aware",
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    snr: float = 30.0,
    nex: Sequence[float] | np.ndarray | float | None = None,
    include_s0: bool = True,
    nex_budget: float | None = None,
    penalty_value: float = 1.0e6,
) -> float:
    """Scalar acquisition-design loss averaged over a prior parameter ensemble ``params``.

    Supported criteria:
    - ``"crlb"`` / ``"a_optimal"``: Weighted mean squared CRLB coefficient of
      variation ``E[sum_j w_j (sigma_j / theta_j)^2]``.
    - ``"d_optimal"``: Mean negative log-determinant of the dimensionless
      parameter-scaled Fisher Information Matrix.
    - ``"calibration_aware"``: Combines global relative CRLB variance with an
      explicit penalty on top-``D*``-tercile relative ``D*``/``f`` variance and
      ill-conditioned voxels to mitigate boundary railing and high-``D*``
      conditional under-coverage.
    """
    theta, _ = _coerce_theta(params)
    res = compute_crlb(
        protocol,
        theta,
        snr=snr,
        nex=nex,
        include_s0=include_s0,
        nex_budget=nex_budget,
    )
    w_arr = np.asarray(weights, dtype=np.float64)
    crit = criterion.strip().lower()

    if crit in ("d", "d_optimal", "d-optimal"):
        fim = np.atleast_3d(res.fim)
        p_dim = fim.shape[-1]
        scales = np.ones((theta.shape[0], p_dim), dtype=np.float64)
        scales[:, :3] = np.maximum(np.abs(theta), 1.0e-5)
        fim_sc = (fim * scales[:, :, None]) * scales[:, None, :]
        sign, logdet = np.linalg.slogdet(fim_sc)
        valid = (sign > 0) & np.isfinite(logdet)
        if not np.any(valid):
            return float(penalty_value)
        unid_frac = 1.0 - float(np.mean(valid))
        return float(-np.mean(logdet[valid]) + penalty_value * unid_frac)

    cv_sq = np.minimum(np.atleast_2d(res.cv) ** 2, penalty_value)
    weighted = cv_sq @ w_arr
    unid_frac = 1.0 - float(np.mean(np.atleast_1d(res.identifiable_mask)))

    if crit in ("crlb", "a", "a_optimal", "a-optimal", "cv_trace"):
        return float(np.mean(weighted) + penalty_value * unid_frac)

    if crit in ("calibration_aware", "dstar_wall", "conditional"):
        dstar = theta[:, 1]
        q_hi = float(np.quantile(dstar, 2.0 / 3.0))
        hi_mask = dstar >= q_hi
        # Penalize top-D* tercile CV^2 on D* and f + log10 condition number barrier
        hi_term = float(np.mean(1.5 * cv_sq[hi_mask, 1] + 0.75 * cv_sq[hi_mask, 2]))
        cond_arr = np.atleast_1d(np.asarray(res.condition_number, dtype=np.float64))
        finite_cond = np.where(np.isfinite(cond_arr), np.log10(np.maximum(cond_arr, 1.0)), 14.0)
        cond_barrier = 0.02 * float(np.mean(np.maximum(finite_cond - 4.0, 0.0) ** 2))
        return float(np.mean(weighted) + 0.65 * hi_term + cond_barrier + penalty_value * unid_frac)

    raise ValueError(
        f"Unknown optimality criterion {criterion!r}; choose 'crlb', 'd_optimal', or 'calibration_aware'."
    )
