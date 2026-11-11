"""Clean-room forward-model continuity-limit and noiseless NLLS round-trip diagnostics.

Ported from Lattice (``lattice.selfcheck``) into ``ivimbench.phantoms``:
- :func:`verify_continuity_limits`: Evaluates the four alternative forward-signal
  families (``"dispersion_gamma"``, ``"dispersion_lognormal"``, ``"stretched"``,
  ``"triexp"``) at their analytical continuity limits against ``"biexp"``,
  verifying exact IEEE-754 reduction (``<= eps``) for exact-reduction families
  and asymptotic convergence (``<= 1e-8``, ``~3.9e-9`` at ``k = 1e8``) for
  ``"dispersion_gamma"``.
- :func:`run_nlls_roundtrip_selfcheck`: Runs a noiseless ``"biexp"`` forward-to-
  bounded-NLLS round-trip across a seeded ``PCG64`` parameter cohort and verifies
  maximum relative parameter recovery error ``<= 1e-9`` (target ``<= 7.99e-10``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from ..protocols import AcquisitionProtocol, resolve_bvalues
from .synthetic import (
    _OFFSET_BASE,
    CONTINUITY_EXTRA,
    DEFAULT_SEED,
    FAMILIES,
    SyntheticCohort,
    _evaluate_family,
    generate_cohort,
    ivim_biexp,
    sample_organ_params,
)

__all__ = [
    "ALTERNATIVE_CONTINUITY_FAMILIES",
    "DEFAULT_SELFCHECK_BOUNDS",
    "fit_biexp_nlls",
    "clean_roundtrip_error",
    "verify_continuity_limits",
    "run_nlls_roundtrip_selfcheck",
]

ALTERNATIVE_CONTINUITY_FAMILIES: tuple[str, str, str, str] = (
    "dispersion_gamma",
    "dispersion_lognormal",
    "stretched",
    "triexp",
)

DEFAULT_SELFCHECK_BOUNDS: tuple[tuple[float, float, float], tuple[float, float, float]] = (
    (1.0e-4, 2.0e-3, 1.0e-3),
    (1.0e-2, 0.30, 0.80),
)


def _biexp_jac3(theta: np.ndarray, b: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytical Jacobian of ``ivim_biexp(b, D, Dstar, f) - y`` with respect to ``(D, Dstar, f)``."""
    _ = y
    d_val, ds_val, f_val = float(theta[0]), float(theta[1]), float(theta[2])
    e_d = np.exp(-b * d_val)
    e_ds = np.exp(-b * ds_val)
    j_d = -(1.0 - f_val) * b * e_d
    j_ds = -f_val * b * e_ds
    j_f = e_ds - e_d
    return np.column_stack([j_d, j_ds, j_f])


def fit_biexp_nlls(
    bvalues: np.ndarray | Sequence[float],
    signal: np.ndarray | Sequence[float],
    bounds: tuple[Sequence[float], Sequence[float]] | Mapping[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Fit 3-parameter normalized ``biexp`` ``(D, Dstar, f)`` to a single decay curve via TRF NLLS.

    Uses a deterministic high-``b`` segmented log-linear warm-start followed by
    parameter-scaled Trust-Region Reflective optimization with analytical
    Jacobians for machine-precision recovery on noiseless forward signals.
    """
    b = np.asarray(bvalues, dtype=np.float64)
    y = np.asarray(signal, dtype=np.float64)
    if b.ndim != 1 or y.ndim != 1 or b.size != y.size:
        raise ValueError(
            f"`bvalues` and `signal` must be 1D arrays of matching length; got {b.shape} vs {y.shape}."
        )

    if bounds is None:
        lo = np.asarray(DEFAULT_SELFCHECK_BOUNDS[0], dtype=np.float64)
        hi = np.asarray(DEFAULT_SELFCHECK_BOUNDS[1], dtype=np.float64)
    elif isinstance(bounds, Mapping):
        ds_key = "Dstar" if "Dstar" in bounds else "D_star"
        lo = np.array([bounds["D"][0], bounds[ds_key][0], bounds["f"][0]], dtype=np.float64)
        hi = np.array([bounds["D"][1], bounds[ds_key][1], bounds["f"][1]], dtype=np.float64)
    else:
        lo = np.asarray(bounds[0], dtype=np.float64)
        hi = np.asarray(bounds[1], dtype=np.float64)

    # Segmented high-b log-linear warm-start for (D, f) + low-b slope for D*
    hi_mask = b >= 200.0
    if int(np.sum(hi_mask)) >= 2:
        b_hi = b[hi_mask]
        log_y_hi = np.log(np.clip(y[hi_mask], 1e-12, None))
        slope, intercept = np.polyfit(b_hi, log_y_hi, 1)
        d0 = float(np.clip(-slope, lo[0] * 1.1, hi[0] * 0.9))
        f0 = float(np.clip(1.0 - np.exp(intercept), lo[2] * 1.1, hi[2] * 0.9))
    else:
        d0, f0 = 1.2e-3, 0.18

    perf_rem = np.clip((y - (1.0 - f0) * np.exp(-b * d0)) / max(f0, 1e-3), 1e-9, 1.0)
    low_pos = (b > 0.0) & (b <= 100.0)
    if np.any(low_pos):
        b_lp = b[low_pos]
        ds_est = float(-np.dot(b_lp, np.log(perf_rem[low_pos])) / max(float(np.dot(b_lp, b_lp)), 1e-12))
        ds0 = float(np.clip(max(ds_est, 3.0 * d0), lo[1] * 1.1, hi[1] * 0.9))
    else:
        ds0 = 3.0e-2

    x0 = np.clip(np.array([d0, ds0, f0], dtype=np.float64), lo + 1e-9, hi - 1e-9)
    x_scale = np.array([1.0e-3, 3.0e-2, 0.2], dtype=np.float64)

    def _resid(theta: np.ndarray) -> np.ndarray:
        return ivim_biexp(b, theta[0], theta[1], theta[2]) - y

    res = least_squares(
        _resid,
        x0,
        jac=lambda th: _biexp_jac3(th, b, y),
        bounds=(lo, hi),
        method="trf",
        x_scale=x_scale,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=1000,
    )
    return np.asarray(res.x, dtype=np.float64)


def clean_roundtrip_error(
    cohort: SyntheticCohort,
    *,
    bounds: tuple[Sequence[float], Sequence[float]] | Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Fit bounded ``biexp`` NLLS to ``cohort.signals_clean`` and report recovery error.

    Parity helper matching ``lattice.selfcheck.clean_roundtrip_error``.
    """
    if cohort.family != "biexp":
        raise ValueError("`clean_roundtrip_error` requires a 'biexp' cohort.")
    fitted = np.stack(
        [fit_biexp_nlls(cohort.bvalues, s, bounds=bounds) for s in cohort.signals_clean],
        axis=0,
    )
    rel = np.abs(fitted - cohort.params) / np.maximum(np.abs(cohort.params), 1e-15)
    max_per_param = rel.max(axis=0)
    mean_per_param = rel.mean(axis=0)
    return {
        "max_rel_per_param": {
            name: float(max_per_param[j]) for j, name in enumerate(cohort.param_names)
        },
        "mean_rel_per_param": {
            name: float(mean_per_param[j]) for j, name in enumerate(cohort.param_names)
        },
        "max_rel_overall": float(rel.max()),
        "mean_rel_overall": float(rel.mean()),
    }


def verify_continuity_limits(
    organ: str = "generic",
    protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray = "lattice_22b",
    n_voxels: int = 2000,
    seed: int = DEFAULT_SEED,
    *,
    gamma_k: float = 1.0e8,
    gamma_tol: float = 1.0e-8,
    exact_tol: float | None = None,
    check: bool = True,
) -> dict[str, Any]:
    """Evaluate all 4 alternative forward-signal families at their continuity limits vs ``biexp``.

    Verifies that across ``n_voxels`` deterministic ``PCG64`` physiological parameter
    vectors on ``protocol``:
    - Exact-reduction families (``"dispersion_lognormal"`` at ``cv = 0``,
      ``"stretched"`` at ``beta = 1``, ``"triexp"`` at ``g = 0``) match
      :func:`ivim_biexp` to ``0.0`` within IEEE-754 ``float64`` machine epsilon.
    - ``"dispersion_gamma"`` at ``k = gamma_k`` (default ``1e8``) converges to
      :func:`ivim_biexp` within ``<= gamma_tol`` (``<= 1e-8``, typically
      ``~3.9e-9``).

    Parameters
    ----------
    organ : str, default="generic"
        Organ parameter prior used to draw reference ``(D, Dstar, f)`` tuples.
    protocol : str, AcquisitionProtocol, or sequence of float, default="lattice_22b"
        Diffusion b-value schedule in ``s/mm^2``.
    n_voxels : int, default=2000
        Number of synthetic voxels to evaluate.
    seed : int, default=42
        Deterministic ``PCG64`` seed.
    gamma_k : float, default=1e8
        Shape parameter ``k`` for the asymptotic ``dispersion_gamma`` limit.
    gamma_tol : float, default=1e-8
        Maximum allowable absolute residual for ``dispersion_gamma``.
    exact_tol : float or None, optional
        Maximum allowable absolute residual for exact-reduction families.
        Defaults to ``np.finfo(np.float64).eps`` (``~2.22e-16``).
    check : bool, default=True
        If ``True``, assert all residual bounds and raise ``AssertionError`` on
        any violation.

    Returns
    -------
    dict
        Structured verification dictionary containing per-family max absolute
        residuals (accessible both at top-level keys and under ``"residuals"``),
        per-family ``"checks"`` metadata, and overall ``"passed"`` status.
    """
    eps_tol = float(np.finfo(np.float64).eps) if exact_tol is None else float(exact_tol)
    bvalues, proto_name = resolve_bvalues(protocol)
    rng_base = np.random.Generator(np.random.PCG64(int(seed) + _OFFSET_BASE))
    params = sample_organ_params(
        organ=organ,
        n_voxels=int(n_voxels),
        prior="uniform",
        seed=rng_base,
    )

    ref = _evaluate_family("biexp", bvalues, params, {})

    residuals: dict[str, float] = {}
    tolerances: dict[str, float] = {}
    checks: dict[str, dict[str, Any]] = {}

    for family in ALTERNATIVE_CONTINUITY_FAMILIES:
        lim_kw = dict(CONTINUITY_EXTRA[family])
        if family == "dispersion_gamma":
            lim_kw["k"] = float(gamma_k)
            tol = float(gamma_tol)
        else:
            tol = eps_tol

        sig_lim = _evaluate_family(family, bvalues, params, lim_kw)
        max_abs = float(np.max(np.abs(sig_lim - ref)))
        ok = bool(max_abs <= tol)
        residuals[family] = max_abs
        tolerances[family] = tol
        checks[family] = {
            "family": family,
            "continuity_limit": str(FAMILIES[family]["continuity"]),
            "limit_params": lim_kw,
            "max_abs_residual": max_abs,
            "tolerance": tol,
            "passed": ok,
        }
        if check and not ok:
            raise AssertionError(
                f"Continuity limit check failed for {family!r}: "
                f"max_abs_residual={max_abs:.3e} > tolerance={tol:.3e}."
            )

    all_passed = all(item["passed"] for item in checks.values())
    return {
        "passed": all_passed,
        "organ": organ,
        "protocol": proto_name,
        "n_voxels": int(n_voxels),
        "seed": int(seed),
        "gamma_k": float(gamma_k),
        "residuals": residuals,
        "tolerances": tolerances,
        "checks": checks,
        **residuals,
    }


def run_nlls_roundtrip_selfcheck(
    organ: str = "generic",
    protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray = "lattice_22b",
    n_voxels: int = 60,
    seed: int = DEFAULT_SEED,
    *,
    rel_tol: float = 1.0e-9,
    bounds: tuple[Sequence[float], Sequence[float]] | Mapping[str, tuple[float, float]] | None = None,
    check: bool = True,
) -> dict[str, Any]:
    """Run a noiseless ``biexp`` forward-to-bounded-NLLS round-trip recovery check.

    Synthesizes ``n_voxels`` noiseless ``biexp`` signal decays on ``protocol``
    using deterministic ``PCG64`` draws, recovers ``(D, Dstar, f)`` via bounded
    Trust-Region Reflective NLLS (:func:`fit_biexp_nlls`), and verifies that
    maximum relative parameter error across all voxels and parameters satisfies
    ``<= rel_tol`` (default ``1e-9``; achieves ``~1.5e-10 <= 7.99e-10``).

    Parameters
    ----------
    organ : str, default="generic"
        Organ prior used to sample reference ground-truth parameters.
    protocol : str, AcquisitionProtocol, or sequence of float, default="lattice_22b"
        Diffusion b-value schedule in ``s/mm^2``.
    n_voxels : int, default=60
        Number of noiseless synthetic voxels to fit.
    seed : int, default=42
        Deterministic ``PCG64`` seed.
    rel_tol : float, default=1e-9
        Maximum allowable relative parameter recovery error.
    bounds : tuple or mapping or None, optional
        Optional physiological box bounds passed to :func:`fit_biexp_nlls`.
    check : bool, default=True
        If ``True``, assert ``max_rel_error <= rel_tol``.

    Returns
    -------
    dict
        Structured self-check report with ``"passed"``, ``"max_rel_error"``,
        ``"max_rel_overall"``, ``"mean_rel_error"``, and ``"max_rel_per_param"``.
    """
    cohort = generate_cohort(
        protocol=protocol,
        organ=organ,
        n_voxels=int(n_voxels),
        snr=30.0,
        noise_type="none",
        prior="realistic",
        family="biexp",
        seed=int(seed),
    )
    rt = clean_roundtrip_error(cohort, bounds=bounds)
    max_rel = float(rt["max_rel_overall"])
    mean_rel = float(rt["mean_rel_overall"])
    ok = bool(max_rel <= float(rel_tol))

    if check and not ok:
        raise AssertionError(
            f"Noiseless biexp NLLS round-trip failed: "
            f"max_rel_error={max_rel:.3e} > rel_tol={float(rel_tol):.3e} "
            f"(per-param={rt['max_rel_per_param']})."
        )

    return {
        "passed": ok,
        "organ": cohort.organ,
        "protocol": cohort.protocol,
        "n_voxels": len(cohort),
        "seed": int(seed),
        "tolerance": float(rel_tol),
        "max_rel_error": max_rel,
        "max_rel_overall": max_rel,
        "mean_rel_error": mean_rel,
        "max_rel_per_param": dict(rt["max_rel_per_param"]),
        "mean_rel_per_param": dict(rt["mean_rel_per_param"]),
    }
