"""Boundary-railing diagnostics and non-parametric bootstrap confidence intervals.

Ported and unified from Caliper (``caliper.baselines``) and Sextant
(``sextant_core.fitting`` and ``sextant_core.calibration``):
- Flags voxel estimates pinned within a relative tolerance of physiological or
  box-optimization bounds (:func:`classify_railing`, :func:`railing_fraction`),
  providing an assumption-free, ground-truth-independent identifiability readout
  for weakly identified parameters (principally ``D*``).
- Computes chunked non-parametric percentile bootstrap confidence intervals for
  railing fractions and percentages (:func:`bootstrap_railing_ci`,
  :func:`evaluate_parameter_railing`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "DEFAULT_PARAM_BOUNDS",
    "DEFAULT_RAIL_TOL",
    "BootstrapCI",
    "classify_railing",
    "railing_fraction",
    "bootstrap_railing_ci",
    "evaluate_parameter_railing",
]

# Standard physical box-optimization bounds: D and Dstar in mm^2/s, f dimensionless.
DEFAULT_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "D": (1.0e-4, 4.0e-3),
    "Dstar": (3.0e-3, 0.20),
    "f": (0.0, 0.70),
}

DEFAULT_RAIL_TOL: float = 1.0e-2


@dataclass(frozen=True)
class BootstrapCI:
    """Non-parametric percentile bootstrap confidence interval for a railing proportion."""

    point: float
    lo: float
    hi: float
    pct: float
    pct_lo: float
    pct_hi: float
    confidence: float = 0.95
    n_boot: int = 2000
    seed: int = 42

    def to_dict(self) -> dict[str, float | int]:
        """Serialize bootstrap confidence interval to a dictionary."""
        return {
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "pct": self.pct,
            "pct_lo": self.pct_lo,
            "pct_hi": self.pct_hi,
            "confidence": self.confidence,
            "n_boot": self.n_boot,
            "seed": self.seed,
        }


def classify_railing(
    estimates: np.ndarray | Sequence[float] | float,
    bounds: tuple[float, float],
    rel_tol: float = DEFAULT_RAIL_TOL,
) -> dict[str, np.ndarray]:
    """Identify estimates lying within ``rel_tol * (upper - lower)`` of box bounds.

    Parameters
    ----------
    estimates : array-like or float
        Fitted parameter values for a single parameter.
    bounds : (lower, upper) tuple of float
        Optimization or physiological box bounds.
    rel_tol : float, default=1e-2
        Tolerance expressed as a fraction of ``upper - lower``.

    Returns
    -------
    dict of str to np.ndarray
        Boolean arrays ``"railed_mask"``, ``"lower_mask"``, ``"upper_mask"``, and
        ``"finite_mask"``.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi <= lo:
        raise ValueError(f"Upper bound must exceed lower bound; got ({lo}, {hi}).")
    if float(rel_tol) < 0.0:
        raise ValueError(f"`rel_tol` must be non-negative; got {rel_tol}.")

    arr = np.asarray(estimates, dtype=np.float64)
    tol = float(rel_tol) * (hi - lo)
    finite = np.isfinite(arr)
    lower_mask = finite & ((arr - lo) <= tol)
    upper_mask = finite & ((hi - arr) <= tol)
    railed_mask = lower_mask | upper_mask
    return {
        "railed_mask": railed_mask,
        "lower_mask": lower_mask,
        "upper_mask": upper_mask,
        "finite_mask": finite,
    }


def railing_fraction(
    estimates: np.ndarray | Sequence[float],
    bounds: tuple[float, float],
    rel_tol: float = DEFAULT_RAIL_TOL,
) -> dict[str, float]:
    """Compute total, lower-bound, and upper-bound railing fractions and percentages."""
    masks = classify_railing(estimates, bounds=bounds, rel_tol=rel_tol)
    finite = masks["finite_mask"]
    if not np.any(finite):
        return {
            "railed_frac": float("nan"),
            "lower_frac": float("nan"),
            "upper_frac": float("nan"),
            "railed_pct": float("nan"),
            "lower_pct": float("nan"),
            "upper_pct": float("nan"),
        }
    r_frac = float(np.mean(masks["railed_mask"][finite]))
    l_frac = float(np.mean(masks["lower_mask"][finite]))
    u_frac = float(np.mean(masks["upper_mask"][finite]))
    return {
        "railed_frac": r_frac,
        "lower_frac": l_frac,
        "upper_frac": u_frac,
        "railed_pct": 100.0 * r_frac,
        "lower_pct": 100.0 * l_frac,
        "upper_pct": 100.0 * u_frac,
    }


def bootstrap_railing_ci(
    railed_mask: np.ndarray | Sequence[bool],
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    chunk: int = 250,
) -> BootstrapCI:
    """Compute a chunked non-parametric percentile bootstrap CI for a binary railing mask."""
    x = np.asarray(railed_mask, dtype=bool).astype(np.float64).ravel()
    n = x.size
    if n == 0:
        raise ValueError("`railed_mask` must not be empty.")
    if int(n_boot) < 1:
        raise ValueError(f"`n_boot` must be >= 1; got {n_boot}.")
    if not (0.0 < float(confidence) < 1.0):
        raise ValueError(f"`confidence` must lie strictly in (0, 1); got {confidence}.")

    rng = np.random.default_rng(int(seed))
    point = float(np.mean(x))
    means = np.empty(int(n_boot), dtype=np.float64)
    done = 0
    while done < int(n_boot):
        m = min(chunk, int(n_boot) - done)
        idx = rng.integers(0, n, size=(m, n))
        means[done : done + m] = x[idx].mean(axis=1)
        done += m

    tail = 0.5 * (1.0 - float(confidence))
    lo, hi = np.percentile(means, [100.0 * tail, 100.0 * (1.0 - tail)])
    return BootstrapCI(
        point=point,
        lo=float(lo),
        hi=float(hi),
        pct=100.0 * point,
        pct_lo=100.0 * float(lo),
        pct_hi=100.0 * float(hi),
        confidence=float(confidence),
        n_boot=int(n_boot),
        seed=int(seed),
    )


def evaluate_parameter_railing(
    point_estimates: np.ndarray,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    param_names: Sequence[str] = ("D", "Dstar", "f"),
    rel_tol: float = DEFAULT_RAIL_TOL,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Evaluate boundary-railing rates and bootstrap CIs across all IVIM parameters.

    Parameters
    ----------
    point_estimates : (n, n_params) np.ndarray
        Fitted parameter matrix ordered according to ``param_names``.
    bounds : mapping of {param_name: (lo, hi)} or None, optional
        Optimization bounds per parameter. Defaults to :data:`DEFAULT_PARAM_BOUNDS`.
    param_names : sequence of str, default=("D", "Dstar", "f")
        Names of columns in ``point_estimates``.
    rel_tol : float, default=1e-2
        Relative tolerance as a fraction of ``hi - lo``.
    n_boot : int, default=2000
        Number of bootstrap resamples per parameter.
    confidence : float, default=0.95
        Confidence level for bootstrap intervals.
    seed : int, default=42
        Base RNG seed (offset deterministically per parameter).

    Returns
    -------
    dict of str to dict
        Per-parameter (plus ``"any"``) railing fractions, percentages, lower/upper
        splits, and bootstrap CIs.
    """
    est = np.asarray(point_estimates, dtype=np.float64)
    if est.ndim != 2 or est.shape[1] != len(param_names):
        raise ValueError(
            f"`point_estimates` must have shape (n, {len(param_names)}); got {est.shape}."
        )
    active_bounds = dict(DEFAULT_PARAM_BOUNDS)
    if bounds is not None:
        active_bounds.update(bounds)

    results: dict[str, dict[str, Any]] = {}
    any_railed = np.zeros(est.shape[0], dtype=bool)
    all_finite = np.ones(est.shape[0], dtype=bool)

    for j, name in enumerate(param_names):
        if name not in active_bounds:
            raise KeyError(f"Missing bounds for parameter {name!r}.")
        b_pair = active_bounds[name]
        masks = classify_railing(est[:, j], bounds=b_pair, rel_tol=rel_tol)
        finite = masks["finite_mask"]
        all_finite &= finite
        any_railed |= masks["railed_mask"]

        fracs = railing_fraction(est[:, j], bounds=b_pair, rel_tol=rel_tol)
        if np.any(finite):
            ci = bootstrap_railing_ci(
                masks["railed_mask"][finite],
                n_boot=n_boot,
                confidence=confidence,
                seed=int(seed) + j,
            )
        else:
            ci = BootstrapCI(
                point=float("nan"),
                lo=float("nan"),
                hi=float("nan"),
                pct=float("nan"),
                pct_lo=float("nan"),
                pct_hi=float("nan"),
                confidence=float(confidence),
                n_boot=int(n_boot),
                seed=int(seed) + j,
            )

        results[name] = {
            **fracs,
            "bounds": tuple(b_pair),
            "ci": ci,
            "ci_lo": ci.lo,
            "ci_hi": ci.hi,
            "pct_ci_lo": ci.pct_lo,
            "pct_ci_hi": ci.pct_hi,
        }

    if np.any(all_finite):
        any_ci = bootstrap_railing_ci(
            any_railed[all_finite],
            n_boot=n_boot,
            confidence=confidence,
            seed=int(seed) + 99,
        )
        any_frac = float(np.mean(any_railed[all_finite]))
    else:
        any_ci = BootstrapCI(
            point=float("nan"),
            lo=float("nan"),
            hi=float("nan"),
            pct=float("nan"),
            pct_lo=float("nan"),
            pct_hi=float("nan"),
            confidence=float(confidence),
            n_boot=int(n_boot),
            seed=int(seed) + 99,
        )
        any_frac = float("nan")

    results["any"] = {
        "railed_frac": any_frac,
        "railed_pct": 100.0 * any_frac,
        "ci": any_ci,
        "ci_lo": any_ci.lo,
        "ci_hi": any_ci.hi,
        "pct_ci_lo": any_ci.pct_lo,
        "pct_ci_hi": any_ci.pct_hi,
    }
    return results
