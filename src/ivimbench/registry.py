"""Extensible IVIM estimator registry and unified prediction protocol.

Enables researchers to benchmark custom NLLS, segmented, Bayesian, conformal,
or deep-learning (``IVIMnet``) estimators in ~5 lines of Python:

.. code-block:: python

    from ivimbench import register_estimator, run_benchmark_suite

    @register_estimator("my_ivimnet")
    def fit_my_ivimnet(signals, bvalues):
        ...
        return {"D": d_hat, "D_star": ds_hat, "f": f_hat, "intervals": (lo, hi)}

    res = run_benchmark_suite(estimators=["my_ivimnet", "segmented", "crlb"])
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy import stats

from .baselines.crlb_estimator import CRLBNormalEstimator
from .baselines.nlls import NLLSEstimator
from .baselines.segmented import SegmentedEstimator
from .metrics.coverage import DEFAULT_QUANTILE_LEVELS, SDConvention, central_interval
from .metrics.railing import DEFAULT_PARAM_BOUNDS, DEFAULT_RAIL_TOL, classify_railing
from .phantoms.synthetic import PARAM_NAMES
from .protocols import resolve_bvalues

__all__ = [
    "DEFAULT_REGISTERED_ESTIMATORS",
    "EstimatorCallable",
    "NormalizedEstimatorOutput",
    "RegisteredEstimator",
    "register_estimator",
    "unregister_estimator",
    "get_estimator",
    "list_estimators",
    "normalize_estimator_output",
    "run_estimator",
]

DEFAULT_REGISTERED_ESTIMATORS: tuple[str, ...] = ("segmented", "nlls", "crlb")


@runtime_checkable
class EstimatorCallable(Protocol):
    """Protocol for any callable fitting ``(signals, bvalues) -> {D, D_star, f, intervals?}``."""

    def __call__(self, signals: np.ndarray, bvalues: np.ndarray) -> Mapping[str, Any]: ...


@dataclass
class NormalizedEstimatorOutput:
    """Standardized estimator output across point estimates, intervals, and diagnostics."""

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    has_intervals: bool = True
    railed: np.ndarray | None = None
    quantiles: np.ndarray | None = None
    q_levels: np.ndarray | None = None
    std: np.ndarray | None = None
    s0: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def D(self) -> np.ndarray:
        return self.point[:, 0]

    @property
    def D_star(self) -> np.ndarray:
        return self.point[:, 1]

    @property
    def Dstar(self) -> np.ndarray:
        return self.point[:, 1]

    @property
    def f(self) -> np.ndarray:
        return self.point[:, 2]

    @property
    def intervals(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.lower, self.upper)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "D": self.D,
            "D_star": self.D_star,
            "Dstar": self.Dstar,
            "f": self.f,
            "point": self.point,
            "intervals": (self.lower, self.upper),
            "lower": self.lower,
            "upper": self.upper,
            "has_intervals": self.has_intervals,
        }
        if self.railed is not None:
            out["railed"] = self.railed
        if self.s0 is not None:
            out["s0"] = self.s0
        if self.std is not None:
            out["std"] = self.std
        out.update(self.extras)
        return out


def _extract_param_col(mapping: Mapping[str, Any], param: str, n: int | None = None) -> np.ndarray:
    if param in mapping:
        raw = mapping[param]
    elif param == "Dstar" and "D_star" in mapping:
        raw = mapping["D_star"]
    elif param == "D_star" and "Dstar" in mapping:
        raw = mapping["Dstar"]
    else:
        raise KeyError(
            f"Estimator output dict missing parameter {param!r} (expected keys 'D', 'D_star'/'Dstar', 'f')."
        )
    arr = np.atleast_1d(np.asarray(raw, dtype=np.float64)).ravel()
    if n is not None and arr.size == 1 and n > 1:
        arr = np.full(n, float(arr[0]), dtype=np.float64)
    if n is not None and arr.size != n:
        raise ValueError(f"Parameter {param!r} has length {arr.size}, expected {n}.")
    return arr


def _extract_bounds_pair(
    iv_obj: Any,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse flexible ``intervals`` formats into ``(lower, upper)`` of shape ``(n, 3)``."""
    if isinstance(iv_obj, Mapping):
        lo_cols: list[np.ndarray] = []
        hi_cols: list[np.ndarray] = []
        for p in PARAM_NAMES:
            key = p if p in iv_obj else ("D_star" if (p == "Dstar" and "D_star" in iv_obj) else p)
            if key not in iv_obj:
                raise KeyError(f"Missing interval entry for {p!r} in `intervals` mapping.")
            entry = iv_obj[key]
            if isinstance(entry, tuple) and len(entry) == 2:
                lo_p = np.atleast_1d(np.asarray(entry[0], dtype=np.float64)).ravel()
                hi_p = np.atleast_1d(np.asarray(entry[1], dtype=np.float64)).ravel()
            else:
                arr_p = np.asarray(entry, dtype=np.float64)
                if arr_p.ndim == 2 and arr_p.shape == (n, 2):
                    lo_p, hi_p = arr_p[:, 0], arr_p[:, 1]
                elif arr_p.ndim == 1 and arr_p.size == 2 and n == 1:
                    lo_p, hi_p = arr_p[:1], arr_p[1:]
                else:
                    raise ValueError(
                        f"Expected interval entry for {p!r} to be (lo, hi) or shape ({n}, 2); got {arr_p.shape}."
                    )
            lo_cols.append(lo_p)
            hi_cols.append(hi_p)
        return np.column_stack(lo_cols), np.column_stack(hi_cols)

    if isinstance(iv_obj, tuple) and len(iv_obj) == 2:
        raw_lo, raw_hi = iv_obj
        if isinstance(raw_lo, Mapping) and isinstance(raw_hi, Mapping):
            lo = np.column_stack([_extract_param_col(raw_lo, p, n) for p in PARAM_NAMES])
            hi = np.column_stack([_extract_param_col(raw_hi, p, n) for p in PARAM_NAMES])
            return lo, hi
        lo = np.atleast_2d(np.asarray(raw_lo, dtype=np.float64))
        hi = np.atleast_2d(np.asarray(raw_hi, dtype=np.float64))
        if lo.shape == (n, 3) and hi.shape == (n, 3):
            return lo, hi
        raise ValueError(f"Interval tuple arrays must have shape ({n}, 3); got {lo.shape}, {hi.shape}.")

    arr = np.asarray(iv_obj, dtype=np.float64)
    if arr.ndim == 3 and arr.shape == (n, 3, 2):
        return arr[:, :, 0], arr[:, :, 1]
    if arr.ndim == 2 and arr.shape == (3, 2) and n == 1:
        return arr[None, :, 0], arr[None, :, 1]
    raise ValueError(
        f"Unsupported `intervals` structure; expected (lower, upper), per-param dict, or ({n}, 3, 2) array."
    )


def normalize_estimator_output(
    raw: Any,
    *,
    n_voxels: int,
    alpha: float = 0.10,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    rail_tol: float = DEFAULT_RAIL_TOL,
) -> NormalizedEstimatorOutput:
    """Normalize raw estimator return values into a :class:`NormalizedEstimatorOutput`."""
    if isinstance(raw, NormalizedEstimatorOutput):
        return raw

    active_bounds = dict(DEFAULT_PARAM_BOUNDS)
    if bounds is not None:
        active_bounds.update(bounds)

    z_crit = float(stats.norm.ppf(1.0 - 0.5 * float(alpha)))
    quantiles: np.ndarray | None = None
    q_levels: np.ndarray | None = None
    std_arr: np.ndarray | None = None
    s0_arr: np.ndarray | None = None
    railed_arr: np.ndarray | None = None
    extras: dict[str, Any] = {}
    has_intervals = True

    if isinstance(raw, Mapping):
        if "point" in raw and not ("D" in raw or "Dstar" in raw or "D_star" in raw):
            pt = np.atleast_2d(np.asarray(raw["point"], dtype=np.float64))
        else:
            d_col = _extract_param_col(raw, "D", n_voxels)
            ds_col = _extract_param_col(raw, "Dstar", n_voxels)
            f_col = _extract_param_col(raw, "f", n_voxels)
            pt = np.column_stack([d_col, ds_col, f_col])

        if pt.shape != (n_voxels, 3):
            raise ValueError(f"Point predictions must have shape ({n_voxels}, 3); got {pt.shape}.")

        if raw.get("s0") is not None:
            s0_arr = np.atleast_1d(np.asarray(raw["s0"], dtype=np.float64)).ravel()
        if raw.get("std") is not None or raw.get("sigma") is not None:
            raw_sd = raw["std"] if raw.get("std") is not None else raw["sigma"]
            if isinstance(raw_sd, Mapping):
                std_arr = np.column_stack([_extract_param_col(raw_sd, p, n_voxels) for p in PARAM_NAMES])
            else:
                std_arr = np.atleast_2d(np.asarray(raw_sd, dtype=np.float64))
        if raw.get("quantiles") is not None:
            quantiles = np.asarray(raw["quantiles"], dtype=np.float64)
            q_levels = np.asarray(raw.get("q_levels", DEFAULT_QUANTILE_LEVELS), dtype=np.float64)

        if raw.get("intervals") is not None:
            lo, hi = _extract_bounds_pair(raw["intervals"], n_voxels)
        elif raw.get("lower") is not None and raw.get("upper") is not None:
            lo, hi = _extract_bounds_pair((raw["lower"], raw["upper"]), n_voxels)
        elif quantiles is not None and q_levels is not None:
            lo, hi = central_interval(quantiles, q_levels, alpha=alpha)
        elif std_arr is not None:
            lo = pt - z_crit * std_arr
            hi = pt + z_crit * std_arr
        else:
            has_intervals = False
            lo = pt.copy()
            hi = pt.copy()

        if raw.get("railed") is not None:
            railed_arr = np.atleast_2d(np.asarray(raw["railed"], dtype=bool))

        reserved = {
            "D",
            "Dstar",
            "D_star",
            "f",
            "point",
            "intervals",
            "lower",
            "upper",
            "std",
            "sigma",
            "quantiles",
            "q_levels",
            "s0",
            "railed",
            "has_intervals",
        }
        extras = {k: v for k, v in raw.items() if k not in reserved}

    elif isinstance(raw, tuple) and len(raw) == 2:
        pt = np.atleast_2d(np.asarray(raw[0], dtype=np.float64))
        lo, hi = _extract_bounds_pair(raw[1], n_voxels)
    else:
        pt = np.atleast_2d(np.asarray(raw, dtype=np.float64))
        if pt.shape != (n_voxels, 3):
            raise ValueError(
                f"Expected estimator output dict or ({n_voxels}, 3) array; got shape {pt.shape}."
            )
        has_intervals = False
        lo = pt.copy()
        hi = pt.copy()

    hi = np.maximum(hi, lo)
    if railed_arr is None:
        railed_cols = [
            classify_railing(pt[:, j], bounds=active_bounds[p], rel_tol=rail_tol)["railed_mask"]
            for j, p in enumerate(PARAM_NAMES)
        ]
        railed_arr = np.column_stack(railed_cols)

    return NormalizedEstimatorOutput(
        point=pt,
        lower=lo,
        upper=hi,
        has_intervals=has_intervals,
        railed=railed_arr,
        quantiles=quantiles,
        q_levels=q_levels,
        std=std_arr,
        s0=s0_arr,
        extras=extras,
    )


def _call_with_supported_kwargs(
    fn: Callable[..., Any],
    signals: np.ndarray,
    bvalues: np.ndarray,
    *,
    alpha: float,
    extra_kwargs: Mapping[str, Any],
) -> Any:
    """Invoke a callable ``fn(signals, bvalues, ...)`` passing only accepted keyword arguments."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(signals, bvalues)

    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    call_kw: dict[str, Any] = {}
    if accepts_var_kw or "alpha" in params:
        call_kw["alpha"] = alpha
    for k, v in extra_kwargs.items():
        if accepts_var_kw or k in params:
            call_kw[k] = v
    return fn(signals, bvalues, **call_kw)


def _instantiate_estimator_class(cls: type[Any], bvalues: np.ndarray, extra_kwargs: Mapping[str, Any]) -> Any:
    """Instantiate an estimator class, forwarding ``protocol``/``bvalues`` if accepted."""
    try:
        sig = inspect.signature(cls)
        params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        init_kw: dict[str, Any] = {}
        if "protocol" in params:
            init_kw["protocol"] = bvalues
        elif "bvalues" in params:
            init_kw["bvalues"] = bvalues
        for k, v in extra_kwargs.items():
            if (accepts_var_kw or k in params) and k not in init_kw:
                init_kw[k] = v
        return cls(**init_kw)
    except (TypeError, ValueError):
        return cls()


@dataclass(frozen=True)
class RegisteredEstimator:
    """Metadata and callable target for a registered IVIM estimator."""

    name: str
    target: Any
    description: str = ""
    tags: tuple[str, ...] = ()

    def run(
        self,
        signals: np.ndarray,
        bvalues: Sequence[float] | np.ndarray | str,
        *,
        alpha: float = 0.10,
        **kwargs: Any,
    ) -> NormalizedEstimatorOutput:
        """Execute this estimator on ``(signals, bvalues)`` and normalize predictions."""
        return run_estimator(self.target, signals, bvalues, alpha=alpha, **kwargs)

    predict = run

    def __call__(
        self,
        signals: np.ndarray,
        bvalues: Sequence[float] | np.ndarray | str,
        *,
        alpha: float = 0.10,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run and return standardized dictionary ``{D, D_star, f, intervals, ...}``."""
        return self.run(signals, bvalues, alpha=alpha, **kwargs).to_dict()


_ESTIMATOR_REGISTRY: dict[str, RegisteredEstimator] = {}
_ESTIMATOR_ALIASES: dict[str, str] = {
    "segmented_nlls": "segmented",
    "two_step_segmented": "segmented",
    "bounded_nlls": "nlls",
    "simultaneous_nlls": "nlls",
    "crlb_normal": "crlb",
    "analytical_crlb": "crlb",
    "maf_posterior": "maf",
}


def _normalize_estimator_name(name: str) -> str:
    cleaned = name.strip().lower().replace("-", "_").replace(" ", "_")
    return _ESTIMATOR_ALIASES.get(cleaned, cleaned)


def register_estimator(
    name_or_target: str | Callable[..., Any] | type[Any] | None = None,
    target: Callable[..., Any] | type[Any] | Any | None = None,
    *,
    name: str | None = None,
    description: str = "",
    tags: Sequence[str] = (),
    overwrite: bool = True,
) -> Any:
    """Register an IVIM estimator callable or class into the global benchmark registry.

    Supports three ergonomic call patterns:
    1. ``@register_estimator("my_net")`` decorator on a function or class,
    2. ``@register_estimator`` bare decorator (uses ``fn.__name__``),
    3. ``register_estimator("my_net", fn_or_instance)`` direct call.

    Any registered callable or class receives ``(signals, bvalues)`` (and optional
    ``alpha`` if present in its signature) and returns
    ``{"D": ..., "D_star": ..., "f": ..., "intervals": (lo, hi)}`` (with
    ``"intervals"`` optional and ``"Dstar"`` accepted as an alias for ``"D_star"``).
    """

    def _do_register(reg_name: str, obj: Any) -> Any:
        norm_key = reg_name.strip().lower().replace("-", "_").replace(" ", "_")
        if not norm_key:
            raise ValueError("Estimator registry name must be non-empty.")
        if not overwrite and norm_key in _ESTIMATOR_REGISTRY:
            raise ValueError(f"Estimator {norm_key!r} is already registered.")
        doc = description or (inspect.getdoc(obj) or "").splitlines()[0] if (description or inspect.getdoc(obj)) else ""
        _ESTIMATOR_REGISTRY[norm_key] = RegisteredEstimator(
            name=norm_key,
            target=obj,
            description=doc,
            tags=tuple(tags),
        )
        return obj

    # Case 1: register_estimator("my_net", fn_or_cls)
    if isinstance(name_or_target, str) and target is not None:
        return _do_register(name_or_target, target)

    # Case 2: @register_estimator("my_net", description=...)
    if isinstance(name_or_target, str) and target is None:
        reg_key = name_or_target

        def _decorator(obj: Any) -> Any:
            return _do_register(reg_key, obj)

        return _decorator

    # Case 3: @register_estimator(name="my_net")
    if name_or_target is None and target is None:
        def _kw_decorator(obj: Any) -> Any:
            resolved = name or getattr(obj, "__name__", None)
            if not resolved:
                raise ValueError("Provide an explicit `name` when registering anonymous objects.")
            return _do_register(str(resolved), obj)

        return _kw_decorator

    # Case 4: @register_estimator (bare decorator)
    if name_or_target is not None and not isinstance(name_or_target, str):
        obj = name_or_target
        resolved = name or getattr(obj, "__name__", obj.__class__.__name__)
        return _do_register(str(resolved), obj)

    raise TypeError("Invalid arguments passed to `register_estimator`.")


def unregister_estimator(name: str) -> None:
    """Remove a registered estimator by name (raises ``KeyError`` if absent)."""
    key = _normalize_estimator_name(name)
    if key not in _ESTIMATOR_REGISTRY:
        raise KeyError(f"Unknown estimator {name!r}.")
    del _ESTIMATOR_REGISTRY[key]


def get_estimator(name: str | RegisteredEstimator) -> RegisteredEstimator:
    """Look up a :class:`RegisteredEstimator` by registry name or alias."""
    if isinstance(name, RegisteredEstimator):
        return name
    key = _normalize_estimator_name(name)
    if key not in _ESTIMATOR_REGISTRY:
        raise KeyError(
            f"Unknown estimator {name!r}. Registered estimators: {sorted(_ESTIMATOR_REGISTRY.keys())}"
        )
    return _ESTIMATOR_REGISTRY[key]


def list_estimators() -> dict[str, RegisteredEstimator]:
    """Return a shallow copy of all registered estimators."""
    return dict(_ESTIMATOR_REGISTRY)


def run_estimator(
    estimator: str | RegisteredEstimator | Callable[..., Any] | Any,
    signals: np.ndarray,
    bvalues: Sequence[float] | np.ndarray | str,
    *,
    alpha: float = 0.10,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    **kwargs: Any,
) -> NormalizedEstimatorOutput:
    """Execute any registered name, callable, class, or baseline object on ``(signals, bvalues)``."""
    sig_2d = np.atleast_2d(np.asarray(signals, dtype=np.float64))
    b_arr, _ = resolve_bvalues(bvalues)
    if sig_2d.shape[1] != b_arr.size:
        raise ValueError(
            f"Signals shape {sig_2d.shape} does not match b-values length {b_arr.size}."
        )
    n = sig_2d.shape[0]

    target = get_estimator(estimator).target if isinstance(estimator, (str, RegisteredEstimator)) else estimator

    if inspect.isclass(target):
        instance = _instantiate_estimator_class(target, b_arr, kwargs)
    else:
        instance = target

    # 1. Native ivimbench estimator objects exposing predict_quantiles / predict_intervals
    if not inspect.isfunction(instance) and (
        hasattr(instance, "predict_quantiles") or hasattr(instance, "predict_intervals")
    ):
        # Re-bind protocol if the instance was created for a different b-value length
        if hasattr(instance, "bvalues") and np.asarray(instance.bvalues).size != b_arr.size:
            instance = _instantiate_estimator_class(instance.__class__, b_arr, kwargs)

        s0_hat: np.ndarray | None = None
        railed: np.ndarray | None = None
        std_hat: np.ndarray | None = None

        if hasattr(instance, "solve"):
            solve_kw: dict[str, Any] = {}
            if "sd_convention" in kwargs:
                try:
                    if "sd_convention" in inspect.signature(instance.solve).parameters:
                        solve_kw["sd_convention"] = kwargs["sd_convention"]
                except (TypeError, ValueError):
                    pass
            solved = instance.solve(sig_2d, **solve_kw)
            pt = np.asarray(solved.params, dtype=np.float64)
            s0_hat = getattr(solved, "s0", None)
            railed = getattr(solved, "railed", None)
            if hasattr(solved, "sigma"):
                std_hat = np.asarray(solved.sigma, dtype=np.float64)
            elif hasattr(solved, "sigma_crlb"):
                std_hat = np.asarray(solved.sigma_crlb, dtype=np.float64)
            z = float(stats.norm.ppf(1.0 - 0.5 * float(alpha)))
            if std_hat is not None:
                lo, hi = pt - z * std_hat, pt + z * std_hat
                levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)
                q_mat = pt[:, :, None] + std_hat[:, :, None] * stats.norm.ppf(levels)[None, None, :]
                return normalize_estimator_output(
                    {
                        "point": pt,
                        "lower": lo,
                        "upper": hi,
                        "std": std_hat,
                        "quantiles": q_mat,
                        "q_levels": levels,
                        "s0": s0_hat,
                        "railed": railed,
                    },
                    n_voxels=n,
                    alpha=alpha,
                    bounds=bounds,
                )

        if hasattr(instance, "predict_quantiles"):
            levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)
            q_mat = np.asarray(instance.predict_quantiles(sig_2d, levels), dtype=np.float64)
            pt = (
                np.asarray(instance.predict(sig_2d), dtype=np.float64)
                if hasattr(instance, "predict")
                else q_mat[:, :, int(np.argmin(np.abs(levels - 0.5)))]
            )
            lo, hi = central_interval(q_mat, levels, alpha=alpha)
            return normalize_estimator_output(
                {"point": pt, "lower": lo, "upper": hi, "quantiles": q_mat, "q_levels": levels},
                n_voxels=n,
                alpha=alpha,
                bounds=bounds,
            )

        lo, hi = instance.predict_intervals(sig_2d, alpha=alpha)
        pt = (
            np.asarray(instance.predict(sig_2d), dtype=np.float64)
            if hasattr(instance, "predict")
            else 0.5 * (np.asarray(lo, dtype=np.float64) + np.asarray(hi, dtype=np.float64))
        )
        return normalize_estimator_output(
            {"point": pt, "lower": lo, "upper": hi},
            n_voxels=n,
            alpha=alpha,
            bounds=bounds,
        )

    # 2. Custom class with fit_predict(signals, bvalues, ...)
    if hasattr(instance, "fit_predict") and callable(instance.fit_predict):
        raw = _call_with_supported_kwargs(
            instance.fit_predict, sig_2d, b_arr, alpha=alpha, extra_kwargs=kwargs
        )
        return normalize_estimator_output(raw, n_voxels=n, alpha=alpha, bounds=bounds)

    # 3. Direct callable fn(signals, bvalues, ...) -> {D, D_star, f, intervals?}
    if callable(instance):
        raw = _call_with_supported_kwargs(instance, sig_2d, b_arr, alpha=alpha, extra_kwargs=kwargs)
        return normalize_estimator_output(raw, n_voxels=n, alpha=alpha, bounds=bounds)

    raise TypeError(
        f"Estimator {estimator!r} must be a registered name, callable `(signals, bvalues) -> dict`, "
        f"or object with `predict_intervals`/`predict_quantiles`/`fit_predict`."
    )


# --------------------------------------------------------------------------- #
# Built-in deterministic pure NumPy/SciPy baseline estimators
# --------------------------------------------------------------------------- #
@register_estimator(
    "segmented",
    description="Deterministic 2-step segmented NLLS baseline (high-b OLS + damped Gauss-Newton D* + asymptotic Jacobian intervals).",
    tags=("baseline", "deterministic", "nlls"),
)
def _builtin_segmented_estimator(
    signals: np.ndarray,
    bvalues: np.ndarray,
    *,
    alpha: float = 0.10,
    b_thresh: float = 200.0,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    est = SegmentedEstimator(
        protocol=bvalues,
        b_thresh=b_thresh,
        bounds=bounds,
        uncertainty="asymptotic",
    )
    solved = est.solve(signals)
    z = float(stats.norm.ppf(1.0 - 0.5 * float(alpha)))
    lo = solved.params - z * solved.sigma
    hi = solved.params + z * solved.sigma
    return {
        "D": solved.params[:, 0],
        "D_star": solved.params[:, 1],
        "f": solved.params[:, 2],
        "intervals": (lo, hi),
        "std": solved.sigma,
        "s0": solved.s0,
    }


@register_estimator(
    "nlls",
    description="Bounded simultaneous 4-parameter TRF NLLS baseline with boundary-railing diagnostics and honest asymptotic intervals.",
    tags=("baseline", "deterministic", "nlls"),
)
def _builtin_nlls_estimator(
    signals: np.ndarray,
    bvalues: np.ndarray,
    *,
    alpha: float = 0.10,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    sd_convention: SDConvention | str = "honest_crlb",
    max_nfev: int = 250,
) -> dict[str, Any]:
    est = NLLSEstimator(
        protocol=bvalues,
        bounds=bounds,
        sd_convention=sd_convention,
        max_nfev=max_nfev,
    )
    solved = est.solve(signals)
    z = float(stats.norm.ppf(1.0 - 0.5 * float(alpha)))
    lo = solved.params - z * solved.sigma
    hi = solved.params + z * solved.sigma
    return {
        "D": solved.params[:, 0],
        "D_star": solved.params[:, 1],
        "f": solved.params[:, 2],
        "intervals": (lo, hi),
        "std": solved.sigma,
        "s0": solved.s0,
        "railed": solved.railed,
        "success": solved.success,
    }


@register_estimator(
    "crlb",
    description="Plug-in analytical Fisher-information CRLB normal-approximation baseline estimator.",
    tags=("baseline", "deterministic", "crlb"),
)
def _builtin_crlb_estimator(
    signals: np.ndarray,
    bvalues: np.ndarray,
    *,
    alpha: float = 0.10,
    snr: float | None = None,
    nex: Sequence[float] | np.ndarray | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    sd_convention: SDConvention | str = "honest_crlb",
) -> dict[str, Any]:
    est = CRLBNormalEstimator(
        protocol=bvalues,
        bounds=bounds,
        snr=snr,
        nex=nex,
        sd_convention=sd_convention,
    )
    solved = est.solve(signals)
    z = float(stats.norm.ppf(1.0 - 0.5 * float(alpha)))
    lo = solved.params - z * solved.sigma_crlb
    hi = solved.params + z * solved.sigma_crlb
    return {
        "D": solved.params[:, 0],
        "D_star": solved.params[:, 1],
        "f": solved.params[:, 2],
        "intervals": (lo, hi),
        "std": solved.sigma_crlb,
        "s0": solved.s0,
        "railed": solved.railed,
        "snr_hat": solved.snr_hat,
    }


# Trigger conditional @register_estimator("maf") when optional ivimbench[torch] is installed
from .baselines import maf as _maf  # noqa: E402, F401
