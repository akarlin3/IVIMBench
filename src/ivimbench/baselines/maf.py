"""Conditional Masked Autoregressive Flow (MAF) posterior IVIM baseline estimator.

Ported and consolidated from Caliper (``caliper.estimator_maf``) into
``ivimbench.baselines``:
- Models the amortized full joint posterior ``p(theta | signal)`` over canonical
  IVIM parameters ``theta = (D, Dstar, f)`` conditioned on multi-``b`` DWI signal
  decays using a stack of conditional MADE affine autoregressive layers
  (Papamakarios et al., NeurIPS 2017).
- Requires only the optional ``ivimbench[torch]`` extra; core ``IVIMBench``
  remains 100% PyTorch-free. When ``torch`` is absent, importing this module
  succeeds cleanly and instantiating :class:`MAFPosteriorEstimator` raises
  ``ImportError("Install ivimbench[torch] to use MAFPosteriorEstimator")``.
- When ``torch`` is installed, :class:`MAFPosteriorEstimator` is automatically
  registered in :mod:`ivimbench.registry` under ``"maf"`` and adheres to the
  per-voxel quantile-prediction contract
  ``(signals, bvalues, quantiles) -> {"D", "D_star", "f", "intervals"}``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..metrics.coverage import DEFAULT_QUANTILE_LEVELS, central_interval
from ..phantoms.synthetic import DEFAULT_SEED, PARAM_NAMES, generate_cohort
from ..protocols import AcquisitionProtocol, resolve_bvalues

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fnn

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised in torch-free core environments
    _HAS_TORCH = False

__all__ = ["MAFPosteriorEstimator", "MAFPosterior"]

_MISSING_TORCH_MSG = "Install ivimbench[torch] to use MAFPosteriorEstimator"


def _to_unconstrained_np(params: np.ndarray) -> np.ndarray:
    """Map physical ``(D, Dstar, f)`` to unconstrained ``(log D, log D*, logit f)``."""
    d_val = np.clip(params[:, 0], 1e-6, None)
    ds_val = np.clip(params[:, 1], 1e-6, None)
    f_val = np.clip(params[:, 2], 1e-4, 1.0 - 1e-4)
    return np.stack(
        [np.log(d_val), np.log(ds_val), np.log(f_val / (1.0 - f_val))],
        axis=1,
    )


if _HAS_TORCH:

    def _from_unconstrained_torch(u: Any) -> Any:
        """Map unconstrained ``(log D, log D*, logit f)`` back to physical ``(D, Dstar, f)``."""
        d_val = torch.exp(torch.clamp(u[..., 0], min=-16.0, max=2.0))
        ds_val = torch.exp(torch.clamp(u[..., 1], min=-14.0, max=2.0))
        f_val = torch.sigmoid(u[..., 2])
        return torch.stack([d_val, ds_val, f_val], dim=-1)

    class _MaskedLinear(nn.Linear):
        def __init__(self, in_f: int, out_f: int, mask: np.ndarray) -> None:
            super().__init__(in_f, out_f)
            self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))

        def forward(self, x: Any) -> Any:
            return Fnn.linear(x, self.weight * self.mask, self.bias)  # type: ignore[operator]

    class _ConditionalMADE(nn.Module):
        """Conditional MADE producing ``(mu, log_scale)`` for an affine autoregressive step."""

        def __init__(self, d: int, c: int, hidden: int, n_hidden: int = 1) -> None:
            super().__init__()
            self.d = d
            self.c = c
            m_in = np.concatenate([np.zeros(c, dtype=int), np.arange(1, d + 1)])
            degrees = [m_in]
            for _ in range(n_hidden):
                hd = (np.arange(hidden) % max(d - 1, 1)) + 1
                degrees.append(hd)
            m_out = np.concatenate([np.arange(1, d + 1), np.arange(1, d + 1)])

            layers: list[nn.Module] = []
            prev = degrees[0]
            in_f = c + d
            for k in range(1, len(degrees)):
                cur = degrees[k]
                mask = (cur[:, None] >= prev[None, :]).astype(np.float32)
                layers.append(_MaskedLinear(in_f, hidden, mask))
                layers.append(nn.ReLU())
                prev = cur
                in_f = hidden
            out_mask = (m_out[:, None] > prev[None, :]).astype(np.float32)
            self.out = _MaskedLinear(in_f, 2 * d, out_mask)
            self.net = nn.Sequential(*layers)

        def forward(self, x: Any, context: Any) -> tuple[Any, Any]:
            h = torch.cat([context, x], dim=-1)
            h = self.net(h)
            out = self.out(h)
            mu, log_scale = out[..., : self.d], out[..., self.d :]
            log_scale = torch.tanh(log_scale)
            return mu, log_scale

    class _MAF(nn.Module):
        """Stack of conditional MADE affine layers with deterministic dimension permutations."""

        def __init__(self, d: int, c: int, hidden: int, n_layers: int, n_hidden: int = 1) -> None:
            super().__init__()
            self.d = d
            self.layers = nn.ModuleList(
                [_ConditionalMADE(d, c, hidden, n_hidden) for _ in range(n_layers)]
            )
            base = np.arange(d)
            perms = [np.roll(base, i) if d > 1 else base for i in range(n_layers)]
            self.register_buffer("perms", torch.as_tensor(np.stack(perms), dtype=torch.long))

        def log_prob(self, x: Any, context: Any) -> Any:
            log_det = torch.zeros(x.shape[0], device=x.device)
            z = x
            for i, layer in enumerate(self.layers):
                perm_idx = self.perms[i]  # type: ignore[index]
                z = z[:, perm_idx]
                mu, log_scale = layer(z, context)
                z = (z - mu) * torch.exp(-log_scale)
                log_det = log_det - log_scale.sum(dim=-1)
            base = -0.5 * (z**2 + float(np.log(2.0 * np.pi)))
            return base.sum(dim=-1) + log_det

        def sample(self, context: Any, n_samples: int, generator: Any | None = None) -> Any:
            nctx = int(context.shape[0])
            ctx = context.repeat_interleave(int(n_samples), dim=0)
            z = torch.randn(
                nctx * int(n_samples),
                self.d,
                generator=generator,
                device=context.device,
            )
            for i in reversed(range(len(self.layers))):
                layer = self.layers[i]
                x = torch.zeros_like(z)
                for _ in range(self.d):
                    mu, log_scale = layer(x, ctx)
                    x = z * torch.exp(log_scale) + mu
                perm_idx = self.perms[i]  # type: ignore[index]
                inv = torch.argsort(perm_idx)
                z = x[:, inv]
            return z.reshape(nctx, int(n_samples), self.d)


@dataclass
class MAFPosteriorEstimator:
    """Conditional Masked Autoregressive Flow posterior estimator over ``(D, Dstar, f)``.

    Adheres to both the standard ``IVIMBench`` baseline API (:meth:`fit`,
    :meth:`predict`, :meth:`predict_intervals`, :meth:`predict_quantiles`) and
    the per-voxel callable contract
    ``est(signals, bvalues, quantiles) -> {"D", "D_star", "f", "intervals"}``.
    """

    protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray = "osipi"
    n_bvalues: int | None = None
    hidden: int = 64
    n_layers: int = 5
    n_hidden: int = 1
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 256
    n_posterior: int = 400
    seed: int = DEFAULT_SEED
    param_names: tuple[str, ...] = PARAM_NAMES
    bvalues: np.ndarray = field(init=False)
    is_fitted_: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not _HAS_TORCH:
            raise ImportError(_MISSING_TORCH_MSG)

        if self.n_bvalues is not None:
            self.bvalues = np.linspace(0.0, 800.0, int(self.n_bvalues), dtype=np.float64)
        else:
            self.bvalues, _ = resolve_bvalues(self.protocol)
            self.n_bvalues = int(self.bvalues.size)

        torch.manual_seed(int(self.seed))
        self.flow = _MAF(
            d=3,
            c=int(self.n_bvalues),
            hidden=int(self.hidden),
            n_layers=int(self.n_layers),
            n_hidden=int(self.n_hidden),
        )
        self._ctx_mean: np.ndarray | None = None
        self._ctx_std: np.ndarray | None = None
        self._u_mean: np.ndarray | None = None
        self._u_std: np.ndarray | None = None
        self.history_: list[float] = []

    def _rebind_bvalues(self, bvalues: Sequence[float] | np.ndarray | str) -> None:
        b_arr, _ = resolve_bvalues(bvalues)
        if b_arr.size != int(self.n_bvalues or 0):
            self.bvalues = b_arr
            self.n_bvalues = int(b_arr.size)
            torch.manual_seed(int(self.seed))
            self.flow = _MAF(
                d=3,
                c=self.n_bvalues,
                hidden=int(self.hidden),
                n_layers=int(self.n_layers),
                n_hidden=int(self.n_hidden),
            )
            self.is_fitted_ = False
        else:
            self.bvalues = b_arr

    def _standardize_ctx(self, signals: np.ndarray) -> np.ndarray:
        assert self._ctx_mean is not None and self._ctx_std is not None
        return (signals - self._ctx_mean) / self._ctx_std

    def _fit_scalers(self, signals: np.ndarray, params: np.ndarray) -> np.ndarray:
        self._ctx_mean = signals.mean(axis=0, keepdims=True)
        self._ctx_std = signals.std(axis=0, keepdims=True) + 1e-6
        u = _to_unconstrained_np(params)
        self._u_mean = u.mean(axis=0, keepdims=True)
        self._u_std = u.std(axis=0, keepdims=True) + 1e-6
        return u

    def fit(
        self,
        signals: np.ndarray | None = None,
        params: np.ndarray | None = None,
        *,
        organ: str = "liver",
        snr: float = 30.0,
        n_synthetic: int = 2000,
        verbose: bool = False,
    ) -> MAFPosteriorEstimator:
        """Train the conditional MAF by maximum likelihood on ``(signals, params)``.

        When ``signals`` or ``params`` is omitted, synthesizes ``n_synthetic``
        ``PCG64``-seeded training voxels on ``self.bvalues`` via
        :func:`~ivimbench.phantoms.synthetic.generate_cohort`.
        """
        if signals is None or params is None:
            cohort = generate_cohort(
                protocol=self.bvalues,
                organ=organ,
                n_voxels=int(n_synthetic),
                snr=float(snr),
                seed=int(self.seed),
            )
            sig_np = np.asarray(cohort.signals, dtype=np.float32)
            par_np = np.asarray(cohort.params, dtype=np.float64)
        else:
            sig_np = np.atleast_2d(np.asarray(signals, dtype=np.float32))
            par_np = np.atleast_2d(np.asarray(params, dtype=np.float64))

        if sig_np.shape[1] != int(self.n_bvalues or 0):
            raise ValueError(
                f"`signals` second dim ({sig_np.shape[1]}) must equal n_bvalues ({self.n_bvalues})."
            )
        if par_np.shape != (sig_np.shape[0], 3):
            raise ValueError(
                f"`params` must have shape ({sig_np.shape[0]}, 3) ordered (D, Dstar, f); got {par_np.shape}."
            )

        u = self._fit_scalers(sig_np, par_np)
        assert self._u_mean is not None and self._u_std is not None
        ctx = torch.as_tensor(self._standardize_ctx(sig_np), dtype=torch.float32)
        x = torch.as_tensor((u - self._u_mean) / self._u_std, dtype=torch.float32)

        opt = torch.optim.Adam(self.flow.parameters(), lr=float(self.lr))
        n = int(x.shape[0])
        rng_pcg = np.random.Generator(np.random.PCG64(int(self.seed)))
        history: list[float] = []
        self.flow.train()

        for ep in range(int(self.epochs)):
            perm_np = rng_pcg.permutation(n)
            perm = torch.as_tensor(perm_np, dtype=torch.long)
            tot = 0.0
            for s in range(0, n, int(self.batch_size)):
                idx = perm[s : s + int(self.batch_size)]
                opt.zero_grad()
                loss = -self.flow.log_prob(x[idx], ctx[idx]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 5.0)
                opt.step()
                tot += float(loss.detach()) * int(len(idx))
            history.append(tot / max(n, 1))
            if verbose and (ep % 10 == 0 or ep == int(self.epochs) - 1):  # pragma: no cover
                print(f"  epoch {ep:3d}  nll={history[-1]:.4f}")

        if not np.isfinite(history[-1]):
            raise RuntimeError("MAFPosteriorEstimator training diverged (non-finite NLL).")

        self.history_ = history
        self.is_fitted_ = True
        return self

    def posterior_samples(self, signals: np.ndarray, *, seed: int | None = None) -> np.ndarray:
        """Draw ``(n, n_posterior, 3)`` posterior samples ordered ``(D, Dstar, f)``."""
        if not self.is_fitted_:
            self.fit()
        assert self._u_mean is not None and self._u_std is not None
        sig_np = np.atleast_2d(np.asarray(signals, dtype=np.float32))
        if sig_np.shape[1] != int(self.n_bvalues or 0):
            raise ValueError(
                f"Signal b-axis ({sig_np.shape[1]}) != n_bvalues ({self.n_bvalues})."
            )

        sample_seed = int(self.seed) + 10007 if seed is None else int(seed)
        gen = torch.Generator().manual_seed(sample_seed)
        self.flow.eval()
        with torch.no_grad():
            ctx = torch.as_tensor(self._standardize_ctx(sig_np), dtype=torch.float32)
            u_std = self.flow.sample(ctx, int(self.n_posterior), generator=gen)
            u = u_std * torch.as_tensor(self._u_std, dtype=torch.float32) + torch.as_tensor(
                self._u_mean, dtype=torch.float32
            )
            nat = _from_unconstrained_torch(u)
        return np.asarray(nat.cpu().numpy(), dtype=np.float64)

    def predict_quantiles(
        self,
        signals: np.ndarray,
        q_levels: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS,
    ) -> np.ndarray:
        """Predict ``(n, 3, n_levels)`` posterior quantiles ordered ``(D, Dstar, f)``."""
        levels = np.asarray(q_levels, dtype=np.float64)
        if np.any((levels <= 0.0) | (levels >= 1.0)):
            raise ValueError("`q_levels` must lie strictly in (0, 1).")
        samp = self.posterior_samples(signals)
        q = np.quantile(samp, levels, axis=1)
        return np.asarray(np.transpose(q, (1, 2, 0)), dtype=np.float64)

    def predict(self, signals: np.ndarray) -> np.ndarray:
        """Return ``(n, 3)`` posterior median point estimates ordered ``(D, Dstar, f)``."""
        return self.predict_quantiles(signals, [0.50])[:, :, 0]

    predict_point = predict

    def predict_intervals(
        self,
        signals: np.ndarray,
        alpha: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return central ``(1 - alpha)`` posterior intervals ``(lower, upper)``."""
        lo_q = 0.5 * float(alpha)
        hi_q = 1.0 - 0.5 * float(alpha)
        q = self.predict_quantiles(signals, [lo_q, hi_q])
        return q[:, :, 0], q[:, :, 1]

    def __call__(
        self,
        signals: np.ndarray,
        bvalues: Sequence[float] | np.ndarray | str | None = None,
        quantiles: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS,
        *,
        alpha: float = 0.10,
    ) -> dict[str, Any]:
        """Execute per-voxel quantile prediction contract ``(signals, bvalues, quantiles)``."""
        if bvalues is not None:
            self._rebind_bvalues(bvalues)
        levels = np.asarray(quantiles, dtype=np.float64)
        q_mat = self.predict_quantiles(signals, levels)
        mid_idx = int(np.argmin(np.abs(levels - 0.50)))
        pt = q_mat[:, :, mid_idx]
        lo, hi = central_interval(q_mat, levels, alpha=alpha)
        return {
            "D": pt[:, 0],
            "D_star": pt[:, 1],
            "Dstar": pt[:, 1],
            "f": pt[:, 2],
            "intervals": (lo, hi),
            "quantiles": q_mat,
            "q_levels": levels,
        }


MAFPosterior = MAFPosteriorEstimator

if _HAS_TORCH:  # pragma: no cover - registered automatically when ivimbench[torch] is installed
    from ..registry import register_estimator

    register_estimator(
        "maf",
        MAFPosteriorEstimator,
        description="Conditional Masked Autoregressive Flow (MAF) amortized posterior baseline (requires ivimbench[torch]).",
        tags=("baseline", "posterior", "flow", "torch"),
    )
