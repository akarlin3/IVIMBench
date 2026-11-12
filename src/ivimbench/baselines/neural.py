"""Plug-and-play physics-informed and quantile-regression neural IVIM baselines.

Provides two fast, self-contained PyTorch reference adapters gated behind the
optional ``ivimbench[torch]`` extra (keeping core ``ivimbench`` 100% pure
NumPy/SciPy/Pandas):

1. :class:`IVIMNetBaseline` (registered as ``"ivimnet"``):
   Physics-informed self-supervised bi-exponential network minimizing exact Rician
   or Gaussian signal reconstruction loss
   ``L(S_hat(b; S0_hat, D_hat, D*_hat, f_hat), y(b))`` with sigmoid box-constrained
   parameter heads and combined MC-Dropout + heteroscedastic dispersion quantile
   outputs.
2. :class:`HeteroscedasticQuantileMLP` (registered as ``"quantile_mlp"``):
   Direct pinball-loss quantile regression network predicting non-crossing
   ``(q_{alpha/2}, q_{0.5}, q_{1 - alpha/2})`` triplets (and full quantile grids)
   for ``(D, D*, f)``, directly compatible with downstream split and Mondrian
   conformalization (:class:`~ivimbench.metrics.coverage.SplitConformalQuantile`
   and ``conformalIVIM``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import stats

from ..metrics.coverage import (
    DEFAULT_QUANTILE_LEVELS,
    SplitConformalQuantile,
    central_interval,
)
from ..metrics.railing import DEFAULT_PARAM_BOUNDS
from ..phantoms.synthetic import DEFAULT_SEED, PARAM_NAMES, generate_cohort
from ..protocols import AcquisitionProtocol, resolve_bvalues

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fnn

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised in torch-free core environments
    _HAS_TORCH = False

__all__ = [
    "IVIMNetBaseline",
    "HeteroscedasticQuantileMLP",
]

_MISSING_TORCH_IVIMNET_MSG = (
    'Install ivimbench[torch] (e.g. `pip install "ivimbench[torch]"`) to use IVIMNetBaseline.'
)
_MISSING_TORCH_QMLP_MSG = (
    'Install ivimbench[torch] (e.g. `pip install "ivimbench[torch]"`) to use HeteroscedasticQuantileMLP.'
)


def _extract_signal_features_np(signals: np.ndarray, bvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized decay features ``(n, 2 * n_b)`` and observed ``s0_obs`` ``(n, 1)``."""
    sig = np.atleast_2d(np.asarray(signals, dtype=np.float64))
    if sig.shape[1] != bvalues.size:
        raise ValueError(
            f"Signal b-axis ({sig.shape[1]}) does not match protocol length ({bvalues.size})."
        )
    b0_idx = int(np.argmin(bvalues))
    s0_obs = np.maximum(sig[:, b0_idx : b0_idx + 1], 1.0e-4)
    ratio = np.clip(sig / s0_obs, 1.0e-4, 2.5)
    log_ratio = np.log(np.clip(ratio, 1.0e-4, 2.5))
    feats = np.concatenate([ratio, 0.25 * log_ratio], axis=1).astype(np.float32)
    return feats, s0_obs.astype(np.float32)


if _HAS_TORCH:

    class _IVIMNetModule(nn.Module):
        """Self-supervised physics-informed encoder with sigmoid box heads and MC-Dropout."""

        def __init__(self, in_dim: int, hidden: int = 64, dropout_p: float = 0.12) -> None:
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.drop1 = nn.Dropout(p=float(dropout_p))
            self.fc2 = nn.Linear(hidden, hidden)
            self.drop2 = nn.Dropout(p=float(dropout_p))
            self.fc3 = nn.Linear(hidden, hidden)
            # 4 physical parameter logits (S0_scale, D, Dstar, f) + 3 heteroscedastic log-scales
            self.head = nn.Linear(hidden, 7)
            nn.init.constant_(self.head.bias[4:], -2.4)

        def forward(self, x: Any) -> tuple[Any, Any]:
            h = Fnn.elu(self.fc1(x))
            h = self.drop1(h)
            h = Fnn.elu(self.fc2(h))
            h = self.drop2(h)
            h = Fnn.elu(self.fc3(h))
            out = self.head(h)
            param_logits = out[:, :4]
            log_disp = torch.clamp(out[:, 4:], min=-6.0, max=0.5)
            return param_logits, log_disp

    class _QuantileMLPModule(nn.Module):
        """Non-crossing heteroscedastic quantile regression MLP predicting (q_lo, q_med, q_hi)."""

        def __init__(self, in_dim: int, hidden: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            # 3 median logits + 3 lower positive offsets + 3 upper positive offsets
            self.head = nn.Linear(hidden, 9)
            with torch.no_grad():
                self.head.bias[3:].fill_(-1.2)

        def forward(self, x: Any) -> tuple[Any, Any, Any]:
            h = self.net(x)
            out = self.head(h)
            m = out[:, :3]
            d_lo = Fnn.softplus(out[:, 3:6]) + 1.0e-3
            d_hi = Fnn.softplus(out[:, 6:9]) + 1.0e-3
            return m, d_lo, d_hi


@dataclass
class IVIMNetBaseline:
    """Physics-informed self-supervised bi-exponential neural IVIM baseline (``ivimbench[torch]``).

    Minimizes exact Rician negative log-likelihood (or Gaussian MSE) signal
    reconstruction loss ``L(S_hat(b; S0_hat, D_hat, D*_hat, f_hat), y(b))`` with
    sigmoid box-constrained physical heads and combined MC-Dropout + learned
    heteroscedastic dispersion quantile outputs.
    """

    protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray = "osipi"
    hidden: int = 64
    dropout_p: float = 0.12
    loss_type: Literal["rician", "gaussian"] = "rician"
    snr: float = 30.0
    lr: float = 2.5e-3
    epochs: int = 45
    batch_size: int = 128
    n_mc: int = 32
    bounds: Mapping[str, tuple[float, float]] | None = None
    s0_scale_bounds: tuple[float, float] = (0.55, 1.55)
    seed: int = DEFAULT_SEED
    param_names: tuple[str, ...] = PARAM_NAMES
    bvalues: np.ndarray = field(init=False)
    is_fitted_: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not _HAS_TORCH:
            raise ImportError(_MISSING_TORCH_IVIMNET_MSG)

        self.bvalues, _ = resolve_bvalues(self.protocol)
        active = dict(DEFAULT_PARAM_BOUNDS)
        if self.bounds is not None:
            active.update(self.bounds)
        self._lo = np.array(
            [active["D"][0], active["Dstar"][0], active["f"][0]], dtype=np.float32
        )
        self._hi = np.array(
            [active["D"][1], active["Dstar"][1], active["f"][1]], dtype=np.float32
        )
        self._span = self._hi - self._lo

        torch.manual_seed(int(self.seed))
        self.model = _IVIMNetModule(
            in_dim=2 * int(self.bvalues.size),
            hidden=int(self.hidden),
            dropout_p=float(self.dropout_p),
        )
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self.history_: list[float] = []

    def _rebind_bvalues(self, bvalues: Sequence[float] | np.ndarray | str) -> None:
        b_arr, _ = resolve_bvalues(bvalues)
        if b_arr.size != self.bvalues.size or not np.allclose(b_arr, self.bvalues):
            self.bvalues = b_arr
            torch.manual_seed(int(self.seed))
            self.model = _IVIMNetModule(
                in_dim=2 * int(self.bvalues.size),
                hidden=int(self.hidden),
                dropout_p=float(self.dropout_p),
            )
            self.is_fitted_ = False

    def _decode_params_torch(
        self, param_logits: Any, s0_obs_t: Any
    ) -> tuple[Any, Any, Any]:
        s0_lo, s0_hi = float(self.s0_scale_bounds[0]), float(self.s0_scale_bounds[1])
        s0_hat = s0_obs_t * (s0_lo + (s0_hi - s0_lo) * torch.sigmoid(param_logits[:, 0:1]))
        u3 = torch.sigmoid(param_logits[:, 1:4])
        lo_t = torch.as_tensor(self._lo, dtype=torch.float32, device=param_logits.device)
        span_t = torch.as_tensor(self._span, dtype=torch.float32, device=param_logits.device)
        theta_phys = lo_t[None, :] + span_t[None, :] * u3
        return s0_hat, theta_phys, u3

    def _forward_physics_torch(self, s0_hat: Any, theta_phys: Any, b_t: Any) -> Any:
        d_col = theta_phys[:, 0:1]
        ds_col = theta_phys[:, 1:2]
        f_col = theta_phys[:, 2:3]
        return s0_hat * (
            f_col * torch.exp(-b_t[None, :] * ds_col)
            + (1.0 - f_col) * torch.exp(-b_t[None, :] * d_col)
        )

    def fit(
        self,
        signals: np.ndarray | None = None,
        params: np.ndarray | None = None,
        *,
        organ: str = "liver",
        n_synthetic: int = 1200,
    ) -> IVIMNetBaseline:
        """Train ``IVIMNetBaseline`` self-supervised on ``signals`` (``params`` unused/optional)."""
        _ = params
        if signals is None:
            cohort = generate_cohort(
                protocol=self.bvalues,
                organ=organ,
                n_voxels=int(n_synthetic),
                snr=float(self.snr),
                noise_type=self.loss_type if self.loss_type in ("rician", "gaussian") else "rician",
                seed=int(self.seed),
            )
            sig_train = np.asarray(cohort.signals, dtype=np.float32)
        else:
            sig_in = np.atleast_2d(np.asarray(signals, dtype=np.float32))
            if sig_in.shape[0] < 96:
                cohort_aug = generate_cohort(
                    protocol=self.bvalues,
                    organ=organ,
                    n_voxels=max(512 - sig_in.shape[0], 256),
                    snr=float(self.snr),
                    seed=int(self.seed) + 101,
                )
                sig_train = np.vstack([sig_in, np.asarray(cohort_aug.signals, dtype=np.float32)])
            else:
                sig_train = sig_in

        feats, s0_obs = _extract_signal_features_np(sig_train, self.bvalues)
        self._x_mean = feats.mean(axis=0, keepdims=True)
        self._x_std = feats.std(axis=0, keepdims=True) + 1.0e-5
        x_norm = (feats - self._x_mean) / self._x_std

        torch.manual_seed(int(self.seed))
        rng_pcg = np.random.Generator(np.random.PCG64(int(self.seed)))

        x_t = torch.as_tensor(x_norm, dtype=torch.float32)
        y_t = torch.as_tensor(sig_train, dtype=torch.float32)
        s0_t = torch.as_tensor(s0_obs, dtype=torch.float32)
        b_t = torch.as_tensor(self.bvalues, dtype=torch.float32)

        opt = torch.optim.Adam(self.model.parameters(), lr=float(self.lr), weight_decay=1.0e-5)
        n = int(x_t.shape[0])
        loss_mode = self.loss_type.strip().lower()
        if loss_mode not in ("rician", "gaussian"):
            raise ValueError(f"`loss_type` must be 'rician' or 'gaussian'; got {self.loss_type!r}.")

        history: list[float] = []
        self.model.train()
        for _ in range(int(self.epochs)):
            perm = torch.as_tensor(rng_pcg.permutation(n), dtype=torch.long)
            ep_loss = 0.0
            for start in range(0, n, int(self.batch_size)):
                idx = perm[start : start + int(self.batch_size)]
                xb, yb, sb = x_t[idx], y_t[idx], s0_t[idx]
                opt.zero_grad()

                logits, log_disp = self.model(xb)
                s0_hat, theta_phys, _ = self._decode_params_torch(logits, sb)
                s_pred = self._forward_physics_torch(s0_hat, theta_phys, b_t)

                sigma_n = torch.clamp(sb / float(self.snr), min=1.0e-3)
                if loss_mode == "rician":
                    # Overflow-safe exact Rician negative log-likelihood surrogate:
                    # (y - S)^2 / (2 sigma^2) - log(i0e(y * S / sigma^2))
                    y_pos = torch.clamp(yb, min=1.0e-6)
                    s_pos = torch.clamp(s_pred, min=1.0e-6)
                    var_n = sigma_n**2
                    z_bessel = (y_pos * s_pos) / var_n
                    i0e_val = torch.clamp(torch.special.i0e(z_bessel), min=1.0e-12)
                    recon_per_vox = (
                        0.5 * ((y_pos - s_pos) ** 2) / var_n - torch.log(i0e_val)
                    ).mean(dim=1, keepdim=True)
                else:
                    recon_per_vox = (((yb - s_pred) ** 2) / (sigma_n**2)).mean(dim=1, keepdim=True)

                # Calibrate heteroscedastic dispersion head to per-voxel residual fit error
                rms_rel = torch.sqrt(torch.mean((yb - s_pred).detach() ** 2, dim=1, keepdim=True)) / sb
                # Higher relative dispersion on D* and f than slow D
                param_mult = torch.tensor([[0.9, 2.2, 1.3]], dtype=torch.float32)
                target_disp = torch.clamp(rms_rel * param_mult, min=0.02, max=0.45)
                disp_pred = torch.exp(log_disp)
                disp_loss = torch.mean((disp_pred - target_disp) ** 2)

                # Soft ordering penalty encouraging D* > 3 * D
                order_pen = torch.mean(Fnn.relu(3.0 * theta_phys[:, 0] - theta_phys[:, 1])) * 10.0

                loss = recon_per_vox.mean() + 0.35 * disp_loss + order_pen
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                ep_loss += float(loss.detach()) * int(len(idx))
            history.append(ep_loss / max(n, 1))

        self.history_ = history
        self.is_fitted_ = True
        return self

    def _mc_and_hetero_samples(self, signals: np.ndarray) -> np.ndarray:
        """Draw ``(n, n_mc, 3)`` deterministic posterior/predictive parameter samples."""
        if not self.is_fitted_ or self._x_mean is None or self._x_std is None:
            self.fit(signals)
        assert self._x_mean is not None and self._x_std is not None

        feats, s0_obs = _extract_signal_features_np(signals, self.bvalues)
        x_norm = (feats - self._x_mean) / self._x_std
        x_t = torch.as_tensor(x_norm, dtype=torch.float32)
        s0_t = torch.as_tensor(s0_obs, dtype=torch.float32)

        # Deterministic seed across both global RNG (for nn.Dropout) and explicit noise draws
        torch.manual_seed(int(self.seed) + 7919)
        gen = torch.Generator().manual_seed(int(self.seed) + 7919)

        n_vox = int(x_t.shape[0])
        m_draws = max(int(self.n_mc), 8)
        x_rep = x_t.repeat_interleave(m_draws, dim=0)
        s0_rep = s0_t.repeat_interleave(m_draws, dim=0)

        self.model.train()  # keep MC-Dropout active during stochastic passes
        with torch.no_grad():
            logits, log_disp = self.model(x_rep)
            _, _, u3 = self._decode_params_torch(logits, s0_rep)
            disp = torch.exp(log_disp)
            eps = torch.randn(u3.shape, generator=gen)
            # Perturb in logit space so physical box support [lo, hi] is strictly preserved
            u3_clamped = torch.clamp(u3, 1.0e-4, 1.0 - 1.0e-4)
            logit_u3 = torch.log(u3_clamped / (1.0 - u3_clamped))
            u3_samp = torch.sigmoid(logit_u3 + disp * eps)
            lo_t = torch.as_tensor(self._lo, dtype=torch.float32)
            span_t = torch.as_tensor(self._span, dtype=torch.float32)
            phys_samp = lo_t[None, :] + span_t[None, :] * u3_samp

        self.model.eval()
        return np.asarray(phys_samp.cpu().numpy(), dtype=np.float64).reshape(n_vox, m_draws, 3)

    def predict_quantiles(
        self,
        signals: np.ndarray,
        q_levels: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS,
    ) -> np.ndarray:
        """Predict ``(n, 3, n_levels)`` quantiles ordered ``(D, Dstar, f)``."""
        levels = np.asarray(q_levels, dtype=np.float64)
        if np.any((levels <= 0.0) | (levels >= 1.0)):
            raise ValueError("`q_levels` must lie strictly in (0, 1).")
        samp = self._mc_and_hetero_samples(signals)
        q = np.quantile(samp, levels, axis=1)
        return np.asarray(np.transpose(q, (1, 2, 0)), dtype=np.float64)

    def predict(self, signals: np.ndarray) -> np.ndarray:
        """Return ``(n, 3)`` median point predictions ordered ``(D, Dstar, f)``."""
        return self.predict_quantiles(signals, [0.50])[:, :, 0]

    predict_point = predict

    def predict_intervals(
        self,
        signals: np.ndarray,
        alpha: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return central ``(1 - alpha)`` predictive intervals ``(lower, upper)``."""
        q = self.predict_quantiles(signals, [0.5 * float(alpha), 1.0 - 0.5 * float(alpha)])
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


@dataclass
class HeteroscedasticQuantileMLP:
    """Direct pinball-loss quantile regression MLP for ``(D, D*, f)`` (``ivimbench[torch]``).

    Directly predicts non-crossing triplet quantiles
    ``(q_{alpha/2}, q_{0.5}, q_{1 - alpha/2})`` via check/pinball loss and supports
    arbitrary quantile grids plus downstream split conformalization
    (:meth:`conformalize`) compatible with ``conformalIVIM``.
    """

    protocol: str | AcquisitionProtocol | Sequence[float] | np.ndarray = "osipi"
    alpha: float = 0.10
    hidden: int = 64
    lr: float = 3.0e-3
    epochs: int = 55
    batch_size: int = 128
    bounds: Mapping[str, tuple[float, float]] | None = None
    seed: int = DEFAULT_SEED
    param_names: tuple[str, ...] = PARAM_NAMES
    bvalues: np.ndarray = field(init=False)
    is_fitted_: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not _HAS_TORCH:
            raise ImportError(_MISSING_TORCH_QMLP_MSG)
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError(f"`alpha` must lie strictly in (0, 1); got {self.alpha}.")

        self.bvalues, _ = resolve_bvalues(self.protocol)
        active = dict(DEFAULT_PARAM_BOUNDS)
        if self.bounds is not None:
            active.update(self.bounds)
        self._lo = np.array(
            [active["D"][0], active["Dstar"][0], active["f"][0]], dtype=np.float64
        )
        self._hi = np.array(
            [active["D"][1], active["Dstar"][1], active["f"][1]], dtype=np.float64
        )
        self._span = self._hi - self._lo

        torch.manual_seed(int(self.seed))
        self.model = _QuantileMLPModule(
            in_dim=2 * int(self.bvalues.size),
            hidden=int(self.hidden),
        )
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self.history_: list[float] = []

    def _rebind_bvalues(self, bvalues: Sequence[float] | np.ndarray | str) -> None:
        b_arr, _ = resolve_bvalues(bvalues)
        if b_arr.size != self.bvalues.size or not np.allclose(b_arr, self.bvalues):
            self.bvalues = b_arr
            torch.manual_seed(int(self.seed))
            self.model = _QuantileMLPModule(
                in_dim=2 * int(self.bvalues.size),
                hidden=int(self.hidden),
            )
            self.is_fitted_ = False

    def fit(
        self,
        signals: np.ndarray | None = None,
        params: np.ndarray | None = None,
        *,
        organ: str = "liver",
        snr: float = 30.0,
        n_synthetic: int = 2400,
    ) -> HeteroscedasticQuantileMLP:
        """Train via multi-quantile pinball loss at ``(alpha/2, 0.5, 1 - alpha/2)``."""
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

        if par_np.shape != (sig_np.shape[0], 3):
            raise ValueError(
                f"`params` must have shape ({sig_np.shape[0]}, 3) ordered (D, Dstar, f); got {par_np.shape}."
            )

        feats, _ = _extract_signal_features_np(sig_np, self.bvalues)
        self._x_mean = feats.mean(axis=0, keepdims=True)
        self._x_std = feats.std(axis=0, keepdims=True) + 1.0e-5
        x_norm = (feats - self._x_mean) / self._x_std

        y_unit = np.clip((par_np - self._lo[None, :]) / self._span[None, :], 1.0e-4, 1.0 - 1.0e-4)

        torch.manual_seed(int(self.seed))
        rng_pcg = np.random.Generator(np.random.PCG64(int(self.seed)))

        x_t = torch.as_tensor(x_norm, dtype=torch.float32)
        y_t = torch.as_tensor(y_unit, dtype=torch.float32)

        tau_lo = 0.5 * float(self.alpha)
        tau_mid = 0.50
        tau_hi = 1.0 - 0.5 * float(self.alpha)

        def _pinball_torch(target: Any, pred: Any, tau: float) -> Any:
            diff = target - pred
            return torch.maximum(tau * diff, (tau - 1.0) * diff).mean()

        opt = torch.optim.Adam(self.model.parameters(), lr=float(self.lr), weight_decay=1.0e-5)
        n = int(x_t.shape[0])
        history: list[float] = []
        self.model.train()

        for _ in range(int(self.epochs)):
            perm = torch.as_tensor(rng_pcg.permutation(n), dtype=torch.long)
            ep_loss = 0.0
            for start in range(0, n, int(self.batch_size)):
                idx = perm[start : start + int(self.batch_size)]
                xb, yb = x_t[idx], y_t[idx]
                opt.zero_grad()

                m, d_lo, d_hi = self.model(xb)
                q_lo_u = torch.sigmoid(m - d_lo)
                q_mid_u = torch.sigmoid(m)
                q_hi_u = torch.sigmoid(m + d_hi)

                loss = (
                    _pinball_torch(yb, q_lo_u, tau_lo)
                    + _pinball_torch(yb, q_mid_u, tau_mid)
                    + _pinball_torch(yb, q_hi_u, tau_hi)
                ) / 3.0
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                ep_loss += float(loss.detach()) * int(len(idx))
            history.append(ep_loss / max(n, 1))

        self.history_ = history
        self.is_fitted_ = True
        return self

    def _forward_logit_heads(self, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_fitted_ or self._x_mean is None or self._x_std is None:
            self.fit()
        assert self._x_mean is not None and self._x_std is not None
        feats, _ = _extract_signal_features_np(signals, self.bvalues)
        x_norm = (feats - self._x_mean) / self._x_std
        self.model.eval()
        with torch.no_grad():
            m, d_lo, d_hi = self.model(torch.as_tensor(x_norm, dtype=torch.float32))
        return (
            np.asarray(m.cpu().numpy(), dtype=np.float64),
            np.asarray(d_lo.cpu().numpy(), dtype=np.float64),
            np.asarray(d_hi.cpu().numpy(), dtype=np.float64),
        )

    def _logit_to_phys(self, logit_arr: np.ndarray) -> np.ndarray:
        sig = 1.0 / (1.0 + np.exp(-np.clip(logit_arr, -40.0, 40.0)))
        return self._lo[None, ...] + self._span[None, ...] * sig

    def predict_triplet(self, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict exact trained ``(q_{alpha/2}, q_{0.5}, q_{1 - alpha/2})`` arrays of shape ``(n, 3)``."""
        m, d_lo, d_hi = self._forward_logit_heads(signals)
        q_lo = self._logit_to_phys(m - d_lo)
        q_med = self._logit_to_phys(m)
        q_hi = self._logit_to_phys(m + d_hi)
        return q_lo, q_med, q_hi

    def predict_quantiles(
        self,
        signals: np.ndarray,
        q_levels: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS,
    ) -> np.ndarray:
        """Predict strictly monotonic ``(n, 3, n_levels)`` quantiles matching the trained triplet."""
        levels = np.asarray(q_levels, dtype=np.float64)
        if np.any((levels <= 0.0) | (levels >= 1.0)):
            raise ValueError("`q_levels` must lie strictly in (0, 1).")

        m, d_lo, d_hi = self._forward_logit_heads(signals)
        z_ref = max(float(stats.norm.ppf(1.0 - 0.5 * float(self.alpha))), 1.0e-6)
        z_grid = stats.norm.ppf(levels) / z_ref  # -1 at alpha/2, 0 at 0.5, +1 at 1 - alpha/2

        spread = np.where(z_grid[None, None, :] < 0.0, d_lo[:, :, None], d_hi[:, :, None])
        logits_q = m[:, :, None] + spread * z_grid[None, None, :]
        sig_q = 1.0 / (1.0 + np.exp(-np.clip(logits_q, -40.0, 40.0)))
        return self._lo[None, :, None] + self._span[None, :, None] * sig_q

    def predict(self, signals: np.ndarray) -> np.ndarray:
        """Return ``(n, 3)`` median ``q_{0.5}`` predictions ordered ``(D, Dstar, f)``."""
        _, q_med, _ = self.predict_triplet(signals)
        return q_med

    predict_point = predict

    def predict_intervals(
        self,
        signals: np.ndarray,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return central ``(1 - alpha)`` quantile intervals ``(lower, upper)``."""
        target_alpha = float(self.alpha) if alpha is None else float(alpha)
        if np.isclose(target_alpha, float(self.alpha)):
            q_lo, _, q_hi = self.predict_triplet(signals)
            return q_lo, q_hi
        q = self.predict_quantiles(signals, [0.5 * target_alpha, 1.0 - 0.5 * target_alpha])
        return q[:, :, 0], q[:, :, 1]

    def conformalize(
        self,
        cal_signals: np.ndarray,
        cal_params: np.ndarray,
        test_signals: np.ndarray,
        *,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Romano-Patterson-Candes ``SplitConformalQuantile`` directly to predicted quantiles."""
        target_alpha = float(self.alpha) if alpha is None else float(alpha)
        levels = (0.5 * target_alpha, 0.50, 1.0 - 0.5 * target_alpha)
        q_cal = self.predict_quantiles(cal_signals, levels)
        q_test = self.predict_quantiles(test_signals, levels)
        q_conf = SplitConformalQuantile(q_levels=levels).calibrate_apply(
            q_cal, cal_params, q_test
        )
        return q_conf[:, :, 0], q_conf[:, :, 2]

    def __call__(
        self,
        signals: np.ndarray,
        bvalues: Sequence[float] | np.ndarray | str | None = None,
        quantiles: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS,
        *,
        alpha: float | None = None,
    ) -> dict[str, Any]:
        """Execute per-voxel quantile prediction contract compatible with ``conformalIVIM``."""
        if bvalues is not None:
            self._rebind_bvalues(bvalues)
        eff_alpha = float(self.alpha) if alpha is None else float(alpha)
        levels = np.asarray(quantiles, dtype=np.float64)
        q_mat = self.predict_quantiles(signals, levels)
        q_lo, q_med, q_hi = self.predict_triplet(signals)
        lo, hi = (
            (q_lo, q_hi)
            if np.isclose(eff_alpha, float(self.alpha))
            else central_interval(q_mat, levels, alpha=eff_alpha)
        )
        return {
            "D": q_med[:, 0],
            "D_star": q_med[:, 1],
            "Dstar": q_med[:, 1],
            "f": q_med[:, 2],
            "intervals": (lo, hi),
            "quantiles": q_mat,
            "q_levels": levels,
            "q_triplet": (q_lo, q_med, q_hi),
        }


from ..registry import register_estimator  # noqa: E402

register_estimator(
    "ivimnet",
    IVIMNetBaseline,
    description="Physics-informed self-supervised bi-exponential network with Rician/Gaussian loss and MC-Dropout/heteroscedastic UQ (requires ivimbench[torch]).",
    tags=("baseline", "neural", "pinn", "self_supervised", "torch"),
)

register_estimator(
    "quantile_mlp",
    HeteroscedasticQuantileMLP,
    description="Direct pinball-loss heteroscedastic quantile regression MLP predicting (q_{alpha/2}, q_{0.5}, q_{1-alpha/2}) for conformalIVIM (requires ivimbench[torch]).",
    tags=("baseline", "neural", "quantile", "conformal", "torch"),
)
