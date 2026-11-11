"""Self-contained heteroscedastic Multi-Layer Perceptron (MLP) IVIM baseline.

Implements a lightweight, deterministic, NumPy-only neural network estimator
(requiring no PyTorch or scikit-learn dependencies) that maps normalized multi-b
DWI signal curves to bounded ``(D, Dstar, f)`` point estimates and learned
heteroscedastic Gaussian uncertainties ``sigma`` via Gaussian negative
log-likelihood (NLL) training with Adam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

from ..metrics.coverage import DEFAULT_QUANTILE_LEVELS
from ..metrics.railing import DEFAULT_PARAM_BOUNDS
from ..phantoms.synthetic import generate_cohort
from ..protocols import AcquisitionProtocol, resolve_bvalues

__all__ = ["MLPEstimator", "MLPIVIMEstimator"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    xc = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-xc))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


@dataclass
class MLPEstimator:
    """Standard heteroscedastic MLP regressor for IVIM ``(D, Dstar, f)`` and UQ.

    Parameters
    ----------
    protocol : str, AcquisitionProtocol, or sequence of float, default="osipi"
        Acquisition b-value schedule in ``s/mm^2``.
    hidden_dims : tuple of int, default=(64, 64)
        Widths of the two ReLU hidden layers.
    lr : float, default=3e-3
        Adam optimizer learning rate.
    epochs : int, default=60
        Total training epochs (first third MSE warmup, remainder heteroscedastic
        Gaussian NLL).
    batch_size : int, default=128
        Mini-batch size.
    weight_decay : float, default=1e-4
        L2 weight regularization coefficient.
    bounds : mapping of str to (float, float) or None, optional
        Output sigmoid bounding box for ``(D, Dstar, f)``.
    seed : int, default=42
        Deterministic RNG seed for weight initialization and mini-batch shuffling.
    """

    protocol: str | AcquisitionProtocol | Sequence[float] = "osipi"
    hidden_dims: tuple[int, int] = (64, 64)
    lr: float = 3e-3
    epochs: int = 60
    batch_size: int = 128
    weight_decay: float = 1e-4
    bounds: Mapping[str, tuple[float, float]] | None = None
    seed: int = 42
    bvalues: np.ndarray = field(init=False)
    is_fitted_: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.bvalues, self._protocol_name = resolve_bvalues(self.protocol)
        if len(self.hidden_dims) != 2:
            raise ValueError("`hidden_dims` must be a 2-tuple of hidden layer sizes, e.g. (64, 64).")
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
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._weights: dict[str, np.ndarray] = {}

    def _preprocess_signals(self, signals: np.ndarray) -> np.ndarray:
        sig = np.atleast_2d(np.asarray(signals, dtype=np.float64))
        if sig.shape[1] != self.bvalues.size:
            raise ValueError(
                f"Signal b-axis ({sig.shape[1]}) != protocol b-values ({self.bvalues.size})."
            )
        b0_idx = int(np.argmin(self.bvalues))
        s0 = np.maximum(sig[:, b0_idx : b0_idx + 1], 1e-4)
        return np.clip(sig / s0, 0.0, 2.5)

    def _init_params(self, in_dim: int, rng: np.random.Generator) -> None:
        h1, h2 = int(self.hidden_dims[0]), int(self.hidden_dims[1])
        self._weights = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, h1)),
            "b1": np.zeros(h1, dtype=np.float64),
            "W2": rng.normal(0.0, np.sqrt(2.0 / h1), size=(h1, h2)),
            "b2": np.zeros(h2, dtype=np.float64),
            "W3": rng.normal(0.0, np.sqrt(1.0 / h2), size=(h2, 6)),
            "b3": np.array([0.0, -0.5, -0.5, -1.5, -1.5, -1.5], dtype=np.float64),
        }

    def fit(
        self,
        signals: np.ndarray | None = None,
        params: np.ndarray | None = None,
        *,
        organ: str = "liver",
        snr: float = 30.0,
        n_synthetic: int = 4000,
    ) -> "MLPEstimator":
        """Train the MLP on ``(signals, params)`` or self-generated synthetic data.

        If ``signals`` or ``params`` are omitted, generates a synthetic training
        cohort of size ``n_synthetic`` matching ``self.bvalues`` so the baseline
        can be used out-of-the-box without external training boilerplate.
        """
        rng = np.random.default_rng(int(self.seed))
        if signals is None or params is None:
            cohort = generate_cohort(
                protocol=self.bvalues,
                organ=organ,
                n_voxels=int(n_synthetic),
                snr=snr,
                seed=int(self.seed) + 500,
            )
            signals = cohort.signals
            params = cohort.params

        x_norm = self._preprocess_signals(signals)
        y_phys = np.asarray(params, dtype=np.float64)
        if y_phys.shape != (x_norm.shape[0], 3):
            raise ValueError(
                f"`params` must have shape ({x_norm.shape[0]}, 3); got {y_phys.shape}."
            )

        self._x_mean = np.mean(x_norm, axis=0, keepdims=True)
        self._x_std = np.maximum(np.std(x_norm, axis=0, keepdims=True), 1e-4)
        x_std = (x_norm - self._x_mean) / self._x_std

        # Normalized targets in [0, 1] inside physiological box
        y_unit = np.clip((y_phys - self._lo[None, :]) / self._span[None, :], 1e-4, 1.0 - 1e-4)

        n = x_std.shape[0]
        self._init_params(x_std.shape[1], rng)

        m_adam = {k: np.zeros_like(v) for k, v in self._weights.items()}
        v_adam = {k: np.zeros_like(v) for k, v in self._weights.items()}
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        step_t = 0
        warmup_epochs = max(int(self.epochs) // 3, 5)

        for epoch in range(int(self.epochs)):
            perm = rng.permutation(n)
            use_nll = epoch >= warmup_epochs
            for start in range(0, n, int(self.batch_size)):
                idx = perm[start : start + int(self.batch_size)]
                xb = x_std[idx]
                yb = y_unit[idx]
                bsz = float(xb.shape[0])

                # Forward pass
                z1 = xb @ self._weights["W1"] + self._weights["b1"]
                a1 = np.maximum(z1, 0.0)
                z2 = a1 @ self._weights["W2"] + self._weights["b2"]
                a2 = np.maximum(z2, 0.0)
                out = a2 @ self._weights["W3"] + self._weights["b3"]

                raw_mu = out[:, :3]
                raw_s = out[:, 3:]
                mu = _sigmoid(raw_mu)  # (B, 3) in (0, 1)
                dmu_draw = mu * (1.0 - mu)

                sigma = _softplus(raw_s) + 5e-3  # (B, 3) in unit space
                dsigma_draw = _sigmoid(raw_s)

                err = mu - yb
                if not use_nll:
                    d_mu = (2.0 / bsz) * err
                    target_sig = np.maximum(np.abs(err), 0.02)
                    d_sigma = (0.5 / bsz) * (sigma - target_sig)
                else:
                    inv_var = 1.0 / (sigma * sigma)
                    d_mu = (1.0 / bsz) * err * inv_var
                    d_sigma = (1.0 / bsz) * (1.0 / sigma - (err * err) / (sigma**3))

                dout = np.hstack([d_mu * dmu_draw, d_sigma * dsigma_draw])

                # Backward pass
                grads: dict[str, np.ndarray] = {}
                grads["W3"] = a2.T @ dout + self.weight_decay * self._weights["W3"]
                grads["b3"] = np.sum(dout, axis=0)

                da2 = dout @ self._weights["W3"].T
                dz2 = da2 * (z2 > 0.0)
                grads["W2"] = a1.T @ dz2 + self.weight_decay * self._weights["W2"]
                grads["b2"] = np.sum(dz2, axis=0)

                da1 = dz2 @ self._weights["W2"].T
                dz1 = da1 * (z1 > 0.0)
                grads["W1"] = xb.T @ dz1 + self.weight_decay * self._weights["W1"]
                grads["b1"] = np.sum(dz1, axis=0)

                step_t += 1
                bc1 = 1.0 - beta1**step_t
                bc2 = 1.0 - beta2**step_t
                for k in self._weights:
                    g = np.clip(grads[k], -5.0, 5.0)
                    m_adam[k] = beta1 * m_adam[k] + (1.0 - beta1) * g
                    v_adam[k] = beta2 * v_adam[k] + (1.0 - beta2) * (g * g)
                    m_hat = m_adam[k] / bc1
                    v_hat = v_adam[k] / bc2
                    self._weights[k] -= float(self.lr) * m_hat / (np.sqrt(v_hat) + eps_adam)

        self.is_fitted_ = True
        return self

    def _forward_predict(self, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted_ or self._x_mean is None or self._x_std is None:
            self.fit()
        assert self._x_mean is not None and self._x_std is not None
        x_norm = self._preprocess_signals(signals)
        x_std = (x_norm - self._x_mean) / self._x_std

        z1 = x_std @ self._weights["W1"] + self._weights["b1"]
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self._weights["W2"] + self._weights["b2"]
        a2 = np.maximum(z2, 0.0)
        out = a2 @ self._weights["W3"] + self._weights["b3"]

        mu_unit = _sigmoid(out[:, :3])
        sigma_unit = _softplus(out[:, 3:]) + 5e-3

        mean_phys = self._lo[None, :] + self._span[None, :] * mu_unit
        std_phys = self._span[None, :] * sigma_unit
        return mean_phys, std_phys

    def predict(self, signals: np.ndarray) -> np.ndarray:
        """Predict ``(n, 3)`` point estimates ``(D, Dstar, f)`` in physical units."""
        mean_phys, _ = self._forward_predict(signals)
        return mean_phys

    predict_point = predict

    def predict_std(self, signals: np.ndarray) -> np.ndarray:
        """Predict ``(n, 3)`` heteroscedastic standard deviations in physical units."""
        _, std_phys = self._forward_predict(signals)
        return std_phys

    def predict_intervals(
        self,
        signals: np.ndarray,
        alpha: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict central ``(1 - alpha)`` intervals ``(lower, upper)`` of shape ``(n, 3)``."""
        mean_phys, std_phys = self._forward_predict(signals)
        z = float(stats.norm.ppf(1.0 - 0.5 * float(alpha)))
        return mean_phys - z * std_phys, mean_phys + z * std_phys

    def predict_quantiles(
        self,
        signals: np.ndarray,
        q_levels: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS,
    ) -> np.ndarray:
        """Predict ``(n, 3, n_levels)`` heteroscedastic Gaussian quantiles."""
        levels = np.asarray(q_levels, dtype=np.float64)
        if np.any((levels <= 0.0) | (levels >= 1.0)):
            raise ValueError("`q_levels` must lie strictly in (0, 1).")
        mean_phys, std_phys = self._forward_predict(signals)
        z = stats.norm.ppf(levels)
        return mean_phys[:, :, None] + std_phys[:, :, None] * z[None, None, :]


MLPIVIMEstimator = MLPEstimator
