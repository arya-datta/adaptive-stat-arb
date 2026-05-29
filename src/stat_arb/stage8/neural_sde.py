r"""Neural SDE: learn the spread's drift as a neural function (NumPy, no torch).

We keep the SDE structure :math:`dX = f_\theta(X)\,dt + \sigma\,dW` and learn
the drift :math:`f_\theta` as a one-hidden-layer ``tanh`` MLP, fit by Adam to
predict the one-step increment :math:`\Delta X` from :math:`X`. The diffusion
:math:`\sigma` is the residual standard deviation. This generalises the linear
OU drift :math:`\kappa(\mu - X)` — and is held to the same out-of-sample bar.

Implemented in pure NumPy (manual forward/backward + Adam) so the project has
no heavy deep-learning dependency; the point is the *discipline of the gate*,
not framework plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..stage1.ou_mle import OUMLEEstimator


class MLP:
    """Minimal 1-hidden-layer tanh MLP (scalar in, scalar out) with Adam."""

    def __init__(self, hidden: int = 16, lr: float = 0.01, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((1, hidden)) * 0.5
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, 1)) * 0.5
        self.b2 = np.zeros(1)
        self.lr = lr
        self._adam = {k: [np.zeros_like(getattr(self, k)), np.zeros_like(getattr(self, k))]
                      for k in ("W1", "b1", "W2", "b2")}
        self._t = 0

    def forward(self, x: np.ndarray):
        z1 = x[:, None] @ self.W1 + self.b1     # (n, H)
        h = np.tanh(z1)
        out = (h @ self.W2 + self.b2)[:, 0]     # (n,)
        return out, (x, h)

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        n = x.size
        out, (x_in, h) = self.forward(x)
        err = out - y                            # dL/dout for MSE/n
        loss = float(np.mean(err ** 2))

        g_out = (2.0 / n) * err                  # (n,)
        gW2 = h.T @ g_out[:, None]               # (H, 1)
        gb2 = np.array([g_out.sum()])
        gh = g_out[:, None] @ self.W2.T          # (n, H)
        gz1 = gh * (1.0 - h ** 2)                # tanh'
        gW1 = x_in[:, None].T @ gz1              # (1, H)
        gb1 = gz1.sum(axis=0)

        self._apply("W1", gW1); self._apply("b1", gb1)
        self._apply("W2", gW2); self._apply("b2", gb2)
        return loss

    def _apply(self, name: str, grad: np.ndarray, b1=0.9, b2=0.999, eps=1e-8) -> None:
        self._t += 0  # global step incremented once per batch in fit()
        m, v = self._adam[name]
        m[...] = b1 * m + (1 - b1) * grad
        v[...] = b2 * v + (1 - b2) * grad ** 2
        t = max(self._t, 1)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        getattr(self, name)[...] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(np.asarray(x, float))[0]


@dataclass
class NeuralSDE:
    """Fit ``dX = f_theta(X) dt + sigma dW`` with a neural drift.

    Attributes set by :meth:`fit`: ``net`` (the MLP on standardised inputs),
    ``sigma`` (diffusion), and the input/target standardisation constants.
    """

    hidden: int = 16
    lr: float = 0.01
    n_epochs: int = 400
    seed: int = 0

    def fit(self, x: np.ndarray, dt: float) -> "NeuralSDE":
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        x_prev, dx = x[:-1], np.diff(x)

        self._xm, self._xs = x_prev.mean(), x_prev.std() + 1e-12
        self._ym, self._ys = dx.mean(), dx.std() + 1e-12
        xs = (x_prev - self._xm) / self._xs
        ys = (dx - self._ym) / self._ys

        self.net = MLP(hidden=self.hidden, lr=self.lr, seed=self.seed)
        for _ in range(self.n_epochs):
            self.net._t += 1
            self.net.train_step(xs, ys)

        resid = dx - self._predict_dx(x_prev)
        self.sigma = float(np.sqrt(np.var(resid) / dt))
        self._resid_var = float(np.var(resid))
        self.dt = float(dt)
        return self

    def _predict_dx(self, x_prev: np.ndarray) -> np.ndarray:
        xs = (np.asarray(x_prev, float) - self._xm) / self._xs
        return self.net.predict(xs) * self._ys + self._ym

    def drift(self, x: np.ndarray) -> np.ndarray:
        """Estimated drift ``f_theta(x)`` (= predicted increment / dt)."""
        return self._predict_dx(np.asarray(x, float)) / self.dt

    def predictive_loglik(self, x: np.ndarray) -> float:
        """One-step Gaussian predictive log-likelihood on ``x``."""
        x = np.asarray(x, float)
        x_prev, dx = x[:-1], np.diff(x)
        pred = self._predict_dx(x_prev)
        resid = dx - pred
        var = max(self._resid_var, 1e-12)
        return float(np.sum(-0.5 * (np.log(2 * np.pi * var) + resid ** 2 / var)))


def _ou_predictive_loglik(x_train: np.ndarray, x_test: np.ndarray, dt: float) -> float:
    """One-step predictive log-likelihood of the classical linear OU on test data."""
    ou = OUMLEEstimator().fit(x_train, dt=dt)
    b = np.exp(-ou.kappa * dt)
    a = ou.mu * (1 - b)
    xt = np.asarray(x_test, float)
    x_prev, x_next = xt[:-1], xt[1:]
    resid_train = np.diff(x_train) - ((a + b * x_train[:-1]) - x_train[:-1])
    var = max(float(np.var(resid_train)), 1e-12)
    resid = x_next - (a + b * x_prev)
    return float(np.sum(-0.5 * (np.log(2 * np.pi * var) + resid ** 2 / var)))


def neural_vs_ou_gate(
    x_train: np.ndarray,
    x_test: np.ndarray,
    dt: float,
    hidden: int = 16,
    n_epochs: int = 400,
) -> dict:
    """Honest out-of-sample comparison: neural SDE vs classical OU.

    Fits both drifts on ``x_train`` and compares one-step predictive
    log-likelihood on held-out ``x_test``. The neural model is adopted only if
    it beats the linear OU out of sample — the roadmap's gate.
    """
    nsde = NeuralSDE(hidden=hidden, n_epochs=n_epochs).fit(x_train, dt)
    ll_neural = nsde.predictive_loglik(x_test)
    ll_ou = _ou_predictive_loglik(x_train, x_test, dt)
    return {
        "ll_neural": ll_neural,
        "ll_ou": ll_ou,
        "neural_wins": bool(ll_neural > ll_ou),
        "delta": float(ll_neural - ll_ou),
    }
