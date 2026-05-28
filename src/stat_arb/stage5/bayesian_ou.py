r"""Bayesian Ornstein-Uhlenbeck estimation (conjugate, exact posterior).

The exact OU discretisation is a Gaussian AR(1),
:math:`x_{t+1} = a + b\,x_t + \varepsilon_t,\ \varepsilon_t\sim\mathcal N(0,v)`.
Place a **Normal-Inverse-Gamma** prior on :math:`(\beta=(a,b),\, v)`:

.. math::

   \beta \mid v \sim \mathcal N(m_0, v V_0), \qquad v \sim \mathrm{IG}(a_0, b_0).

This is conjugate, so the posterior is Normal-Inverse-Gamma in closed form —
no MCMC approximation is required. We draw posterior samples of
:math:`(a,b,v)` and map each to :math:`(\kappa,\mu,\sigma)` via the same
transform as the Stage 1 MLE. The spread of the :math:`\kappa` posterior is
the quantity Stage 5 trades on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils import infer_dt


@dataclass
class BayesianOUPosterior:
    """Normal-Inverse-Gamma posterior over the OU AR(1) parameters.

    ``mn, Vn`` parameterise ``beta | v ~ N(mn, v Vn)``; ``an, bn`` give
    ``v ~ InvGamma(an, bn)``. Use :meth:`sample` to obtain draws of
    :math:`(\\kappa,\\mu,\\sigma)`.
    """

    mn: np.ndarray       # (2,) posterior mean of (a, b)
    Vn: np.ndarray       # (2, 2)
    an: float
    bn: float
    dt: float
    n_obs: int

    def sample(self, n_draws: int = 5000, seed: int | None = None) -> dict:
        """Draw posterior samples and map to ``(kappa, mu, sigma)``.

        Draws with :math:`b \\notin (0,1)` are non-stationary; we keep them
        but expose ``p_stationary`` so callers can act on reversion
        uncertainty. ``kappa/mu/sigma`` arrays contain only the stationary
        draws (the economically meaningful ones).
        """
        rng = np.random.default_rng(seed)
        v = 1.0 / rng.gamma(shape=self.an, scale=1.0 / self.bn, size=n_draws)
        L = np.linalg.cholesky(self.Vn)
        z = rng.standard_normal((n_draws, 2))
        beta = self.mn[None, :] + np.sqrt(v)[:, None] * (z @ L.T)
        a, b = beta[:, 0], beta[:, 1]

        stationary = (b > 0) & (b < 1)
        p_stationary = float(stationary.mean())

        bs, as_, vs = b[stationary], a[stationary], v[stationary]
        kappa = -np.log(bs) / self.dt
        mu = as_ / (1.0 - bs)
        sigma = np.sqrt(np.clip(vs * 2.0 * kappa / (1.0 - bs**2), 0, None))

        return {
            "kappa": kappa, "mu": mu, "sigma": sigma,
            "p_stationary": p_stationary, "n_valid": int(stationary.sum()),
        }

    def summary(self, n_draws: int = 5000, seed: int = 0) -> dict:
        """Posterior means/stds plus a diffuseness measure for ``kappa``."""
        s = self.sample(n_draws=n_draws, seed=seed)
        k = s["kappa"]
        if k.size == 0:
            return {"kappa_mean": float("nan"), "kappa_cv": float("inf"),
                    "p_stationary": s["p_stationary"]}
        return {
            "kappa_mean": float(k.mean()),
            "kappa_std": float(k.std()),
            "kappa_cv": float(k.std() / k.mean()) if k.mean() > 0 else float("inf"),
            "mu_mean": float(s["mu"].mean()),
            "sigma_mean": float(s["sigma"].mean()),
            "p_stationary": s["p_stationary"],
        }


class BayesianOU:
    """Conjugate Bayesian OU estimator.

    Parameters
    ----------
    prior_precision:
        Scales the prior precision :math:`V_0^{-1} = \\text{prior\\_precision}\\cdot I`.
        Small values (default ``1e-3``) give a weak prior so the posterior is
        data-dominated (close to the MLE) while still being proper.
    a0, b0:
        Inverse-Gamma prior shape/scale for the innovation variance.
    """

    def __init__(self, prior_precision: float = 1e-3, a0: float = 1e-3, b0: float = 1e-3) -> None:
        self.prior_precision = float(prior_precision)
        self.a0 = float(a0)
        self.b0 = float(b0)

    def fit(self, series: pd.Series | np.ndarray, dt: float | None = None) -> BayesianOUPosterior:
        if isinstance(series, pd.Series):
            if dt is None:
                dt = infer_dt(series.index)
            x = series.dropna().to_numpy(float)
        else:
            x = np.asarray(series, float)
            x = x[np.isfinite(x)]
            if dt is None:
                raise ValueError("Must supply dt when passing a plain array.")
        if x.size < 30:
            raise ValueError("Need >= 30 observations for Bayesian OU.")

        x_prev, x_next = x[:-1], x[1:]
        n = x_prev.size
        X = np.column_stack([np.ones(n), x_prev])
        y = x_next

        V0_inv = self.prior_precision * np.eye(2)
        m0 = np.array([0.0, 0.9])  # weak prior centred on a persistent AR(1)

        Vn = np.linalg.inv(V0_inv + X.T @ X)
        mn = Vn @ (V0_inv @ m0 + X.T @ y)
        an = self.a0 + n / 2.0
        bn = self.b0 + 0.5 * float(
            y @ y + m0 @ V0_inv @ m0 - mn @ np.linalg.inv(Vn) @ mn
        )
        bn = max(bn, 1e-12)

        return BayesianOUPosterior(mn=mn, Vn=Vn, an=an, bn=bn, dt=float(dt), n_obs=int(n + 1))
