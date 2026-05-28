"""Synthetic Ornstein-Uhlenbeck and cointegrated-pair generators.

Used heavily by the test suite: every estimator is verified to recover
known parameters from a process it was designed for. The simulation uses
the *exact* OU transition density rather than Euler discretisation, so
recovered ``kappa`` is unbiased even for coarse ``dt``.

These classes are stateful (they cache the last simulated path so they
can act as a :class:`DataSource`), so they are deliberately *not* frozen
dataclasses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DataSource


class SyntheticOU(DataSource):
    r"""Simulate an OU process ``dX_t = kappa*(mu - X_t)dt + sigma*dW_t``.

    The exact discrete-time transition is Gaussian:

    .. math::

       X_{t+\Delta t} \mid X_t \sim \mathcal{N}\!\left(
         \mu + (X_t - \mu) e^{-\kappa \Delta t},\;
         \tfrac{\sigma^{2}}{2\kappa}\bigl(1 - e^{-2\kappa \Delta t}\bigr)
       \right).

    Sampling from this directly (rather than Euler) means the recovered
    MLE has no discretisation bias regardless of ``dt`` — critical for
    Stage 1's "MLE recovers truth" gate test.

    Parameters
    ----------
    kappa:
        Mean-reversion speed (must be positive for stationarity).
    mu:
        Long-run mean.
    sigma:
        Instantaneous volatility.
    dt:
        Time step in years (e.g. ``1/252`` for daily).
    """

    def __init__(self, kappa: float, mu: float, sigma: float, dt: float = 1.0 / 252.0) -> None:
        if kappa <= 0:
            raise ValueError("kappa must be > 0 for a stationary OU process.")
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        self.kappa = float(kappa)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self._cached: pd.DataFrame | None = None

    def simulate(
        self,
        n: int,
        x0: float | None = None,
        seed: int | None = None,
        start: str | pd.Timestamp = "2010-01-04",
        freq: str = "B",
    ) -> pd.DataFrame:
        """Draw a path of length ``n`` and return it as a one-column frame.

        ``x0`` defaults to a draw from the stationary distribution
        :math:`\\mathcal{N}(\\mu, \\sigma^{2}/(2\\kappa))`, which lets the
        chain skip its burn-in.
        """
        rng = np.random.default_rng(seed)
        decay = np.exp(-self.kappa * self.dt)
        sigma_eps = self.sigma * np.sqrt((1.0 - decay**2) / (2.0 * self.kappa))
        stationary_sd = self.sigma / np.sqrt(2.0 * self.kappa)

        x = np.empty(n)
        x[0] = self.mu + stationary_sd * rng.standard_normal() if x0 is None else x0
        noise = rng.standard_normal(n - 1)
        for t in range(1, n):
            x[t] = self.mu + (x[t - 1] - self.mu) * decay + sigma_eps * noise[t - 1]

        index = pd.date_range(start=start, periods=n, freq=freq)
        self._cached = pd.DataFrame({"spread": x}, index=index)
        return self._cached

    def frame(self) -> pd.DataFrame:
        if self._cached is None:
            raise RuntimeError("Call .simulate(n, ...) before iterating SyntheticOU.")
        return self._cached


class SyntheticPair(DataSource):
    """Two log-prices ``Y`` and ``X`` whose linear combination is OU.

    Specifically: ``log_Y_t = beta * log_X_t + spread_t``, where
    ``log_X_t`` is a random walk with drift and ``spread_t`` is an OU
    process. Useful for end-to-end tests of cointegration screening +
    estimation + backtesting.
    """

    def __init__(
        self,
        beta: float,
        spread_kappa: float,
        spread_mu: float,
        spread_sigma: float,
        x_drift: float = 0.05,
        x_sigma: float = 0.2,
        dt: float = 1.0 / 252.0,
    ) -> None:
        self.beta = float(beta)
        self.spread_kappa = float(spread_kappa)
        self.spread_mu = float(spread_mu)
        self.spread_sigma = float(spread_sigma)
        self.x_drift = float(x_drift)
        self.x_sigma = float(x_sigma)
        self.dt = float(dt)
        self._cached: pd.DataFrame | None = None

    def simulate(self, n: int, seed: int | None = None) -> pd.DataFrame:
        # The spread and the X random walk must be *independent*; spawning
        # two child seeds avoids the subtle bug where reusing one seed makes
        # them share noise and biases the OLS hedge ratio.
        seed_spread, seed_x = np.random.SeedSequence(seed).spawn(2)
        spread = SyntheticOU(
            self.spread_kappa, self.spread_mu, self.spread_sigma, self.dt
        ).simulate(n=n, seed=seed_spread)["spread"].to_numpy()

        # GBM-style random walk for log X
        rng = np.random.default_rng(seed_x)
        innov = rng.standard_normal(n) * self.x_sigma * np.sqrt(self.dt)
        log_x = np.cumsum(np.full(n, self.x_drift * self.dt) + innov) + np.log(100.0)
        log_y = self.beta * log_x + spread

        index = pd.date_range(start="2010-01-04", periods=n, freq="B")
        self._cached = pd.DataFrame(
            {"Y": np.exp(log_y), "X": np.exp(log_x)}, index=index
        )
        return self._cached

    def frame(self) -> pd.DataFrame:
        if self._cached is None:
            raise RuntimeError("Call .simulate(n, ...) before iterating SyntheticPair.")
        return self._cached
