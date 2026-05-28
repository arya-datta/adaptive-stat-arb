"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb.data import SyntheticOU, SyntheticPair

# True parameters used across the OU recovery tests.
TRUE_KAPPA = 2.0
TRUE_MU = 0.0
TRUE_SIGMA = 0.3
DT = 1.0 / 252.0


@pytest.fixture
def ou_spread() -> pd.Series:
    """A 10-year daily OU path with known parameters."""
    sim = SyntheticOU(kappa=TRUE_KAPPA, mu=TRUE_MU, sigma=TRUE_SIGMA, dt=DT)
    return sim.simulate(n=2520, seed=0)["spread"]


@pytest.fixture
def ou_source() -> SyntheticOU:
    """The :class:`DataSource` form of the same series."""
    sim = SyntheticOU(kappa=TRUE_KAPPA, mu=TRUE_MU, sigma=TRUE_SIGMA, dt=DT)
    sim.simulate(n=2520, seed=0)
    return sim


@pytest.fixture
def cointegrated_pair() -> SyntheticPair:
    """Two log-prices whose linear combination is OU."""
    pair = SyntheticPair(
        beta=1.2,
        spread_kappa=3.0, spread_mu=0.0, spread_sigma=0.15,
        x_drift=0.05, x_sigma=0.2,
    )
    pair.simulate(n=1500, seed=42)
    return pair


@pytest.fixture
def random_returns_matrix() -> np.ndarray:
    """A (T=1000, N=200) matrix of independent zero-mean Gaussian returns.

    Used by PBO/DSR tests: any "best" strategy here is pure noise, so
    overfitting metrics should flag it.
    """
    rng = np.random.default_rng(2026)
    return rng.normal(0.0, 0.01, size=(1000, 200))


@pytest.fixture
def buy_and_hold_data() -> pd.DataFrame:
    """50 days of a single ticker rising linearly from 100 to 149.

    Buy-and-hold equity should multiply by ``149/100``. The numbers are
    chosen so the gate test ties out *exactly* to hand calculation.
    """
    prices = np.arange(100, 150, dtype=float)
    index = pd.date_range("2024-01-02", periods=50, freq="B")
    return pd.DataFrame({"ASSET": prices}, index=index)
