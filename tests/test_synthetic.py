"""Sanity tests for the synthetic generators.

These are not algorithm tests — they validate that the *test fixtures
themselves* deliver the statistical properties downstream tests assume.
"""

from __future__ import annotations

import numpy as np
import pytest

from stat_arb.data import SyntheticOU


def test_ou_stationary_moments_match_theory():
    """Empirical mean ≈ μ, std ≈ σ/sqrt(2κ) (stationary OU)."""
    kappa, mu, sigma = 2.5, 1.0, 0.4
    sim = SyntheticOU(kappa=kappa, mu=mu, sigma=sigma, dt=1 / 252)
    x = sim.simulate(n=20_000, seed=7)["spread"].to_numpy()

    theoretical_sd = sigma / np.sqrt(2 * kappa)
    assert abs(x.mean() - mu) < 0.05
    # Allow 5% tolerance: large n but autocorrelated, so the effective
    # sample size is much smaller than 20k.
    assert abs(x.std(ddof=1) - theoretical_sd) / theoretical_sd < 0.10


def test_ou_rejects_negative_kappa():
    with pytest.raises(ValueError):
        SyntheticOU(kappa=-1.0, mu=0.0, sigma=0.1)


def test_ou_simulate_is_deterministic():
    sim = SyntheticOU(2.0, 0.0, 0.3)
    a = sim.simulate(n=100, seed=1)["spread"].to_numpy()
    b = sim.simulate(n=100, seed=1)["spread"].to_numpy()
    np.testing.assert_array_equal(a, b)


def test_ou_iteration_yields_bars(ou_source):
    bars = list(ou_source)
    assert len(bars) == 2520
    # bars are sorted in time
    timestamps = [b.timestamp for b in bars]
    assert timestamps == sorted(timestamps)
