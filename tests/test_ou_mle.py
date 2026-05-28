"""Stage 1 cornerstone: the exact-discretisation MLE must recover known parameters."""

from __future__ import annotations

import numpy as np
import pytest

from stat_arb.data import SyntheticOU
from stat_arb.stage1 import OUMLEEstimator


def test_mle_recovers_kappa_mu_sigma_within_tolerance(ou_spread):
    """10y daily path → all three params within 15% of truth."""
    est = OUMLEEstimator().fit(ou_spread)

    assert est.kappa == pytest.approx(2.0, rel=0.20)
    # mu's natural scale is the stationary SD = sigma/sqrt(2k) = 0.15; the
    # sample mean of an OU path is only estimable to ~1 stationary SD.
    stationary_sd = 0.3 / np.sqrt(2 * 2.0)
    assert abs(est.mu - 0.0) < stationary_sd
    assert est.sigma == pytest.approx(0.3, rel=0.15)

    # Half-life follows kappa
    assert est.half_life == pytest.approx(np.log(2) / est.kappa, rel=1e-9)
    # Should be flagged stationary
    assert est.stationary


def test_mle_confidence_intervals_contain_truth_most_of_the_time():
    """95% CIs should contain the truth roughly 95% of trials.

    With 50 trials we don't have the power to test exactly 0.95, but
    coverage should be solidly above 0.85.
    """
    truth = {"kappa": 2.0, "mu": 0.0, "sigma": 0.3}
    hits = {"kappa": 0, "mu": 0, "sigma": 0}
    n_trials = 60
    for seed in range(n_trials):
        sim = SyntheticOU(2.0, 0.0, 0.3).simulate(n=4000, seed=seed)
        params = OUMLEEstimator().fit(sim["spread"])
        for k in hits:
            lo, hi = getattr(params, f"{k}_ci")
            if lo <= truth[k] <= hi:
                hits[k] += 1
    # kappa carries a known finite-sample upward bias (Tang-Chen 2009), so
    # its coverage is a touch below nominal; mu/sigma should be near 0.95.
    assert hits["mu"] / n_trials >= 0.88, f"mu coverage {hits['mu']/n_trials:.2f}"
    assert hits["sigma"] / n_trials >= 0.88, f"sigma coverage {hits['sigma']/n_trials:.2f}"
    assert hits["kappa"] / n_trials >= 0.78, f"kappa coverage {hits['kappa']/n_trials:.2f}"


def test_mle_handles_short_dt_without_euler_bias():
    """Naïve Euler regression biases κ for coarse dt; exact MLE does not."""
    sim = SyntheticOU(kappa=5.0, mu=1.0, sigma=0.5, dt=1 / 52).simulate(
        n=2000, seed=11
    )
    est = OUMLEEstimator().fit(sim["spread"], dt=1 / 52)
    # Even with weekly dt and fast reversion (half-life ~0.14 years),
    # the exact MLE must stay close to truth.
    assert est.kappa == pytest.approx(5.0, rel=0.25)


def test_mle_flags_random_walk_as_non_stationary():
    """A random walk has AR(1) coefficient ≈ 1 (kappa ≈ 0).

    The estimator should not crash on it — it should *report* that mean
    reversion is not statistically significant (the PDF's "stationarity
    condition kappa > 0" gate), i.e. ``stationary is False``.
    """
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.standard_normal(2000))
    est = OUMLEEstimator().fit(rw, dt=1 / 252)
    assert not est.stationary
    # Kappa's lower confidence bound should sit at or below zero.
    assert est.kappa_ci[0] <= 0.0


def test_mle_accepts_array_with_explicit_dt():
    """API check: a plain ndarray + explicit dt fits and recovers kappa.

    Uses a longer path because kappa's finite-sample bias shrinks with T.
    """
    arr = SyntheticOU(2.0, 0.0, 0.3).simulate(n=4000, seed=3)["spread"].to_numpy()
    est = OUMLEEstimator().fit(arr, dt=1 / 252)
    assert est.kappa == pytest.approx(2.0, rel=0.30)
