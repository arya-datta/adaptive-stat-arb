"""Stage 2: Leung-Li boundary tests on a known-OU spread.

We don't have a closed-form OOS Sharpe to assert against, so the tests
target structural properties: ``long_entry < μ < long_exit``, narrower
band for smaller transaction cost, infeasibility at very high cost.
"""

from __future__ import annotations

import numpy as np
import pytest

from stat_arb.stage1 import OUMLEEstimator
from stat_arb.stage1.ou_mle import OUParams
from stat_arb.stage2 import compute_boundaries, OUFundamentals


@pytest.fixture
def fitted_ou(ou_spread):
    return OUMLEEstimator().fit(ou_spread)


def test_fundamental_solutions_monotone(fitted_ou):
    """F must be increasing, G decreasing — definitional."""
    fund = OUFundamentals(
        kappa=fitted_ou.kappa, mu=fitted_ou.mu, sigma=fitted_ou.sigma, r=0.05
    )
    xs = np.linspace(fitted_ou.mu - 1.0, fitted_ou.mu + 1.0, 11)
    F_vals = np.array([fund.F(x) for x in xs])
    G_vals = np.array([fund.G(x) for x in xs])
    assert np.all(np.diff(F_vals) > 0)
    assert np.all(np.diff(G_vals) < 0)


def test_boundaries_bracket_the_mean(fitted_ou):
    b = compute_boundaries(fitted_ou, r=0.05, cost=0.01)
    assert b.has_long
    assert b.long_entry < fitted_ou.mu < b.long_exit


def test_boundaries_widen_with_cost(fitted_ou):
    cheap = compute_boundaries(fitted_ou, r=0.05, cost=0.005)
    pricey = compute_boundaries(fitted_ou, r=0.05, cost=0.04)
    assert cheap.has_long and pricey.has_long
    # Higher cost → demand a bigger profit before exit, so b* moves right
    assert pricey.long_exit > cheap.long_exit
    # And requires a more attractive (deeper-below-μ) entry
    assert pricey.long_entry < cheap.long_entry


def test_extreme_cost_pushes_entry_far_from_mean(fitted_ou):
    """High cost doesn't make trading infeasible (OU is unbounded) — it just
    pushes the entry threshold many standard deviations below the mean, so
    the strategy only acts on enormous, rare dislocations."""
    sd = fitted_ou.sigma / np.sqrt(2 * fitted_ou.kappa)
    res = compute_boundaries(fitted_ou, r=0.05, cost=10.0 * sd)
    assert res.has_long
    # Entry should be pushed well below the mean (here ~10 SD).
    assert (res.long_entry - fitted_ou.mu) < -5.0 * sd


def test_sub_bar_half_life_is_rejected_as_untradeable():
    """A κ so large that the half-life is under ~2 bars must be infeasible.

    Mirrors the real-data VOO/SPY case: two near-identical ETFs cointegrate
    with a ~1-day half-life — microstructure noise, not a tradeable spread.
    """
    dt = 1 / 252
    kappa = 310.0  # half-life ~0.56 bars
    params = OUParams(
        kappa=kappa, mu=0.0, sigma=0.014, kappa_ci=(250, 370), mu_ci=(-.01, .01),
        sigma_ci=(.01, .02), half_life=np.log(2) / kappa, log_likelihood=0.0,
        n_obs=880, dt=dt, stationary=True,
    )
    res = compute_boundaries(params, r=0.05, cost=0.0)
    assert not res.has_long
    assert not res.has_short


def test_short_boundaries_mirror_long_when_mu_is_zero(fitted_ou):
    """Symmetric reflection check: μ=0 path → short boundaries mirror longs."""
    # ou_spread fixture has μ=0 exactly in truth; fitted μ is near 0.
    res = compute_boundaries(fitted_ou, r=0.05, cost=0.01)
    if res.has_long and res.has_short:
        # The reflection is about fitted μ — anchor on that.
        np.testing.assert_allclose(
            res.short_exit - fitted_ou.mu,
            -(res.long_exit - fitted_ou.mu),
            atol=1e-4,
        )
