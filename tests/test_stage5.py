"""Stage 5 tests: Bayesian OU posterior + Liu-West particle filter + the
uncertainty-scaling gate (scaled book must not be riskier than the point book)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb.data import SyntheticOU, SyntheticPair
from stat_arb.data.base import DataSource
from stat_arb.engine import Backtester, LinearCostModel
from stat_arb.stage1 import OUMLEEstimator, engle_granger
from stat_arb.stage5 import BayesianOU, ParticleFilterOU, UncertaintyScaledStrategy


from stat_arb.data import InMemorySource as FrameSource  # shared frame adapter


def test_bayesian_posterior_recovers_kappa():
    spread = SyntheticOU(kappa=3.0, mu=0.0, sigma=0.2).simulate(n=3000, seed=0)["spread"]
    post = BayesianOU().fit(spread)
    s = post.summary(n_draws=4000, seed=0)
    assert s["kappa_mean"] == pytest.approx(3.0, rel=0.25)
    assert s["p_stationary"] > 0.9


def test_bayesian_posterior_wider_with_less_data():
    """Less data -> more diffuse posterior (higher kappa coefficient of variation)."""
    short = SyntheticOU(kappa=3.0, mu=0.0, sigma=0.2).simulate(n=120, seed=1)["spread"]
    long = SyntheticOU(kappa=3.0, mu=0.0, sigma=0.2).simulate(n=4000, seed=1)["spread"]
    cv_short = BayesianOU().fit(short).summary(seed=0)["kappa_cv"]
    cv_long = BayesianOU().fit(long).summary(seed=0)["kappa_cv"]
    assert cv_short > cv_long


def test_particle_filter_recovers_kappa_online():
    """The Liu-West filter is an *approximate* online posterior, so we only
    require the reversion speed to land in the right ballpark (within ~2x) and
    the process to be flagged stationary -- not the precision of the batch fit."""
    dt = 1 / 252
    spread = SyntheticOU(kappa=3.0, mu=0.0, sigma=0.2, dt=dt).simulate(n=3000, seed=0)["spread"]
    pf = ParticleFilterOU(n_particles=2000, seed=0)
    pf.seed_from_ranges()
    report = None
    for x in spread.to_numpy():
        report = pf.step(float(x))
    kappa_yr = report["kappa_mean"] / dt
    assert 1.5 <= kappa_yr <= 6.0          # within ~2x of the true 3.0
    assert report["p_stationary"] > 0.9


def test_particle_filter_seed_from_posterior():
    spread = SyntheticOU(kappa=3.0, mu=0.0, sigma=0.2).simulate(n=1000, seed=3)["spread"]
    post = BayesianOU().fit(spread)
    pf = ParticleFilterOU(n_particles=500, seed=0)
    pf.seed_from_posterior(post)
    rep = pf.step(float(spread.iloc[0]))
    assert "kappa_mean" in rep and np.isfinite(rep["p_stationary"])


def test_uncertainty_scaling_does_not_increase_risk():
    """The gate: the uncertainty-scaled book should not have *higher* volatility
    or a worse drawdown than the equivalent point-estimate book."""
    pair = SyntheticPair(beta=1.0, spread_kappa=6.0, spread_mu=0.0,
                         spread_sigma=0.12).simulate(n=2000, seed=4)
    cols = list(pair.columns)
    eg = engle_granger(np.log(pair[cols[0]]), np.log(pair[cols[1]]))
    est = OUMLEEstimator().fit(eg.spread)
    spec = eg.to_pair_spec(cols[0], cols[1])

    def run(scale: bool):
        pf = ParticleFilterOU(n_particles=800, seed=1)
        strat = UncertaintyScaledStrategy(est, spec, pf, entry_z=1.5, stop_z=4.0,
                                          scale_by_uncertainty=scale)
        return Backtester(strat, LinearCostModel(bps=5, half_spread_bps=2)).run(
            FrameSource(pair)
        ).summary()

    scaled, point = run(True), run(False)
    assert scaled["ann_vol"] <= point["ann_vol"] + 1e-9
    assert scaled["max_drawdown"] >= point["max_drawdown"] - 1e-9
