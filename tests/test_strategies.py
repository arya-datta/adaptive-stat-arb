"""Integration tests: run the Stage 1 / Stage 2 strategies through the engine.

These guard the pairs-trading refactor — a cointegration spread is a
zero-centred log-residual, so the strategies must trade the two legs (not
a fictitious 'spread instrument'). A regression to single-instrument
sizing would blow up volatility and drawdown, which these tests catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb.data import SyntheticPair
from stat_arb.data.base import DataSource
from stat_arb.engine import Backtester, LinearCostModel
from stat_arb.stage1 import OUMLEEstimator, ZScoreStrategy, engle_granger
from stat_arb.stage2 import OptimalStoppingStrategy


from stat_arb.data import InMemorySource as FrameSource  # shared frame adapter


@pytest.fixture
def fitted_pair():
    """A cointegrated pair, split IS/OOS, with the IS OU fit and pair spec."""
    pair = SyntheticPair(
        beta=1.0, spread_kappa=8.0, spread_mu=0.0, spread_sigma=0.12,
        x_drift=0.05, x_sigma=0.2,
    )
    raw = pair.simulate(n=2520, seed=5)
    cols = list(raw.columns)
    split = len(raw) // 2
    is_log = np.log(raw.iloc[:split])
    eg = engle_granger(is_log[cols[0]], is_log[cols[1]])
    est = OUMLEEstimator().fit(eg.spread)
    spec = eg.to_pair_spec(cols[0], cols[1], use_log=True)
    return est, spec, raw.iloc[split:]


def test_zscore_strategy_runs_with_sane_risk(fitted_pair):
    est, spec, oos_raw = fitted_pair
    strat = ZScoreStrategy(est, spec, entry_z=1.5, exit_z=0.0, stop_z=4.0)
    result = Backtester(strat, LinearCostModel(bps=5, half_spread_bps=2)).run(
        FrameSource(oos_raw)
    )
    s = result.summary()
    # Trading legs (dollar-neutral, gross≈1) must keep risk bounded — a
    # single-instrument regression would show >100% vol and ~-100% drawdown.
    assert s["ann_vol"] < 1.0
    assert s["max_drawdown"] > -0.6
    assert s["num_trades"] >= 2            # it actually trades
    assert np.isfinite(s["sharpe"])


def test_optimal_stopping_strategy_runs_with_sane_risk(fitted_pair):
    est, spec, oos_raw = fitted_pair
    sd = est.sigma / np.sqrt(2 * est.kappa)
    strat = OptimalStoppingStrategy(est, spec, r=0.05, cost=0.05 * sd)
    result = Backtester(strat, LinearCostModel(bps=5, half_spread_bps=2)).run(
        FrameSource(oos_raw)
    )
    s = result.summary()
    assert s["ann_vol"] < 1.0
    assert s["max_drawdown"] > -0.6
    assert np.isfinite(s["sharpe"])


def test_both_strategies_are_dollar_neutral_in_weights(fitted_pair):
    """Long-spread weights should net to ~0 dollars (β=1 here)."""
    est, spec, _ = fitted_pair
    w = spec.leg_weights(direction=+1, gross=1.0)
    # beta≈1 → equal and opposite
    assert w[spec.y_symbol] > 0 and w[spec.x_symbol] < 0
    assert abs(w[spec.y_symbol] + w[spec.x_symbol]) < 0.05
    assert abs(w[spec.y_symbol]) + abs(w[spec.x_symbol]) == pytest.approx(1.0, abs=1e-9)
