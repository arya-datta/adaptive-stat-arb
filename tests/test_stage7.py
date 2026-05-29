"""Stage 7 tests: microstructure cost model (square-root impact, partial fills)
and Almgren-Chriss scheduling (TWAP limit, front-loading, cost/risk frontier)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb.data import SyntheticPair
from stat_arb.data.base import DataSource
from stat_arb.engine import Backtester, ZeroCostModel
from stat_arb.engine.events import OrderEvent
from stat_arb.stage1 import OUMLEEstimator, ZScoreStrategy, engle_granger
from stat_arb.stage7 import (
    MicrostructureCostModel, almgren_chriss_schedule, execution_frontier,
    twap_schedule, vwap_schedule,
)


class FrameSource(DataSource):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def frame(self) -> pd.DataFrame:
        return self._df


# -------------------- cost model --------------------
def test_square_root_impact_scaling():
    """Impact (price fraction) should scale as sqrt(participation)."""
    m = MicrostructureCostModel(half_spread_bps=0, commission_bps=0, latency_bps=0,
                                impact_coef=1.0, daily_vol=0.02, adv=1_000_000,
                                participation_cap=1.0)
    f1 = m.fill(OrderEvent(pd.Timestamp("2020-01-01"), "X", 10_000), 100.0)
    f4 = m.fill(OrderEvent(pd.Timestamp("2020-01-01"), "X", 40_000), 100.0)
    imp1 = f1.price / 100.0 - 1.0
    imp4 = f4.price / 100.0 - 1.0
    # 4x size -> 2x impact (square-root law).
    assert imp4 / imp1 == pytest.approx(2.0, rel=0.05)


def test_partial_fill_caps_quantity():
    m = MicrostructureCostModel(adv=1_000_000, participation_cap=0.05)
    f = m.fill(OrderEvent(pd.Timestamp("2020-01-01"), "X", 200_000), 100.0)
    assert abs(f.quantity) == pytest.approx(0.05 * 1_000_000)   # capped at 50k


def test_microstructure_costlier_than_frictionless():
    pair = SyntheticPair(beta=1.0, spread_kappa=8.0, spread_mu=0.0,
                         spread_sigma=0.12).simulate(n=1500, seed=4)
    cols = list(pair.columns)
    eg = engle_granger(np.log(pair[cols[0]]), np.log(pair[cols[1]]))
    est = OUMLEEstimator().fit(eg.spread)
    spec = eg.to_pair_spec(cols[0], cols[1])

    def run(model):
        strat = ZScoreStrategy(est, spec, entry_z=1.5, exit_z=0.0, stop_z=4.0)
        return Backtester(strat, model).run(FrameSource(pair)).summary()

    free = run(ZeroCostModel())
    # Low ADV makes impact bite -> realistic execution must cost more.
    costed = run(MicrostructureCostModel(adv=200_000, participation_cap=0.2,
                                         impact_coef=1.0, daily_vol=0.02))
    assert costed["total_return"] < free["total_return"]
    assert costed["total_commission"] >= 0


# -------------------- Almgren-Chriss --------------------
def test_ac_reduces_to_twap_when_risk_neutral():
    sched = almgren_chriss_schedule(total_shares=1_000, n_steps=10, sigma=0.3,
                                    eta=1e-6, lam=0.0)
    twap = twap_schedule(1_000, 10)
    np.testing.assert_allclose(sched.holdings, twap.holdings, atol=1e-6)
    assert sched.kappa == 0.0


def test_ac_front_loads_with_risk_aversion():
    slow = almgren_chriss_schedule(1_000, 20, sigma=0.3, eta=1e-6, lam=1e-3)
    fast = almgren_chriss_schedule(1_000, 20, sigma=0.3, eta=1e-6, lam=1e-1)
    mid = 10
    # Higher risk aversion -> fewer shares still held at the midpoint.
    assert fast.holdings[mid] < slow.holdings[mid]
    assert fast.kappa > slow.kappa


def test_execution_frontier_tradeoff():
    lambdas = np.array([1e-4, 1e-3, 1e-2, 1e-1])
    frontier = execution_frontier(1_000, 20, sigma=0.3, eta=1e-6, lambdas=lambdas)
    costs = [c for c, _ in frontier]
    stds = [s for _, s in frontier]
    # More urgency -> higher expected impact cost, lower timing-risk std.
    assert costs[0] < costs[-1]
    assert stds[0] > stds[-1]


def test_schedules_sum_to_total():
    assert twap_schedule(500, 10).trades.sum() == pytest.approx(500)
    vp = np.array([1, 3, 5, 3, 1], float)
    assert vwap_schedule(500, vp).trades.sum() == pytest.approx(500)
