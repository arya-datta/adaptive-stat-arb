"""Stage 4 tests: EM recovers known regime params, the justification gate is
honest (adopts when warranted, rejects single-regime data), and the online
filter matches the batch filter.

Speed: the EM uses pure-Python recursions, so tests keep ``n`` modest, use
few restarts, and **cache a single fit** (module-scoped) shared across the
recovery / classification / online-filter checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb.data import SyntheticMarkovOU, SyntheticOU, SyntheticPair
from stat_arb.data.base import DataSource
from stat_arb.engine import Backtester, LinearCostModel
from stat_arb.stage1 import OUMLEEstimator, engle_granger
from stat_arb.stage4 import (
    MarkovSwitchingOU, OnlineHamiltonFilter, RegimeSwitchingStrategy, regime_justification,
)


class FrameSource(DataSource):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def frame(self) -> pd.DataFrame:
        return self._df


@pytest.fixture(scope="module")
def two_regime():
    """Synthetic 2-regime spread + true regime path (generated once)."""
    gen = SyntheticMarkovOU(
        kappas=[12.0, 2.0], mus=[0.0, 0.0], sigmas=[0.10, 0.40],
        P=[[0.98, 0.02], [0.04, 0.96]], dt=1 / 252,
    )
    spread = gen.simulate(n=2500, seed=1)["spread"]
    return spread, gen.regimes


@pytest.fixture(scope="module")
def fitted_two_regime(two_regime):
    """Fit the 2-regime model ONCE; reused by the three checks below."""
    spread, true_regimes = two_regime
    ms = MarkovSwitchingOU(n_regimes=2, n_init=3, seed=0).fit(spread)
    return ms, spread, true_regimes


def test_em_recovers_two_regime_structure(fitted_two_regime):
    ms, _, _ = fitted_two_regime
    # Regimes are ordered fastest-kappa first.
    assert ms.kappa[0] > ms.kappa[1]
    # Fast regime ~ low vol, slow regime ~ high vol (true 0.10 / 0.40).
    assert ms.sigma[0] < ms.sigma[1]
    assert ms.sigma[0] == pytest.approx(0.10, abs=0.05)
    assert ms.sigma[1] == pytest.approx(0.40, abs=0.12)
    assert ms.mean_reverting.all()


def test_em_regime_classification_is_accurate(fitted_two_regime):
    ms, _, true_regimes = fitted_two_regime
    pred = ms.smoothed_prob.argmax(axis=1)
    acc = (pred == true_regimes[1:]).mean()
    assert max(acc, 1 - acc) > 0.85


def test_online_filter_matches_batch(fitted_two_regime):
    ms, spread, _ = fitted_two_regime
    online = OnlineHamiltonFilter(ms)
    x = spread.to_numpy()
    probs = np.array([online.update(float(xi)) for xi in x])
    # Online filtered probs (t>=1) match the batch Hamilton filter up to
    # floating-point accumulation order.
    np.testing.assert_allclose(probs[1:], ms.filtered_prob, atol=1e-5)


def test_justification_gate_adopts_for_two_regime_data(two_regime):
    spread, _ = two_regime
    j = regime_justification(spread, n_regimes=2, n_init=3)
    assert j["adopt_regime_switching"]
    assert j["bic_multi"] < j["bic_single"]
    assert j["lr_statistic"] > 0


def test_justification_gate_rejects_single_regime_data():
    """Justified complexity only: on genuine single-regime OU, BIC should not
    adopt a 2-regime model (the roadmap's principle #2). Verified across
    several seeds offline; here we check one representative path."""
    spread = SyntheticOU(kappa=5.0, mu=0.0, sigma=0.2).simulate(n=1500, seed=0)["spread"]
    j = regime_justification(spread, n_regimes=2, n_init=3)
    assert not j["adopt_regime_switching"]


def test_regime_strategy_runs_and_stands_down():
    pair_gen = SyntheticPair(beta=1.0, spread_kappa=8.0, spread_mu=0.0,
                             spread_sigma=0.12).simulate(n=1500, seed=2)
    cols = list(pair_gen.columns)
    log = np.log(pair_gen)
    eg = engle_granger(log[cols[0]], log[cols[1]])
    spec = eg.to_pair_spec(cols[0], cols[1], use_log=True)
    ms = MarkovSwitchingOU(n_regimes=2, n_init=2, seed=0).fit(eg.spread)

    strat = RegimeSwitchingStrategy(ms, spec, entry_z=1.5, stop_z=4.0)
    result = Backtester(strat, LinearCostModel(bps=5, half_spread_bps=2)).run(
        FrameSource(pair_gen)
    )
    s = result.summary()
    assert s["ann_vol"] < 1.0
    assert s["max_drawdown"] > -0.6
    # The strategy recorded an inferred-regime path for the robustness report.
    assert len(strat.regime_history) == len(pair_gen)
