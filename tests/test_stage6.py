"""Stage 6 tests: PCA factor recovery, Ledoit-Wolf shrinkage, HRP, residual
s-scores, the cross-sectional strategy, VECM, and multiple-testing controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb.data import SyntheticFactorMarket
from stat_arb.data.base import DataSource
from stat_arb.engine import Backtester, LinearCostModel
from stat_arb.stage6 import (
    EigenportfolioStrategy, fit_vecm, hierarchical_risk_parity, ledoit_wolf,
    pca_factors, residual_sscores,
)
from stat_arb.validation import benjamini_hochberg, harvey_liu_zhu_hurdle
from stat_arb.validation.stationarity import adf_test


class FrameSource(DataSource):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def frame(self) -> pd.DataFrame:
        return self._df


@pytest.fixture(scope="module")
def factor_market():
    gen = SyntheticFactorMarket(
        n_stocks=15, n_factors=3, factor_vol=0.012,
        resid_kappa=12.0, resid_sigma=0.04,
    )
    return gen.simulate(n=1200, seed=0)


# -------------------- PCA --------------------
def test_pca_top_factors_explain_most_variance(factor_market):
    R = np.diff(np.log(factor_market.to_numpy()), axis=0)
    fac = pca_factors(R, n_factors=3)
    # 3 true common factors -> top 3 components dominate.
    assert fac.explained_variance_ratio[:3].sum() > 0.5
    assert np.all(np.diff(fac.eigenvalues) <= 1e-9)   # descending


# -------------------- Ledoit-Wolf --------------------
def test_ledoit_wolf_shrinkage_and_conditioning():
    rng = np.random.default_rng(0)
    # N close to T -> sample cov ill-conditioned; shrinkage should help.
    X = rng.standard_normal((40, 30))
    lw = ledoit_wolf(X)
    assert 0.0 <= lw["shrinkage"] <= 1.0
    sample_cov = np.cov(X, rowvar=False)
    cond_sample = np.linalg.cond(sample_cov)
    cond_shrunk = np.linalg.cond(lw["cov"])
    assert cond_shrunk < cond_sample


# -------------------- HRP --------------------
def test_hrp_weights_valid(factor_market):
    R = np.diff(np.log(factor_market.to_numpy()), axis=0)
    cov = ledoit_wolf(R)["cov"]
    w = hierarchical_risk_parity(cov)
    assert w.shape[0] == cov.shape[0]
    assert np.all(w >= -1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


# -------------------- residual s-scores --------------------
def test_residual_sscores_detects_reverting_residuals(factor_market):
    window = factor_market.to_numpy()[-120:]
    scores = residual_sscores(window, n_factors=3)
    # The synthetic residuals are fast-reverting (kappa=12), so most names
    # should be flagged tradeable with finite s-scores.
    assert scores.reverting.sum() >= 5
    assert np.all(np.isfinite(scores.sscore))
    assert scores.explained_variance > 0.4


# -------------------- strategy --------------------
def test_eigenportfolio_strategy_runs_dollar_neutral(factor_market):
    symbols = list(factor_market.columns)
    strat = EigenportfolioStrategy(symbols, n_factors=3, lookback=60,
                                   recalc_every=5, s_entry=1.25, s_close=0.5)
    result = Backtester(strat, LinearCostModel(bps=5, half_spread_bps=2)).run(
        FrameSource(factor_market)
    )
    s = result.summary()
    assert np.isfinite(s["sharpe"])
    assert s["ann_vol"] < 1.0
    # When holding positions, the book is ~dollar-neutral (equal long/short notional).
    pos = result.positions.iloc[-1]
    prices = factor_market.iloc[-1]
    net = float((pos * prices.reindex(pos.index)).sum())
    gross = float((pos.abs() * prices.reindex(pos.index)).sum())
    if gross > 0:
        assert abs(net) / gross < 0.25


# -------------------- VECM --------------------
def test_vecm_finds_cointegration_in_common_trend_system():
    """4 log-price series sharing ONE common random walk are cointegrated
    (rank ~ N-1); VECM should detect rank >= 1 and yield a stationary spread."""
    rng = np.random.default_rng(1)
    n = 800
    common = np.cumsum(rng.standard_normal(n) * 0.02)
    loadings = np.array([1.0, 0.8, 1.2, 0.6])
    noise = rng.standard_normal((n, 4)) * 0.05
    logp = common[:, None] * loadings[None, :] + noise + np.log(50)
    df = pd.DataFrame(logp, columns=[f"A{i}" for i in range(4)],
                      index=pd.date_range("2015-01-01", periods=n, freq="B"))
    res = fit_vecm(df, k_ar_diff=1, det_order=0)
    assert res.rank >= 1
    assert adf_test(res.spread(df)).stationary_at_5pct


# -------------------- multiple testing --------------------
def test_benjamini_hochberg_controls_discoveries():
    # 5 genuine signals (tiny p) among 95 nulls (uniform p).
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 1e-4, 5), rng.uniform(0, 1, 95)])
    out = benjamini_hochberg(p, fdr=0.10)
    assert out["n_reject"] >= 5            # recovers the true signals
    assert out["n_reject"] < 30            # without flooding with false positives


def test_harvey_liu_zhu_hurdle():
    # Sharpe 1.0 over 4 years daily: t ~ 2 -> fails the t>3 buy-side hurdle.
    weak = harvey_liu_zhu_hurdle(sharpe=1.0, n_obs=1008)
    assert not weak["passes"]
    strong = harvey_liu_zhu_hurdle(sharpe=2.5, n_obs=2520)
    assert strong["passes"]
