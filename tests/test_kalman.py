"""Stage 3 tests: the Kalman hedge must track a known time-varying beta and
add value over rolling-OLS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb.data import SyntheticPair
from stat_arb.data.base import DataSource
from stat_arb.engine import Backtester, LinearCostModel
from stat_arb.stage3 import KalmanHedge, KalmanZScoreStrategy, rolling_ols_hedge


class FrameSource(DataSource):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def frame(self) -> pd.DataFrame:
        return self._df


def _drifting_beta_series(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n) * 0.01) + 5.0
    beta_true = 1.0 + 0.5 * np.linspace(0, 1, n)        # 1.0 -> 1.5
    y = beta_true * x + rng.standard_normal(n) * 0.02
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    return pd.Series(y, idx), pd.Series(x, idx), beta_true


def test_kalman_tracks_drifting_beta():
    y, x, beta_true = _drifting_beta_series()
    kf = KalmanHedge.fit_mle(y, x)
    out = kf.filter(y, x)
    # Beta should rise from ~1.0 toward ~1.5 (filter lags but follows).
    assert out["beta"].iloc[50] < out["beta"].iloc[-1]
    assert abs(out["beta"].iloc[-1] - beta_true[-1]) < 0.25


def test_kalman_beats_rolling_ols_spread_variance():
    """The Stage 3 value claim: lower-variance spread than rolling-OLS."""
    y, x, _ = _drifting_beta_series()
    kf = KalmanHedge.fit_mle(y, x)
    kal = kf.filter(y, x)["spread"].to_numpy()[100:]
    roll = rolling_ols_hedge(y, x, window=60)["spread"].to_numpy()[100:]
    assert np.nanvar(kal) < np.nanvar(roll)


def test_kalman_fit_mle_returns_positive_params():
    y, x, _ = _drifting_beta_series(seed=3)
    kf = KalmanHedge.fit_mle(y, x)
    assert kf.q >= 0 and kf.r > 0


def test_kalman_strategy_runs_with_sane_risk():
    pair = SyntheticPair(beta=1.0, spread_kappa=8.0, spread_mu=0.0,
                         spread_sigma=0.12).simulate(n=2500, seed=4)
    cols = list(pair.columns)
    # Calibrate q, r by MLE so the innovation z-score is unit-scaled (an
    # arbitrary r leaves z compressed and the book never trades -> NaN Sharpe).
    kf = KalmanHedge.fit_mle(np.log(pair[cols[0]]), np.log(pair[cols[1]]))
    strat = KalmanZScoreStrategy(kf, cols[0], cols[1], entry_z=1.5, exit_z=0.0,
                                 stop_z=4.0, warmup=20)
    result = Backtester(strat, LinearCostModel(bps=5, half_spread_bps=2)).run(
        FrameSource(pair)
    )
    s = result.summary()
    # Dollar-neutral legs keep gross at 1.0 regardless of beta, so risk is bounded.
    assert s["ann_vol"] < 1.0
    assert s["max_drawdown"] > -0.6
    assert s["num_trades"] >= 1
    assert np.isfinite(s["sharpe"])
