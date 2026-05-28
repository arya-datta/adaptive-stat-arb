"""Cointegration screen tests on synthetic pairs of known structure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb.data import SyntheticPair
from stat_arb.stage1 import engle_granger, johansen


def test_engle_granger_finds_cointegration_in_synthetic_pair(cointegrated_pair):
    df = np.log(cointegrated_pair.frame())
    res = engle_granger(df["Y"], df["X"])
    assert res.cointegrated_at_5pct
    # Hedge ratio close to truth (β = 1.2 with mild OLS bias).
    assert res.beta == pytest.approx(1.2, abs=0.1)


def test_engle_granger_rejects_independent_random_walks():
    rng = np.random.default_rng(123)
    n = 1000
    idx = pd.date_range("2010-01-04", periods=n, freq="B")
    y = pd.Series(np.cumsum(rng.standard_normal(n)), index=idx, name="y")
    x = pd.Series(np.cumsum(rng.standard_normal(n)), index=idx, name="x")
    res = engle_granger(y, x)
    assert not res.cointegrated_at_5pct


def test_johansen_finds_rank_one_in_synthetic_pair(cointegrated_pair):
    df = np.log(cointegrated_pair.frame())
    res = johansen(df, det_order=0, k_ar_diff=1)
    assert res.rank >= 1, f"Johansen failed to detect rank ≥ 1; got {res.rank}"
