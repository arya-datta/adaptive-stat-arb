"""Stage 0 gate test: a deterministic buy-and-hold backtest ties out to hand-calc."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb.data import CSVDataSource
from stat_arb.data.base import DataSource
from stat_arb.engine import (
    Backtester, LinearCostModel, MarketEvent, SignalEvent, Strategy, ZeroCostModel,
)


# A minimal DataSource that wraps a fixed DataFrame for testing.
class _FrameSource(DataSource):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def frame(self) -> pd.DataFrame:
        return self.df


class _AlwaysFullyInvested(Strategy):
    """Hold 100% of the single symbol from bar 0."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._sent = False

    def reset(self) -> None:
        self._sent = False

    def on_bar(self, event: MarketEvent) -> SignalEvent | None:
        if self._sent:
            return None
        self._sent = True
        return SignalEvent(timestamp=event.timestamp, target_weights={self.symbol: 1.0})


def test_zero_cost_buy_and_hold_ties_to_handcalc(buy_and_hold_data):
    """Equity at T ties out exactly, accounting for the one-bar fill delay.

    The engine sizes the order on bar 0 (decision price P0) but fills it on
    bar 1 (price P_fill). So shares = capital / P0, and the small price move
    between decision and fill leaves a residual cash balance. This is the
    intended no-look-ahead convention, and the gate documents it precisely.
    """
    initial = 1_000_000.0
    prices = buy_and_hold_data["ASSET"]
    p0 = float(prices.iloc[0])       # decision/sizing price (bar 0)
    p_fill = float(prices.iloc[1])   # fill price (bar 1, one-bar delay)
    p_final = float(prices.iloc[-1])

    result = Backtester(
        strategy=_AlwaysFullyInvested("ASSET"),
        execution=ZeroCostModel(),
        initial_capital=initial,
        allow_fractional=True,
    ).run(_FrameSource(buy_and_hold_data))

    shares = initial / p0
    cash = initial - shares * p_fill
    expected_final = cash + shares * p_final

    np.testing.assert_allclose(result.equity.iloc[-1], expected_final, rtol=1e-9)
    # Exactly one fill (the initial buy) and no drawdown on a rising series.
    assert len(result.fills) == 1


def test_linear_cost_model_reduces_equity_vs_zero_cost(buy_and_hold_data):
    """A non-zero cost model must produce strictly less equity at T."""
    src = _FrameSource(buy_and_hold_data)
    a = Backtester(
        _AlwaysFullyInvested("ASSET"), ZeroCostModel(), initial_capital=1e6,
    ).run(src)
    b = Backtester(
        _AlwaysFullyInvested("ASSET"),
        LinearCostModel(bps=10, half_spread_bps=5),
        initial_capital=1e6,
    ).run(src)
    assert b.equity.iloc[-1] < a.equity.iloc[-1]


def test_summary_reports_expected_keys(buy_and_hold_data):
    result = Backtester(
        _AlwaysFullyInvested("ASSET"), ZeroCostModel(), initial_capital=1e6,
    ).run(_FrameSource(buy_and_hold_data))
    summary = result.summary()
    assert {
        "total_return", "cagr", "ann_vol", "sharpe",
        "max_drawdown", "turnover_annual", "num_trades", "total_commission",
    } <= set(summary)
    # A monotonically rising price means no drawdown.
    assert summary["max_drawdown"] == pytest.approx(0.0, abs=1e-9)


def test_csv_loader_roundtrip(tmp_path, buy_and_hold_data):
    """CSV adapter reproduces the same frame we feed it."""
    f = tmp_path / "wide.csv"
    buy_and_hold_data.reset_index(names="date").to_csv(f, index=False)
    src = CSVDataSource(f)
    out = src.frame()
    np.testing.assert_allclose(
        out["ASSET"].to_numpy(dtype=float),
        buy_and_hold_data["ASSET"].to_numpy(dtype=float),
    )
    assert list(out.index) == list(buy_and_hold_data.index)
