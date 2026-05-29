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


# Shared in-memory frame adapter (was a one-off shim).
from stat_arb.data import InMemorySource as _FrameSource  # noqa: E402


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


def test_returns_are_finite_through_zero_equity():
    """Bankruptcy edge: equity touching zero must not poison returns with inf/nan."""
    from stat_arb.engine import BacktestResult
    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    equity = pd.Series([1_000_000.0, 500_000.0, 0.0, 100_000.0], index=idx)
    res = BacktestResult(equity=equity, positions=pd.DataFrame(index=idx))
    assert np.isfinite(res.returns.to_numpy()).all()
    assert np.isfinite(res.log_returns.to_numpy()).all()


def test_equity_carries_last_price_through_a_gap():
    """A missing/NaN quote on a held name must not collapse equity to cash —
    the position is valued at its last good price (no fabricated drawdown)."""
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    prices = pd.DataFrame({"ASSET": [100.0, 101.0, 102.0, np.nan, 104.0, 105.0]}, index=idx)
    result = Backtester(_AlwaysFullyInvested("ASSET"), ZeroCostModel(),
                        initial_capital=1_000_000.0).run(_FrameSource(prices))
    eq = result.equity
    gap_equity = float(eq.iloc[3])             # the NaN bar
    # Position bought at bar 1 (≈101); on the gap bar it should still be marked
    # near the neighbouring levels, far above a cash-only ~1e6 collapse artifact.
    assert gap_equity > 1.005e6
    assert eq.iloc[2] * 0.97 < gap_equity < eq.iloc[4] * 1.03
    # No NaN in the equity curve.
    assert eq.notna().all()


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
