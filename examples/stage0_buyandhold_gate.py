"""Stage 0 gate: a deterministic buy-and-hold backtest that ties to hand-calc.

Per the roadmap: "A deterministic, reproducible backtest of a trivial
strategy that ties out to hand calculation. If this isn't clean, nothing
downstream is trustworthy."

Run:
    python examples/stage0_buyandhold_gate.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.data.base import DataSource
from stat_arb.engine import (
    Backtester, MarketEvent, SignalEvent, Strategy, ZeroCostModel, LinearCostModel,
)


class FrameSource(DataSource):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def frame(self) -> pd.DataFrame:
        return self.df


class BuyAndHold(Strategy):
    """Allocate 100% to one symbol on the first bar, then never trade again."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._sent = False

    def reset(self) -> None:
        self._sent = False

    def on_bar(self, event: MarketEvent) -> SignalEvent | None:
        if self._sent:
            return None
        self._sent = True
        return SignalEvent(event.timestamp, {self.symbol: 1.0})


def main() -> None:
    # A linearly rising price from 100 to 149 over 50 business days.
    prices = pd.DataFrame(
        {"ASSET": np.arange(100, 150, dtype=float)},
        index=pd.date_range("2024-01-02", periods=50, freq="B"),
    )
    source = FrameSource(prices)
    initial = 1_000_000.0

    result = Backtester(BuyAndHold("ASSET"), ZeroCostModel(), initial_capital=initial).run(source)

    # Hand calculation (mirrors the engine's one-bar fill delay).
    p0, p_fill, p_final = prices["ASSET"].iloc[[0, 1, -1]]
    shares = initial / p0
    cash = initial - shares * p_fill
    expected_final = cash + shares * p_final

    print("=== Stage 0 buy-and-hold gate ===")
    print(f"shares bought         : {shares:,.4f}")
    print(f"engine final equity   : {result.equity.iloc[-1]:,.2f}")
    print(f"hand-calc final equity: {expected_final:,.2f}")
    print(f"match                 : {np.isclose(result.equity.iloc[-1], expected_final)}")
    print(f"total return          : {result.summary()['total_return']:.4%}")
    print(f"max drawdown          : {result.summary()['max_drawdown']:.4%}")

    # And with realistic costs, equity must be strictly lower.
    costed = Backtester(
        BuyAndHold("ASSET"), LinearCostModel(bps=10, half_spread_bps=5),
        initial_capital=initial,
    ).run(source)
    print(f"\nwith 10bps + 5bps half-spread, final equity: {costed.equity.iloc[-1]:,.2f}")
    print(f"cost drag             : {result.equity.iloc[-1] - costed.equity.iloc[-1]:,.2f}")


if __name__ == "__main__":
    main()
