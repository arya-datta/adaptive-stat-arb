"""Event-loop backtester.

The loop, in one paragraph:

    For every bar ``t`` (after a warm-up gap) we (1) mark the portfolio to
    market at ``prices_t``, (2) settle any orders pending from bar ``t-1``
    using ``prices_t`` (so no fill is ever priced from data the strategy
    has already seen), (3) feed ``MarketEvent_t`` to the strategy and, if
    a :class:`SignalEvent` comes back, ask the portfolio to translate it
    into orders that *queue* until bar ``t+1``. This single deferral is
    what kills the most common look-ahead bug — orders priced at the same
    close that triggered them.
"""

from __future__ import annotations

from collections import deque

import pandas as pd

from ..data import DataSource
from .accounting import BacktestResult
from .events import MarketEvent, OrderEvent
from .execution import ExecutionModel, LinearCostModel
from .portfolio import Portfolio
from .strategy import Strategy


class Backtester:
    """Glue between data source, strategy, portfolio, and execution model.

    Parameters
    ----------
    strategy:
        The user-supplied :class:`Strategy`.
    execution:
        Defaults to :class:`LinearCostModel` (5 bps + 2 bps half-spread).
    initial_capital:
        Starting equity.
    """

    def __init__(
        self,
        strategy: Strategy,
        execution: ExecutionModel | None = None,
        initial_capital: float = 1_000_000.0,
        allow_fractional: bool = True,
    ) -> None:
        self.strategy = strategy
        self.execution = execution if execution is not None else LinearCostModel()
        self.portfolio = Portfolio(
            initial_capital=initial_capital, allow_fractional=allow_fractional
        )

    def run(self, data: DataSource) -> BacktestResult:
        self.strategy.reset()

        pending: deque[OrderEvent] = deque()
        equity_rows: list[tuple[pd.Timestamp, float]] = []
        position_rows: list[tuple[pd.Timestamp, dict[str, float]]] = []
        fills: list = []

        for bar in data:
            # 1. Fill orders queued at the previous bar at *this* bar's price.
            while pending:
                order = pending.popleft()
                fill_price = float(bar.prices.get(order.symbol, float("nan")))
                if not (fill_price == fill_price) or fill_price <= 0:
                    continue  # symbol untradable today; drop the order
                fill = self.execution.fill(order, fill_price)
                self.portfolio.apply_fill(fill)
                fills.append(fill)

            # 2. Mark the book to market.
            equity_rows.append((bar.timestamp, self.portfolio.equity(bar.prices)))
            position_rows.append((bar.timestamp, dict(self.portfolio.positions)))

            # 3. Ask the strategy for a new signal and queue resulting orders.
            event = MarketEvent(timestamp=bar.timestamp, prices=bar.prices)
            signal = self.strategy.on_bar(event)
            if signal is not None:
                for order in self.portfolio.orders_from_signal(signal, bar.prices):
                    pending.append(order)

        equity = pd.Series(
            [v for _, v in equity_rows],
            index=pd.DatetimeIndex([t for t, _ in equity_rows]),
            name="equity",
        )
        positions = pd.DataFrame(
            [p for _, p in position_rows],
            index=pd.DatetimeIndex([t for t, _ in position_rows]),
        ).fillna(0.0)

        return BacktestResult(
            equity=equity,
            positions=positions,
            fills=fills,
            initial_capital=self.portfolio.initial_capital,
        )
