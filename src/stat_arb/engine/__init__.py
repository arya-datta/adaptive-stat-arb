"""Stage 0: modular event-driven backtest engine.

Components are kept deliberately separate so each can be swapped:

* :mod:`events`      — value objects flowing between components
* :mod:`strategy`    — signal generation (subclass :class:`Strategy`)
* :mod:`portfolio`   — converts target weights → orders
* :mod:`execution`   — order → fill, including the Stage 0 cost stub
* :mod:`accounting`  — equity curve, trade log, performance metrics
* :mod:`backtester`  — the event loop tying it all together

The signal → portfolio → execution → accounting chain is enforced by the
event loop in :class:`backtester.Backtester`; bypassing it (e.g. for a
vectorised PnL) is the kind of shortcut that creates the look-ahead bugs
the staged roadmap was built to prevent.
"""

from .events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from .strategy import Strategy
from .portfolio import Portfolio
from .execution import ExecutionModel, LinearCostModel, ZeroCostModel
from .accounting import BacktestResult
from .backtester import Backtester

__all__ = [
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "Strategy",
    "Portfolio",
    "ExecutionModel",
    "LinearCostModel",
    "ZeroCostModel",
    "BacktestResult",
    "Backtester",
]
