"""Portfolio: positions, cash, and signal → order translation.

The portfolio holds the canonical state — cash and per-symbol share counts
— and is the only component that may produce :class:`OrderEvent`\\s. It
reconciles the strategy's *requested* weights against actuals using the
mark-to-market equity, integerises the resulting deltas (fractional shares
allowed for simplicity; toggleable for equities), and emits one
:class:`OrderEvent` per symbol that needs adjusting.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import pandas as pd

from .events import SignalEvent, OrderEvent, FillEvent


class Portfolio:
    """Track cash and positions; translate target weights into orders.

    Parameters
    ----------
    initial_capital:
        Starting cash in account currency.
    allow_fractional:
        If False, share quantities are floor-rounded to an integer. The
        residual stays as cash so equity is conserved.
    weight_tol:
        Minimum absolute weight change required to trigger a trade. Avoids
        thrashing on numerically tiny rebalances.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        allow_fractional: bool = True,
        weight_tol: float = 1e-4,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.allow_fractional = allow_fractional
        self.weight_tol = float(weight_tol)

        self.cash: float = float(initial_capital)
        self.positions: dict[str, float] = defaultdict(float)
        # Last finite, positive price seen per symbol — used to value a position
        # on a bar where its quote is missing/NaN (a halt or data gap) instead of
        # marking it to zero, which would fabricate a drawdown-and-recovery.
        self._last_prices: dict[str, float] = {}

    def equity(self, prices: pd.Series) -> float:
        """Mark-to-market equity using ``prices`` (symbol → price).

        Symbols absent or non-finite in ``prices`` are valued at their last
        known good price (carried forward); only if a symbol has *never* been
        priced is it treated as zero.
        """
        # Refresh the last-good-price cache from this bar's valid quotes.
        for sym, px in prices.items():
            fpx = float(px)
            if fpx == fpx and fpx > 0:        # finite and positive
                self._last_prices[sym] = fpx

        mtm = 0.0
        for sym, qty in self.positions.items():
            px = float(prices.get(sym, float("nan")))
            if not (px == px) or px <= 0:     # missing/NaN/non-positive this bar
                px = self._last_prices.get(sym, 0.0)
            mtm += qty * px
        return self.cash + mtm

    def apply_fill(self, fill: FillEvent) -> None:
        """Update cash and positions for a :class:`FillEvent`."""
        notional = fill.quantity * fill.price
        self.cash -= notional + fill.commission
        self.positions[fill.symbol] += fill.quantity

    def orders_from_signal(
        self,
        signal: SignalEvent,
        prices: pd.Series,
    ) -> Iterator[OrderEvent]:
        """Yield orders that move current weights toward ``signal.target_weights``.

        Orders are timestamped with ``signal.timestamp`` (the bar the
        decision was made on); the backtester delays fills to the *next*
        bar's price.

        Note on exposure drift: a position is sized once, when a target weight
        changes, and is *not* re-balanced to a constant weight every bar. So a
        strategy's nominal ``gross`` holds at entry but drifts with price until
        the next signal — realistic for low-turnover books, but it means
        ``gross`` is a target at trade time, not a per-bar invariant.
        """
        equity = self.equity(prices)
        if equity <= 0:
            return  # bankrupt; emit no further trades

        for symbol, weight in signal.target_weights.items():
            price = float(prices.get(symbol, float("nan")))
            if price <= 0 or not (price == price):  # NaN check
                continue

            target_qty = weight * equity / price
            if not self.allow_fractional:
                target_qty = float(int(target_qty))  # truncate toward zero

            delta = target_qty - self.positions.get(symbol, 0.0)
            delta_weight = delta * price / equity
            if abs(delta_weight) < self.weight_tol:
                continue

            yield OrderEvent(
                timestamp=signal.timestamp,
                symbol=symbol,
                quantity=delta,
            )

        # Also flatten positions that the signal omits (means "hold none").
        held = set(self.positions) - set(signal.target_weights)
        for symbol in held:
            qty = self.positions[symbol]
            if abs(qty) < 1e-12:
                continue
            price = float(prices.get(symbol, float("nan")))
            if price <= 0 or not (price == price):
                continue
            if abs(qty * price / equity) < self.weight_tol:
                continue
            yield OrderEvent(
                timestamp=signal.timestamp,
                symbol=symbol,
                quantity=-qty,
            )
