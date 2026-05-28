"""Execution / cost models.

The Stage 0 model is deliberately crude (linear in notional: fixed bps fee
plus a half-spread paid on every trade). The point is that *no Sharpe is
ever cost-free* — even the trivial Stage 0 gate report is post-cost. The
realistic model (square-root market impact, Almgren-Chriss scheduling)
arrives in Stage 7.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .events import OrderEvent, FillEvent


class ExecutionModel(ABC):
    """Turn an :class:`OrderEvent` into a :class:`FillEvent`."""

    @abstractmethod
    def fill(self, order: OrderEvent, fill_price: float) -> FillEvent:
        """Produce a fill at ``fill_price`` (the next-bar reference price)."""


class ZeroCostModel(ExecutionModel):
    """Frictionless fills. Use only for the buy-and-hold gate test."""

    def fill(self, order: OrderEvent, fill_price: float) -> FillEvent:
        return FillEvent(
            timestamp=order.timestamp,
            symbol=order.symbol,
            quantity=order.quantity,
            price=fill_price,
            commission=0.0,
        )


class LinearCostModel(ExecutionModel):
    """Fixed-bps commission + half-spread slippage.

    Parameters
    ----------
    bps:
        Round-trip-equivalent commission, charged on **notional** of each
        side. ``5`` means 5 bps per side.
    half_spread_bps:
        Half the bid-ask spread, paid on every trade. Buys pay
        ``fill_price * (1 + half_spread_bps/1e4)``; sells receive
        ``fill_price * (1 - half_spread_bps/1e4)``.
    """

    def __init__(self, bps: float = 5.0, half_spread_bps: float = 2.0) -> None:
        if bps < 0 or half_spread_bps < 0:
            raise ValueError("bps and half_spread_bps must be non-negative.")
        self.bps = bps
        self.half_spread_bps = half_spread_bps

    def fill(self, order: OrderEvent, fill_price: float) -> FillEvent:
        side = 1.0 if order.quantity > 0 else -1.0
        slip = self.half_spread_bps / 1e4
        executed_price = fill_price * (1.0 + side * slip)
        commission = abs(order.quantity) * executed_price * (self.bps / 1e4)
        return FillEvent(
            timestamp=order.timestamp,
            symbol=order.symbol,
            quantity=order.quantity,
            price=executed_price,
            commission=commission,
        )
