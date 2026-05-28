"""Event value objects.

Four event types flow through the engine in this fixed order:

    MarketEvent → SignalEvent → OrderEvent → FillEvent

Each step is intentionally a separate object so look-ahead leakage can
only happen by an explicit timestamp violation — which the engine asserts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """A new bar of prices."""

    timestamp: pd.Timestamp
    prices: pd.Series  # index=symbol, value=price


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """The strategy's requested portfolio composition for the *next* bar.

    Weights are fractions of total equity, signed (negative = short).
    They do not need to sum to 1; the engine treats the remainder as cash.
    """

    timestamp: pd.Timestamp
    target_weights: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """An instruction to trade ``quantity`` shares (signed) of ``symbol``.

    Created by the portfolio when it reconciles current vs target weights.
    Held for one bar before fill — preventing the strategy from acting on a
    price it has already used to make its decision.
    """

    timestamp: pd.Timestamp
    symbol: str
    quantity: float


@dataclass(frozen=True, slots=True)
class FillEvent:
    """The execution model's response to an :class:`OrderEvent`."""

    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    price: float       # the price actually paid (already includes half-spread)
    commission: float  # additional fixed/proportional fee in account currency
