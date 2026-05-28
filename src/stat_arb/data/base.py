"""Common data abstractions.

A :class:`DataSource` is the minimum the backtest engine needs: an iterable
of ``Bar`` events in chronological order plus the raw frame for batch
estimation. Concrete sources (CSV, yfinance, synthetic) live in sibling
modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import pandas as pd

PriceFrame = pd.DataFrame  # alias for readability; long×wide DataFrame


@dataclass(frozen=True, slots=True)
class Bar:
    """A single-timestamp price observation across the universe.

    Attributes
    ----------
    timestamp:
        Bar end time.
    prices:
        Mapping symbol → adjusted price. ``NaN`` is allowed and signals
        "no quote at this timestamp" (e.g. pre-IPO).
    """

    timestamp: pd.Timestamp
    prices: pd.Series

    def __getitem__(self, symbol: str) -> float:
        return float(self.prices[symbol])


class DataSource(ABC):
    """Abstract base class. Implementations expose a frame plus an iterator.

    The contract is intentionally thin:

    * :meth:`frame` returns the full point-in-time-adjusted history.
    * :meth:`__iter__` yields ``Bar`` instances in chronological order.

    Both methods must agree: ``iter`` is just a row-wise view of ``frame``.
    """

    @abstractmethod
    def frame(self) -> PriceFrame:
        """Return the price history. Index must be a sorted ``DatetimeIndex``."""

    def __iter__(self) -> Iterator[Bar]:
        df = self.frame()
        if not df.index.is_monotonic_increasing:
            raise ValueError("DataSource frame must be sorted by timestamp.")
        for ts, row in df.iterrows():
            yield Bar(timestamp=ts, prices=row)

    @property
    def symbols(self) -> list[str]:
        return list(self.frame().columns)

    def __len__(self) -> int:
        return len(self.frame())
