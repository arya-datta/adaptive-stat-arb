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


class InMemorySource(DataSource):
    """Wrap an in-memory price :class:`~pandas.DataFrame` as a ``DataSource``.

    The canonical adapter for backtesting a frame you already hold (synthetic
    data, a slice of a larger history, or a constructed spread). Replaces the
    ad-hoc one-off ``FrameSource`` shims that otherwise get re-declared in
    every example and test.
    """

    def __init__(self, frame: PriceFrame) -> None:
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.copy()
            frame.index = pd.to_datetime(frame.index)
        self._frame = frame

    def frame(self) -> PriceFrame:
        return self._frame
