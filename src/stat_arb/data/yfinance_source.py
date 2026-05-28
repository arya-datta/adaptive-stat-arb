"""yfinance → :class:`DataSource` adapter.

Returns **adjusted close** prices (splits + dividends) so backtests are
free of corporate-action artefacts. We accept the survivorship-bias caveat
that yfinance's universe is the *current* listing, not point-in-time — for
serious research use a paid survivorship-bias-free vendor (CRSP, Norgate)
plugged in via :class:`CSVDataSource`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .base import DataSource, PriceFrame


class YFinanceDataSource(DataSource):
    """Fetch daily adjusted-close prices from Yahoo Finance.

    Parameters
    ----------
    symbols:
        Iterable of tickers.
    start, end:
        Date strings (inclusive start, exclusive end per yfinance).
    cache_dir:
        Optional on-disk cache. Re-runs with the same ``(symbol, start, end)``
        hit the parquet cache and skip the network — crucial for iterative
        backtests and offline reproducibility.
    """

    def __init__(
        self,
        symbols: Iterable[str],
        start: str,
        end: str,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._symbols = list(symbols)
        self.start = start
        self.end = end
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._frame: PriceFrame | None = None

    def frame(self) -> PriceFrame:
        if self._frame is not None:
            return self._frame
        cache_path = self._cache_path()
        if cache_path and cache_path.exists():
            self._frame = pd.read_parquet(cache_path)
            return self._frame

        import yfinance as yf  # local import keeps the dep optional

        raw = yf.download(
            self._symbols,
            start=self.start,
            end=self.end,
            auto_adjust=True,      # apply splits + dividends in place
            progress=False,
            group_by="ticker",
        )

        # yfinance returns wide multi-index when >1 ticker; flatten to close
        if isinstance(raw.columns, pd.MultiIndex):
            closes = pd.DataFrame(
                {sym: raw[sym]["Close"] for sym in self._symbols if sym in raw.columns.levels[0]}
            )
        else:
            closes = raw[["Close"]].rename(columns={"Close": self._symbols[0]})

        closes.index = pd.to_datetime(closes.index)
        closes = closes.sort_index().dropna(how="all")

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            closes.to_parquet(cache_path)

        self._frame = closes
        return self._frame

    def _cache_path(self) -> Path | None:
        if not self.cache_dir:
            return None
        key = "_".join(sorted(self._symbols)) + f"_{self.start}_{self.end}.parquet"
        return self.cache_dir / key
