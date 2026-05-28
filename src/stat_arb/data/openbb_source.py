"""OpenBB → :class:`DataSource` adapter.

OpenBB's Python SDK exposes a multi-provider equity endpoint:
``openbb.equity.price.historical(symbol, start, end, provider=...)``.
Using OpenBB means the project gains free access to Polygon / FMP /
AlphaVantage etc. without rewriting the loader — only the provider string
changes.

This module is optional; install with ``pip install -e ".[openbb]"``.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .base import DataSource, PriceFrame


class OpenBBDataSource(DataSource):
    """Fetch daily adjusted-close prices via OpenBB's equity endpoint.

    Parameters
    ----------
    symbols:
        Iterable of tickers.
    start, end:
        ISO date strings.
    provider:
        OpenBB provider name (``"yfinance"``, ``"fmp"``, ``"polygon"``, ...).
        Defaults to ``"yfinance"`` to match :class:`YFinanceDataSource` while
        going through OpenBB's normalised schema.
    """

    def __init__(
        self,
        symbols: Iterable[str],
        start: str,
        end: str,
        provider: str = "yfinance",
    ) -> None:
        self._symbols = list(symbols)
        self.start = start
        self.end = end
        self.provider = provider
        self._frame: PriceFrame | None = None

    def frame(self) -> PriceFrame:
        if self._frame is not None:
            return self._frame

        from openbb import obb  # local import keeps the dep optional

        series = {}
        for symbol in self._symbols:
            response = obb.equity.price.historical(
                symbol=symbol,
                start_date=self.start,
                end_date=self.end,
                provider=self.provider,
            )
            df = response.to_df()
            # OpenBB normalises the close column name across providers.
            close_col = "close" if "close" in df.columns else df.columns[0]
            series[symbol] = df[close_col]

        out = pd.DataFrame(series)
        out.index = pd.to_datetime(out.index)
        self._frame = out.sort_index().dropna(how="all")
        return self._frame
