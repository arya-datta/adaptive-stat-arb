"""CSV file → :class:`DataSource` adapter.

Expects either:

* a single wide CSV with a date column and one column per symbol, or
* a directory containing one ``<symbol>.csv`` per name, each with a
  ``date,adj_close`` schema (yfinance-style).

The loader applies no extra adjustment: it trusts the file. Point-in-time
correctness is the caller's responsibility for CSV-supplied data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import DataSource, PriceFrame


class CSVDataSource(DataSource):
    """Load prices from CSV.

    Parameters
    ----------
    path:
        Either a file path (wide CSV) or a directory (one CSV per symbol).
    date_col:
        Name of the date column (default ``"date"``).
    price_col:
        Used only in per-symbol mode (default ``"adj_close"``).
    """

    def __init__(
        self,
        path: str | Path,
        date_col: str = "date",
        price_col: str = "adj_close",
    ) -> None:
        self.path = Path(path)
        self.date_col = date_col
        self.price_col = price_col
        self._frame: PriceFrame | None = None

    def frame(self) -> PriceFrame:
        if self._frame is not None:
            return self._frame

        if self.path.is_dir():
            self._frame = self._load_directory()
        else:
            self._frame = self._load_wide_file()

        self._frame = self._frame.sort_index().ffill()
        return self._frame

    def _load_wide_file(self) -> PriceFrame:
        df = pd.read_csv(self.path)
        if self.date_col not in df.columns:
            raise KeyError(f"Date column {self.date_col!r} not found in {self.path}.")
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        return df.set_index(self.date_col)

    def _load_directory(self) -> PriceFrame:
        series = {}
        for csv_path in sorted(self.path.glob("*.csv")):
            symbol = csv_path.stem
            df = pd.read_csv(csv_path)
            if self.date_col not in df.columns or self.price_col not in df.columns:
                raise KeyError(
                    f"{csv_path}: expected columns {self.date_col!r} and "
                    f"{self.price_col!r}; got {list(df.columns)}."
                )
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            series[symbol] = df.set_index(self.date_col)[self.price_col]
        if not series:
            raise FileNotFoundError(f"No CSV files in {self.path}.")
        return pd.DataFrame(series)
