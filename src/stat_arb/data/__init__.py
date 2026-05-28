"""Data sources for prices and synthetic OU spreads.

All loaders return a ``pandas.DataFrame`` indexed by ``DatetimeIndex`` with
one column per symbol. Adjustments for splits/dividends are applied by the
real-data loaders; ``SyntheticOU`` returns a one-column ``spread`` frame.

The split between loaders is deliberate: the engine only consumes the
``DataSource`` interface, so swapping yfinance for CSV or OpenBB is a single
constructor change.
"""

from .base import DataSource, Bar, PriceFrame
from .synthetic import (
    SyntheticOU, SyntheticPair, SyntheticMarkovOU, SyntheticFactorMarket,
)
from .csv_source import CSVDataSource

# Optional sources — fail soft so the package still imports without them.
try:  # pragma: no cover - import-time only
    from .yfinance_source import YFinanceDataSource
except ImportError:  # pragma: no cover
    YFinanceDataSource = None  # type: ignore[assignment]

try:  # pragma: no cover
    from .openbb_source import OpenBBDataSource
except ImportError:  # pragma: no cover
    OpenBBDataSource = None  # type: ignore[assignment]

__all__ = [
    "DataSource",
    "Bar",
    "PriceFrame",
    "SyntheticOU",
    "SyntheticPair",
    "SyntheticMarkovOU",
    "SyntheticFactorMarket",
    "CSVDataSource",
    "YFinanceDataSource",
    "OpenBBDataSource",
]
