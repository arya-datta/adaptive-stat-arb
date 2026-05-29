"""Time-step inference and calendar constants.

Stat-arb on equity data conventionally uses *trading-day* time:
``dt = 1/252``. Calendar time (``1/365.25``) shifts OU half-lives by ~30%
and gives misleading mean-reversion speeds, so we snap to the trading-day
convention when the index looks daily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BUSINESS_DAYS_PER_YEAR = 252


def infer_dt(index: pd.Index) -> float:
    """Infer the time step (in years) from a ``DatetimeIndex``.

    Detection rule:

    * Median spacing 0.5-3 days → daily data → ``1/252``
    * Median spacing 5-9 days  → weekly data → ``1/52``
    * Median spacing 25-35 days → monthly → ``1/12``
    * Otherwise → calendar fallback: ``median_days / 365.25``

    Raises
    ------
    ValueError
        If ``index`` is not a ``DatetimeIndex`` or has fewer than two points.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("infer_dt requires a pandas DatetimeIndex.")
    if len(index) < 2:
        raise ValueError("Need at least two timestamps to infer dt.")

    deltas_days = np.diff(index.values).astype("timedelta64[s]").astype(float) / 86400.0
    median_days = float(np.median(deltas_days))

    if 0.5 <= median_days <= 3.0:
        return 1.0 / BUSINESS_DAYS_PER_YEAR
    if 5.0 <= median_days <= 9.0:
        return 1.0 / 52.0
    if 25.0 <= median_days <= 35.0:
        return 1.0 / 12.0

    # No recognised calendar: fall back to calendar time, but warn — an
    # irregular index silently mis-scales every downstream kappa/half-life.
    import warnings
    warnings.warn(
        f"infer_dt: median spacing {median_days:.2f} days matches no standard "
        "calendar (daily/weekly/monthly); falling back to calendar-time "
        "dt = median_days/365.25. Pass dt explicitly if this is wrong.",
        stacklevel=2,
    )
    return median_days / 365.25
