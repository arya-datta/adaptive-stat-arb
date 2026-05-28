"""ADF and KPSS wrappers — paired by design.

ADF rejects unit-root (null: non-stationary). KPSS *fails to reject*
stationarity (null: stationary). Used together they discriminate:

    ADF rejects + KPSS doesn't reject → stationary (good)
    ADF rejects + KPSS rejects       → fractionally integrated / borderline
    ADF doesn't  + KPSS doesn't     → too little data to tell
    ADF doesn't  + KPSS rejects     → non-stationary (do not trade as a spread)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass(frozen=True)
class StationarityResult:
    statistic: float
    pvalue: float
    n_lags: int
    n_obs: int
    critical_values: dict[str, float]
    stationary_at_5pct: bool

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        verdict = "stationary" if self.stationary_at_5pct else "non-stationary"
        return (
            f"{self.__class__.__name__}: stat={self.statistic:.3f}, "
            f"p={self.pvalue:.4f}, lags={self.n_lags}, n={self.n_obs} -> {verdict}"
        )


def _to_array(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 20:
        raise ValueError("Need at least 20 finite observations for ADF/KPSS.")
    return arr


def adf_test(x: pd.Series | np.ndarray, regression: str = "c") -> StationarityResult:
    """Augmented Dickey-Fuller test (H0: unit root)."""
    arr = _to_array(x)
    stat, pvalue, n_lags, n_obs, crit, _ = adfuller(arr, regression=regression, autolag="AIC")
    return StationarityResult(
        statistic=float(stat),
        pvalue=float(pvalue),
        n_lags=int(n_lags),
        n_obs=int(n_obs),
        critical_values={k: float(v) for k, v in crit.items()},
        stationary_at_5pct=pvalue < 0.05,  # reject null = stationary
    )


def kpss_test(x: pd.Series | np.ndarray, regression: str = "c") -> StationarityResult:
    """KPSS test (H0: stationary). We flip the verdict accordingly."""
    arr = _to_array(x)
    stat, pvalue, n_lags, crit = kpss(arr, regression=regression, nlags="auto")
    return StationarityResult(
        statistic=float(stat),
        pvalue=float(pvalue),
        n_lags=int(n_lags),
        n_obs=int(arr.size),
        critical_values={k: float(v) for k, v in crit.items()},
        stationary_at_5pct=pvalue >= 0.05,  # *fail* to reject null = stationary
    )
