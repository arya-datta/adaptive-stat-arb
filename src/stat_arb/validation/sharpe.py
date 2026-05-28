r"""Sharpe ratio with the Lo (2002) autocorrelation correction.

Lo, A. W. (2002). *The Statistics of Sharpe Ratios.* Financial Analysts
Journal, 58(4), 36-52.

For iid returns with sample size :math:`T`,

.. math::

   \mathrm{SE}(\widehat{SR}) \approx
     \sqrt{\frac{1 + \tfrac{1}{2}\widehat{SR}^2}{T}}.

With autocorrelation, scale by :math:`\sqrt{\eta(q)}` where

.. math::

   \eta(q) = 1 + 2\sum_{k=1}^{q}\bigl(1 - k/q\bigr)\rho_k,

and :math:`\rho_k` is the lag-k autocorrelation of returns. ``q`` is a
truncation lag (Newey-West-style); we default to
:math:`\lfloor 4(T/100)^{2/9} \rfloor` per Andrews (1991).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..utils import BUSINESS_DAYS_PER_YEAR


def _clean(returns: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 30:
        raise ValueError(f"Need ≥30 returns for Sharpe inference; got {arr.size}.")
    return arr


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free: float = 0.0,
    periods_per_year: int = BUSINESS_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio. ``risk_free`` is per-period."""
    r = _clean(returns) - risk_free
    std = r.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(r.mean() / std * np.sqrt(periods_per_year))


def _autocorr_factor(r: np.ndarray, q: int | None = None) -> float:
    T = r.size
    if q is None:
        q = max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    q = min(q, T - 2)

    centred = r - r.mean()
    denom = float(np.dot(centred, centred))
    if denom == 0:
        return 1.0

    factor = 1.0
    for k in range(1, q + 1):
        rho = float(np.dot(centred[:-k], centred[k:]) / denom)
        factor += 2.0 * (1.0 - k / q) * rho
    return max(factor, 1e-6)  # guard against numerical negatives


def sharpe_se_lo(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = BUSINESS_DAYS_PER_YEAR,
    q: int | None = None,
) -> float:
    """Lo (2002) standard error of the *annualised* Sharpe ratio."""
    r = _clean(returns)
    sr = sharpe_ratio(r, periods_per_year=periods_per_year)
    if not np.isfinite(sr):
        return float("nan")

    # Convert annualised SR to per-period scale for the variance formula.
    sr_period = sr / np.sqrt(periods_per_year)
    iid_var = (1.0 + 0.5 * sr_period**2) / r.size
    eta = _autocorr_factor(r, q=q)
    se_period = float(np.sqrt(eta * iid_var))
    return se_period * np.sqrt(periods_per_year)


def sharpe_ci_lo(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
    periods_per_year: int = BUSINESS_DAYS_PER_YEAR,
    q: int | None = None,
) -> tuple[float, float, float]:
    """Return ``(SR, lower, upper)`` at the requested confidence level.

    The Lo SE is asymptotically normal, so the interval is symmetric.
    """
    sr = sharpe_ratio(returns, periods_per_year=periods_per_year)
    se = sharpe_se_lo(returns, periods_per_year=periods_per_year, q=q)
    z = float(norm.ppf(0.5 + confidence / 2.0))
    return float(sr), float(sr - z * se), float(sr + z * se)
