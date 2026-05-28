r"""Engle-Granger and Johansen cointegration screens.

Two complementary tests:

* **Engle-Granger (1987).** Regress :math:`Y_t = \alpha + \beta X_t + u_t`
  by OLS, then ADF the residual. Simple, asymmetric (swapping ``Y`` and
  ``X`` changes the answer), but the residual *is* the hedged spread —
  exactly what we want to feed the OU MLE.
* **Johansen (1988, 1991).** Eigen-analysis of a VECM. Symmetric, allows
  multiple cointegrating vectors, and is the basis of Stage 6's
  multivariate work. We use the *trace* statistic.

The two tests will sometimes disagree on short, noisy samples — that's
informative, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass(frozen=True)
class EngleGrangerResult:
    alpha: float
    beta: float
    spread: pd.Series
    coint_statistic: float
    pvalue: float
    cointegrated_at_5pct: bool

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        verdict = "COINTEGRATED" if self.cointegrated_at_5pct else "NOT cointegrated"
        return (
            f"Engle-Granger: beta={self.beta:.4f}, alpha={self.alpha:.4f}, "
            f"p={self.pvalue:.4f} -> {verdict}"
        )

    def to_pair_spec(self, y_symbol: str, x_symbol: str, use_log: bool = True):
        """Build a :class:`stat_arb.stage1.pair.PairSpec` from this fit."""
        from .pair import PairSpec

        return PairSpec(
            y_symbol=y_symbol, x_symbol=x_symbol,
            alpha=self.alpha, beta=self.beta, use_log=use_log,
        )


def engle_granger(y: pd.Series, x: pd.Series) -> EngleGrangerResult:
    """Engle-Granger two-step.

    Returns the hedge ratio and the residual spread (from OLS), plus the
    cointegration verdict. The p-value comes from
    :func:`statsmodels.tsa.stattools.coint`, which uses MacKinnon critical
    values appropriate for a *pre-estimated* cointegrating vector — plain
    ``adfuller`` on the residual over-rejects because it ignores that
    ``beta`` was fitted from the same data.
    """
    if len(y) != len(x):
        raise ValueError("y and x must have the same length.")
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()

    # OLS for the hedge ratio and the tradable residual spread.
    X = sm.add_constant(df["x"].values)
    model = sm.OLS(df["y"].values, X).fit()
    alpha, beta = float(model.params[0]), float(model.params[1])
    spread = df["y"] - alpha - beta * df["x"]
    spread.name = "spread"

    # Engle-Granger test with the correct (residual-based) critical values.
    stat, pvalue, _ = coint(df["y"].values, df["x"].values, trend="c", autolag="AIC")

    return EngleGrangerResult(
        alpha=alpha,
        beta=beta,
        spread=spread,
        coint_statistic=float(stat),
        pvalue=float(pvalue),
        cointegrated_at_5pct=pvalue < 0.05,
    )


@dataclass(frozen=True)
class JohansenResult:
    rank: int                     # cointegration rank at 5% level
    trace_statistics: np.ndarray
    trace_critical_values: np.ndarray   # shape (k, 3): 90/95/99%
    eigenvectors: np.ndarray            # columns = cointegrating vectors
    eigenvalues: np.ndarray


def johansen(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> JohansenResult:
    """Johansen trace test on a wide ``DataFrame`` of log-prices.

    Parameters
    ----------
    prices:
        Wide frame; each column is a series (e.g. log-prices of one asset).
    det_order:
        ``-1`` = no deterministic terms, ``0`` = constant (most common),
        ``1`` = constant + linear trend.
    k_ar_diff:
        Lag order in differences of the VECM. ``1`` is the usual default.
    """
    arr = prices.dropna().values
    if arr.shape[0] < 50:
        raise ValueError("Need ≥50 observations for a reliable Johansen test.")
    res = coint_johansen(arr, det_order=det_order, k_ar_diff=k_ar_diff)

    trace_stat = np.asarray(res.lr1)
    crit_vals = np.asarray(res.cvt)  # shape (k, 3)
    rank = int((trace_stat > crit_vals[:, 1]).sum())  # 5% critical column

    return JohansenResult(
        rank=rank,
        trace_statistics=trace_stat,
        trace_critical_values=crit_vals,
        eigenvectors=np.asarray(res.evec),
        eigenvalues=np.asarray(res.eig),
    )
