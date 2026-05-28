r"""Vector Error-Correction Model for multi-asset cointegration.

The Johansen test from Stage 1 identifies the cointegration *rank*; the VECM
estimates the cointegrating vectors :math:`\beta` (the stationary linear
combinations) and the error-correction speeds :math:`\alpha` (how fast each
asset adjusts back). A cointegrating vector applied to the level prices yields
a multi-asset stationary spread that the OU / optimal-stopping machinery can
trade — the multi-asset generalisation of the Stage 1 pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM

from ..stage1.cointegration import johansen


@dataclass
class VECMResult:
    beta: np.ndarray             # (N, rank) cointegrating vectors
    alpha: np.ndarray            # (N, rank) error-correction loadings
    rank: int
    columns: list[str]

    def spread(self, prices: pd.DataFrame, which: int = 0) -> pd.Series:
        """Project prices onto the ``which``-th cointegrating vector."""
        b = self.beta[:, which]
        s = prices[self.columns].to_numpy() @ b
        return pd.Series(s, index=prices.index, name=f"vecm_spread_{which}")


def fit_vecm(
    prices: pd.DataFrame,
    k_ar_diff: int = 1,
    det_order: int = 0,
    coint_rank: int | None = None,
) -> VECMResult:
    """Fit a VECM, auto-selecting the cointegration rank via Johansen if needed.

    Parameters
    ----------
    prices:
        Wide frame of (log-)prices; one column per asset.
    k_ar_diff:
        Lags in differences.
    det_order:
        Deterministic-term order passed through to Johansen / VECM.
    coint_rank:
        Cointegration rank. If ``None``, taken from the Johansen trace test.
    """
    cols = list(prices.columns)
    df = prices.dropna()
    if coint_rank is None:
        coint_rank = max(johansen(df, det_order=det_order, k_ar_diff=k_ar_diff).rank, 1)

    model = VECM(df.values, k_ar_diff=k_ar_diff, coint_rank=coint_rank,
                 deterministic="ci" if det_order == 0 else "co")
    res = model.fit()
    return VECMResult(
        beta=np.asarray(res.beta),
        alpha=np.asarray(res.alpha),
        rank=int(coint_rank),
        columns=cols,
    )
