"""Pair specification: hedge ratio + spread definition + leg weights.

A cointegration residual is a *dimensionless, zero-centred* quantity, not
a tradable price. You realise a position in it by trading the two legs:
long the spread = long ``Y`` and short ``beta`` units of ``X``. This class
centralises that bookkeeping so both the Stage 1 (±z) and Stage 2
(optimal-stopping) strategies share one definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairSpec:
    r"""Defines spread ``= f(Y) - alpha - beta * f(X)`` and the trade that holds it.

    Parameters
    ----------
    y_symbol, x_symbol:
        Column names of the two legs in incoming price series.
    alpha, beta:
        Cointegration intercept and hedge ratio (from Engle-Granger).
    use_log:
        If ``True`` (default), the spread is on log-prices — the usual
        convention for equities, so the hedge is return-stationary.
    """

    y_symbol: str
    x_symbol: str
    alpha: float
    beta: float
    use_log: bool = True

    def spread(self, prices: pd.Series) -> float:
        """Compute the scalar spread value from a bar's price vector."""
        py = float(prices[self.y_symbol])
        px = float(prices[self.x_symbol])
        if self.use_log:
            if py <= 0 or px <= 0:
                return float("nan")
            return np.log(py) - self.alpha - self.beta * np.log(px)
        return py - self.alpha - self.beta * px

    def leg_weights(self, direction: int, gross: float) -> dict[str, float]:
        r"""Map a signed spread position to dollar weights on the two legs.

        ``direction = +1`` means *long the spread* (bet it rises): long
        ``Y``, short ``X``. Weights are scaled so the gross exposure
        (``|w_Y| + |w_X|``) equals ``gross``; the short leg carries the
        ``beta`` hedge so the portfolio return tracks the spread change.
        """
        if direction == 0:
            return {self.y_symbol: 0.0, self.x_symbol: 0.0}
        g = gross / (1.0 + abs(self.beta))
        return {
            self.y_symbol: direction * g,
            self.x_symbol: -direction * self.beta * g,
        }
