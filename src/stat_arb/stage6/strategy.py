r"""Cross-sectional eigenportfolio strategy (Stage 6).

By the fundamental law of active management (:math:`\mathrm{IR}\approx
\mathrm{IC}\sqrt{\mathrm{breadth}}`), many weakly-predictive residuals beat a
handful of pairs. This strategy, following Avellaneda-Lee, runs a rolling PCA
on the universe, computes each name's residual s-score, and holds a
dollar-neutral long-short book: long the cheap residuals (s < -entry), short
the rich ones (s > +entry), closing near s = 0.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from ..engine import MarketEvent, SignalEvent, Strategy
from .eigenportfolio import residual_sscores


class EigenportfolioStrategy(Strategy):
    r"""Rolling-PCA cross-sectional mean-reversion on idiosyncratic residuals.

    Parameters
    ----------
    symbols:
        Universe (column names in the market events).
    n_factors:
        PCA factors projected out before computing residuals.
    lookback:
        Estimation window (bars) for PCA + residual OU.
    recalc_every:
        Recompute s-scores every this many bars (caps per-bar cost).
    s_entry, s_close:
        Open a position when ``|s| > s_entry``; close when ``|s| < s_close``.
    gross:
        Total gross exposure (split equally across the long and short sides).
    """

    def __init__(
        self,
        symbols: list[str],
        n_factors: int = 3,
        lookback: int = 60,
        recalc_every: int = 5,
        s_entry: float = 1.25,
        s_close: float = 0.5,
        gross: float = 1.0,
    ) -> None:
        if s_entry <= s_close:
            raise ValueError("s_entry must be > s_close.")
        self.symbols = list(symbols)
        self.n_factors = int(n_factors)
        self.lookback = int(lookback)
        self.recalc_every = int(recalc_every)
        self.s_entry = float(s_entry)
        self.s_close = float(s_close)
        self.gross = float(gross)

        self._buf: deque[np.ndarray] = deque(maxlen=lookback)
        self._dir = {s: 0 for s in self.symbols}
        self._bar = 0
        self.explained_variance_history: list[float] = []

    def reset(self) -> None:
        self._buf = deque(maxlen=self.lookback)
        self._dir = {s: 0 for s in self.symbols}
        self._bar = 0
        self.explained_variance_history = []

    def on_bar(self, event: MarketEvent) -> SignalEvent | None:
        row = np.array([float(event.prices.get(s, np.nan)) for s in self.symbols])
        if np.any(~np.isfinite(row)) or np.any(row <= 0):
            return None
        self._buf.append(row)
        self._bar += 1

        if len(self._buf) < self.lookback or self._bar % self.recalc_every != 0:
            return None

        window = np.array(self._buf)                  # (lookback, N)
        scores = residual_sscores(window, n_factors=self.n_factors)
        self.explained_variance_history.append(scores.explained_variance)

        # Update per-name target direction with open/close bands.
        for i, sym in enumerate(self.symbols):
            s = scores.sscore[i]
            if not scores.reverting[i]:
                self._dir[sym] = 0
                continue
            cur = self._dir[sym]
            if abs(s) < self.s_close:
                self._dir[sym] = 0
            elif s > self.s_entry:
                self._dir[sym] = -1     # residual rich -> short
            elif s < -self.s_entry:
                self._dir[sym] = +1     # residual cheap -> long
            else:
                self._dir[sym] = cur

        return SignalEvent(event.timestamp, self._dollar_neutral_weights())

    # ------------------------------------------------------------------ #
    def _dollar_neutral_weights(self) -> dict[str, float]:
        longs = [s for s, d in self._dir.items() if d > 0]
        shorts = [s for s, d in self._dir.items() if d < 0]
        weights = {s: 0.0 for s in self.symbols}
        # Each side gets half the gross, equally weighted within the side.
        if longs:
            w = (self.gross / 2.0) / len(longs)
            for s in longs:
                weights[s] = w
        if shorts:
            w = (self.gross / 2.0) / len(shorts)
            for s in shorts:
                weights[s] = -w
        return weights
