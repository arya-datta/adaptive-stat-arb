"""Accounting: equity curve, trade log, performance summary.

Lightweight on purpose — heavier statistical reporting (Deflated Sharpe,
PBO, etc.) lives in :mod:`stat_arb.validation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..utils import BUSINESS_DAYS_PER_YEAR
from .events import FillEvent


@dataclass
class BacktestResult:
    """Output of :meth:`stat_arb.engine.Backtester.run`."""

    equity: pd.Series                       # mark-to-market equity, one point per bar
    positions: pd.DataFrame                  # symbol-wise share counts, one row per bar
    fills: list[FillEvent] = field(default_factory=list)
    initial_capital: float = 1_000_000.0

    # ------------------------------------------------------------------ #
    # Derived series                                                     #
    # ------------------------------------------------------------------ #
    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)

    @property
    def log_returns(self) -> pd.Series:
        return np.log(self.equity / self.equity.shift(1)).fillna(0.0)

    # ------------------------------------------------------------------ #
    # Summary metrics                                                    #
    # ------------------------------------------------------------------ #
    def summary(self, periods_per_year: int = BUSINESS_DAYS_PER_YEAR) -> dict:
        """Return a dict of headline metrics.

        ``sharpe`` here is the *uncorrected* annualised ratio; honest CIs
        require :func:`stat_arb.validation.sharpe.sharpe_ci_lo` and the
        Deflated Sharpe correction in :mod:`stat_arb.validation.deflated_sharpe`.
        """
        r = self.returns
        if r.std(ddof=1) == 0:
            sharpe = float("nan")
        else:
            sharpe = r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)

        total_return = self.equity.iloc[-1] / self.equity.iloc[0] - 1.0
        years = len(r) / periods_per_year if periods_per_year else 1.0
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
        ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
        max_dd = self._max_drawdown(self.equity)
        turnover = self._annualised_turnover(periods_per_year)
        total_commission = sum(f.commission for f in self.fills)

        return {
            "total_return":     float(total_return),
            "cagr":             float(cagr),
            "ann_vol":          float(ann_vol),
            "sharpe":           float(sharpe),
            "max_drawdown":     float(max_dd),
            "turnover_annual":  float(turnover),
            "num_trades":       len(self.fills),
            "total_commission": float(total_commission),
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        return float(drawdown.min())

    def _annualised_turnover(self, periods_per_year: int) -> float:
        """Sum of |notional| traded, divided by mean equity and rescaled."""
        if not self.fills or len(self.equity) < 2:
            return 0.0
        traded_notional = sum(abs(f.quantity * f.price) for f in self.fills)
        mean_equity = float(self.equity.mean())
        if mean_equity == 0:
            return float("nan")
        years = len(self.equity) / periods_per_year
        return traded_notional / mean_equity / max(years, 1e-9)
