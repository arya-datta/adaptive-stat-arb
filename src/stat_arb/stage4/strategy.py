r"""Regime-conditional pairs strategy (Stage 4).

Uses a fixed Stage-1 hedge (:class:`PairSpec`) to form the spread, then runs
an online Hamilton filter to infer the active regime each bar. Trading is
*conditional on the regime*:

* In a mean-reverting regime, trade the ±z rule using **that regime's**
  :math:`(\mu_s, \sigma_s)`.
* In a non-mean-reverting regime (or one excluded by the caller), **stand
  down** — flatten and wait. This is the core Stage 4 behaviour: don't fight
  a spread that has stopped reverting.

The strategy records its inferred-regime path so the caller can produce the
roadmap's *regime-conditional robustness* report (performance by regime).
"""

from __future__ import annotations

import numpy as np

from ..engine import MarketEvent, SignalEvent, Strategy
from ..stage1.pair import PairSpec
from .regime_switching import RegimeOUParams, OnlineHamiltonFilter


class RegimeSwitchingStrategy(Strategy):
    r"""±z trading gated by the inferred regime.

    Parameters
    ----------
    params:
        Fitted :class:`RegimeOUParams`.
    pair:
        :class:`PairSpec` for the spread and legs.
    entry_z, exit_z, stop_z:
        Z-score thresholds (computed with the *active regime's* moments).
    gross:
        Target gross exposure when in a position.
    tradeable_regimes:
        Indices of regimes in which to trade. Defaults to every
        mean-reverting regime (``params.mean_reverting``).
    min_confidence:
        Minimum filtered probability of the active regime required to act;
        below it the strategy treats the regime as ambiguous and stands down.
    """

    def __init__(
        self,
        params: RegimeOUParams,
        pair: PairSpec,
        entry_z: float = 1.5,
        exit_z: float = 0.0,
        stop_z: float | None = 4.0,
        gross: float = 1.0,
        tradeable_regimes: list[int] | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        if entry_z <= exit_z:
            raise ValueError("entry_z must be > exit_z.")
        self.params = params
        self.pair = pair
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.stop_z = float(stop_z) if stop_z is not None else None
        self.gross = float(gross)
        self.min_confidence = float(min_confidence)
        if tradeable_regimes is None:
            tradeable_regimes = [s for s in range(params.n_regimes)
                                 if params.mean_reverting[s]]
        self.tradeable = set(tradeable_regimes)

        # Per-regime stationary SD for the z-score (inf where not reverting).
        self._sd = np.where(
            params.mean_reverting,
            params.sigma / np.sqrt(2.0 * np.where(params.kappa > 0, params.kappa, np.nan)),
            np.inf,
        )

        self._filter = OnlineHamiltonFilter(params)
        self._position = 0
        self.regime_history: list[tuple] = []

    def reset(self) -> None:
        self._filter = OnlineHamiltonFilter(self.params)
        self._position = 0
        self.regime_history = []

    def on_bar(self, event: MarketEvent) -> SignalEvent | None:
        x = self.pair.spread(event.prices)
        if not np.isfinite(x):
            return None

        probs = self._filter.update(x)
        active = int(np.argmax(probs))
        confidence = float(probs[active])
        self.regime_history.append((event.timestamp, active, confidence))

        new_position = self._next_position(x, active, confidence)
        if new_position == self._position:
            return None
        self._position = new_position
        return SignalEvent(event.timestamp, self.pair.leg_weights(new_position, self.gross))

    # ------------------------------------------------------------------ #
    def _next_position(self, x: float, active: int, confidence: float) -> int:
        cur = self._position

        # Stand down: ambiguous regime or a non-tradeable (non-reverting) one.
        if confidence < self.min_confidence or active not in self.tradeable:
            return 0

        mu_s = self.params.mu[active]
        sd_s = self._sd[active]
        if not np.isfinite(sd_s) or sd_s <= 0:
            return 0
        z = (x - mu_s) / sd_s

        if self.stop_z is not None and abs(z) >= self.stop_z:
            return 0
        # Directional exit: close on reversion through the exit level.
        if cur == -1 and z <= self.exit_z:
            return 0
        if cur == +1 and z >= -self.exit_z:
            return 0
        if cur == 0:
            if z >= self.entry_z:
                return -1
            if z <= -self.entry_z:
                return +1
        return cur
