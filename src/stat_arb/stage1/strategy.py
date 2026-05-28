r"""±z baseline pairs strategy on an OU spread.

The strategy observes the two legs of a cointegrated pair, reconstructs
the spread :math:`X_t` via a :class:`~stat_arb.stage1.pair.PairSpec`,
forms the z-score :math:`z_t = (X_t - \mu)/\sigma_{\text{ss}}`, and trades
the legs to take a long/short *spread* position:

* :math:`z_t \le -e` → spread is cheap → **long the spread** (long Y, short X).
* :math:`z_t \ge +e` → spread is rich → **short the spread**.
* :math:`|z_t| \le x` → flat.

This is *deliberately naive* — its job is to be the documented benchmark
that the Stage 2 optimal-stopping rule must beat after costs and DSR
adjustment. Where Stage 2 derives the boundaries from the SDE, Stage 1
just sets them by convention.
"""

from __future__ import annotations

import numpy as np

from ..engine import MarketEvent, SignalEvent, Strategy
from .ou_mle import OUParams
from .pair import PairSpec


class ZScoreStrategy(Strategy):
    """Symmetric ±z entry, mean exit, optional stop on extreme z.

    Parameters
    ----------
    params:
        Fitted :class:`OUParams` for the spread. ``mu`` and the stationary
        SD :math:`\\sigma/\\sqrt{2\\kappa}` define the z-score.
    pair:
        :class:`PairSpec` describing the spread and the two legs to trade.
    entry_z, exit_z:
        Absolute z-scores to enter / flatten (``exit_z`` defaults to 0).
    stop_z:
        Optional hard stop in z-score units (``None`` disables).
    gross:
        Target gross exposure (``|w_Y| + |w_X|``) when in a position.
    """

    def __init__(
        self,
        params: OUParams,
        pair: PairSpec,
        entry_z: float = 1.5,
        exit_z: float = 0.0,
        stop_z: float | None = None,
        gross: float = 1.0,
    ) -> None:
        if entry_z <= exit_z:
            raise ValueError("entry_z must be > exit_z.")
        if stop_z is not None and stop_z <= entry_z:
            raise ValueError("stop_z must be > entry_z (or None to disable).")

        self.params = params
        self.pair = pair
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.stop_z = float(stop_z) if stop_z is not None else None
        self.gross = float(gross)

        self._sd = params.sigma / np.sqrt(2.0 * params.kappa)
        self._position: int = 0  # +1 long spread, 0 flat, -1 short spread

    def reset(self) -> None:
        self._position = 0

    def on_bar(self, event: MarketEvent) -> SignalEvent | None:
        spread = self.pair.spread(event.prices)
        if not np.isfinite(spread):
            return None

        z = (spread - self.params.mu) / self._sd
        new_position = self._next_position(z)
        if new_position == self._position:
            return None

        self._position = new_position
        return SignalEvent(
            timestamp=event.timestamp,
            target_weights=self.pair.leg_weights(new_position, self.gross),
        )

    # ------------------------------------------------------------------ #
    def _next_position(self, z: float) -> int:
        """Map the current z-score to a desired spread position direction."""
        cur = self._position

        if self.stop_z is not None and abs(z) >= self.stop_z:
            return 0
        if cur != 0 and abs(z) <= self.exit_z:
            return 0
        if cur == 0:
            if z >= self.entry_z:
                return -1  # rich → short the spread
            if z <= -self.entry_z:
                return +1  # cheap → long the spread
        return cur
