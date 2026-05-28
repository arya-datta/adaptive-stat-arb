r"""Pairs strategy that consumes Leung-Li boundaries.

Distinct from :class:`stat_arb.stage1.ZScoreStrategy` in two ways:

1. The thresholds are *derived* from the OU SDE, transaction cost, and
   discount rate — no ``entry_z`` knob to tune.
2. There is no enforced symmetry around :math:`\mu`. Costs are explicit,
   so the long and short sides have their own thresholds.

Like the Stage 1 baseline, it trades the two legs of the pair (via a
:class:`~stat_arb.stage1.pair.PairSpec`) rather than a fictitious
"spread instrument".

The gate (per the PDF, Stage 2): show this strategy beats the naive ±z
baseline *after costs and after the DSR adjustment*, or honestly report
that it does not for this particular spread.
"""

from __future__ import annotations

import numpy as np

from ..engine import MarketEvent, SignalEvent, Strategy
from ..stage1.ou_mle import OUParams
from ..stage1.pair import PairSpec
from .optimal_stopping import OptimalStoppingBoundaries, compute_boundaries


class OptimalStoppingStrategy(Strategy):
    """Long if spread ≤ d*; exit ≥ b*. Symmetric short side optional.

    Parameters
    ----------
    params:
        Fitted :class:`OUParams`.
    pair:
        :class:`PairSpec` describing the spread and the two legs.
    r:
        Discount rate used in the Leung-Li problem.
    cost:
        Round-trip transaction cost in *spread* units (not bps — the bps
        cost goes through the execution model). Drives how far the entry
        and exit boundaries sit from the mean.
    include_short:
        Mirror the boundaries for short positions when feasible.
    gross:
        Target gross exposure when in a position.
    """

    def __init__(
        self,
        params: OUParams,
        pair: PairSpec,
        *,
        r: float = 0.05,
        cost: float = 0.0,
        include_short: bool = True,
        gross: float = 1.0,
    ) -> None:
        self.params = params
        self.pair = pair
        self.r = float(r)
        self.cost = float(cost)
        self.gross = float(gross)

        self.boundaries: OptimalStoppingBoundaries = compute_boundaries(
            params, r=r, cost=cost, include_short=include_short,
        )
        self._position: int = 0

    def reset(self) -> None:
        self._position = 0

    def on_bar(self, event: MarketEvent) -> SignalEvent | None:
        x = self.pair.spread(event.prices)
        if not np.isfinite(x):
            return None

        new_position = self._next_position(x)
        if new_position == self._position:
            return None
        self._position = new_position
        return SignalEvent(
            timestamp=event.timestamp,
            target_weights=self.pair.leg_weights(new_position, self.gross),
        )

    # ------------------------------------------------------------------ #
    def _next_position(self, x: float) -> int:
        b = self.boundaries
        cur = self._position

        # Exit logic dominates entry.
        if cur == +1 and b.has_long and x >= b.long_exit:
            return 0
        if cur == -1 and b.has_short and x <= b.short_exit:
            return 0

        # Entry only from flat (the Leung-Li framework doesn't price a
        # re-entry decision while already in a position).
        if cur == 0:
            if b.has_long and x <= b.long_entry:
                return +1
            if b.has_short and x >= b.short_entry:
                return -1
        return cur
