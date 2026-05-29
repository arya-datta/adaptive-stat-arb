r"""Microstructure-aware execution model (Stage 7).

Drop-in replacement for the Stage-0 :class:`LinearCostModel` that adds the
frictions that actually decide whether a short-horizon edge is real:

* **Bid-ask bounce** — a half-spread paid on every trade.
* **Square-root market impact** — the empirical law
  :math:`\Delta p / p \approx Y\,\sigma\sqrt{Q/V}` (Almgren et al.): impact
  grows with the square root of participation (order size ``Q`` over average
  daily volume ``V``), scaled by daily volatility ``sigma`` and a constant
  ``Y``. Doubling size does *not* double slippage, but large orders are
  punished super-linearly in *total* cost.
* **Latency slippage** — a fixed adverse drift between decision and fill.
* **Partial fills** — only ``participation_cap * ADV`` shares clear in one
  bar; the unfilled remainder is dropped (the strategy re-attempts next bar
  via its target-weight reconciliation).

Because it implements the same :class:`ExecutionModel` interface, any prior
stage's strategy can be re-run through it unchanged.
"""

from __future__ import annotations

import numpy as np

from ..engine.events import OrderEvent, FillEvent
from ..engine.execution import ExecutionModel


class MicrostructureCostModel(ExecutionModel):
    r"""Square-root impact + half-spread + latency + partial fills.

    Parameters
    ----------
    half_spread_bps:
        Half the quoted bid-ask spread, paid on every trade (bps).
    commission_bps:
        Proportional commission on filled notional (bps).
    impact_coef:
        The ``Y`` constant in the square-root law (empirically ~0.5-1.5).
    daily_vol:
        Representative daily return volatility used to scale impact.
    adv:
        Average daily volume in **shares** (sets the participation scale).
    participation_cap:
        Max fraction of ``adv`` that can fill in a single bar; excess is a
        partial fill (remainder dropped).
    latency_bps:
        Fixed adverse slippage from decision-to-fill latency (bps).
    """

    def __init__(
        self,
        half_spread_bps: float = 2.0,
        commission_bps: float = 1.0,
        impact_coef: float = 1.0,
        daily_vol: float = 0.02,
        adv: float = 1_000_000.0,
        participation_cap: float = 0.1,
        latency_bps: float = 0.5,
    ) -> None:
        if adv <= 0 or participation_cap <= 0:
            raise ValueError("adv and participation_cap must be positive.")
        self.half_spread_bps = float(half_spread_bps)
        self.commission_bps = float(commission_bps)
        self.impact_coef = float(impact_coef)
        self.daily_vol = float(daily_vol)
        self.adv = float(adv)
        self.participation_cap = float(participation_cap)
        self.latency_bps = float(latency_bps)

    def fill(self, order: OrderEvent, fill_price: float) -> FillEvent:
        side = 1.0 if order.quantity > 0 else -1.0
        requested = abs(order.quantity)

        # Partial fill: cap by available liquidity this bar.
        max_fill = self.participation_cap * self.adv
        filled = min(requested, max_fill)
        participation = filled / self.adv

        # Square-root temporary impact (price fraction).
        impact_frac = self.impact_coef * self.daily_vol * np.sqrt(participation)
        spread_frac = self.half_spread_bps / 1e4
        latency_frac = self.latency_bps / 1e4
        slip = side * (spread_frac + impact_frac + latency_frac)

        executed_price = fill_price * (1.0 + slip)
        commission = filled * executed_price * (self.commission_bps / 1e4)

        return FillEvent(
            timestamp=order.timestamp,
            symbol=order.symbol,
            quantity=side * filled,
            price=executed_price,
            commission=commission,
        )
