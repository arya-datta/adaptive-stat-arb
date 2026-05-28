r"""Uncertainty-scaled pairs strategy (Stage 5).

Trades the same OU spread as Stage 1, but the *position size* is scaled by
how confident the posterior is about mean reversion. A Liu-West particle
filter runs online; when the posterior over the reversion speed is diffuse
(high coefficient of variation) or stationarity is ambiguous
(low ``p_stationary``), exposure is reduced.

Gate (per the roadmap): the uncertainty-scaled book should show better
drawdown / tail behaviour than the equivalent point-estimate book at
comparable gross return.
"""

from __future__ import annotations

import numpy as np

from ..engine import MarketEvent, SignalEvent, Strategy
from ..stage1.ou_mle import OUParams
from ..stage1.pair import PairSpec
from .particle_filter import ParticleFilterOU


class UncertaintyScaledStrategy(Strategy):
    r"""±z entry with confidence-scaled sizing from an online particle filter.

    Parameters
    ----------
    params:
        Point-estimate :class:`OUParams` (for the z-score mean/SD).
    pair:
        :class:`PairSpec` for the spread and legs.
    particle_filter:
        A seeded :class:`ParticleFilterOU` (reset at the start of each run).
    entry_z, exit_z, stop_z:
        Z-score thresholds.
    gross:
        Maximum gross exposure (scaled down by confidence).
    warmup:
        Bars to let the filter settle before trading.
    scale_by_uncertainty:
        If ``False``, the confidence multiplier is forced to 1.0 — i.e. the
        point-estimate book used as the gate's control.
    """

    def __init__(
        self,
        params: OUParams,
        pair: PairSpec,
        particle_filter: ParticleFilterOU,
        entry_z: float = 1.5,
        exit_z: float = 0.0,
        stop_z: float | None = 4.0,
        gross: float = 1.0,
        warmup: int = 30,
        scale_by_uncertainty: bool = True,
    ) -> None:
        if entry_z <= exit_z:
            raise ValueError("entry_z must be > exit_z.")
        self.params = params
        self.pair = pair
        self.pf = particle_filter
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.stop_z = float(stop_z) if stop_z is not None else None
        self.gross = float(gross)
        self.warmup = int(warmup)
        self.scale_by_uncertainty = bool(scale_by_uncertainty)

        self._sd = params.sigma / np.sqrt(2.0 * params.kappa)
        self._position = 0
        self._bar = 0
        self._confidence = 1.0
        self.confidence_history: list[float] = []

    def reset(self) -> None:
        # Re-seed the particle cloud so each run is independent and reproducible.
        self.pf.seed_from_ranges()
        self._position = 0
        self._bar = 0
        self._confidence = 1.0
        self.confidence_history = []

    def on_bar(self, event: MarketEvent) -> SignalEvent | None:
        spread = self.pair.spread(event.prices)
        if not np.isfinite(spread):
            return None

        report = self.pf.step(spread)          # online posterior update (data <= t)
        self._confidence = self._confidence_from(report)
        self.confidence_history.append(self._confidence)
        self._bar += 1
        if self._bar <= self.warmup:
            return None

        z = (spread - self.params.mu) / self._sd
        new_position = self._next_position(z)

        # Always re-issue weights when in a position (confidence may have moved
        # even if the discrete direction has not).
        target = self._leg_weights(new_position)
        self._position = new_position
        return SignalEvent(event.timestamp, target)

    # ------------------------------------------------------------------ #
    def _confidence_from(self, report: dict) -> float:
        """Map posterior diffuseness to a multiplier in (0, 1]."""
        if not self.scale_by_uncertainty:
            return 1.0
        p_stat = report.get("p_stationary", 1.0)
        cv = report.get("kappa_cv", 0.0)
        if not np.isfinite(cv):
            return 0.0
        # Down-weight by both stationarity probability and reversion-speed CV.
        return float(p_stat / (1.0 + cv))

    def _leg_weights(self, direction: int) -> dict[str, float]:
        g = self.gross * self._confidence
        return self.pair.leg_weights(direction, g) if direction != 0 else \
            self.pair.leg_weights(0, g)

    def _next_position(self, z: float) -> int:
        cur = self._position
        if self.stop_z is not None and abs(z) >= self.stop_z:
            return 0
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
