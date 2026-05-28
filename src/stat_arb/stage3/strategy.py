r"""Kalman z-score pairs strategy (Stage 3).

Trades the *dynamic* spread — the Kalman innovation :math:`e_t` — rather
than a static-:math:`\beta` residual. The filter runs online inside the
event loop: each bar updates the hedge :math:`\beta_t` and produces a
standardised spread :math:`z_t = e_t/\sqrt{S_t}` that drives entry/exit.
Leg weights use the *current* filtered :math:`\beta_t`, so the hedge
adapts as the relationship drifts.

The Stage 3 gate is to beat Stage 1's static hedge — usually visible as a
lower-variance spread and a more stable half-life, not merely more
machinery.
"""

from __future__ import annotations

import numpy as np

from ..engine import MarketEvent, SignalEvent, Strategy
from .kalman_cointegration import KalmanHedge


class KalmanZScoreStrategy(Strategy):
    r"""±z entry on the Kalman innovation; mean exit; optional stop.

    Parameters
    ----------
    kalman:
        A configured :class:`KalmanHedge` (e.g. from ``KalmanHedge.fit_mle``).
        It is reset at the start of every backtest.
    y_symbol, x_symbol:
        The two legs. ``Y`` is the dependent (filtered) leg.
    use_log:
        Filter on log-prices (default) for return-stationary hedging.
    entry_z, exit_z, stop_z:
        Standardised-spread thresholds (``stop_z=None`` disables the stop).
    gross:
        Target gross exposure when in a position.
    warmup:
        Number of initial bars to skip trading while the diffuse prior
        settles (the filter still updates).
    """

    def __init__(
        self,
        kalman: KalmanHedge,
        y_symbol: str,
        x_symbol: str,
        use_log: bool = True,
        entry_z: float = 1.5,
        exit_z: float = 0.0,
        stop_z: float | None = 4.0,
        gross: float = 1.0,
        warmup: int = 30,
    ) -> None:
        if entry_z <= exit_z:
            raise ValueError("entry_z must be > exit_z.")
        if stop_z is not None and stop_z <= entry_z:
            raise ValueError("stop_z must be > entry_z (or None).")

        self.kalman = kalman
        self.y_symbol = y_symbol
        self.x_symbol = x_symbol
        self.use_log = use_log
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.stop_z = float(stop_z) if stop_z is not None else None
        self.gross = float(gross)
        self.warmup = int(warmup)

        self._position = 0
        self._beta = 1.0
        self._bar = 0
        self._warmup_innov: list[float] = []
        self._e_mean = 0.0
        self._e_std = 1.0

    def reset(self) -> None:
        self.kalman.reset()
        self._position = 0
        self._beta = 1.0
        self._bar = 0
        self._warmup_innov = []
        self._e_mean = 0.0
        self._e_std = 1.0

    def on_bar(self, event: MarketEvent) -> SignalEvent | None:
        if self.y_symbol not in event.prices or self.x_symbol not in event.prices:
            return None
        py = float(event.prices[self.y_symbol])
        px = float(event.prices[self.x_symbol])
        if not (np.isfinite(py) and np.isfinite(px)) or py <= 0 or px <= 0:
            return None

        y = np.log(py) if self.use_log else py
        x = np.log(px) if self.use_log else px

        out = self.kalman.step(y, x)   # filter ALWAYS updates (no look-ahead: uses data <= t)
        self._beta = out["beta"]
        self._bar += 1
        e = out["spread"]              # the dynamic-hedge innovation

        # Calibrate the innovation's scale over the warmup window, then trade
        # the *empirically* standardised innovation. This is robust to the
        # filter's internal S_t, which is unreliable because the random-walk
        # hedge model is only an approximation of an OU residual.
        if self._bar <= self.warmup:
            self._warmup_innov.append(e)
            return None
        if self._bar == self.warmup + 1:
            arr = np.asarray(self._warmup_innov)
            self._e_mean = float(arr.mean())
            self._e_std = float(arr.std(ddof=1)) or 1.0

        z = (e - self._e_mean) / self._e_std
        new_position = self._next_position(z)
        if new_position == self._position:
            return None
        self._position = new_position
        return SignalEvent(event.timestamp, self._leg_weights(new_position))

    # ------------------------------------------------------------------ #
    def _leg_weights(self, direction: int) -> dict[str, float]:
        if direction == 0:
            return {self.y_symbol: 0.0, self.x_symbol: 0.0}
        g = self.gross / (1.0 + abs(self._beta))
        return {
            self.y_symbol: direction * g,
            self.x_symbol: -direction * self._beta * g,
        }

    def _next_position(self, z: float) -> int:
        cur = self._position
        if self.stop_z is not None and abs(z) >= self.stop_z:
            return 0
        # Directional exit (see ZScoreStrategy): close on reversion through
        # the exit level, not the measure-zero ``abs(z) <= exit_z``.
        if cur == -1 and z <= self.exit_z:
            return 0
        if cur == +1 and z >= -self.exit_z:
            return 0
        if cur == 0:
            if z >= self.entry_z:
                return -1  # innovation rich → short the spread
            if z <= -self.entry_z:
                return +1  # innovation cheap → long the spread
        return cur
