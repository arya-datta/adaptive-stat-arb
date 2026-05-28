"""Strategy base class.

A :class:`Strategy` is a pure function of the bar stream: it observes
:class:`MarketEvent`\\s in order and optionally emits :class:`SignalEvent`\\s.
Concrete strategies live in ``stat_arb.stage1.strategy`` (±z baseline)
and ``stat_arb.stage2.strategy`` (Leung-Li optimal stopping).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .events import MarketEvent, SignalEvent


class Strategy(ABC):
    """Override :meth:`on_bar` to react to each market tick.

    A strategy must be **stateless with respect to time travel** — that is,
    its decisions on bar ``t`` may depend only on data observable at or
    before ``t``. The engine does not enforce this beyond delaying fills by
    one bar; pulling future values out of a cached frame is on you.
    """

    @abstractmethod
    def on_bar(self, event: MarketEvent) -> SignalEvent | None:
        """Return a :class:`SignalEvent` (or ``None`` to hold) for this bar."""

    def reset(self) -> None:
        """Reset internal state. Default is a no-op; override if you cache."""
