r"""Stage 7: microstructure-aware execution.

Make the Sharpe mean something. Short-horizon stat-arb edges live or die on
execution, so the Stage-0 cost stub (fixed bps + half-spread) is replaced by:

* :mod:`execution` — :class:`MicrostructureCostModel`: bid-ask bounce, a
  **square-root market-impact** law, latency slippage, and **partial fills**
  (you cannot always trade your full target into the available liquidity).
* :mod:`almgren_chriss` — optimal execution scheduling: TWAP, VWAP, and the
  Almgren-Chriss trajectory trading impact off against timing risk, plus the
  cost/variance efficient frontier.

The deliverable is honest: re-run prior stages through realistic execution and
report which edges survive the round trip. The post-microstructure Deflated
Sharpe is the number you actually defend.

Reference: Almgren & Chriss (2000); standard market-microstructure literature.
"""

from .execution import MicrostructureCostModel
from .almgren_chriss import (
    almgren_chriss_schedule,
    twap_schedule,
    vwap_schedule,
    execution_frontier,
    ExecutionSchedule,
)

__all__ = [
    "MicrostructureCostModel",
    "almgren_chriss_schedule",
    "twap_schedule",
    "vwap_schedule",
    "execution_frontier",
    "ExecutionSchedule",
]
