"""Stage 2: derived (not guessed) entry/exit boundaries.

Following Leung & Li, *Optimal Mean Reversion Trading* (2015): formulate
entry and liquidation as an *optimal double-stopping* problem on the OU
process, solve the resulting free-boundary problem for the optimal
take-profit, optimal entry, and (optionally) a stop-loss. The single
biggest mathematical upgrade in the roadmap.

* :mod:`optimal_stopping` — closed-form fundamental solutions ``F``/``G``
  and root-finders for the boundaries.
* :mod:`strategy` — engine-ready strategy that consumes those boundaries.
"""

from .optimal_stopping import (
    OUFundamentals,
    OptimalStoppingBoundaries,
    compute_boundaries,
)
from .strategy import OptimalStoppingStrategy

__all__ = [
    "OUFundamentals",
    "OptimalStoppingBoundaries",
    "compute_boundaries",
    "OptimalStoppingStrategy",
]
