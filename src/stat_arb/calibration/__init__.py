"""Parallelised parameter-sweep harness.

Stage 0's last build item per the roadmap. Used to recalibrate thousands
of (pair × window) combinations without blocking on a single core, and to
collect the per-trial Sharpe distribution that
:func:`stat_arb.validation.deflated_sharpe_ratio` needs to penalise the
multiple-testing search.
"""

from .harness import grid_search, ParameterGrid, TrialResult

__all__ = ["grid_search", "ParameterGrid", "TrialResult"]
