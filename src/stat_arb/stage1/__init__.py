"""Stage 1: static cointegration + continuous-time OU MLE + ±z baseline.

The deliverable is a *rigorous* baseline — a strategy whose every metric
later stages will be benchmarked against. The cointegration tests live in
:mod:`cointegration`, the exact-discretisation OU MLE in :mod:`ou_mle`,
and the symmetric-threshold strategy in :mod:`strategy`.
"""

from .cointegration import (
    engle_granger,
    johansen,
    EngleGrangerResult,
    JohansenResult,
)
from .ou_mle import OUParams, OUMLEEstimator
from .pair import PairSpec
from .strategy import ZScoreStrategy

__all__ = [
    "engle_granger",
    "johansen",
    "EngleGrangerResult",
    "JohansenResult",
    "OUParams",
    "OUMLEEstimator",
    "PairSpec",
    "ZScoreStrategy",
]
