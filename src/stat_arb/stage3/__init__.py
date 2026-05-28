"""Stage 3: Kalman-filtered dynamic cointegration.

Static :math:`\\beta` is the weakest assumption in Stage 1. Here the hedge
ratio is allowed to breathe: the pair is cast as a state-space model with a
time-varying hedge ratio, estimated online with a Kalman filter. The
filtered residual (the one-step forecast error) becomes the new spread fed
into the OU / optimal-stopping machinery.

* :mod:`kalman_cointegration` — the Kalman hedge filter + rolling-OLS benchmark.
* :mod:`strategy` — engine-ready strategy trading the Kalman z-score.

Reference: Elliott, van der Hoek & Malcolm (2005); standard linear
state-space / Kalman theory.
"""

from .kalman_cointegration import (
    KalmanHedge,
    KalmanState,
    rolling_ols_hedge,
)
from .strategy import KalmanZScoreStrategy

__all__ = [
    "KalmanHedge",
    "KalmanState",
    "rolling_ols_hedge",
    "KalmanZScoreStrategy",
]
