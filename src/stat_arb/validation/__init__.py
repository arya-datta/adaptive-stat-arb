"""The validation spine — applied at every stage.

The PDF makes the bar explicit: every stage reports the *same* honesty
metrics, so that adding machinery proves it earns its place rather than
inflating a single backtested Sharpe.

* :mod:`stationarity`     — ADF / KPSS wrappers
* :mod:`sharpe`           — Lo (2002) Sharpe standard errors under autocorrelation
* :mod:`deflated_sharpe`  — Bailey & López de Prado (2014) DSR
* :mod:`purged_cv`        — López de Prado purged k-fold with embargo
* :mod:`walk_forward`     — anchored & rolling walk-forward generators
* :mod:`pbo`              — Combinatorially-symmetric CV (Bailey et al. 2017)
"""

from .stationarity import adf_test, kpss_test, StationarityResult
from .sharpe import sharpe_ratio, sharpe_se_lo, sharpe_ci_lo
from .deflated_sharpe import deflated_sharpe_ratio, expected_max_sharpe
from .purged_cv import PurgedKFold
from .walk_forward import walk_forward_splits
from .pbo import probability_of_backtest_overfitting
from .multiple_testing import (
    benjamini_hochberg,
    harvey_liu_zhu_hurdle,
    sharpe_pvalue,
)

__all__ = [
    "adf_test",
    "kpss_test",
    "StationarityResult",
    "sharpe_ratio",
    "sharpe_se_lo",
    "sharpe_ci_lo",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "PurgedKFold",
    "walk_forward_splits",
    "probability_of_backtest_overfitting",
    "benjamini_hochberg",
    "harvey_liu_zhu_hurdle",
    "sharpe_pvalue",
]
