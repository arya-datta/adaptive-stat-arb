"""Stage 6: multivariate stat-arb (from pairs to a portfolio engine).

Breadth. By the fundamental law of active management
(:math:`\\mathrm{IR}\\approx\\mathrm{IC}\\sqrt{\\mathrm{breadth}}`), many
independent residuals beat a handful of pairs.

* :mod:`eigenportfolio` — PCA factors + Avellaneda-Lee residual s-scores.
* :mod:`covariance`     — Ledoit-Wolf shrinkage and Hierarchical Risk Parity.
* :mod:`vecm`           — multi-asset cointegration (Johansen -> VECM).
* :mod:`strategy`       — cross-sectional dollar-neutral mean-reversion.

Validation here leans on the multiple-testing controls in
:mod:`stat_arb.validation.multiple_testing` (Benjamini-Hochberg FDR,
Harvey-Liu-Zhu t>3) plus the Probability of Backtest Overfitting — essential
once hundreds of candidate residuals are screened.

References: Avellaneda & Lee (2010); Ledoit & Wolf (2004); López de Prado
(2016); Harvey, Liu & Zhu (2016).
"""

from .eigenportfolio import pca_factors, residual_sscores, PCAFactors, ResidualScores
from .covariance import ledoit_wolf, hierarchical_risk_parity
from .vecm import fit_vecm, VECMResult
from .strategy import EigenportfolioStrategy

__all__ = [
    "pca_factors",
    "residual_sscores",
    "PCAFactors",
    "ResidualScores",
    "ledoit_wolf",
    "hierarchical_risk_parity",
    "fit_vecm",
    "VECMResult",
    "EigenportfolioStrategy",
]
