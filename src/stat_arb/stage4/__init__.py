"""Stage 4: regime-switching OU (hidden-Markov dynamics).

Mean-reversion speed, level, and volatility shift across market regimes.
A latent regime :math:`S_t` governs the OU parameters:

.. math:: dX_t = \\kappa_{S_t}(\\mu_{S_t} - X_t)\\,dt + \\sigma_{S_t}\\,dW_t.

We fit by EM (Baum-Welch) with a Hamilton filter for the forward pass and a
Kim smoother for the backward pass, inferring regime probabilities online.

**Justification gate (roadmap principle #2).** Regime-switching is adopted
only if it is *justified*: :func:`regime_justification` runs a likelihood-
ratio / BIC comparison of the single-regime OU against the K-regime model.
A regime-switching model with no statistical support is worse than a clean
baseline.

* :mod:`regime_switching` — EM fitter, online filter, justification gate.
* :mod:`strategy` — trades conditional on the active regime; stands down in
  non-mean-reverting regimes.

Reference: Hamilton (1989); standard HMM / EM treatments.
"""

from .regime_switching import (
    MarkovSwitchingOU,
    RegimeOUParams,
    OnlineHamiltonFilter,
    regime_justification,
)
from .strategy import RegimeSwitchingStrategy

__all__ = [
    "MarkovSwitchingOU",
    "RegimeOUParams",
    "OnlineHamiltonFilter",
    "regime_justification",
    "RegimeSwitchingStrategy",
]
