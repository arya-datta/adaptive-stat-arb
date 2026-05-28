r"""Stage 5: Bayesian OU / particle filtering (uncertainty-aware trading).

Trade your *confidence*, not just your point estimate. Two complementary
tools:

* :mod:`bayesian_ou` — conjugate Bayesian linear regression on the exact OU
  discretisation gives the **exact posterior** over :math:`(\kappa,\mu,\sigma)`
  (no MCMC approximation needed; we expose posterior sampling).
* :mod:`particle_filter` — a Liu-West particle filter learns the parameter
  posterior **online**, naturally handling drift and the non-Gaussian
  parameter space.

The payoff is :mod:`strategy.UncertaintyScaledStrategy`: scale exposure down
when the posterior over the reversion speed is diffuse or stationarity is
ambiguous. Gate: the uncertainty-scaled book should show better drawdown /
tail behaviour than the point-estimate book at comparable gross return.
"""

from .bayesian_ou import BayesianOU, BayesianOUPosterior
from .particle_filter import ParticleFilterOU
from .strategy import UncertaintyScaledStrategy

__all__ = [
    "BayesianOU",
    "BayesianOUPosterior",
    "ParticleFilterOU",
    "UncertaintyScaledStrategy",
]
