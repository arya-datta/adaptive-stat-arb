r"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

When :math:`N` independent strategies are tried, the *expected maximum*
Sharpe under the null (zero true skill) grows like :math:`O(\sqrt{\log N})`.
Reporting the winning Sharpe without deflating for the search is the
single most common stat-arb mistake the roadmap is designed to prevent.

We implement the closed-form approximation:

.. math::

   \mathrm{DSR} = \Phi\!\Bigl(
     \frac{(\widehat{SR} - SR_0)\sqrt{T-1}}
          {\sqrt{1 - \gamma_3 \widehat{SR} + (\gamma_4 - 1)/4\, \widehat{SR}^2}}
   \Bigr),

with :math:`SR_0 = \sqrt{V[\widehat{SR}]}\bigl((1-\gamma_E)\Phi^{-1}(1-1/N)
+ \gamma_E \Phi^{-1}(1 - 1/(Ne))\bigr)`, :math:`\gamma_E` Euler-Mascheroni,
:math:`V[\widehat{SR}]` the cross-sectional variance over the search,
:math:`\gamma_3,\gamma_4` the skew and kurtosis of returns, and :math:`T`
the sample size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

EULER_MASCHERONI = 0.5772156649015328606


def expected_max_sharpe(n_trials: int, sr_variance_across_trials: float) -> float:
    """Closed-form :math:`E[\\max_i \\widehat{SR}_i]` from Bailey-LdP (2014)."""
    if n_trials < 1:
        raise ValueError("n_trials must be ≥ 1.")
    if sr_variance_across_trials < 0:
        raise ValueError("Variance must be non-negative.")
    if n_trials == 1:
        return 0.0
    sd = float(np.sqrt(sr_variance_across_trials))
    a = float(norm.ppf(1.0 - 1.0 / n_trials))
    b = float(norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
    return sd * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b)


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    n_trials: int,
    sr_variance_across_trials: float | None = None,
    trial_sharpes: np.ndarray | None = None,
) -> dict:
    """Return Deflated Sharpe and its components.

    Provide *one* of ``sr_variance_across_trials`` or ``trial_sharpes``
    (the array of Sharpes from each of the ``n_trials`` candidate
    strategies you searched over). Without it, deflation is impossible —
    the whole point of the metric is that the search cost is information.

    Returns
    -------
    dict with keys: ``sr``, ``sr0`` (expected max under null), ``dsr``
    (probability the true SR exceeds ``sr0``), ``T``, ``skew``, ``kurt``.
    """
    if sr_variance_across_trials is None and trial_sharpes is None:
        raise ValueError(
            "Provide either sr_variance_across_trials or trial_sharpes — "
            "DSR is meaningless without the multiple-testing context."
        )
    if sr_variance_across_trials is None:
        ts = np.asarray(trial_sharpes, dtype=float)
        sr_variance_across_trials = float(np.var(ts, ddof=1)) if ts.size > 1 else 0.0

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = r.size
    if T < 30:
        raise ValueError(f"Need ≥30 return observations; got {T}.")

    std = r.std(ddof=1)
    if std == 0:
        return {"sr": float("nan"), "sr0": 0.0, "dsr": float("nan"),
                "T": T, "skew": 0.0, "kurt": 3.0}

    sr_period = r.mean() / std
    gamma3 = float(skew(r, bias=False))
    gamma4 = float(kurtosis(r, fisher=False, bias=False))  # *non-excess*

    sr0 = expected_max_sharpe(n_trials, sr_variance_across_trials)

    denom = 1.0 - gamma3 * sr_period + ((gamma4 - 1.0) / 4.0) * sr_period**2
    if denom <= 0:
        # Pathological higher moments — fall back to iid-normal SE.
        denom = 1.0

    z = (sr_period - sr0) * np.sqrt(T - 1) / np.sqrt(denom)
    return {
        "sr":   float(sr_period),
        "sr0":  float(sr0),
        "dsr":  float(norm.cdf(z)),
        "T":    int(T),
        "skew": gamma3,
        "kurt": gamma4,
    }
