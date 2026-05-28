r"""Multiple-testing control for many candidate signals.

Stage 6 trades hundreds of idiosyncratic residuals, so the false-discovery
problem becomes first-order: with enough candidates, some will look
significant by chance. Two complementary controls:

* **Benjamini-Hochberg (1995)** — controls the *false discovery rate* (the
  expected fraction of false positives among rejections). Less conservative
  than Bonferroni, appropriate when you expect several true signals.
* **Harvey-Liu-Zhu (2016)** — argues that in finance the conventional
  ``t > 2`` bar is far too lax given decades of data mining, and proposes a
  ``t > 3`` hurdle. We expose it as a simple, documented threshold.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def benjamini_hochberg(pvalues: np.ndarray, fdr: float = 0.10) -> dict:
    r"""Benjamini-Hochberg FDR control.

    Sort p-values ascending; find the largest ``k`` with
    :math:`p_{(k)} \le \tfrac{k}{m}\,q`; reject all hypotheses with
    :math:`p \le p_{(k)}`.

    Returns
    -------
    dict with ``reject`` (bool mask, original order), ``threshold`` (the
    p-value cutoff), and ``n_reject``.
    """
    p = np.asarray(pvalues, float)
    m = p.size
    if m == 0:
        return {"reject": np.array([], dtype=bool), "threshold": 0.0, "n_reject": 0}

    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, m + 1) / m) * fdr
    passed = ranked <= thresholds
    if not passed.any():
        cutoff = 0.0
    else:
        k = np.max(np.where(passed)[0])     # largest index that passes
        cutoff = ranked[k]

    reject = p <= cutoff
    return {"reject": reject, "threshold": float(cutoff), "n_reject": int(reject.sum())}


def harvey_liu_zhu_hurdle(
    sharpe: float,
    n_obs: int,
    periods_per_year: int = 252,
    t_hurdle: float = 3.0,
) -> dict:
    r"""Apply the Harvey-Liu-Zhu ``t > 3`` hurdle to an annualised Sharpe.

    The t-statistic of a Sharpe over ``n_obs`` periods is approximately
    :math:`t = SR_{\text{per-period}}\sqrt{n}`. We convert the annualised
    Sharpe back to per-period scale and compare against ``t_hurdle``.
    """
    sr_period = sharpe / np.sqrt(periods_per_year)
    t_stat = sr_period * np.sqrt(n_obs)
    return {
        "t_stat": float(t_stat),
        "passes": bool(abs(t_stat) > t_hurdle),
        "t_hurdle": float(t_hurdle),
    }


def sharpe_pvalue(sharpe: float, n_obs: int, periods_per_year: int = 252) -> float:
    """Two-sided p-value for an annualised Sharpe under the zero-skill null."""
    sr_period = sharpe / np.sqrt(periods_per_year)
    t_stat = sr_period * np.sqrt(n_obs)
    return float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
