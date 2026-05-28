r"""PCA eigenportfolios and Avellaneda-Lee residual s-scores.

Following Avellaneda & Lee (2010): extract systematic factors by PCA on the
return correlation matrix, regress each name on the factor returns, and trade
the **mean-reverting idiosyncratic residual**. The cumulative residual of each
name is modelled as an OU process; its standardised level is the *s-score*,

.. math:: s_i = \frac{X_i - m_i}{\sigma_{\mathrm{eq},i}}, \qquad
          \sigma_{\mathrm{eq},i} = \frac{\sigma_i}{\sqrt{2\kappa_i}},

with entry/exit thresholds applied cross-sectionally. Only names whose
residual reverts fast enough (short half-life relative to the lookback) are
traded — slow reverters are noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAFactors:
    eigenvalues: np.ndarray       # (N,) descending
    eigenvectors: np.ndarray      # (N, N), columns = components
    explained_variance_ratio: np.ndarray
    n_factors: int

    @property
    def top_explained(self) -> float:
        return float(self.explained_variance_ratio[: self.n_factors].sum())


def pca_factors(returns: np.ndarray, n_factors: int) -> PCAFactors:
    """PCA on the correlation matrix of ``returns`` (T x N)."""
    R = np.asarray(returns, float)
    std = R.std(axis=0, ddof=1)
    std = np.where(std > 0, std, 1.0)
    Z = (R - R.mean(axis=0)) / std
    corr = (Z.T @ Z) / (R.shape[0] - 1)
    vals, vecs = np.linalg.eigh(corr)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    evr = vals / vals.sum()
    return PCAFactors(eigenvalues=vals, eigenvectors=vecs,
                      explained_variance_ratio=evr, n_factors=n_factors)


@dataclass
class ResidualScores:
    sscore: np.ndarray            # (N,) current s-score per name
    kappa: np.ndarray             # (N,) per-step reversion speed of the residual
    half_life: np.ndarray         # (N,) in bars
    reverting: np.ndarray         # (N,) bool: tradeable (fast enough, b in (0,1))
    explained_variance: float


def residual_sscores(
    window_prices: np.ndarray,
    n_factors: int = 3,
    max_half_life_bars: float | None = None,
) -> ResidualScores:
    """Compute Avellaneda-Lee s-scores at the end of a price window.

    Parameters
    ----------
    window_prices:
        ``(L, N)`` array of prices over the estimation window.
    n_factors:
        Number of PCA factors to project out.
    max_half_life_bars:
        Names whose residual half-life exceeds this are flagged non-tradeable.
        Defaults to half the window length (Avellaneda-Lee require reliable,
        fast reversion).
    """
    P = np.asarray(window_prices, float)
    L, N = P.shape
    R = np.diff(np.log(P), axis=0)          # (L-1, N) log returns
    if max_half_life_bars is None:
        max_half_life_bars = L / 2.0

    # --- PCA factor returns ---
    fac = pca_factors(R, n_factors)
    std = R.std(axis=0, ddof=1)
    std = np.where(std > 0, std, 1.0)
    Q = fac.eigenvectors[:, :n_factors] / std[:, None]   # eigenportfolio weights
    F = R @ Q                                              # (L-1, K) factor returns

    # --- regress each name on factors; cumulative residual -> OU ---
    G = np.column_stack([np.ones(F.shape[0]), F])         # design with intercept
    coef, *_ = np.linalg.lstsq(G, R, rcond=None)          # (K+1, N)
    resid = R - G @ coef                                  # (L-1, N) idiosyncratic returns
    X = np.cumsum(resid, axis=0)                          # (L-1, N) cumulative residual

    # --- vectorised AR(1) OU fit per name on X ---
    x_prev, x_next = X[:-1], X[1:]                        # (L-2, N)
    n = x_prev.shape[0]
    xbar = x_prev.mean(axis=0)
    ybar = x_next.mean(axis=0)
    Sxx = ((x_prev - xbar) ** 2).sum(axis=0)
    Sxy = ((x_prev - xbar) * (x_next - ybar)).sum(axis=0)
    Sxx = np.where(Sxx > 0, Sxx, np.nan)
    b = Sxy / Sxx
    a = ybar - b * xbar
    resid_ar = x_next - (a + b * x_prev)
    v = (resid_ar ** 2).sum(axis=0) / n

    with np.errstate(invalid="ignore", divide="ignore"):
        reverting = (b > 0) & (b < 1) & np.isfinite(b)
        kappa = np.where(reverting, -np.log(np.clip(b, 1e-9, None)), 0.0)
        m = np.where(np.abs(1 - b) > 1e-9, a / (1 - b), np.nan)
        sigma_eq = np.sqrt(np.clip(v / np.clip(1 - b**2, 1e-9, None), 0, None))
        half_life = np.where(kappa > 0, np.log(2) / kappa, np.inf)
        sscore = np.where(sigma_eq > 0, (X[-1] - m) / sigma_eq, 0.0)

    tradeable = reverting & (half_life <= max_half_life_bars) & np.isfinite(sscore)
    sscore = np.where(tradeable, sscore, 0.0)

    return ResidualScores(
        sscore=sscore, kappa=kappa, half_life=half_life,
        reverting=tradeable, explained_variance=fac.top_explained,
    )
