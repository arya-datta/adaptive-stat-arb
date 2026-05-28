r"""Exact-discretisation maximum-likelihood for the Ornstein-Uhlenbeck SDE.

Model: :math:`dX_t = \kappa(\mu - X_t)\,dt + \sigma\,dW_t`.

The naive approach (regressing :math:`\Delta X_i` on :math:`X_i`) discretises
the SDE via Euler and inherits an :math:`O(\Delta t)` bias in
:math:`\kappa`. We instead exploit the *exact* Gaussian transition
density: conditional on :math:`X_i`,

.. math::

   X_{i+1} \sim \mathcal{N}\!\bigl(
     \mu + (X_i - \mu)\,e^{-\kappa\Delta t},\;
     \tfrac{\sigma^2}{2\kappa}(1 - e^{-2\kappa\Delta t})
   \bigr).

This is structurally an AR(1) with intercept, so OLS on
:math:`X_{i+1} = a + b X_i + \varepsilon_i` recovers the *exact* MLE for
any :math:`\Delta t`. We then back out :math:`(\kappa, \mu, \sigma)` and
form Fisher-information confidence intervals via the delta method.

References: Tang & Chen (2009); Phillips (1972). The roadmap PDF flags
the discrete-AR(1) shortcut as the canonical mistake.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..utils import infer_dt


@dataclass(frozen=True)
class OUParams:
    """Point estimates and 95% Fisher-information confidence intervals.

    Attributes
    ----------
    kappa, mu, sigma:
        Point estimates.
    kappa_ci, mu_ci, sigma_ci:
        ``(lower, upper)`` 95% intervals from the delta-method asymptotics.
    half_life:
        :math:`\\ln 2 / \\kappa` in the same time units as ``dt``.
    log_likelihood:
        Maximised log-likelihood at the MLE.
    n_obs:
        Number of *observations* (one transition needs two prices).
    dt:
        Time step (years) used to fit.
    stationary:
        ``True`` iff :math:`\\kappa > 0` and the lower CI for :math:`\\kappa`
        also exceeds 0 — i.e. mean-reversion is statistically significant.
    """

    kappa: float
    mu: float
    sigma: float
    kappa_ci: tuple[float, float]
    mu_ci: tuple[float, float]
    sigma_ci: tuple[float, float]
    half_life: float
    log_likelihood: float
    n_obs: int
    dt: float
    stationary: bool

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"OUParams(kappa={self.kappa:.4f} [{self.kappa_ci[0]:.4f}, "
            f"{self.kappa_ci[1]:.4f}], mu={self.mu:.4f} "
            f"[{self.mu_ci[0]:.4f}, {self.mu_ci[1]:.4f}], "
            f"sigma={self.sigma:.4f} [{self.sigma_ci[0]:.4f}, "
            f"{self.sigma_ci[1]:.4f}], half_life={self.half_life:.2f}, "
            f"stationary={self.stationary})"
        )


class OUMLEEstimator:
    """Fit an OU process by exact-discretisation MLE.

    Example
    -------
    >>> from stat_arb.data import SyntheticOU
    >>> sim = SyntheticOU(kappa=2.0, mu=0.0, sigma=0.3).simulate(2000, seed=0)
    >>> params = OUMLEEstimator().fit(sim["spread"])
    >>> abs(params.kappa - 2.0) < 0.3
    True
    """

    def __init__(self, confidence: float = 0.95) -> None:
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1).")
        self.confidence = confidence

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def fit(
        self,
        series: pd.Series | np.ndarray,
        dt: float | None = None,
    ) -> OUParams:
        """Fit the OU SDE to ``series``. ``dt`` defaults to inference from index."""
        if isinstance(series, pd.Series):
            if dt is None:
                dt = infer_dt(series.index)
            x = series.dropna().to_numpy(dtype=float)
        else:
            x = np.asarray(series, dtype=float)
            x = x[np.isfinite(x)]
            if dt is None:
                raise ValueError("Must supply dt when passing a plain array.")

        if x.size < 30:
            raise ValueError(f"Need ≥30 observations to fit OU MLE; got {x.size}.")
        if dt <= 0:
            raise ValueError("dt must be > 0.")

        # --- exact-discretisation OLS gives the MLE for (a, b, var_eps) ---
        x_prev, x_next = x[:-1], x[1:]
        n = x_prev.size
        x_bar, y_bar = x_prev.mean(), x_next.mean()
        Sxx = float(((x_prev - x_bar) ** 2).sum())
        Sxy = float(((x_prev - x_bar) * (x_next - y_bar)).sum())

        if Sxx == 0:
            raise ValueError("Predictor has zero variance; cannot fit OU.")

        b_hat = Sxy / Sxx
        a_hat = y_bar - b_hat * x_bar
        resid = x_next - (a_hat + b_hat * x_prev)
        sigma_eps_sq = float((resid ** 2).sum()) / n  # MLE (not unbiased) — matches likelihood

        if b_hat <= 0:
            # OU implies b = exp(-kappa*dt) ∈ (0, 1). Negative b is a sign the
            # series isn't OU (e.g. anti-persistent noise) — abort honestly.
            raise ValueError(
                f"OLS AR(1) coefficient is {b_hat:.4f}; series is not OU-like "
                "(should be in (0, 1)). Inspect stationarity before fitting."
            )

        # --- transform to (kappa, mu, sigma) ---
        kappa = -np.log(b_hat) / dt
        mu = a_hat / (1.0 - b_hat) if b_hat < 1.0 else x_bar
        sigma_sq = sigma_eps_sq * (2.0 * kappa) / (1.0 - b_hat**2)
        sigma = float(np.sqrt(max(sigma_sq, 0.0)))

        # --- delta-method covariance for (kappa, mu, sigma) ---
        kappa_ci, mu_ci, sigma_ci = self._delta_method_cis(
            n=n, x_prev=x_prev, sigma_eps_sq=sigma_eps_sq,
            a=a_hat, b=b_hat, kappa=kappa, mu=mu, sigma=sigma, dt=dt,
        )

        # --- log-likelihood at the MLE ---
        log_lik = -0.5 * n * (np.log(2 * np.pi * sigma_eps_sq) + 1.0)

        half_life = float(np.log(2.0) / kappa)
        stationary = kappa > 0 and kappa_ci[0] > 0

        return OUParams(
            kappa=float(kappa),
            mu=float(mu),
            sigma=sigma,
            kappa_ci=(float(kappa_ci[0]), float(kappa_ci[1])),
            mu_ci=(float(mu_ci[0]), float(mu_ci[1])),
            sigma_ci=(float(sigma_ci[0]), float(sigma_ci[1])),
            half_life=half_life,
            log_likelihood=float(log_lik),
            n_obs=int(n + 1),
            dt=float(dt),
            stationary=bool(stationary),
        )

    # ------------------------------------------------------------------ #
    # Internal: Fisher-information for (a, b, sigma_eps^2) → (kappa, mu, sigma)
    # ------------------------------------------------------------------ #
    def _delta_method_cis(
        self,
        *,
        n: int,
        x_prev: np.ndarray,
        sigma_eps_sq: float,
        a: float,
        b: float,
        kappa: float,
        mu: float,
        sigma: float,
        dt: float,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """95% asymptotic CIs via the Fisher info of the Gaussian AR(1).

        Parameter ordering is :math:`(a, b, v)` where :math:`v = \\sigma_\\varepsilon^2`.
        The information matrix is block-diagonal in :math:`v`:

        .. math::

           I = \\begin{pmatrix}
             n/v & \\sum x / v & 0 \\\\
             \\sum x / v & \\sum x^2 / v & 0 \\\\
             0 & 0 & n/(2v^2)
           \\end{pmatrix}.

        We invert :math:`I` for the asymptotic covariance, then apply the
        Jacobian of :math:`(a, b, v) \\mapsto (\\kappa, \\mu, \\sigma)`.
        """
        v = sigma_eps_sq
        sum_x = float(x_prev.sum())
        sum_x2 = float((x_prev ** 2).sum())

        # Covariance of OLS (a, b) is v * (X'X)^{-1} with X = [1, x_prev].
        # det(X'X) = n * sum_x2 - (sum_x)^2 = n * Sxx.
        det_xx = n * sum_x2 - sum_x ** 2
        if det_xx <= 0:
            nan = (float("nan"), float("nan"))
            return nan, nan, nan

        var_a = v * sum_x2 / det_xx
        var_b = v * n / det_xx
        cov_ab = -v * sum_x / det_xx
        var_v = 2.0 * v ** 2 / n  # exact for Gaussian AR(1) MLE

        cov_abv = np.array([
            [var_a, cov_ab, 0.0],
            [cov_ab, var_b, 0.0],
            [0.0,    0.0,   var_v],
        ])

        # Jacobian d(kappa, mu, sigma) / d(a, b, v)
        # kappa = -ln(b)/dt           → dκ/db = -1/(b dt), others 0
        # mu    = a/(1-b)             → dμ/da = 1/(1-b), dμ/db = a/(1-b)^2
        # sigma = sqrt(v * 2κ / (1-b^2))
        #        Let g(b, v) = v * 2κ(b) / (1-b^2). Then σ = sqrt(g).
        #        dσ/dv = (1/(2σ)) * 2κ/(1-b^2)
        #        dσ/db = (1/(2σ)) * v * d/db [2κ/(1-b^2)]
        if abs(1.0 - b) < 1e-10 or b <= 0:
            nan = (float("nan"), float("nan"))
            return nan, nan, nan

        dkappa_db = -1.0 / (b * dt)
        dmu_da = 1.0 / (1.0 - b)
        dmu_db = a / (1.0 - b) ** 2

        # d/db [2κ/(1-b^2)] where κ = -ln(b)/dt
        # = [2 dκ/db (1-b^2) - 2κ * (-2b)] / (1-b^2)^2
        # = [2 * (-1/(b dt)) * (1-b^2) + 4κb] / (1-b^2)^2
        one_minus_b2 = 1.0 - b ** 2
        d_coeff_db = (2.0 * dkappa_db * one_minus_b2 + 4.0 * kappa * b) / one_minus_b2 ** 2
        # σ^2 = v * 2κ/(1-b^2). Guard against tiny σ during early iterations.
        sigma_sq = sigma ** 2
        if sigma_sq < 1e-20:
            nan = (float("nan"), float("nan"))
            return nan, nan, nan
        dsigma_dv = (1.0 / (2.0 * sigma)) * 2.0 * kappa / one_minus_b2
        dsigma_db = (1.0 / (2.0 * sigma)) * v * d_coeff_db

        J = np.array([
            [0.0,    dkappa_db, 0.0],
            [dmu_da, dmu_db,    0.0],
            [0.0,    dsigma_db, dsigma_dv],
        ])

        cov_kms = J @ cov_abv @ J.T
        z = float(norm.ppf(0.5 + self.confidence / 2.0))

        kappa_se = float(np.sqrt(max(cov_kms[0, 0], 0.0)))
        mu_se = float(np.sqrt(max(cov_kms[1, 1], 0.0)))
        sigma_se = float(np.sqrt(max(cov_kms[2, 2], 0.0)))

        return (
            (kappa - z * kappa_se, kappa + z * kappa_se),
            (mu - z * mu_se, mu + z * mu_se),
            (sigma - z * sigma_se, sigma + z * sigma_se),
        )
