r"""Kalman filter for a time-varying hedge ratio.

State-space model for the pair (on log-prices by convention):

.. math::

   Y_t &= \alpha_t + \beta_t X_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal N(0, R) \\
   \theta_t &= \theta_{t-1} + w_t, \qquad \theta_t = (\alpha_t, \beta_t)',\;\; w_t \sim \mathcal N(0, Q).

The observation matrix is :math:`F_t = (1,\, X_t)`. The Kalman recursions
yield the filtered hedge :math:`\theta_t`, the **innovation**
:math:`e_t = Y_t - F_t\theta_{t|t-1}` (the dynamic spread), its variance
:math:`S_t`, and the standardised spread :math:`z_t = e_t/\sqrt{S_t}`.

Trading the innovation (rather than a static-:math:`\beta` residual) lets
the relationship drift — the Stage 1 weakness this stage targets. ``Q`` and
``R`` can be set directly or estimated by maximum likelihood on the
one-step innovations (:meth:`KalmanHedge.fit_mle`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class KalmanState:
    """Filter state: hedge vector ``theta = [alpha, beta]`` and covariance ``P``."""

    theta: np.ndarray            # shape (2,)
    P: np.ndarray                # shape (2, 2)


class KalmanHedge:
    r"""Online Kalman filter for ``Y = alpha_t + beta_t * X + noise``.

    Parameters
    ----------
    q:
        State-transition variance scale; :math:`Q = q\,I_2`. Larger ``q``
        lets the hedge ratio move faster. Typical range ``1e-6``–``1e-3``.
    r:
        Observation-noise variance :math:`R` (scalar).
    init_theta, init_P:
        Initial state mean and covariance. Defaults to a diffuse prior
        (``theta=0``, ``P=I``). Warm-start with an OLS estimate for a
        shorter transient.
    """

    def __init__(
        self,
        q: float = 1e-5,
        r: float = 1e-3,
        init_theta: np.ndarray | None = None,
        init_P: np.ndarray | None = None,
    ) -> None:
        if q < 0 or r <= 0:
            raise ValueError("q must be >= 0 and r must be > 0.")
        self.q = float(q)
        self.r = float(r)
        self.Q = q * np.eye(2)
        self._init_theta = (
            np.zeros(2) if init_theta is None else np.asarray(init_theta, float)
        )
        self._init_P = np.eye(2) if init_P is None else np.asarray(init_P, float)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.state = KalmanState(theta=self._init_theta.copy(), P=self._init_P.copy())

    def step(self, y: float, x: float) -> dict:
        r"""Advance one bar with observation ``(y, x)`` and return diagnostics.

        Returns keys: ``alpha``, ``beta`` (filtered state *after* update),
        ``spread`` (the innovation :math:`e_t`), ``innov_var`` (:math:`S_t`),
        and ``z`` (standardised spread).
        """
        theta, P = self.state.theta, self.state.P

        # Predict
        P_pred = P + self.Q                     # theta is a random walk
        F = np.array([1.0, x])

        # Innovation
        y_hat = float(F @ theta)
        e = y - y_hat
        S = float(F @ P_pred @ F + self.r)      # scalar innovation variance

        # Update
        K = (P_pred @ F) / S                    # Kalman gain (2,)
        theta_new = theta + K * e
        P_new = P_pred - np.outer(K, F) @ P_pred

        self.state = KalmanState(theta=theta_new, P=P_new)
        z = e / np.sqrt(S) if S > 0 else 0.0
        return {
            "alpha": float(theta_new[0]),
            "beta": float(theta_new[1]),
            "spread": float(e),
            "innov_var": float(S),
            "z": float(z),
        }

    # ------------------------------------------------------------------ #
    def filter(
        self,
        y: pd.Series | np.ndarray,
        x: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        """Run the filter over full series; return a per-bar diagnostics frame.

        The filter is reset first, so this is a pure function of the inputs.
        """
        y_arr = np.asarray(y, float)
        x_arr = np.asarray(x, float)
        if y_arr.shape != x_arr.shape:
            raise ValueError("y and x must have the same shape.")
        self.reset()
        rows = [self.step(float(yi), float(xi)) for yi, xi in zip(y_arr, x_arr)]
        index = y.index if isinstance(y, pd.Series) else None
        return pd.DataFrame(rows, index=index)

    # ------------------------------------------------------------------ #
    @classmethod
    def fit_mle(
        cls,
        y: pd.Series | np.ndarray,
        x: pd.Series | np.ndarray,
        burn_in: float = 0.1,
        init_theta: np.ndarray | None = None,
    ) -> "KalmanHedge":
        r"""Estimate ``(q, r)`` by maximising the Gaussian innovation likelihood.

        The prediction-error decomposition gives

        .. math:: \log L = -\tfrac12\sum_t\bigl(\log 2\pi S_t + e_t^2/S_t\bigr).

        We optimise over :math:`(\log q, \log r)` but **floor** ``r`` at a tiny
        fraction of ``Var(y)``. The floor matters: with a random-walk state and
        a fully free ``r``, the likelihood is degenerate — it is maximised by
        driving ``r \to 0`` so the state absorbs everything. Downstream
        robustness (directional exits, *empirical* innovation standardisation in
        :class:`KalmanZScoreStrategy`) handles the rest, so a responsive filter
        is safe. The first ``burn_in`` fraction of innovations is skipped so the
        diffuse-prior transient doesn't dominate the objective.
        """
        y_arr = np.asarray(y, float)
        x_arr = np.asarray(x, float)
        n = y_arr.size
        skip = int(burn_in * n)

        X = np.column_stack([np.ones(n), x_arr])
        beta0, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
        if init_theta is None:
            init_theta = beta0
        r_floor = max(1e-8 * float(np.var(y_arr)), 1e-12)

        def neg_loglik(log_params: np.ndarray) -> float:
            q = float(np.exp(log_params[0]))
            r = max(float(np.exp(log_params[1])), r_floor)
            kf = cls(q=q, r=r, init_theta=init_theta, init_P=np.eye(2))
            ll = 0.0
            for i in range(n):
                out = kf.step(float(y_arr[i]), float(x_arr[i]))
                if i < skip:
                    continue
                S = out["innov_var"]
                e = out["spread"]
                if S <= 0:
                    return 1e12
                ll += -0.5 * (np.log(2 * np.pi * S) + e * e / S)
            return -ll

        resid_var = max(float(np.var(y_arr - X @ beta0)), r_floor)
        res = minimize(
            neg_loglik,
            x0=np.log([1e-5, resid_var]),
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-2, "maxiter": 400},
        )
        q_hat = float(np.exp(res.x[0]))
        r_hat = max(float(np.exp(res.x[1])), r_floor)
        return cls(q=q_hat, r=r_hat, init_theta=init_theta, init_P=np.eye(2))


def rolling_ols_hedge(
    y: pd.Series,
    x: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """Benchmark: rolling-OLS hedge ratio and the resulting spread.

    Returns a frame with ``alpha``, ``beta`` (rolling estimates) and
    ``spread`` (``y - alpha - beta*x``). Used to demonstrate that the
    Kalman estimate adds value — lower spread variance, more stable
    half-life — rather than just complexity.
    """
    if window < 5:
        raise ValueError("window must be >= 5.")
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    alpha = pd.Series(index=df.index, dtype=float)
    beta = pd.Series(index=df.index, dtype=float)

    yv, xv = df["y"].to_numpy(), df["x"].to_numpy()
    for i in range(window, len(df) + 1):
        xs = xv[i - window:i]
        ys = yv[i - window:i]
        X = np.column_stack([np.ones(window), xs])
        coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
        beta.iloc[i - 1] = coef[1]
        alpha.iloc[i - 1] = coef[0]

    spread = df["y"] - alpha - beta * df["x"]
    return pd.DataFrame({"alpha": alpha, "beta": beta, "spread": spread})
