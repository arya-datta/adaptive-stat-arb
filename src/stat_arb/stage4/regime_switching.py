r"""Markov-switching OU via EM (Hamilton filter + Kim smoother).

A latent :math:`K`-state Markov chain governs a Gaussian AR(1) — the exact
discretisation of a regime-specific OU process. For regime :math:`s`,

.. math:: x_{t+1} = a_s + b_s x_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal N(0, v_s),

with :math:`b_s = e^{-\kappa_s \Delta t}`, :math:`a_s = \mu_s(1-b_s)`,
:math:`v_s = \sigma_s^2(1-b_s^2)/(2\kappa_s)`.

EM:

* **E-step** — Hamilton filter (forward) gives filtered regime
  probabilities and the log-likelihood; the Kim smoother (backward) gives
  smoothed marginals and pairwise probabilities.
* **M-step** — transition matrix from smoothed pairwise probabilities;
  per-regime :math:`(a_s, b_s)` by weighted least squares; :math:`v_s` by
  weighted residual variance.

Multiple random restarts guard against local optima. Regimes are returned
ordered by mean-reversion speed (fastest :math:`\kappa` first) for a stable,
reproducible labelling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..utils import infer_dt
from ..stage1.ou_mle import OUMLEEstimator


# -------------------------------------------------------------------- #
# Fitted-parameter container                                           #
# -------------------------------------------------------------------- #
@dataclass
class RegimeOUParams:
    """Fitted regime-switching OU parameters and inference artefacts."""

    kappa: np.ndarray            # (K,) per-regime mean-reversion speed
    mu: np.ndarray               # (K,) per-regime long-run level
    sigma: np.ndarray            # (K,) per-regime volatility
    ar_a: np.ndarray             # (K,) AR(1) intercept
    ar_b: np.ndarray             # (K,) AR(1) slope = exp(-kappa*dt)
    ar_v: np.ndarray             # (K,) AR(1) innovation variance
    P: np.ndarray                # (K, K) transition matrix
    stationary: np.ndarray       # (K,) stationary regime distribution
    mean_reverting: np.ndarray   # (K,) bool: 0 < b_s < 1
    half_life: np.ndarray        # (K,) ln 2 / kappa (inf if not reverting)
    log_likelihood: float
    n_regimes: int
    n_obs: int
    dt: float
    aic: float
    bic: float
    filtered_prob: np.ndarray = field(default=None, repr=False)   # (n, K)
    smoothed_prob: np.ndarray = field(default=None, repr=False)   # (n, K)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = [f"RegimeOUParams(K={self.n_regimes}, LL={self.log_likelihood:.1f}, "
                 f"BIC={self.bic:.1f})"]
        for s in range(self.n_regimes):
            tag = "revert" if self.mean_reverting[s] else "NON-revert"
            hl = f"{self.half_life[s]*252:.0f}d" if self.mean_reverting[s] else "inf"
            lines.append(
                f"  regime {s}: kappa={self.kappa[s]:.2f} mu={self.mu[s]:.4f} "
                f"sigma={self.sigma[s]:.4f} half-life={hl} stay={self.P[s, s]:.2f} [{tag}]"
            )
        return "\n".join(lines)


# -------------------------------------------------------------------- #
# Online Hamilton filter (for the strategy)                            #
# -------------------------------------------------------------------- #
class OnlineHamiltonFilter:
    """Stream regime probabilities one observation at a time.

    Initialised from a fitted :class:`RegimeOUParams`; ``update(x_t)`` returns
    the filtered regime distribution after seeing ``x_t`` (using only data up
    to ``t`` — safe for live trading inside the event loop).
    """

    def __init__(self, params: RegimeOUParams) -> None:
        self.a = params.ar_a
        self.b = params.ar_b
        self.v = params.ar_v
        self.P = params.P
        self.xi = params.stationary.copy()
        self._prev_x: float | None = None

    def update(self, x_t: float) -> np.ndarray:
        if self._prev_x is None:
            self._prev_x = x_t
            return self.xi
        mean = self.a + self.b * self._prev_x
        eta = np.exp(-0.5 * (x_t - mean) ** 2 / self.v) / np.sqrt(2 * np.pi * self.v)
        eta = np.clip(eta, 1e-300, None)
        pred = self.P.T @ self.xi
        joint = pred * eta
        total = joint.sum()
        self.xi = joint / total if total > 0 else self.xi
        self._prev_x = x_t
        return self.xi


# -------------------------------------------------------------------- #
# EM fitter                                                            #
# -------------------------------------------------------------------- #
class MarkovSwitchingOU:
    """Fit a K-regime Markov-switching OU by EM with random restarts."""

    def __init__(
        self,
        n_regimes: int = 2,
        max_iter: int = 300,
        tol: float = 1e-6,
        n_init: int = 8,
        seed: int = 0,
    ) -> None:
        if n_regimes < 1:
            raise ValueError("n_regimes must be >= 1.")
        self.n_regimes = n_regimes
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.seed = seed

    def fit(self, series: pd.Series | np.ndarray, dt: float | None = None) -> RegimeOUParams:
        if isinstance(series, pd.Series):
            if dt is None:
                dt = infer_dt(series.index)
            x = series.dropna().to_numpy(float)
        else:
            x = np.asarray(series, float)
            x = x[np.isfinite(x)]
            if dt is None:
                raise ValueError("Must supply dt when passing a plain array.")
        if x.size < 50:
            raise ValueError("Need >= 50 observations for regime-switching EM.")

        x_prev, x_next = x[:-1], x[1:]
        K = self.n_regimes

        # Single regime → closed-form (delegate to Stage 1).
        if K == 1:
            ou = OUMLEEstimator().fit(x, dt=dt)
            return self._single_regime_params(ou, x_prev, x_next, dt)

        best = None
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_init):
            init = self._init_params(x_prev, x_next, K, rng)
            result = self._em(x_prev, x_next, dt, *init)
            if best is None or result[4] > best[4]:  # compare log-likelihood
                best = result

        a, b, v, P, ll, filt, smooth = best
        return self._assemble(a, b, v, P, ll, filt, smooth, x_prev.size, dt)

    # ------------------------------------------------------------------ #
    # EM internals                                                       #
    # ------------------------------------------------------------------ #
    def _init_params(self, x_prev, x_next, K, rng):
        """OLS-based start with perturbed slopes and spread variances."""
        X = np.column_stack([np.ones_like(x_prev), x_prev])
        coef, *_ = np.linalg.lstsq(X, x_next, rcond=None)
        a0, b0 = coef
        resid = x_next - (a0 + b0 * x_prev)
        v0 = float(np.var(resid))

        a = np.full(K, a0)
        # Spread slopes around the global estimate; spread variances geometrically.
        b = np.clip(b0 * (1.0 + 0.1 * rng.standard_normal(K)), 0.01, 0.999)
        v = v0 * np.linspace(0.5, 2.0, K) * (1.0 + 0.1 * rng.standard_normal(K))
        v = np.clip(v, 1e-10, None)

        P = np.full((K, K), 0.05 / max(K - 1, 1))
        np.fill_diagonal(P, 0.95)
        return a, b, v, P

    def _em(self, x_prev, x_next, dt, a, b, v, P):
        a, b, v, P = a.copy(), b.copy(), v.copy(), P.copy()
        n, K = x_prev.size, a.size
        prev_ll = -np.inf
        filt = smooth = None

        for _ in range(self.max_iter):
            # --- densities eta[t, s] ---
            mean = a[None, :] + b[None, :] * x_prev[:, None]          # (n, K)
            resid = x_next[:, None] - mean
            eta = np.exp(-0.5 * resid**2 / v[None, :]) / np.sqrt(2 * np.pi * v[None, :])
            eta = np.clip(eta, 1e-300, None)

            # --- Hamilton filter (forward) ---
            pi = _stationary(P)
            filt = np.empty((n, K))
            pred = np.empty((n, K))
            xi = pi
            ll = 0.0
            for t in range(n):
                pr = P.T @ xi
                pred[t] = pr
                joint = pr * eta[t]
                total = joint.sum()
                ll += np.log(total)
                xi = joint / total
                filt[t] = xi

            # --- Kim smoother (backward) ---
            smooth = np.empty((n, K))
            smooth[-1] = filt[-1]
            pair_sum = np.zeros((K, K))
            for t in range(n - 2, -1, -1):
                ratio = smooth[t + 1] / np.clip(pred[t + 1], 1e-300, None)   # (K,)
                smooth[t] = filt[t] * (P @ ratio)
                pair_sum += (filt[t][:, None] * P) * ratio[None, :]

            # --- M-step ---
            denom = smooth[:-1].sum(axis=0)                              # (K,)
            P = pair_sum / np.clip(denom[:, None], 1e-300, None)
            P /= P.sum(axis=1, keepdims=True)

            for s in range(K):
                w = smooth[:, s]
                Sw = w.sum()
                Swx = (w * x_prev).sum()
                Swy = (w * x_next).sum()
                Swxx = (w * x_prev * x_prev).sum()
                Swxy = (w * x_prev * x_next).sum()
                det = Sw * Swxx - Swx**2
                if abs(det) < 1e-300:
                    continue
                b[s] = (Sw * Swxy - Swx * Swy) / det
                a[s] = (Swy - b[s] * Swx) / Sw
                r = x_next - (a[s] + b[s] * x_prev)
                v[s] = max((w * r * r).sum() / Sw, 1e-12)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return a, b, v, P, ll, filt, smooth

    # ------------------------------------------------------------------ #
    # Assembly                                                           #
    # ------------------------------------------------------------------ #
    def _assemble(self, a, b, v, P, ll, filt, smooth, n, dt) -> RegimeOUParams:
        K = a.size
        kappa, mu, sigma, reverting, half_life = self._to_ou(a, b, v, dt)

        # Order regimes by mean-reversion speed (fastest first) for stable labels.
        order = np.argsort(-kappa)
        a, b, v = a[order], b[order], v[order]
        kappa, mu, sigma = kappa[order], mu[order], sigma[order]
        reverting, half_life = reverting[order], half_life[order]
        P = P[np.ix_(order, order)]
        filt, smooth = filt[:, order], smooth[:, order]

        n_params = K * 3 + K * (K - 1)   # regime AR params + free transitions
        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + n_params * np.log(n)

        return RegimeOUParams(
            kappa=kappa, mu=mu, sigma=sigma, ar_a=a, ar_b=b, ar_v=v,
            P=P, stationary=_stationary(P), mean_reverting=reverting,
            half_life=half_life, log_likelihood=float(ll), n_regimes=K,
            n_obs=int(n + 1), dt=float(dt), aic=float(aic), bic=float(bic),
            filtered_prob=filt, smoothed_prob=smooth,
        )

    @staticmethod
    def _to_ou(a, b, v, dt):
        K = a.size
        kappa = np.zeros(K); mu = np.zeros(K); sigma = np.zeros(K)
        reverting = np.zeros(K, dtype=bool); half_life = np.full(K, np.inf)
        for s in range(K):
            if 0.0 < b[s] < 1.0:
                kappa[s] = -np.log(b[s]) / dt
                mu[s] = a[s] / (1.0 - b[s])
                sigma[s] = np.sqrt(v[s] * 2.0 * kappa[s] / (1.0 - b[s] ** 2))
                reverting[s] = True
                half_life[s] = np.log(2.0) / kappa[s]
            else:
                # Non-mean-reverting (unit-root-ish) regime: dX = sigma dW, so
                # the one-step innovation variance is sigma^2 * dt -> recover the
                # diffusion as sqrt(v/dt) (the 1/sqrt(dt) factor matters: a slow
                # high-vol regime near b=1 otherwise looks deceptively low-vol).
                kappa[s] = 0.0
                mu[s] = a[s] / (1.0 - b[s]) if b[s] != 1.0 else np.nan
                sigma[s] = np.sqrt(v[s] / dt)
        return kappa, mu, sigma, reverting, half_life

    def _single_regime_params(self, ou, x_prev, x_next, dt) -> RegimeOUParams:
        b = float(np.exp(-ou.kappa * dt))
        a = ou.mu * (1 - b)
        v = float(np.var(x_next - (a + b * x_prev)))
        n_params = 3
        ll = ou.log_likelihood
        return RegimeOUParams(
            kappa=np.array([ou.kappa]), mu=np.array([ou.mu]), sigma=np.array([ou.sigma]),
            ar_a=np.array([a]), ar_b=np.array([b]), ar_v=np.array([v]),
            P=np.array([[1.0]]), stationary=np.array([1.0]),
            mean_reverting=np.array([ou.stationary]),
            half_life=np.array([ou.half_life]), log_likelihood=float(ll),
            n_regimes=1, n_obs=int(x_prev.size + 1), dt=float(dt),
            aic=float(-2 * ll + 2 * n_params),
            bic=float(-2 * ll + n_params * np.log(x_prev.size)),
        )


def _stationary(P: np.ndarray) -> np.ndarray:
    """Stationary distribution: left eigenvector of P for eigenvalue 1."""
    K = P.shape[0]
    if K == 1:
        return np.array([1.0])
    vals, vecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, idx])
    pi = np.clip(pi, 0, None)
    s = pi.sum()
    return pi / s if s > 0 else np.full(K, 1.0 / K)


# -------------------------------------------------------------------- #
# Justification gate (roadmap principle #2)                            #
# -------------------------------------------------------------------- #
def regime_justification(
    series: pd.Series | np.ndarray,
    dt: float | None = None,
    n_regimes: int = 2,
    **fit_kwargs,
) -> dict:
    r"""Decide whether regime-switching is *justified* over a single regime.

    Fits the 1-regime OU and the ``n_regimes``-regime model on the same data
    and compares them. Returns log-likelihoods, AIC/BIC, the likelihood-ratio
    statistic, and a recommendation.

    **Caveat (Davies' problem).** Under the single-regime null the transition
    probabilities are unidentified, so the LR statistic does *not* follow a
    standard :math:`\chi^2`. We therefore base the recommendation on **BIC**
    (which penalises the extra parameters and is robust to this issue) and
    report the LR statistic for context only.
    """
    one = MarkovSwitchingOU(n_regimes=1).fit(series, dt=dt)
    many = MarkovSwitchingOU(n_regimes=n_regimes, **fit_kwargs).fit(series, dt=dt)

    lr_stat = 2.0 * (many.log_likelihood - one.log_likelihood)
    extra_params = (n_regimes * 3 + n_regimes * (n_regimes - 1)) - 3
    adopt = many.bic < one.bic

    return {
        "ll_single": one.log_likelihood,
        "ll_multi": many.log_likelihood,
        "lr_statistic": float(lr_stat),
        "extra_params": int(extra_params),
        "aic_single": one.aic, "aic_multi": many.aic,
        "bic_single": one.bic, "bic_multi": many.bic,
        "adopt_regime_switching": bool(adopt),
        "single": one, "multi": many,
    }
