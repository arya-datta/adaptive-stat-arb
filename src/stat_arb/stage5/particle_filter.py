r"""Liu-West particle filter for online OU parameter learning.

A bootstrap particle filter degenerates when used for *fixed* parameters
(no state noise to rejuvenate the cloud). The Liu-West (2001) filter fixes
this with a kernel-shrinkage step: particles are shrunk toward their mean
and jittered by a matched-variance Gaussian kernel, so the parameter
posterior is learned online without collapsing.

Particles live in ``(a, b, log v)`` space (the AR(1) parameters; ``log v``
keeps the variance positive). Each step ingests a transition
:math:`(x_t, x_{t+1})`, reweights by its Gaussian likelihood, and resamples
when the effective sample size drops. We expose the online posterior mean
and spread of :math:`\kappa` — the input to uncertainty-aware sizing.
"""

from __future__ import annotations

import numpy as np


class ParticleFilterOU:
    r"""Online Bayesian learning of OU parameters via a Liu-West filter.

    Parameters
    ----------
    n_particles:
        Size of the particle cloud.
    delta:
        Liu-West discount (0.95-0.99). Shrinkage ``a = (3*delta-1)/(2*delta)``
        and kernel variance ``h^2 = 1 - a^2``. Lower ``delta`` = faster
        adaptation, more jitter.
    seed:
        RNG seed.
    """

    def __init__(self, n_particles: int = 2000, delta: float = 0.97, seed: int = 0) -> None:
        if not 0.9 <= delta < 1.0:
            raise ValueError("delta should be in [0.9, 1.0).")
        self.n = int(n_particles)
        self.delta = float(delta)
        self.a_shrink = (3.0 * delta - 1.0) / (2.0 * delta)
        self.h2 = 1.0 - self.a_shrink**2
        self.rng = np.random.default_rng(seed)
        self._theta: np.ndarray | None = None   # (n, 3): a, b, log v
        self._w: np.ndarray | None = None
        self._prev_x: float | None = None

    # ------------------------------------------------------------------ #
    def seed_from_ranges(
        self,
        a_range=(-0.05, 0.05),
        b_range=(0.80, 0.999),
        logv_range=(-12.0, -4.0),
    ) -> None:
        """Initialise a diffuse particle cloud from uniform ranges."""
        a = self.rng.uniform(*a_range, self.n)
        b = self.rng.uniform(*b_range, self.n)
        logv = self.rng.uniform(*logv_range, self.n)
        self._theta = np.column_stack([a, b, logv])
        self._w = np.full(self.n, 1.0 / self.n)
        self._prev_x = None

    def seed_from_posterior(self, posterior, seed: int = 0) -> None:
        """Initialise from a :class:`BayesianOUPosterior` (Bayes -> particles)."""
        rng = np.random.default_rng(seed)
        v = 1.0 / rng.gamma(shape=posterior.an, scale=1.0 / posterior.bn, size=self.n)
        L = np.linalg.cholesky(posterior.Vn)
        z = rng.standard_normal((self.n, 2))
        beta = posterior.mn[None, :] + np.sqrt(v)[:, None] * (z @ L.T)
        self._theta = np.column_stack([beta[:, 0], beta[:, 1], np.log(np.clip(v, 1e-12, None))])
        self._w = np.full(self.n, 1.0 / self.n)
        self._prev_x = None

    # ------------------------------------------------------------------ #
    def step(self, x_t: float) -> dict:
        """Advance one observation; return the online parameter posterior."""
        if self._theta is None:
            raise RuntimeError("Call seed_from_ranges()/seed_from_posterior() first.")
        if self._prev_x is None:
            self._prev_x = x_t
            return self._report()

        x_prev = self._prev_x
        theta, w = self._theta, self._w

        # Liu-West kernel shrinkage toward the weighted mean.
        theta_bar = np.average(theta, axis=0, weights=w)
        cov = np.cov(theta.T, aweights=w)
        # Guard against cloud collapse: if the weighted covariance degenerates
        # to (near) zero — e.g. after resampling concentrates the weight — a
        # zero kernel produces no jitter and the filter freezes. A tiny ridge
        # keeps the proposal non-degenerate (and the matrix positive definite).
        cov = np.atleast_2d(np.nan_to_num(cov, nan=0.0))
        cov[np.diag_indices_from(cov)] += 1e-12
        m = self.a_shrink * theta + (1.0 - self.a_shrink) * theta_bar[None, :]

        # Propose jittered particles.
        jitter = self.rng.multivariate_normal(np.zeros(3), self.h2 * cov, size=self.n)
        prop = m + jitter
        a, b, logv = prop[:, 0], prop[:, 1], prop[:, 2]
        v = np.exp(np.clip(logv, -30, 5))

        # Reweight by the transition likelihood of x_t given x_prev.
        resid = x_t - (a + b * x_prev)
        loglik = -0.5 * (np.log(2 * np.pi * v) + resid**2 / v)
        logw = np.log(np.clip(w, 1e-300, None)) + loglik
        logw -= logw.max()
        w_new = np.exp(logw)
        w_new /= w_new.sum()

        # Resample when ESS is low.
        ess = 1.0 / np.sum(w_new**2)
        if ess < self.n / 2.0:
            idx = self.rng.choice(self.n, size=self.n, p=w_new)
            prop = prop[idx]
            w_new = np.full(self.n, 1.0 / self.n)

        self._theta, self._w = prop, w_new
        self._prev_x = x_t
        return self._report()

    # ------------------------------------------------------------------ #
    def _report(self) -> dict:
        a, b = self._theta[:, 0], self._theta[:, 1]
        w = self._w
        dt = 1.0  # report kappa in per-step units; caller divides by dt if needed
        stationary = (b > 0) & (b < 1)
        p_stat = float(np.sum(w[stationary]))
        if p_stat <= 1e-9:
            return {"kappa_mean": 0.0, "kappa_std": float("inf"),
                    "kappa_cv": float("inf"), "p_stationary": p_stat}
        ws = w[stationary] / w[stationary].sum()
        kappa = -np.log(b[stationary])      # per-step
        k_mean = float(np.sum(ws * kappa))
        k_var = float(np.sum(ws * (kappa - k_mean) ** 2))
        k_std = np.sqrt(max(k_var, 0.0))
        return {
            "kappa_mean": k_mean,
            "kappa_std": k_std,
            "kappa_cv": k_std / k_mean if k_mean > 0 else float("inf"),
            "p_stationary": p_stat,
        }
