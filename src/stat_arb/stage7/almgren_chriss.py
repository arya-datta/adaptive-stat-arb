r"""Almgren-Chriss optimal execution scheduling (Stage 7).

Liquidating (or building) a position of :math:`X` shares over :math:`N`
intervals trades **market impact** (trade fast, pay slippage) against
**timing risk** (trade slow, ride volatility). Almgren & Chriss (2000) give the
mean-variance-optimal trajectory in closed form. The holdings path is

.. math::

   x_j = X\,\frac{\sinh\!\big(\kappa (T - t_j)\big)}{\sinh(\kappa T)},
   \qquad \kappa = \sqrt{\lambda\,\sigma^2 / \eta},

where :math:`\eta` is the temporary-impact coefficient, :math:`\sigma` the
volatility, and :math:`\lambda` the risk aversion. As :math:`\lambda \to 0`
(risk-neutral) :math:`\kappa \to 0` and the schedule collapses to **TWAP**
(linear liquidation); as :math:`\lambda` grows the schedule front-loads to cut
risk at the cost of impact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExecutionSchedule:
    """A liquidation schedule and its cost/risk profile."""

    holdings: np.ndarray         # (N+1,) shares remaining at each time node
    trades: np.ndarray           # (N,) shares executed per interval (>0 = sell)
    expected_cost: float         # implementation shortfall (permanent + temporary)
    variance: float              # variance of execution cost (timing risk)
    kappa: float                 # urgency parameter (0 = TWAP)

    @property
    def cost_std(self) -> float:
        return float(np.sqrt(max(self.variance, 0.0)))


def almgren_chriss_schedule(
    total_shares: float,
    n_steps: int,
    sigma: float,
    eta: float,
    lam: float,
    tau: float = 1.0,
    gamma: float = 0.0,
) -> ExecutionSchedule:
    r"""Mean-variance-optimal liquidation trajectory.

    Parameters
    ----------
    total_shares:
        Position size ``X`` to liquidate.
    n_steps:
        Number of trading intervals ``N``.
    sigma:
        Volatility per interval (price units per share).
    eta:
        Temporary-impact coefficient (cost per unit trading rate).
    lam:
        Risk aversion ``lambda`` (0 = risk-neutral / TWAP).
    tau:
        Interval length.
    gamma:
        Permanent-impact coefficient (linear).
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    X, N, T = float(total_shares), int(n_steps), n_steps * tau
    t = np.arange(N + 1) * tau

    if lam <= 0 or eta <= 0:
        kappa = 0.0
        holdings = X * (1.0 - t / T)                     # TWAP (linear)
    else:
        kappa = float(np.sqrt(lam * sigma**2 / eta))
        # sinh(k(T-t))/sinh(kT) rewritten to avoid overflow for large kT:
        #   = exp(-k t) * (1 - exp(-2k(T-t))) / (1 - exp(-2kT))
        a = kappa * (T - t)
        denom = 1.0 - np.exp(-2.0 * kappa * T)
        ratio = np.exp(-kappa * t) * (1.0 - np.exp(-2.0 * a)) / denom
        holdings = X * ratio
    holdings[0] = X
    holdings[-1] = 0.0

    trades = -np.diff(holdings)                          # shares sold per interval
    expected_cost = 0.5 * gamma * X**2 + (eta / tau) * float(np.sum(trades**2))
    variance = float(sigma**2 * tau * np.sum(holdings[1:] ** 2))
    return ExecutionSchedule(holdings=holdings, trades=trades,
                             expected_cost=expected_cost, variance=variance, kappa=kappa)


def twap_schedule(total_shares: float, n_steps: int) -> ExecutionSchedule:
    """Time-Weighted Average Price: equal-sized child orders."""
    X, N = float(total_shares), int(n_steps)
    trades = np.full(N, X / N)
    holdings = np.concatenate([[X], X - np.cumsum(trades)])
    holdings[-1] = 0.0
    return ExecutionSchedule(holdings=holdings, trades=trades,
                             expected_cost=float("nan"), variance=float("nan"), kappa=0.0)


def vwap_schedule(total_shares: float, volume_profile: np.ndarray) -> ExecutionSchedule:
    """Volume-Weighted Average Price: child orders track the volume curve."""
    X = float(total_shares)
    vp = np.asarray(volume_profile, float)
    if np.any(vp < 0) or vp.sum() <= 0:
        raise ValueError("volume_profile must be non-negative with positive sum.")
    weights = vp / vp.sum()
    trades = X * weights
    holdings = np.concatenate([[X], X - np.cumsum(trades)])
    holdings[-1] = 0.0
    return ExecutionSchedule(holdings=holdings, trades=trades,
                             expected_cost=float("nan"), variance=float("nan"), kappa=0.0)


def execution_frontier(
    total_shares: float,
    n_steps: int,
    sigma: float,
    eta: float,
    lambdas: np.ndarray,
    tau: float = 1.0,
    gamma: float = 0.0,
) -> list[tuple[float, float]]:
    """Sweep risk aversion to trace the (expected-cost, cost-std) frontier."""
    out = []
    for lam in np.asarray(lambdas, float):
        sched = almgren_chriss_schedule(total_shares, n_steps, sigma, eta, lam, tau, gamma)
        out.append((sched.expected_cost, sched.cost_std))
    return out
