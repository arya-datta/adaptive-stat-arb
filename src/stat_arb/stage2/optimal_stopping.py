r"""Leung-Li optimal entry/exit boundaries for an OU spread.

Setup. Hold the OU spread :math:`X_t`, transaction cost :math:`c_s` to
exit, :math:`c_b` to enter, discount rate :math:`r > 0`.

Sell (liquidation) value:

.. math::

   V(x) = \sup_{\tau \ge 0}\; \mathbb{E}_x\bigl[e^{-r\tau}(X_\tau - c_s)\bigr].

The ODE :math:`\mathcal{L}V - rV = 0` (with :math:`\mathcal{L}` the OU
generator) has two linearly independent positive solutions

.. math::

   F(x) = \int_0^\infty u^{r/\kappa - 1}\,
            \exp\!\bigl(\tfrac{\sqrt{2\kappa}}{\sigma}(x-\mu) u
                        - \tfrac{u^2}{2}\bigr)\,du
   \qquad (\text{increasing in } x), \\[2pt]
   G(x) = \int_0^\infty u^{r/\kappa - 1}\,
            \exp\!\bigl(-\tfrac{\sqrt{2\kappa}}{\sigma}(x-\mu) u
                        - \tfrac{u^2}{2}\bigr)\,du
   \qquad (\text{decreasing in } x).

Smooth-pasting gives :math:`V(x) = (b^* - c_s) F(x)/F(b^*)` for
:math:`x \le b^*`, and the optimal liquidation level :math:`b^*` solves

.. math::

   F(b^*) = (b^* - c_s)\, F'(b^*).

The entry value :math:`J(x) = \sup_\tau \mathbb{E}_x[e^{-r\tau}(V(X_\tau)
- X_\tau - c_b)]` is solved analogously using :math:`G`. Optimal entry
:math:`d^*` solves

.. math::

   G(d^*)\bigl(V'(d^*) - 1\bigr) = G'(d^*)\bigl(V(d^*) - d^* - c_b\bigr).

References:
Leung & Li (2015); Avellaneda & Lee (2010) for the stat-arb context.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from ..stage1.ou_mle import OUParams


# -------------------------------------------------------------------- #
# Fundamental solutions                                                #
# -------------------------------------------------------------------- #
@dataclass(frozen=True)
class OUFundamentals:
    """Closures over the OU parameters and discount rate.

    Centralises the integral evaluations so the boundary root-finders
    don't recompute the prefactor on every call.
    """

    kappa: float
    mu: float
    sigma: float
    r: float                # discount rate (per year)

    @property
    def _alpha(self) -> float:
        return self.r / self.kappa

    @property
    def _scale(self) -> float:
        """:math:`\\sqrt{2\\kappa}/\\sigma`."""
        return float(np.sqrt(2.0 * self.kappa) / self.sigma)

    # -- core integrals -------------------------------------------------
    def F(self, x: float) -> float:
        return self._integral(x, sign=+1, power_offset=0)

    def G(self, x: float) -> float:
        return self._integral(x, sign=-1, power_offset=0)

    def F_prime(self, x: float) -> float:
        return self._scale * self._integral(x, sign=+1, power_offset=1)

    def G_prime(self, x: float) -> float:
        return -self._scale * self._integral(x, sign=-1, power_offset=1)

    # -- machinery ------------------------------------------------------
    def _integral(self, x: float, *, sign: int, power_offset: int) -> float:
        """Compute ``∫_0^∞ u^(α + offset - 1) exp(sign · s · u - u²/2) du``.

        ``s = scale · (x - μ)``. We integrate with ``scipy.quad`` over a
        finite domain chosen so the truncation error is below ``1e-12``.
        """
        s = sign * self._scale * (x - self.mu)
        alpha = self._alpha
        # Locate the integrand's peak: d/du[(α+offset-1)ln u + s u - u²/2] = 0
        # → u² - s u - (α + offset - 1) = 0
        c = (alpha + power_offset) - 1.0
        disc = s * s + 4.0 * c
        if disc < 0:
            u_star = max(0.5, s)
        else:
            u_star = max(0.0, (s + np.sqrt(disc)) / 2.0)
        # The integrand decays like exp(-u²/2 + s u) past u_star; choose
        # an upper bound 8 standard deviations beyond the peak.
        u_max = max(u_star + 8.0, 12.0)

        # The factor u^(α-1) is integrably singular at u=0 when α < 1
        # (i.e. r < κ, the common case). Pull it into scipy.quad's algebraic
        # weight so the remaining integrand is smooth — this both removes the
        # "probably divergent" warning and improves accuracy.
        def core(u: float) -> float:
            return float(u ** power_offset * np.exp(s * u - 0.5 * u * u))

        val, _ = quad(
            core, 0.0, u_max,
            weight="alg", wvar=(alpha - 1.0, 0.0), limit=200,
        )
        return float(val)


# -------------------------------------------------------------------- #
# Boundary solver                                                      #
# -------------------------------------------------------------------- #
@dataclass(frozen=True)
class OptimalStoppingBoundaries:
    """All four boundaries (long + short) in the spread's original units.

    ``long_entry`` and ``long_exit`` are the Leung-Li :math:`d^*` and
    :math:`b^*` (long position, profit at the upper boundary). The short
    side mirrors via the reflection :math:`x \\to 2\\mu - x`.

    Setting ``has_long=False`` or ``has_short=False`` indicates the
    solver could not find a finite economically-meaningful boundary
    (e.g. transaction costs too high to support a position).
    """

    long_entry: float
    long_exit: float
    short_entry: float
    short_exit: float
    has_long: bool
    has_short: bool

    def __str__(self) -> str:  # pragma: no cover
        L = (f"long: enter<={self.long_entry:.4f}, exit>={self.long_exit:.4f}"
             if self.has_long else "long: infeasible")
        S = (f"short: enter>={self.short_entry:.4f}, exit<={self.short_exit:.4f}"
             if self.has_short else "short: infeasible")
        return f"OptimalStoppingBoundaries({L}; {S})"


def compute_boundaries(
    params: OUParams,
    *,
    r: float = 0.05,
    cost: float = 0.0,
    include_short: bool = True,
) -> OptimalStoppingBoundaries:
    """Solve Leung-Li for the OU process described by ``params``.

    Parameters
    ----------
    params:
        Fitted :class:`OUParams` from :class:`stat_arb.stage1.OUMLEEstimator`.
    r:
        Annualised discount rate. Higher ``r`` makes the agent more
        impatient and narrows the wait region.
    cost:
        Round-trip transaction cost in spread units. The buy-side
        ``c_b`` and sell-side ``c_s`` are both set to ``cost`` here for
        symmetry; pass two separate costs by calling the lower-level
        :func:`_solve_long` and :func:`_solve_short` directly.
    include_short:
        Whether to compute the short-side boundaries by reflection.

    Notes
    -----
    If the fitted half-life is shorter than ~2 observation bars
    (:math:`\\kappa\\,\\Delta t` large), the spread reverts faster than a
    daily strategy can act on it. Such fits are untradeable in this
    framework — and numerically degenerate for the fundamental-solution
    integrals — so we return an all-infeasible result rather than chase a
    spurious boundary.
    """
    # Economic + numerical guard: reject sub-bar half-lives up front.
    half_life_bars = params.half_life / params.dt if params.dt > 0 else np.inf
    if not np.isfinite(half_life_bars) or half_life_bars < 2.0:
        nan = float("nan")
        return OptimalStoppingBoundaries(
            long_entry=nan, long_exit=nan, short_entry=nan, short_exit=nan,
            has_long=False, has_short=False,
        )

    fund = OUFundamentals(
        kappa=params.kappa, mu=params.mu, sigma=params.sigma, r=r
    )
    long_entry, long_exit, has_long = _solve_long(fund, c_b=cost, c_s=cost)

    if include_short and has_long:
        # OU is symmetric about its *own* mean μ, and the costs here are
        # symmetric (c_b = c_s), so the short position is the long position
        # reflected about μ: X' = 2μ − X is the identical OU law. Hence the
        # short boundaries are a mirror image — no second solve required.
        mu = params.mu
        short_entry = 2.0 * mu - long_entry   # enter short when spread is high
        short_exit = 2.0 * mu - long_exit     # exit short when spread is low
        has_short = True
    else:
        short_entry, short_exit, has_short = float("nan"), float("nan"), False

    return OptimalStoppingBoundaries(
        long_entry=long_entry,
        long_exit=long_exit,
        short_entry=short_entry,
        short_exit=short_exit,
        has_long=has_long,
        has_short=has_short,
    )


# -------------------------------------------------------------------- #
# Internal: long-only Leung-Li                                         #
# -------------------------------------------------------------------- #
def _solve_long(
    fund: OUFundamentals,
    *,
    c_b: float,
    c_s: float,
) -> tuple[float, float, bool]:
    """Return ``(d*, b*, ok)`` for the long-only problem."""
    sd = fund.sigma / np.sqrt(2.0 * fund.kappa)        # stationary SD
    mu = fund.mu

    # --- (1) liquidation boundary b* ---------------------------------
    # Optimality: F(b) = (b - c_s) F'(b)  ⇒  h(b) := F(b) - (b - c_s) F'(b) = 0.
    # h(μ) > 0 (we're below the take-profit region); h(μ + 6σ_ss) < 0 (well into it).
    def h_exit(b: float) -> float:
        return fund.F(b) - (b - c_s) * fund.F_prime(b)

    b_lo = mu + 1e-6
    b_hi = mu + 8.0 * sd + max(c_s, 0.0) * 2.0
    try:
        b_star = brentq(h_exit, b_lo, b_hi, xtol=1e-8, rtol=1e-8)
    except ValueError:
        return float("nan"), float("nan"), False

    F_b = fund.F(b_star)

    # --- value function V(x) for x ≤ b* ------------------------------
    def V(x: float) -> float:
        if x >= b_star:
            return x - c_s
        return (b_star - c_s) * fund.F(x) / F_b

    def V_prime(x: float) -> float:
        if x >= b_star:
            return 1.0
        return (b_star - c_s) * fund.F_prime(x) / F_b

    # --- (2) entry boundary d* ---------------------------------------
    # Optimality: G(d)(V'(d) − 1) = G'(d)(V(d) − d − c_b)
    def h_entry(d: float) -> float:
        return fund.G(d) * (V_prime(d) - 1.0) - fund.G_prime(d) * (V(d) - d - c_b)

    d_lo = mu - 8.0 * sd - max(c_b, 0.0) * 2.0
    d_hi = b_star - 1e-6
    try:
        d_star = brentq(h_entry, d_lo, d_hi, xtol=1e-8, rtol=1e-8)
    except ValueError:
        return float("nan"), float(b_star), False

    # Economic-feasibility check: entry must beat the do-nothing option
    if V(d_star) - d_star - c_b <= 0:
        return float("nan"), float(b_star), False

    return float(d_star), float(b_star), True
