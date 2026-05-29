"""Stage 2 gate: do derived (Leung-Li) boundaries beat the naive ±z rule?

The roadmap is explicit: "Show the optimal boundaries beat the naive
thresholds after costs and after the Deflated-Sharpe adjustment — or
honestly report that they don't for this spread (a valid, informative
result)."

This script fits an OU on in-sample data, then runs *both* strategies on
identical out-of-sample data with identical costs (trading both legs of
the pair), and prints the head-to-head. It draws no triumphant
conclusion: it reports whichever way the numbers fall.

Run:
    python examples/stage2_optimal_stopping.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.data import SyntheticPair
from stat_arb.data.base import DataSource
from stat_arb.engine import Backtester, LinearCostModel
from stat_arb.stage1 import OUMLEEstimator, ZScoreStrategy, engle_granger
from stat_arb.stage2 import OptimalStoppingStrategy, compute_boundaries
from stat_arb.validation import deflated_sharpe_ratio, sharpe_ci_lo


from stat_arb.data import InMemorySource as FrameSource  # shared frame adapter


def run(strategy, oos_prices) -> dict:
    result = Backtester(strategy, LinearCostModel(bps=5, half_spread_bps=2)).run(
        FrameSource(oos_prices)
    )
    s = result.summary()
    sr, lo, hi = sharpe_ci_lo(result.returns)
    dsr = deflated_sharpe_ratio(result.returns, n_trials=50, sr_variance_across_trials=0.25)
    s.update({"sharpe_lo": lo, "sharpe_hi": hi, "dsr": dsr["dsr"]})
    return s


def main() -> None:
    pair = SyntheticPair(
        beta=1.0, spread_kappa=8.0, spread_mu=0.0, spread_sigma=0.12,
        x_drift=0.05, x_sigma=0.2,
    )
    raw = pair.simulate(n=2520, seed=3)
    cols = list(raw.columns)
    split = len(raw) // 2
    is_raw, oos_raw = raw.iloc[:split], raw.iloc[split:]
    is_log = np.log(is_raw)

    eg = engle_granger(is_log[cols[0]], is_log[cols[1]])
    est = OUMLEEstimator().fit(eg.spread)
    spec = eg.to_pair_spec(cols[0], cols[1], use_log=True)
    print("=== OU fit (in-sample) ===")
    print(est)

    sd = est.sigma / np.sqrt(2 * est.kappa)
    cost_units = 0.05 * sd
    boundaries = compute_boundaries(est, r=0.05, cost=cost_units)
    print("\n=== Derived boundaries (Leung-Li) ===")
    print(boundaries)
    print(f"(stationary SD = {sd:.4f}; entry z = {(boundaries.long_entry-est.mu)/sd:.2f}, "
          f"exit z = {(boundaries.long_exit-est.mu)/sd:.2f})")

    naive = run(ZScoreStrategy(est, spec, entry_z=1.5, exit_z=0.0, stop_z=4.0), oos_raw)
    optimal = run(OptimalStoppingStrategy(est, spec, r=0.05, cost=cost_units), oos_raw)

    print("\n=== Head-to-head (out-of-sample, post-cost) ===")
    metrics = ["total_return", "sharpe", "sharpe_lo", "sharpe_hi", "dsr",
               "max_drawdown", "num_trades", "turnover_annual"]
    print(f"{'metric':18s} {'pmz baseline':>14s} {'Leung-Li':>14s}")
    for m in metrics:
        print(f"{m:18s} {naive[m]:>14.4f} {optimal[m]:>14.4f}")

    print("\n=== Verdict ===")
    better_sharpe = optimal["sharpe"] > naive["sharpe"]
    better_dsr = optimal["dsr"] > naive["dsr"]
    if better_sharpe and better_dsr:
        print("Leung-Li boundaries beat the baseline on both Sharpe and DSR.")
    elif not better_sharpe and not better_dsr:
        print("Leung-Li did NOT beat the baseline here -- an honest negative result.")
    else:
        print("Mixed: one metric favours each. Inspect costs/half-life before elevating.")


if __name__ == "__main__":
    main()
