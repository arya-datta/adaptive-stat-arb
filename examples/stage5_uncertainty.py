"""Stage 5: trade your confidence, not just your point estimate.

Demonstrates the gate: an *uncertainty-scaled* book (exposure shrunk when the
posterior over the reversion speed is diffuse or stationarity is ambiguous)
should show better drawdown / tail behaviour than the point-estimate book at
comparable gross return.

1. Bayesian OU posterior (conjugate, exact) on the in-sample spread.
2. Liu-West particle filter learns the posterior online.
3. Backtest the uncertainty-scaled strategy vs its point-estimate control.

Run:
    python examples/stage5_uncertainty.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.data import SyntheticPair
from stat_arb.data.base import DataSource
from stat_arb.engine import Backtester, LinearCostModel
from stat_arb.stage1 import OUMLEEstimator, engle_granger
from stat_arb.stage5 import BayesianOU, ParticleFilterOU, UncertaintyScaledStrategy
from stat_arb.validation import sharpe_ci_lo


from stat_arb.data import InMemorySource as FrameSource  # shared frame adapter


def main() -> None:
    pair = SyntheticPair(beta=1.0, spread_kappa=6.0, spread_mu=0.0,
                         spread_sigma=0.12).simulate(n=3000, seed=4)
    cols = list(pair.columns)
    split = len(pair) // 2
    is_log = np.log(pair.iloc[:split])
    oos = pair.iloc[split:]

    eg = engle_granger(is_log[cols[0]], is_log[cols[1]])
    est = OUMLEEstimator().fit(eg.spread)
    spec = eg.to_pair_spec(cols[0], cols[1])

    # --- Bayesian posterior (uncertainty quantification) ---
    post = BayesianOU().fit(eg.spread)
    print("=== Bayesian OU posterior (in-sample) ===")
    s = post.summary()
    print(f"  kappa  : {s['kappa_mean']:.2f} +/- {s.get('kappa_std', float('nan')):.2f}  "
          f"(CV {s['kappa_cv']:.2f})")
    print(f"  P(mean-reverting) : {s['p_stationary']:.3f}")

    cost = LinearCostModel(bps=5, half_spread_bps=2)

    def run(scale: bool):
        pf = ParticleFilterOU(n_particles=1000, seed=1)
        strat = UncertaintyScaledStrategy(est, spec, pf, entry_z=1.5, exit_z=0.0,
                                          stop_z=4.0, scale_by_uncertainty=scale)
        res = Backtester(strat, cost).run(FrameSource(oos))
        sr, lo, hi = sharpe_ci_lo(res.returns)
        return res.summary(), (sr, lo, hi), float(np.mean(strat.confidence_history))

    point, point_ci, _ = run(False)
    scaled, scaled_ci, conf = run(True)

    print("\n=== OOS: point-estimate book vs uncertainty-scaled book ===")
    hdr = f"{'metric':16s} {'point-estimate':>16s} {'uncertainty-scaled':>20s}"
    print(hdr)
    for k in ["total_return", "ann_vol", "sharpe", "max_drawdown", "num_trades"]:
        print(f"{k:16s} {point[k]:>16.4f} {scaled[k]:>20.4f}")
    print(f"{'sharpe_CI':16s} {f'[{point_ci[1]:.2f},{point_ci[2]:.2f}]':>16s} "
          f"{f'[{scaled_ci[1]:.2f},{scaled_ci[2]:.2f}]':>20s}")
    print(f"mean confidence multiplier (scaled book): {conf:.3f}")

    print("\n=== Verdict ===")
    better_dd = scaled["max_drawdown"] >= point["max_drawdown"]
    lower_vol = scaled["ann_vol"] <= point["ann_vol"]
    if better_dd and lower_vol:
        print("Uncertainty scaling improved drawdown and volatility -- it earned its place.")
    else:
        print("Uncertainty scaling did not improve tail behaviour here (honest result).")


if __name__ == "__main__":
    main()
