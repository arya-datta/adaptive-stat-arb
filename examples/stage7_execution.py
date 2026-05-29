"""Stage 7: does the edge survive realistic execution?

Two parts:
1. Almgren-Chriss scheduling -- the impact-vs-timing-risk efficient frontier,
   and how the optimal trajectory front-loads as risk aversion rises.
2. The honest re-run: take the Stage 1 baseline and price it through
   frictionless -> Stage-0 linear stub -> Stage-7 microstructure model
   (square-root impact + partial fills). The post-microstructure Sharpe is the
   number you actually defend.

Run:
    python examples/stage7_execution.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.data import SyntheticPair
from stat_arb.data.base import DataSource
from stat_arb.engine import Backtester, LinearCostModel, ZeroCostModel
from stat_arb.stage1 import OUMLEEstimator, ZScoreStrategy, engle_granger
from stat_arb.stage7 import (
    MicrostructureCostModel, almgren_chriss_schedule, execution_frontier, twap_schedule,
)
from stat_arb.validation import deflated_sharpe_ratio, sharpe_ci_lo


class FrameSource(DataSource):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def frame(self) -> pd.DataFrame:
        return self._df


def main() -> None:
    # --- 1. Almgren-Chriss schedule + frontier ---
    print("=== Almgren-Chriss execution (liquidate 100k shares over 20 steps) ===")
    sigma, eta = 0.5, 0.5
    twap = twap_schedule(100_000, 20)
    ac = almgren_chriss_schedule(100_000, n_steps=20, sigma=sigma, eta=eta, lam=1e-2)
    print(f"TWAP holds {twap.holdings[10]:,.0f} at the midpoint; "
          f"AC holds {ac.holdings[10]:,.0f} (front-loaded, kappa={ac.kappa:.3f})")
    print("\nimpact-vs-timing-risk frontier (more urgency -> more cost, less risk):")
    for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
        c, sd = execution_frontier(100_000, 20, sigma=sigma, eta=eta, lambdas=[lam])[0]
        print(f"  lambda={lam:.0e}:  E[impact]={c:,.3e}   timing_std={sd:,.3e}")

    # --- 2. Edge survival under progressively realistic execution ---
    pair = SyntheticPair(beta=1.0, spread_kappa=8.0, spread_mu=0.0,
                         spread_sigma=0.12).simulate(n=2500, seed=4)
    cols = list(pair.columns)
    split = len(pair) // 2
    eg = engle_granger(np.log(pair.iloc[:split][cols[0]]), np.log(pair.iloc[:split][cols[1]]))
    est = OUMLEEstimator().fit(eg.spread)
    spec = eg.to_pair_spec(cols[0], cols[1])
    oos = pair.iloc[split:]

    def run(model, label):
        strat = ZScoreStrategy(est, spec, entry_z=1.5, exit_z=0.0, stop_z=4.0)
        res = Backtester(strat, model).run(FrameSource(oos))
        sr, lo, hi = sharpe_ci_lo(res.returns)
        dsr = deflated_sharpe_ratio(res.returns, n_trials=50, sr_variance_across_trials=0.25)
        s = res.summary()
        return label, sr, lo, hi, dsr["dsr"], s["total_return"]

    models = [
        (ZeroCostModel(), "frictionless"),
        (LinearCostModel(bps=5, half_spread_bps=2), "Stage 0 linear stub"),
        (MicrostructureCostModel(adv=150_000, participation_cap=0.2, impact_coef=1.0,
                                 daily_vol=0.02, half_spread_bps=2, commission_bps=1),
         "Stage 7 microstructure"),
    ]
    print("\n=== Edge survival (Stage 1 baseline, OOS) ===")
    print(f"{'execution model':24s} {'Sharpe':>8s} {'95% CI':>16s} {'DSR':>6s} {'return':>9s}")
    for model, label in models:
        _, sr, lo, hi, dsr, ret = run(model, label)
        print(f"{label:24s} {sr:>8.2f} {f'[{lo:.2f},{hi:.2f}]':>16s} {dsr:>6.2f} {ret:>9.2%}")

    print("\n=== Verdict ===")
    print("The number you defend is the post-microstructure Sharpe/DSR -- mid-price-only "
          "edges\nthat evaporate under square-root impact and partial fills get retired honestly.")


if __name__ == "__main__":
    main()
