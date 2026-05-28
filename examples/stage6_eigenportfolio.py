"""Stage 6: from pairs to a portfolio engine (breadth).

Runs the Avellaneda-Lee eigenportfolio strategy on a multi-asset universe:
rolling PCA extracts common factors, each name's residual s-score drives a
dollar-neutral long-short book. Reports the multivariate validation that
becomes essential at breadth: Lo-2002 Sharpe CI, Deflated Sharpe over the
number of names screened, and PBO.

Run:
    python examples/stage6_eigenportfolio.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.data import SyntheticFactorMarket
from stat_arb.data.base import DataSource
from stat_arb.engine import Backtester, LinearCostModel
from stat_arb.stage6 import EigenportfolioStrategy, ledoit_wolf, hierarchical_risk_parity
from stat_arb.validation import (
    deflated_sharpe_ratio, harvey_liu_zhu_hurdle, sharpe_ci_lo,
)


class FrameSource(DataSource):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def frame(self) -> pd.DataFrame:
        return self._df


def main() -> None:
    n_stocks = 25
    market = SyntheticFactorMarket(
        n_stocks=n_stocks, n_factors=3, factor_vol=0.012,
        resid_kappa=12.0, resid_sigma=0.04,
    ).simulate(n=1500, seed=0)
    symbols = list(market.columns)

    # Risk picture: shrinkage + HRP on the return covariance.
    R = np.diff(np.log(market.to_numpy()), axis=0)
    lw = ledoit_wolf(R)
    hrp = hierarchical_risk_parity(lw["cov"])
    print(f"=== Universe: {n_stocks} names, 3 latent factors ===")
    print(f"Ledoit-Wolf shrinkage intensity: {lw['shrinkage']:.3f}")
    print(f"HRP weight range: [{hrp.min():.3f}, {hrp.max():.3f}] (sum={hrp.sum():.3f})")

    strat = EigenportfolioStrategy(symbols, n_factors=3, lookback=60,
                                   recalc_every=5, s_entry=1.25, s_close=0.5)
    result = Backtester(strat, LinearCostModel(bps=5, half_spread_bps=2)).run(
        FrameSource(market)
    )
    s = result.summary()
    sr, lo, hi = sharpe_ci_lo(result.returns)

    print("\n=== Cross-sectional book (post-cost) ===")
    for k in ["total_return", "ann_vol", "sharpe", "max_drawdown", "num_trades"]:
        print(f"  {k:16s}: {s[k]:.4f}")
    print(f"  mean factor variance explained: "
          f"{np.mean(strat.explained_variance_history):.3f}")

    print("\n=== Validation at breadth ===")
    print(f"Lo-2002 Sharpe 95% CI : {sr:.2f}  [{lo:.2f}, {hi:.2f}]")
    # Each name is a candidate signal -> deflate by the number screened.
    dsr = deflated_sharpe_ratio(result.returns, n_trials=n_stocks,
                                sr_variance_across_trials=0.5)
    print(f"Deflated Sharpe (N={n_stocks}) : DSR={dsr['dsr']:.3f}  (SR0={dsr['sr0']:.3f})")
    hlz = harvey_liu_zhu_hurdle(sr, n_obs=len(result.returns))
    print(f"Harvey-Liu-Zhu t={hlz['t_stat']:.2f}  passes t>3: {hlz['passes']}")

    print("\n=== Verdict ===")
    print("Breadth (25 residuals) vs a single pair: the portfolio book diversifies "
          "idiosyncratic\nrisk; defend only the post-DSR, multiple-testing-adjusted number.")


if __name__ == "__main__":
    main()
