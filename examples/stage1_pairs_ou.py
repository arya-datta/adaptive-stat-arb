"""Stage 1: cointegration screen + continuous-time OU MLE + ±z baseline.

Pipeline:
    1. Build a (synthetic) cointegrated pair.   [swap in YFinanceDataSource for real data]
    2. Engle-Granger + Johansen screening.
    3. Fit the OU SDE to the in-sample spread by exact-discretisation MLE.
    4. Backtest the ±z baseline out-of-sample, post-cost.
    5. Report the validation spine: ADF (IS/OOS), Lo-2002 Sharpe CI, Deflated Sharpe.

Run:
    python examples/stage1_pairs_ou.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.data import SyntheticPair
from stat_arb.data.base import DataSource
from stat_arb.engine import Backtester, LinearCostModel
from stat_arb.stage1 import OUMLEEstimator, ZScoreStrategy, engle_granger, johansen
from stat_arb.validation import (
    adf_test, deflated_sharpe_ratio, sharpe_ci_lo,
)


class FrameSource(DataSource):
    """Wrap a price frame (the two legs) as a DataSource."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def frame(self) -> pd.DataFrame:
        return self._df


def build_pair() -> pd.DataFrame:
    # ---- Synthetic (reproducible, offline) ----
    # Fast-reverting spread (half-life ~22 trading days) so the pair is
    # unambiguously cointegrated even on the in-sample half.
    pair = SyntheticPair(
        beta=1.2, spread_kappa=8.0, spread_mu=0.0, spread_sigma=0.12,
        x_drift=0.05, x_sigma=0.2,
    )
    return pair.simulate(n=2520, seed=7)

    # ---- Real data (uncomment; needs `pip install -e ".[yfinance]"`) ----
    # from stat_arb.data import YFinanceDataSource
    # return YFinanceDataSource(["KO", "PEP"], "2014-01-01", "2024-01-01",
    #                           cache_dir="data/cache").frame()


def main() -> None:
    raw_prices = build_pair()                 # actual (positive) leg prices
    log_prices = np.log(raw_prices)
    split = len(raw_prices) // 2
    cols = list(raw_prices.columns)           # [Y, X]

    is_log, oos_raw = log_prices.iloc[:split], raw_prices.iloc[split:]

    # --- (2) cointegration screen on the in-sample window ---
    eg = engle_granger(is_log[cols[0]], is_log[cols[1]])
    joh = johansen(is_log, det_order=0, k_ar_diff=1)
    print("=== Stage 1: screening (in-sample) ===")
    print(eg)
    print(f"Johansen rank @5% : {joh.rank}")

    if not eg.cointegrated_at_5pct:
        print("Not cointegrated in-sample -- stop here (honest negative result).")
        return

    # --- (3) OU MLE on the in-sample spread ---
    est = OUMLEEstimator().fit(eg.spread)
    print("\n=== OU MLE (in-sample) ===")
    print(est)
    print(f"half-life (years): {est.half_life:.3f}  (~{est.half_life*252:.0f} trading days)")

    # Build the pair spec from the IS fit (no look-ahead into OOS).
    pair = eg.to_pair_spec(y_symbol=cols[0], x_symbol=cols[1], use_log=True)

    # OOS spread, reconstructed with the IS hedge ratio, for stationarity check.
    oos_log = np.log(oos_raw)
    oos_spread = oos_log[cols[0]] - eg.alpha - eg.beta * oos_log[cols[1]]

    # --- validation: stationarity on both windows ---
    print("\n=== Stationarity (validation spine) ===")
    print("IS :", adf_test(eg.spread))
    print("OOS:", adf_test(oos_spread))

    # --- (4) backtest the ±z baseline OOS, post-cost, trading both legs ---
    strat = ZScoreStrategy(est, pair, entry_z=1.5, exit_z=0.0, stop_z=4.0)
    result = Backtester(strat, LinearCostModel(bps=5, half_spread_bps=2)).run(
        FrameSource(oos_raw)
    )
    s = result.summary()
    print("\n=== +/-z baseline (out-of-sample, post-cost) ===")
    for k, v in s.items():
        print(f"  {k:18s}: {v:,.4f}")

    # --- (5) honest Sharpe inference ---
    sr, lo, hi = sharpe_ci_lo(result.returns)
    print(f"\nLo-2002 Sharpe 95% CI : {sr:.3f}  [{lo:.3f}, {hi:.3f}]")
    # Pretend we screened ~50 candidate pairs to get here.
    dsr = deflated_sharpe_ratio(result.returns, n_trials=50, sr_variance_across_trials=0.25)
    print(f"Deflated Sharpe (N=50): DSR={dsr['dsr']:.3f}  (SR0={dsr['sr0']:.3f})")


if __name__ == "__main__":
    main()
