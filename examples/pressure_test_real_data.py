"""Pressure test: run Stages 0-2 on real market data.

Screens several economically-coherent candidate pairs, fits the OU model
in-sample, and reports honest out-of-sample, post-cost performance for
both the Stage 1 (+/-z) and Stage 2 (Leung-Li) strategies — including the
multiple-testing-aware Deflated Sharpe across all pairs tried.

This is where the synthetic gates meet reality: most real pairs will fail
the cointegration screen or fail to reproduce OOS, and the honest path is
to say so.

Run (needs `pip install -e ".[yfinance]"`):
    python examples/pressure_test_real_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.data import YFinanceDataSource
from stat_arb.data.base import DataSource
from stat_arb.engine import Backtester, LinearCostModel
from stat_arb.stage1 import OUMLEEstimator, ZScoreStrategy, engle_granger, johansen
from stat_arb.stage2 import OptimalStoppingStrategy, compute_boundaries
from stat_arb.validation import adf_test, deflated_sharpe_ratio, sharpe_ci_lo

START, END = "2018-01-01", "2024-12-31"
IS_FRACTION = 0.5  # ~3.5y in-sample / ~3.5y out-of-sample
CANDIDATES = [
    ("GLD", "IAU"),   # two gold ETFs — near-identical underlying (tight)
    ("VOO", "SPY"),   # two S&P 500 ETFs (tight)
    ("QQQ", "XLK"),   # nasdaq-100 vs tech sector
    ("KO", "PEP"),    # beverages
    ("XOM", "CVX"),   # oil majors
    ("EWA", "EWC"),   # Australia vs Canada country ETFs (classic)
]


from stat_arb.data import InMemorySource as FrameSource  # shared frame adapter


def fetch(y: str, x: str) -> pd.DataFrame:
    df = YFinanceDataSource([y, x], START, END, cache_dir="data/cache").frame()
    return df[[y, x]].dropna()


def backtest(strategy, oos_raw, n_trials) -> dict:
    res = Backtester(strategy, LinearCostModel(bps=5, half_spread_bps=2)).run(
        FrameSource(oos_raw)
    )
    s = res.summary()
    if res.returns.std(ddof=1) > 0 and len(res.returns) >= 30:
        sr, lo, hi = sharpe_ci_lo(res.returns)
        dsr = deflated_sharpe_ratio(
            res.returns, n_trials=n_trials, sr_variance_across_trials=0.5
        )["dsr"]
    else:
        sr, lo, hi, dsr = float("nan"), float("nan"), float("nan"), float("nan")
    s.update({"sharpe_lo": lo, "sharpe_hi": hi, "dsr": dsr})
    return s


def main() -> None:
    n_trials = len(CANDIDATES)
    print(f"Pressure-testing {n_trials} pairs, {START}..{END}\n" + "=" * 64)

    survivors = []
    for y, x in CANDIDATES:
        try:
            raw = fetch(y, x)
        except Exception as exc:  # noqa: BLE001
            print(f"\n{y}/{x}: fetch failed ({exc!r})")
            continue
        if len(raw) < 500:
            print(f"\n{y}/{x}: too few overlapping rows ({len(raw)}) — skip")
            continue

        log = np.log(raw)
        split = int(len(raw) * IS_FRACTION)
        is_log, oos_raw = log.iloc[:split], raw.iloc[split:]

        eg = engle_granger(is_log[y], is_log[x])
        joh = johansen(is_log, det_order=0, k_ar_diff=1)
        print(f"\n{y}/{x}  ({len(raw)} rows, IS={split})")
        print(f"  {eg}")
        print(f"  Johansen rank@5%: {joh.rank}")

        if not eg.cointegrated_at_5pct:
            print("  -> not cointegrated in-sample; retire honestly.")
            continue

        est = OUMLEEstimator().fit(eg.spread)
        if not est.stationary:
            print(f"  -> OU fit not significantly mean-reverting "
                  f"(kappa CI {est.kappa_ci}); retire.")
            continue
        print(f"  OU: kappa={est.kappa:.2f} half-life={est.half_life*252:.0f}d "
              f"sigma={est.sigma:.4f}")

        # OOS stationarity (the real test — does the relationship persist?)
        oos_spread = np.log(oos_raw[y]) - eg.alpha - eg.beta * np.log(oos_raw[x])
        oos_adf = adf_test(oos_spread)
        print(f"  OOS spread ADF: p={oos_adf.pvalue:.3f} "
              f"({'stationary' if oos_adf.stationary_at_5pct else 'NOT stationary'})")

        pair = eg.to_pair_spec(y, x, use_log=True)
        sd = est.sigma / np.sqrt(2 * est.kappa)
        naive = backtest(
            ZScoreStrategy(est, pair, entry_z=1.5, exit_z=0.0, stop_z=4.0),
            oos_raw, n_trials,
        )
        opt = backtest(
            OptimalStoppingStrategy(est, pair, r=0.05, cost=0.05 * sd),
            oos_raw, n_trials,
        )
        survivors.append((f"{y}/{x}", naive, opt))
        print(f"  OOS +/-z   : Sharpe {naive['sharpe']:+.2f} "
              f"[{naive['sharpe_lo']:+.2f},{naive['sharpe_hi']:+.2f}]  "
              f"DSR {naive['dsr']:.2f}  maxDD {naive['max_drawdown']:.1%}  "
              f"trades {naive['num_trades']:.0f}")
        print(f"  OOS LeungLi: Sharpe {opt['sharpe']:+.2f} "
              f"[{opt['sharpe_lo']:+.2f},{opt['sharpe_hi']:+.2f}]  "
              f"DSR {opt['dsr']:.2f}  maxDD {opt['max_drawdown']:.1%}  "
              f"trades {opt['num_trades']:.0f}")

    # ---- summary ----
    print("\n" + "=" * 64 + "\nSUMMARY")
    if not survivors:
        print("No pair survived the in-sample screen + stationarity gate.")
        return
    print(f"{len(survivors)}/{n_trials} pairs passed screening. "
          "Deflated Sharpe accounts for all pairs tried.")
    best = max(survivors, key=lambda t: t[2]["dsr"] if np.isfinite(t[2]["dsr"]) else -1)
    print(f"Best by Leung-Li DSR: {best[0]} (DSR={best[2]['dsr']:.2f}, "
          f"OOS Sharpe={best[2]['sharpe']:+.2f})")
    n_pos_dsr = sum(1 for _, _, o in survivors if np.isfinite(o["dsr"]) and o["dsr"] > 0.95)
    print(f"Pairs with Leung-Li DSR > 0.95 (defensible after multiple testing): "
          f"{n_pos_dsr}/{len(survivors)}")


if __name__ == "__main__":
    main()
