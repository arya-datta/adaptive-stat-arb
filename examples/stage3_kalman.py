"""Stage 3: does a Kalman dynamic hedge beat the Stage 1 static hedge?

Builds a pair whose hedge ratio *drifts* over time (the realistic case that
breaks static beta), then compares:

* Stage 1: static OLS hedge + OU MLE + ±z baseline.
* Stage 3: Kalman time-varying hedge + z-score on the innovation.

Reports spread variance, half-life stability, and OOS post-cost Sharpe for
both — so the added machinery has to earn its place.

Run:
    python examples/stage3_kalman.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.data import SyntheticPair
from stat_arb.data.base import DataSource
from stat_arb.engine import Backtester, LinearCostModel
from stat_arb.stage1 import OUMLEEstimator, ZScoreStrategy, engle_granger
from stat_arb.stage3 import KalmanHedge, KalmanZScoreStrategy, rolling_ols_hedge
from stat_arb.validation import sharpe_ci_lo


from stat_arb.data import InMemorySource as FrameSource  # shared frame adapter


def drifting_pair(n=2520, seed=11) -> pd.DataFrame:
    """The canonical Stage-3 scenario: a hedge ratio that is *stable in-sample*
    then *drifts out-of-sample*.

    beta = 1.0 over the first (in-sample) half, then ramps smoothly to 1.25
    over the second (out-of-sample) half. So the Stage 1 static OLS fits a
    clean beta ~ 1.0 in-sample (and warm-starts the Kalman there), but that
    fixed hedge goes stale OOS — its residual picks up a (1.0 - beta_t)*X
    term and stops mean-reverting. The Kalman hedge tracks the drift and
    keeps the spread stationary. X is a moderate random walk.
    """
    from stat_arb.data import SyntheticOU
    rng = np.random.default_rng(seed)
    log_x = np.cumsum(rng.standard_normal(n) * 0.012) + np.log(50.0)
    half = n // 2
    beta_t = np.concatenate([
        np.full(half, 1.0),
        1.0 + 0.25 * np.linspace(0, 1, n - half),     # 1.00 -> 1.25 OOS
    ])
    spread = SyntheticOU(kappa=4.0, mu=0.0, sigma=0.10).simulate(n=n, seed=seed + 1)["spread"].to_numpy()
    log_y = beta_t * log_x + spread
    idx = pd.date_range("2010-01-04", periods=n, freq="B")
    return pd.DataFrame({"Y": np.exp(log_y), "X": np.exp(log_x)}, index=idx)


def half_life(spread: np.ndarray) -> float:
    s = spread[np.isfinite(spread)]
    ds = np.diff(s)
    lag = s[:-1]
    beta = np.polyfit(lag, ds, 1)[0]
    return np.log(2) / -beta if beta < 0 else np.inf


def main() -> None:
    raw = drifting_pair()
    cols = list(raw.columns)
    split = len(raw) // 2
    is_raw, oos_raw = raw.iloc[:split], raw.iloc[split:]
    is_log = np.log(is_raw)

    # ----- Stage 1 static hedge -----
    eg = engle_granger(is_log[cols[0]], is_log[cols[1]])
    est = OUMLEEstimator().fit(eg.spread)
    static_spec = eg.to_pair_spec(cols[0], cols[1], use_log=True)
    static_oos_spread = (np.log(oos_raw[cols[0]]) - eg.alpha
                         - eg.beta * np.log(oos_raw[cols[1]])).to_numpy()

    # ----- Stage 3 Kalman hedge (fit q,r in-sample) -----
    kf = KalmanHedge.fit_mle(is_log[cols[0]], is_log[cols[1]])
    kal_oos = kf.filter(np.log(oos_raw[cols[0]]), np.log(oos_raw[cols[1]]))
    kal_oos_spread = kal_oos["spread"].to_numpy()

    print("=== Hedge comparison (out-of-sample) ===")
    print(f"static beta (Stage 1)      : {eg.beta:.3f}  (fixed)")
    print(f"Kalman beta (Stage 3)      : {kal_oos['beta'].iloc[0]:.3f} -> "
          f"{kal_oos['beta'].iloc[-1]:.3f}  (adapts)")
    print(f"spread variance  static    : {np.nanvar(static_oos_spread):.5f}")
    print(f"spread variance  Kalman    : {np.nanvar(kal_oos_spread):.5f}")
    print(f"half-life (days) static    : {half_life(static_oos_spread)*1:.1f}")
    print(f"half-life (days) Kalman    : {half_life(kal_oos_spread)*1:.1f}")

    # ----- OOS backtests, post-cost -----
    cost = LinearCostModel(bps=5, half_spread_bps=2)
    r_static = Backtester(
        ZScoreStrategy(est, static_spec, entry_z=1.5, exit_z=0.0, stop_z=4.0), cost
    ).run(FrameSource(oos_raw))
    r_kalman = Backtester(
        KalmanZScoreStrategy(kf, cols[0], cols[1], entry_z=1.5, exit_z=0.0, stop_z=4.0),
        cost,
    ).run(FrameSource(oos_raw))

    print("\n=== OOS backtest (post-cost) ===")
    for name, res in (("Stage 1 static", r_static), ("Stage 3 Kalman", r_kalman)):
        sr, lo, hi = sharpe_ci_lo(res.returns)
        s = res.summary()
        print(f"{name:16s}: Sharpe {sr:+.2f} [{lo:+.2f},{hi:+.2f}]  "
              f"ret {s['total_return']:+.2%}  maxDD {s['max_drawdown']:.1%}  "
              f"trades {s['num_trades']:.0f}")

    print("\n=== Verdict ===")
    if np.nanvar(kal_oos_spread) < np.nanvar(static_oos_spread):
        print("Kalman produced a lower-variance spread under a drifting hedge.")
    else:
        print("Kalman did not reduce spread variance here -- static hedge sufficed.")


if __name__ == "__main__":
    main()
