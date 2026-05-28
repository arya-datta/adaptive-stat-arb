# Architecture

The repository implements Stages 0–2 of the staged roadmap. The guiding
constraint is the roadmap's first principle — *validation is the spine* —
so the design keeps signal generation, execution, accounting, and
statistical validation in separate, independently testable components.

## Module map

```
src/stat_arb/
├── data/          # DataSource abstraction + loaders
│   ├── base.py        DataSource ABC, Bar value object
│   ├── synthetic.py   SyntheticOU, SyntheticPair  (exact-transition simulation)
│   ├── csv_source.py  CSVDataSource (wide file or per-symbol directory)
│   ├── yfinance_source.py  adjusted-close, parquet cache
│   └── openbb_source.py    multi-provider via OpenBB
├── engine/        # Stage 0: event-driven backtester
│   ├── events.py      MarketEvent → SignalEvent → OrderEvent → FillEvent
│   ├── strategy.py    Strategy ABC
│   ├── portfolio.py   cash/positions, weights → orders
│   ├── execution.py   ExecutionModel ABC; ZeroCostModel, LinearCostModel
│   ├── accounting.py  BacktestResult (equity, trades, summary metrics)
│   └── backtester.py  the event loop
├── validation/    # the spine, used by every stage
│   ├── stationarity.py   ADF + KPSS
│   ├── sharpe.py         Lo-2002 SE & CI under autocorrelation
│   ├── deflated_sharpe.py Bailey-LdP DSR
│   ├── purged_cv.py      PurgedKFold with embargo
│   ├── walk_forward.py   anchored & rolling splits
│   └── pbo.py            CSCV Probability of Backtest Overfitting
├── stage1/        # static cointegration + continuous-time OU
│   ├── cointegration.py  Engle-Granger, Johansen
│   ├── ou_mle.py         exact-discretisation MLE + Fisher-info CIs
│   ├── pair.py           PairSpec: spread definition + leg weights
│   └── strategy.py       ZScoreStrategy (±z baseline)
├── stage2/        # optimal stopping
│   ├── optimal_stopping.py  F/G fundamentals + boundary root-finders
│   └── strategy.py          OptimalStoppingStrategy
└── calibration/   # parallel parameter sweeps
    └── harness.py     grid_search over a ProcessPoolExecutor
```

## The event-driven flow (Stage 0)

For every bar `t` (a `MarketEvent`), the `Backtester` loop runs:

1. **Settle pending orders.** Orders queued on bar `t-1` are filled at bar
   `t`'s price. This single one-bar deferral is what eliminates the most
   common look-ahead bug — an order priced at the same close that
   triggered it.
2. **Mark to market.** The portfolio's equity is recorded at `t`.
3. **Generate signals.** The `Strategy` sees the `MarketEvent` and may emit
   a `SignalEvent` (target weights). The `Portfolio` reconciles target vs
   actual weights into `OrderEvent`s, which queue for bar `t+1`.

Costs are charged from the very first backtest via the `ExecutionModel`
(Stage 0's `LinearCostModel` = fixed bps + half-spread), so no reported
Sharpe is ever cost-free. The crude model is replaced by a
microstructure-aware one in Stage 7 (future work) without touching the
strategy or accounting code.

## Why pairs trade two legs

A cointegration spread is a *dimensionless, zero-centred log-residual*, not
a tradable price — you cannot "buy `equity/price` shares" of something that
oscillates through zero. So the strategies (`ZScoreStrategy`,
`OptimalStoppingStrategy`) emit dollar weights on the **two legs** of the
pair via `PairSpec.leg_weights`: long-spread = long `Y`, short `beta·X`,
scaled to a target gross exposure. The multi-asset `Portfolio` then handles
the legs as ordinary positively-priced instruments, and the spread's P&L
emerges naturally from the two legs with realistic per-leg costs.

## How the stages compose

* **Stage 1 must beat Stage 0.** The `±z` baseline is deliberately naive so
  Stage 2 has a documented benchmark.
* **Stage 2 must beat Stage 1.** `OptimalStoppingStrategy` derives its
  entry/exit boundaries from the SDE + costs and is compared head-to-head
  with `ZScoreStrategy` on identical OOS data (see
  `examples/stage2_optimal_stopping.py`). The script reports the honest
  verdict — including "the optimal rule did not win here".

Every stage reports the same honesty metrics from `validation/`:
out-of-sample, post-cost, Deflated-Sharpe-adjusted, with Lo-2002 Sharpe
confidence intervals rather than point estimates.

## Data sources are swappable

The engine consumes only the `DataSource` interface (`frame()` + iteration
of `Bar`s). Swapping `SyntheticPair` for `YFinanceDataSource`,
`CSVDataSource`, or `OpenBBDataSource` is a one-line constructor change in
the examples. Optional dependencies (`yfinance`, `openbb`) fail soft: the
package imports fine without them.
