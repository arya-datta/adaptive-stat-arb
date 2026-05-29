# Adaptive Statistical Arbitrage (Stages 0–2)

Implementation of the first three stages of the *Adaptive Statistical Arbitrage
via Regime-Switching Ornstein–Uhlenbeck Processes* roadmap:

| Stage | Title                                                          | Status |
|-------|----------------------------------------------------------------|--------|
| 0     | Data spine + event-driven backtest engine                      | ✓      |
| 1     | Static cointegration + continuous-time OU MLE                  | ✓      |
| 2     | Optimal stopping (Leung & Li free-boundary)                    | ✓      |
| 3     | Kalman-filtered dynamic cointegration                          | ✓      |
| 4     | Regime-switching OU (HMM via EM / Hamilton filter)             | ✓      |
| 5     | Bayesian OU / particle filtering (uncertainty-aware sizing)     | ✓      |
| 6     | Multivariate stat-arb (PCA eigenportfolios, VECM, HRP)         | ✓      |
| 7     | Microstructure-aware execution (square-root impact, Almgren-Chriss) | ✓ |
| 8     | Deep-learning extensions                                       | ✗ (future) |

The three governing principles of the roadmap apply throughout:

1. **Validation is the spine.** Every stage reports the same honesty metrics —
   out-of-sample, post-cost, overfitting-adjusted. See
   [`src/stat_arb/validation`](src/stat_arb/validation).
2. **Justified complexity only.** No machinery is added without a test that
   shows it earns its place. Each `stage*/` package ships with a strategy that
   *must beat* the previous stage on identical OOS data.
3. **Each stage is a self-contained, validated deliverable.** Stages 1–2 cannot
   be trusted if Stage 0 is wrong; the Stage 0 gate (a buy-and-hold backtest
   that ties out to hand calculation) lives in
   [`tests/test_engine.py`](tests/test_engine.py).

## Installation

```bash
pip install -e .                  # core
pip install -e ".[yfinance,dev]"  # + market data + tests
```

## Quickstart

```python
import numpy as np
from stat_arb.data import SyntheticPair
from stat_arb.data.base import DataSource
from stat_arb.stage1 import OUMLEEstimator, ZScoreStrategy, engle_granger
from stat_arb.engine import Backtester, LinearCostModel


class FrameSource(DataSource):
    def __init__(self, df): self._df = df
    def frame(self): return self._df


# 1. Simulate a cointegrated pair (or swap in YFinanceDataSource / CSV / OpenBB)
raw = SyntheticPair(beta=1.0, spread_kappa=8.0, spread_mu=0.0,
                    spread_sigma=0.12).simulate(n=2520, seed=0)
cols = list(raw.columns)

# 2. Screen for cointegration, then fit the OU SDE to the spread
eg  = engle_granger(np.log(raw[cols[0]]), np.log(raw[cols[1]]))
est = OUMLEEstimator().fit(eg.spread)
print(est)   # kappa, mu, sigma with 95% CIs, half-life, stationarity flag

# 3. Backtest the ±z baseline — trades BOTH legs, post-cost
pair     = eg.to_pair_spec(cols[0], cols[1])
strategy = ZScoreStrategy(est, pair, entry_z=1.5, exit_z=0.0)
result   = Backtester(strategy, LinearCostModel(bps=5)).run(FrameSource(raw))
print(result.summary())
```

A cointegration spread is a zero-centred log-residual, not a tradable price,
so strategies trade the two legs of the pair rather than a fictitious "spread
instrument" — see [`docs/architecture.md`](docs/architecture.md).

See [`examples/`](examples) for full Stage 0/1/2 demos and
[`docs/math.md`](docs/math.md) for the mathematical derivations.

## Layout

```
src/stat_arb/
├── data/         # SyntheticOU, CSV, yfinance, OpenBB loaders
├── engine/       # Stage 0: events, strategy, portfolio, execution, accounting
├── validation/   # Stationarity, Lo-2002 Sharpe SE, Deflated Sharpe, purged-CV, PBO, walk-forward
├── stage1/       # Engle-Granger, Johansen, continuous-time OU MLE, ±z strategy
├── stage2/       # Leung-Li optimal entry/exit boundaries via OU fundamental solutions
├── stage3/       # Kalman dynamic hedge (time-varying beta), rolling-OLS benchmark
├── stage4/       # Markov-switching OU via EM/Hamilton filter, justification gate
├── stage5/       # Bayesian OU (conjugate), Liu-West particle filter, uncertainty sizing
├── stage6/       # PCA eigenportfolios (Avellaneda-Lee), VECM, Ledoit-Wolf/HRP
├── stage7/       # Microstructure cost model (sqrt impact, partial fills) + Almgren-Chriss
├── calibration/  # Parallel parameter sweeps
└── utils/
tests/            # pytest suite with synthetic-recovery and gate tests
examples/         # Runnable demos per stage
docs/             # architecture.md and math.md
```

## Running tests

```bash
pytest                       # full suite (excluding network-marked)
pytest -m "not network"      # skip yfinance/OpenBB tests
pytest tests/test_ou_mle.py  # one stage
```
