# Code-Review Fixes

A self-review pass over the Adaptive Statistical Arbitrage codebase, conducted
after all eight roadmap stages were built. Ten findings — correctness bugs,
design gaps, and statistical-rigor gaps — were identified and fixed on a
dedicated `fix/review` branch. This document is the durable record (it is the
one tracked file in the otherwise-gitignored prep package, so it survives to
`main` on merge).

**Verification:** full suite **77 passing, exit 0** after the fixes (+8 new
regression tests over the prior 69).

## The fixes

| # | Finding | Fix | Verified by |
|---|---------|-----|-------------|
| **1** | Deflated Sharpe fed an *annualized* trial-Sharpe variance into a *per-observation* formula → `DSR ≈ 0.00` everywhere (every demo) | `periods_per_year` param; interpret inputs as annualized (library convention) and convert to per-obs internally; report both scales; warn on implausibly large `SR0` | Stage 1 DSR **0.00 → 0.85**, Stage 6 **→ 0.045**; new regression + warning tests |
| **2** | `Portfolio.equity()` marked a missing/NaN quote at **0.0** → fabricated drawdown-and-recovery on gapped real data | carry the last finite positive price forward; only a never-priced symbol is treated as zero | new gap-pricing test |
| **3** | PBO `omega` used `/N` and counted the strategy against itself (off-convention) | standard López de Prado `omega = rank/(N+1) ∈ (0,1)`; logits finite without clipping | finite-logits + "real-edge → low-PBO" tests |
| **4** | Stage 6's Ledoit-Wolf shrinkage + HRP were **built and tested but never used** by the strategy (naive equal-weight per side) | added `weighting="hrp"`: each side's gross allocated by HRP on a Ledoit-Wolf covariance over the lookback; example uses it | 2 new HRP tests |
| **5** | `gross = 1.0` only true at entry (notional drifts between signals) | documented the exposure-drift caveat in `orders_from_signal` | — |
| **6** | `num_trades` counts fills (legs), not round-trips — misleading | added honest `num_fills`; documented both count legs | suite |
| **7** | The `FrameSource` `DataSource` shim was re-declared ~15× across examples/tests | added shared `InMemorySource` to `stat_arb.data`; replaced the duplicates with an alias import | suite |
| **8** | No value-level tests on the validation primitives (why #1 slipped) | added numeric DSR/PBO regression guards | the new tests |
| **9** | Regime "rejects single-regime data" claim tested on only one seed | parametrized the gate test over 3 seeds | suite |
| **10** | Synthetic in-simulation Sharpes overstated as "edge" | README "Honest limitations" section; the real-data pressure test is the reality check | — |
| minor | `apply_fill` untyped; `infer_dt` silently mis-scales an irregular index | typed as `FillEvent`; `infer_dt` now warns on no-calendar fallback | — |

## Why #1 mattered most

It sat inside the validation spine — the part of the project I repeatedly call
its differentiator — and it was silent: the formula was right, the inputs were
the wrong scale, so every reported Deflated Sharpe collapsed to `0.00`. That
made "honest DSR ≈ 0" look like a finding when it was an artifact. After the
fix the numbers are real and *more* informative: Stage 1's strong single pair
survives deflation (**0.85**), while the 25-residual breadth book honestly does
**not** (**0.045**) — which is the genuine result the bug had been masking.

## Branch / merge

All fixes live on `fix/review` as five thematic commits:

1. `validation correctness (DSR scale, PBO omega)`
2. `engine valuation + accounting honesty`
3. `wire Ledoit-Wolf/HRP into the eigenportfolio book`
4. `shared InMemorySource adapter; de-duplicate frame shims`
5. `infer_dt warning, regime-gate multi-seed, README limitations`

Intended to merge to `main` (fast-forward), preserving the five-commit history.
