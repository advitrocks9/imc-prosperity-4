# IMC Prosperity 4

Algorithmic trading agent for IMC's [Prosperity 4](https://prosperity.imc.com/) competition (April 2026). A single Python class trades against a simulated market each tick: stable products, drifting mid-prices, basket spreads, options vouchers, and cross-exchange arbitrage. Five rounds, one algo file under 100 KB.

This repo is the working code we built and iterated during the competition: strategies, the research that calibrated them, the manual-challenge solvers, and the harness used to A/B test variants before submission.

## What's in here

```
trader.py                 Dispatcher: per-product strategy selection + state plumbing
datamodel.py              IMC's TradingState / Order / OrderDepth types
strategies/               33 pluggable strategy modules (see below)
analysis/                 Research scripts: EDA, pair cointegration, IV vs RV, bot fingerprinting
manual/                   Solvers for the four manual-challenge archetypes
scripts/                  Build inliner, A/B harness, round-opening pipeline
tests/                    Black-Scholes regression suite
```

## Strategies

Each module is a self-contained class implementing `update(state, position) -> list[Order]`. `trader.py` instantiates the right one per product and threads shared state through `_base.StateManager`.

| Archetype | Modules | Approach |
|---|---|---|
| Stable | `stable.py`, `stable_anchored.py`, `stable_skew.py`, `stable_ladder.py` | Hardcoded fair value, three-phase order generation (TAKE, CLEAR, MAKE) |
| Drifting mid | `drifting.py`, `hyd_mm.py`, `vel_mm.py` | Adaptive FV from `wall_mid` (deepest-level midpoint), VWAP confirmation |
| Basket / spread arb | `basket.py`, `pair_arb.py` | Welford online mean of spread, enter on z-score above threshold, exit on revert |
| Options | `bs_options.py`, `bs_voucher.py`, `bs_voucher_smile.py`, `voucher_zscore.py`, `voucher_delta_hedge.py` | Black-Scholes IV, smile fit (quadratic in moneyness), trade vs theoretical, delta-hedge underlying |
| Conversion | `conversion.py` | Cross-exchange arb with min-edge filter, sell at `floor(foreign_bid + 0.5)` |
| Signal-driven | `signal.py`, `counterparty_copy.py`, `aggressor_fade.py`, `fade_bias.py` | DualEMA z-score, copy informed counterparties, fade aggressive flow |
| Regime | `regime_adaptive.py`, `regime_lock.py` | Switch parameter sets on detected regime change |

`strategies/_base.py` holds the shared primitives every module uses:

- `wall_mid`, `vwap_mid`: fair-value estimators. `wall_mid` was the empirical winner across products tested on Prosperity 3 historical data.
- `EMA`, `RingBuffer`, `DualEMAZScore`: time-series primitives.
- `OliviaTracker`: informed-bot detector that keys off known counterparty fingerprints.
- `build_orders`: three-phase order generator (TAKE existing edge, then CLEAR inventory, then MAKE quotes).
- `StateManager`: `traderData` JSON wrapper with a 50 KB budget guard (the silent-truncation cliff).
- Black-Scholes suite: `bs_call`, `bs_delta`, `bs_vega`, `implied_vol` via Newton-Raphson with bisection fallback.
- `BasketSpreadTracker`: Welford online mean and variance for spread arb.
- `conversion_arb`: cross-exchange edge calculator.

## Manual-challenge solvers

Each round has a one-shot optimization problem worth roughly 30% of the round's score. Four canonical archetypes from prior competitions, all solved analytically:

| File | Problem | Method |
|---|---|---|
| `manual/fx_arb.py` | Best currency-conversion sequence given an N by N rate matrix | Bellman-Ford in log-space (max product of rates becomes max sum of log-rates), DP over hop count |
| `manual/auction.py` | Two-bid sealed-bid auction against a known reserve distribution | EV optimization over the bid grid |
| `manual/allocation.py` | N-player Nash-equilibrium allocation across reward tiles | Mixed strategy with behavioral prior |
| `manual/news_portfolio.py` | Allocate 100% capital across N assets under quadratic transaction fees | Closed-form KKT (no `cvxpy` dependency) |

All four were verified against the released Prosperity 2 ground-truth solutions before being used live.

## Research and calibration

`analysis/research_*.py` are the scripts that produced the calibrated parameters in `trader.py`. They run on the Prosperity 1, 2, and 3 historical CSV releases (which IMC publishes after each season) and answer questions like:

- `research_drifting_fv.py`: which fair-value estimator minimizes follow-the-mid error on each product? Answer: `wall_mid` beat microprice, OFI, and vol-adaptive variants on every drifting product tested.
- `research_basket.py`: does the basket spread mean-revert via the basket adjusting or the constituents adjusting? Answer: basket only, so trade the basket and not the legs.
- `research_options.py`: is the options PnL vol alpha or directional delta? Answer: directional, via stale smile in the bot quotes.
- `research_bots.py`: fingerprint each named counterparty by trade size, time-of-day, and information ratio.
- `research_conversion.py`: calibrate the min-edge threshold and rounding rule for cross-exchange fills.
- `research_cross_signals.py`: correlation matrix across products and observation channels.

`analysis/parameter_search.py` runs walk-forward grid search and reports the **landscape smoothness** of each parameter. Flat plateaus mean the optimum generalizes; sharp peaks mean it overfits.

## Build and submission flow

The competition mandates a single Python file under 100 KB. `scripts/build.py` inlines all imported strategy modules into a flat `submission.py`, deduplicates symbols, and prints the byte size. Variants for A/B testing are produced by the same inliner with substituted parameter blocks.

The two A/B harnesses both run on local CSV data:

- `scripts/ab_test.py`: fair-value-based MM versus market-follow (best_bid+1 / best_ask-1) on a chosen product.
- `scripts/skeleton_test.py`: sanity-check any strategy archetype against historical data.

`scripts/replay_match_modes.py` runs the same submission under both `--match-trades all` (generous) and `--match-trades none` (strict). The strict mode is the empirical floor for portal performance: passive quotes that look profitable under the generous mode but break under strict matching tend to fail in production.

## Running it

```bash
uv sync
uv run prosperity4btest trader.py 0       # backtest tutorial
uv run scripts/build.py                   # inline to build/submission.py
uv run scripts/ab_test.py data/tutorial/ TOMATOES --limit 80
uv run manual/fx_arb.py                   # currency arb solver
uv run pytest tests/                      # Black-Scholes regression
```

Requires Python 3.12, `uv`, and the `prosperity4btest` package (in `pyproject.toml`).

## Notes for readers

- Historical Prosperity CSVs aren't checked in. IMC redistributes them each season; I pull from `data/p1/`, `data/p2/`, `data/p3/` locally.
- The strategy zoo is intentionally large. Most modules were dead ends; the rest were combined per-round. Keeping them in tree shows what was tried, not just what shipped.
- Final shipped trader uses a subset selected per round via the dispatch logic at the top of `trader.py`.

## License

MIT.
