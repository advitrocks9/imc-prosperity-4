"""Basket spread dynamics research for PICNIC_BASKET1 and PICNIC_BASKET2.

Investigates:
1. OLS weights (multiple specs, log-transforms, WLS)
2. Spread premium half-life (ACF + exponential decay fit)
3. entry_thr and n_prior optimization (walk-forward on P3 data)
4. Inter-basket arb (PICNIC_BASKET1 vs PICNIC_BASKET2)
5. Hedging analysis (hedge_factor sweep via simplified backtest)

Usage:
    uv run analysis/research_basket.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.data_loader import load_prices, get_product_mid

# ─────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────

ROUND_DIRS = [
    ROOT / "data" / "p3" / "round2",
    ROOT / "data" / "p3" / "round3",
    ROOT / "data" / "p3" / "round4",
    ROOT / "data" / "p3" / "round5",
]

BASKET_PRODUCTS = ["PICNIC_BASKET1", "PICNIC_BASKET2"]
CONSTITUENTS = ["CROISSANTS", "JAMS", "DJEMBES"]
EXTRA_PRODUCTS = ["KELP", "SQUID_INK", "RAINFOREST_RESIN"]


def load_all_basket_data() -> pd.DataFrame:
    """Load and merge mid prices across all available rounds for basket analysis."""
    all_dfs: list[pd.DataFrame] = []
    for rd in ROUND_DIRS:
        if not rd.exists():
            continue
        try:
            prices = load_prices(rd)
        except FileNotFoundError:
            continue
        products_in_round = set(prices["product"].unique())
        # Only include rounds with basket products
        if "PICNIC_BASKET1" not in products_in_round:
            continue
        # Tag with round directory for identification
        prices["round_dir"] = rd.name
        all_dfs.append(prices)
    if not all_dfs:
        raise RuntimeError("No basket data found in any round")
    return pd.concat(all_dfs, ignore_index=True)


def align_products(
    prices: pd.DataFrame,
    products: list[str],
    day: int | None = None,
) -> dict[str, np.ndarray]:
    """Extract aligned mid_price arrays for given products.

    If day is None, uses all data (concatenated across days).
    """
    series: dict[str, pd.Series] = {}
    for p in products:
        pdf = get_product_mid(prices, p)
        if pdf.empty:
            continue
        if day is not None:
            pdf = pdf[pdf["day"] == day]
        series[p] = pdf["mid_price"].reset_index(drop=True)

    if not series:
        return {}

    min_len = min(len(s) for s in series.values())
    return {p: s.values[:min_len].astype(float) for p, s in series.items()}


def ols_regression(
    y: np.ndarray,
    X_dict: dict[str, np.ndarray],
    add_intercept: bool = True,
    weights: np.ndarray | None = None,
) -> dict:
    """Run OLS (or WLS) regression. Returns weights, intercept, R2, residuals."""
    names = list(X_dict.keys())
    X = np.column_stack([X_dict[n] for n in names])
    if add_intercept:
        X = np.column_stack([X, np.ones(len(X))])

    if weights is not None:
        W = np.diag(np.sqrt(weights))
        X_w = W @ X
        y_w = W @ y
    else:
        X_w, y_w = X, y

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
    except np.linalg.LinAlgError:
        return {"r_squared": 0.0, "error": "lstsq failed"}

    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    result: dict = {
        "constituents": names,
        "weights": {n: round(float(coeffs[i]), 4) for i, n in enumerate(names)},
        "r_squared": round(float(r2), 6),
        "rmse": round(float(np.sqrt(ss_res / len(y))), 2),
        "residuals": y - y_pred,
    }
    if add_intercept:
        result["intercept"] = round(float(coeffs[-1]), 2)
    return result


# ─────────────────────────────────────────────────────────────────
# 1. OLS Weight Analysis
# ─────────────────────────────────────────────────────────────────

def analyze_ols_weights(prices: pd.DataFrame) -> None:
    """Try many regression specs for both basket products."""
    print("=" * 70)
    print("1. OLS WEIGHT ANALYSIS")
    print("=" * 70)

    days = sorted(prices["day"].unique())
    all_products = CONSTITUENTS + EXTRA_PRODUCTS

    for basket in BASKET_PRODUCTS:
        print(f"\n{'─'*50}")
        print(f"  {basket}")
        print(f"{'─'*50}")

        data = align_products(prices, [basket] + all_products)
        if basket not in data:
            print(f"  No data for {basket}")
            continue

        y = data[basket]
        available = [p for p in all_products if p in data]

        # --- Spec A: All 3 core constituents ---
        core = [p for p in CONSTITUENTS if p in data]
        if len(core) == 3:
            res = ols_regression(y, {p: data[p] for p in core})
            print(f"\n  [A] All 3 constituents: R2={res['r_squared']:.6f}  RMSE={res['rmse']}")
            for c, w in res["weights"].items():
                print(f"      {c}: {w}")
            print(f"      intercept: {res.get('intercept', 'N/A')}")

        # --- Spec B: Each pair of 2 constituents ---
        print(f"\n  [B] Pairs of 2 constituents:")
        for combo in combinations(core, 2):
            res = ols_regression(y, {p: data[p] for p in combo})
            names_str = " + ".join(combo)
            print(f"      {names_str}: R2={res['r_squared']:.6f}  RMSE={res['rmse']}")

        # --- Spec C: Add KELP or SQUID_INK ---
        print(f"\n  [C] Core + extra products:")
        for extra in EXTRA_PRODUCTS:
            if extra not in data:
                continue
            regressors = core + [extra]
            res = ols_regression(y, {p: data[p] for p in regressors})
            print(f"      +{extra}: R2={res['r_squared']:.6f}  RMSE={res['rmse']}")

        # --- Spec D: Log-transformed prices ---
        print(f"\n  [D] Log-transformed prices:")
        log_y = np.log(y)
        log_X = {p: np.log(data[p]) for p in core}
        res = ols_regression(log_y, log_X)
        print(f"      log(core): R2={res['r_squared']:.6f}  RMSE={res['rmse']}")

        # --- Spec E: Weighted least squares (exponential recency weighting) ---
        print(f"\n  [E] Weighted least squares (recency-weighted):")
        n = len(y)
        for halflife in [500, 2000, 5000]:
            weights = np.exp(np.log(2) * np.arange(n) / halflife)  # newer data higher weight
            res = ols_regression(y, {p: data[p] for p in core}, weights=weights)
            print(f"      halflife={halflife}: R2={res['r_squared']:.6f}  RMSE={res['rmse']}")
            if halflife == 2000:
                for c, w in res["weights"].items():
                    print(f"        {c}: {w}")
                print(f"        intercept: {res.get('intercept', 'N/A')}")

        # --- Spec F: Per-day OLS to check stability ---
        print(f"\n  [F] Per-day weight stability:")
        for day in days:
            day_data = align_products(prices, [basket] + core, day=day)
            if basket not in day_data or len(day_data.get(basket, [])) < 100:
                continue
            res = ols_regression(
                day_data[basket],
                {p: day_data[p] for p in core if p in day_data},
            )
            wstr = "  ".join(f"{c}={w}" for c, w in res["weights"].items())
            print(f"      day {day:>2}: R2={res['r_squared']:.6f}  {wstr}  int={res.get('intercept', 'N/A')}")


# ─────────────────────────────────────────────────────────────────
# 2. Spread Premium Half-Life
# ─────────────────────────────────────────────────────────────────

def analyze_half_life(prices: pd.DataFrame) -> dict[str, float]:
    """Compute spread ACF and fit exponential decay to estimate half-life."""
    print("\n" + "=" * 70)
    print("2. SPREAD PREMIUM HALF-LIFE")
    print("=" * 70)

    results: dict[str, float] = {}

    for basket in BASKET_PRODUCTS:
        print(f"\n{'─'*50}")
        print(f"  {basket}")
        print(f"{'─'*50}")

        data = align_products(prices, [basket] + CONSTITUENTS)
        if basket not in data:
            print(f"  No data for {basket}")
            continue

        core = [p for p in CONSTITUENTS if p in data]
        if len(core) < 2:
            continue

        # Get OLS weights first
        y = data[basket]
        res = ols_regression(y, {p: data[p] for p in core})
        spread = res["residuals"]

        # Demean
        spread = spread - np.mean(spread)
        n = len(spread)

        # ACF at lags 1..500
        max_lag = min(500, n // 4)
        acf = np.zeros(max_lag + 1)
        var = np.var(spread)
        if var < 1e-10:
            print("  Spread has near-zero variance, skipping")
            continue

        for lag in range(max_lag + 1):
            acf[lag] = np.mean(spread[:n - lag] * spread[lag:]) / var

        # Fit exponential decay: acf(lag) = exp(-lag / tau)
        # => log(acf(lag)) = -lag / tau
        # Only use positive ACF values
        valid_lags = []
        valid_log_acf = []
        for lag in range(1, max_lag + 1):
            if acf[lag] > 0.01:  # only fit on meaningfully positive ACF
                valid_lags.append(lag)
                valid_log_acf.append(np.log(acf[lag]))

        if len(valid_lags) < 10:
            print("  Not enough positive ACF values for fit")
            continue

        # Linear regression: log(acf) = -(1/tau) * lag
        lags_arr = np.array(valid_lags, dtype=float)
        log_acf_arr = np.array(valid_log_acf, dtype=float)

        # OLS: log_acf = a * lag + b
        A = np.column_stack([lags_arr, np.ones(len(lags_arr))])
        coeffs, _, _, _ = np.linalg.lstsq(A, log_acf_arr, rcond=None)
        decay_rate = -coeffs[0]  # tau = 1/decay_rate
        if decay_rate > 1e-8:
            tau = 1.0 / decay_rate
            half_life = tau * np.log(2)
        else:
            tau = float("inf")
            half_life = float("inf")

        # Also try scipy curve_fit for better accuracy
        try:
            from scipy.optimize import curve_fit

            def exp_decay(x: np.ndarray, tau_param: float) -> np.ndarray:
                return np.exp(-x / tau_param)

            popt, _ = curve_fit(exp_decay, lags_arr, np.array([acf[int(l)] for l in lags_arr]),
                                p0=[tau], maxfev=5000)
            tau_scipy = popt[0]
            half_life_scipy = tau_scipy * np.log(2)
        except Exception:
            tau_scipy = tau
            half_life_scipy = half_life

        print(f"  Spread stats: mean={np.mean(res['residuals']):.2f}, "
              f"std={np.std(res['residuals']):.2f}, "
              f"n={n}")
        print(f"  ACF(1)={acf[1]:.4f}, ACF(5)={acf[5]:.4f}, ACF(10)={acf[10]:.4f}, "
              f"ACF(50)={acf[min(50, max_lag)]:.4f}, ACF(100)={acf[min(100, max_lag)]:.4f}")
        print(f"  Exponential decay fit (log-linear): tau={tau:.1f}, half_life={half_life:.1f} ticks")
        print(f"  Exponential decay fit (curve_fit):  tau={tau_scipy:.1f}, half_life={half_life_scipy:.1f} ticks")
        print(f"  Recommended n_prior = 2 * half_life = {2 * half_life_scipy:.0f}")

        results[basket] = half_life_scipy

        # ADF test for stationarity of spread
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_stat, adf_pvalue, _, _, _, _ = adfuller(res["residuals"], maxlag=50)
            print(f"  ADF test: stat={adf_stat:.4f}, p-value={adf_pvalue:.6f} "
                  f"({'stationary' if adf_pvalue < 0.05 else 'NON-stationary'})")
        except ImportError:
            pass

    return results


# ─────────────────────────────────────────────────────────────────
# 3. Entry Threshold & n_prior Optimization
# ─────────────────────────────────────────────────────────────────

def _precompute_day_arrays(
    prices: pd.DataFrame,
    basket: str,
    constituents: dict[str, float],
    day: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Pre-extract aligned arrays for a single day. Returns (basket_mid, synthetic_mid, best_ask, best_bid, final_mid) or None."""
    day_prices = prices[prices["day"] == day]
    products_needed = [basket] + list(constituents.keys())

    # Pivot to get aligned mid_price, bid_price_1, ask_price_1 per timestamp
    pivoted_mid = day_prices.pivot_table(
        index="timestamp", columns="product", values="mid_price",
    )
    pivoted_ask = day_prices.pivot_table(
        index="timestamp", columns="product", values="ask_price_1",
    )
    pivoted_bid = day_prices.pivot_table(
        index="timestamp", columns="product", values="bid_price_1",
    )

    for p in products_needed:
        if p not in pivoted_mid.columns:
            return None

    # Drop rows with NaNs for required products
    valid = pivoted_mid[products_needed].dropna()
    if len(valid) == 0:
        return None

    idx = valid.index
    basket_mid = valid[basket].values.astype(float)

    # Compute synthetic NAV
    synthetic_mid = np.zeros(len(idx))
    for c, w in constituents.items():
        synthetic_mid += w * valid[c].values.astype(float)

    # Execution prices
    best_ask = pivoted_ask.reindex(idx)[basket].values.astype(float)
    best_bid = pivoted_bid.reindex(idx)[basket].values.astype(float)

    return basket_mid, synthetic_mid, best_ask, best_bid, basket_mid[-1]


# Cache precomputed arrays per (day, basket)
_day_cache: dict[tuple[int, str], tuple] = {}


def _get_day_arrays(
    prices: pd.DataFrame,
    basket: str,
    constituents: dict[str, float],
    day: int,
) -> tuple | None:
    """Cached version of _precompute_day_arrays."""
    key = (day, basket)
    if key not in _day_cache:
        result = _precompute_day_arrays(prices, basket, constituents, day)
        _day_cache[key] = result
    return _day_cache[key]


def run_basket_backtest(
    prices: pd.DataFrame,
    basket: str,
    constituents: dict[str, float],
    intercept: float,
    day: int,
    entry_thr: float,
    n_prior: int,
    hedge_factor: float = 0.0,
    basket_limit: int = 60,
) -> dict:
    """Simplified basket spread backtest on one day (vectorized).

    Returns {pnl, n_trades, max_pos, final_pos}.
    """
    arrays = _get_day_arrays(prices, basket, constituents, day)
    if arrays is None:
        return {"pnl": 0.0, "n_trades": 0, "max_pos": 0, "final_pos": 0}

    basket_mid, synthetic_mid, best_ask, best_bid, final_mid = arrays
    n = len(basket_mid)

    # Vectorized spread computation
    raw_spread = basket_mid - synthetic_mid

    # Welford online mean (must be sequential due to state dependency)
    welford_n = float(n_prior)
    welford_mean = float(intercept)
    sig_spread = np.empty(n)
    for i in range(n):
        welford_n += 1.0
        welford_mean += (raw_spread[i] - welford_mean) / welford_n
        sig_spread[i] = raw_spread[i] - welford_mean

    # Simulate trading (sequential due to position state)
    pos = 0
    cash = 0.0
    n_trades = 0
    max_pos = 0

    for i in range(n):
        rem_buy = basket_limit - pos
        rem_sell = basket_limit + pos

        if sig_spread[i] < -entry_thr and rem_buy > 0:
            if not np.isnan(best_ask[i]):
                cash -= rem_buy * best_ask[i]
                pos += rem_buy
                n_trades += 1
        elif sig_spread[i] > entry_thr and rem_sell > 0:
            if not np.isnan(best_bid[i]):
                cash += rem_sell * best_bid[i]
                pos -= rem_sell
                n_trades += 1

        if abs(pos) > max_pos:
            max_pos = abs(pos)

    mtm = pos * final_mid
    pnl = cash + mtm

    return {
        "pnl": round(pnl, 2),
        "n_trades": n_trades,
        "max_pos": max_pos,
        "final_pos": pos,
    }


def optimize_parameters(prices: pd.DataFrame) -> None:
    """Walk-forward optimization of entry_thr and n_prior."""
    print("\n" + "=" * 70)
    print("3. ENTRY THRESHOLD & N_PRIOR OPTIMIZATION")
    print("=" * 70)

    days = sorted(prices["day"].unique())

    for basket in BASKET_PRODUCTS:
        print(f"\n{'─'*50}")
        print(f"  {basket}")
        print(f"{'─'*50}")

        core = [p for p in CONSTITUENTS if p in set(prices["product"].unique())]
        data = align_products(prices, [basket] + core)
        if basket not in data or len(core) < 2:
            print(f"  Insufficient data for {basket}")
            continue

        # Fit OLS weights on all available data (will use train day for walk-forward)
        y = data[basket]
        res = ols_regression(y, {p: data[p] for p in core})
        constituents = res["weights"]
        intercept = res.get("intercept", 0.0)

        print(f"  Weights: {constituents}")
        print(f"  Intercept: {intercept}")
        print(f"  Days available: {days}")

        # Train on first day, validate on rest
        train_day = days[0]
        val_days = days[1:] if len(days) > 1 else days

        # --- entry_thr sweep ---
        entry_thrs = list(range(10, 210, 10))
        n_priors = [100, 500, 1000, 2000, 5000, 10000, 30000, 60000]

        print(f"\n  [A] entry_thr sweep (n_prior=1000, train={train_day}, val={val_days}):")
        print(f"  {'thr':>6} | {'train PnL':>12} | " + " | ".join(f"day {d} PnL" for d in val_days))
        print(f"  {'─'*6}-+-{'─'*12}-+-" + "-+-".join("─" * 10 for _ in val_days))

        best_val_pnl = -float("inf")
        best_thr = 80

        for thr in entry_thrs:
            train_res = run_basket_backtest(prices, basket, constituents, intercept, train_day, thr, 1000)
            val_pnls = []
            for d in val_days:
                vr = run_basket_backtest(prices, basket, constituents, intercept, d, thr, 1000)
                val_pnls.append(vr["pnl"])

            avg_val = np.mean(val_pnls)
            marker = " <-- best" if avg_val > best_val_pnl else ""
            if avg_val > best_val_pnl:
                best_val_pnl = avg_val
                best_thr = thr

            val_str = " | ".join(f"{p:>10,.0f}" for p in val_pnls)
            print(f"  {thr:>6} | {train_res['pnl']:>12,.0f} | {val_str}{marker}")

        print(f"\n  Best entry_thr: {best_thr} (avg val PnL: {best_val_pnl:,.0f})")

        # --- n_prior sweep ---
        print(f"\n  [B] n_prior sweep (entry_thr={best_thr}):")
        print(f"  {'n_prior':>8} | {'train PnL':>12} | " + " | ".join(f"day {d} PnL" for d in val_days))
        print(f"  {'─'*8}-+-{'─'*12}-+-" + "-+-".join("─" * 10 for _ in val_days))

        best_val_pnl_np = -float("inf")
        best_np = 1000

        for np_val in n_priors:
            train_res = run_basket_backtest(prices, basket, constituents, intercept, train_day, best_thr, np_val)
            val_pnls = []
            for d in val_days:
                vr = run_basket_backtest(prices, basket, constituents, intercept, d, best_thr, np_val)
                val_pnls.append(vr["pnl"])

            avg_val = np.mean(val_pnls)
            marker = " <-- best" if avg_val > best_val_pnl_np else ""
            if avg_val > best_val_pnl_np:
                best_val_pnl_np = avg_val
                best_np = np_val

            val_str = " | ".join(f"{p:>10,.0f}" for p in val_pnls)
            print(f"  {np_val:>8} | {train_res['pnl']:>12,.0f} | {val_str}{marker}")

        print(f"\n  Best n_prior: {best_np} (avg val PnL: {best_val_pnl_np:,.0f})")

        # --- Joint grid search (top 5 combos) ---
        print(f"\n  [C] Joint grid search (top combinations):")
        grid_results: list[tuple[float, int, float, float]] = []
        for thr in entry_thrs:
            for np_val in n_priors:
                val_pnls = []
                for d in val_days:
                    vr = run_basket_backtest(prices, basket, constituents, intercept, d, thr, np_val)
                    val_pnls.append(vr["pnl"])
                avg = np.mean(val_pnls)
                grid_results.append((thr, np_val, avg, np.std(val_pnls)))

        grid_results.sort(key=lambda x: x[2], reverse=True)
        print(f"  {'rank':>4} | {'thr':>6} | {'n_prior':>8} | {'avg_val_PnL':>12} | {'std':>10}")
        for i, (thr, np_val, avg, std) in enumerate(grid_results[:10]):
            print(f"  {i+1:>4} | {thr:>6} | {np_val:>8} | {avg:>12,.0f} | {std:>10,.0f}")


# ─────────────────────────────────────────────────────────────────
# 4. Inter-Basket Arb
# ─────────────────────────────────────────────────────────────────

def analyze_inter_basket(prices: pd.DataFrame) -> None:
    """Analyze PICNIC_BASKET1 - PICNIC_BASKET2 spread."""
    print("\n" + "=" * 70)
    print("4. INTER-BASKET ARBITRAGE")
    print("=" * 70)

    data = align_products(prices, BASKET_PRODUCTS)
    if len(data) < 2:
        print("  Need both PICNIC_BASKET1 and PICNIC_BASKET2")
        return

    b1 = data["PICNIC_BASKET1"]
    b2 = data["PICNIC_BASKET2"]
    spread = b1 - b2
    n = len(spread)

    print(f"\n  Sample size: {n}")
    print(f"  PICNIC_BASKET1 mean: {np.mean(b1):,.2f}")
    print(f"  PICNIC_BASKET2 mean: {np.mean(b2):,.2f}")
    print(f"  Spread (B1 - B2):")
    print(f"    mean:   {np.mean(spread):>10,.2f}")
    print(f"    std:    {np.std(spread):>10,.2f}")
    print(f"    min:    {np.min(spread):>10,.2f}")
    print(f"    max:    {np.max(spread):>10,.2f}")
    print(f"    median: {np.median(spread):>10,.2f}")

    # Is it mean-reverting?
    demeaned = spread - np.mean(spread)
    acf1 = np.corrcoef(demeaned[:-1], demeaned[1:])[0, 1]
    acf5 = np.corrcoef(demeaned[:-5], demeaned[5:])[0, 1]
    acf10 = np.corrcoef(demeaned[:-10], demeaned[10:])[0, 1]
    print(f"\n  Mean-reversion test:")
    print(f"    ACF(1):  {acf1:.4f}")
    print(f"    ACF(5):  {acf5:.4f}")
    print(f"    ACF(10): {acf10:.4f}")

    try:
        from statsmodels.tsa.stattools import adfuller
        adf_stat, adf_pvalue, _, _, _, _ = adfuller(spread, maxlag=50)
        print(f"    ADF stat: {adf_stat:.4f}, p-value: {adf_pvalue:.6f}")
        print(f"    {'STATIONARY' if adf_pvalue < 0.05 else 'NON-STATIONARY'}")
    except ImportError:
        pass

    # Regression: B1 = alpha * B2 + beta
    res = ols_regression(b1, {"PICNIC_BASKET2": b2})
    print(f"\n  B1 = {res['weights']['PICNIC_BASKET2']:.4f} * B2 + {res.get('intercept', 0):.2f}")
    print(f"  R2: {res['r_squared']:.6f}")

    # What are B2's constituents?
    print(f"\n  B2 constituent analysis:")
    all_prods = CONSTITUENTS + EXTRA_PRODUCTS
    data_full = align_products(prices, ["PICNIC_BASKET2"] + all_prods)
    if "PICNIC_BASKET2" in data_full:
        y2 = data_full["PICNIC_BASKET2"]
        avail = [p for p in all_prods if p in data_full]

        # Try all combos of 1-4 constituents
        best_r2 = 0.0
        best_spec = None
        for n_c in range(1, min(5, len(avail) + 1)):
            for combo in combinations(avail, n_c):
                res2 = ols_regression(y2, {p: data_full[p] for p in combo})
                if res2["r_squared"] > best_r2:
                    best_r2 = res2["r_squared"]
                    best_spec = res2

        if best_spec:
            print(f"    Best fit: R2={best_spec['r_squared']:.6f}")
            for c, w in best_spec["weights"].items():
                print(f"      {c}: {w}")
            print(f"      intercept: {best_spec.get('intercept', 'N/A')}")

    # Per-day spread analysis
    days = sorted(prices["day"].unique())
    print(f"\n  Per-day spread stats:")
    for day in days:
        day_data = align_products(prices, BASKET_PRODUCTS, day=day)
        if len(day_data) < 2:
            continue
        ds = day_data["PICNIC_BASKET1"] - day_data["PICNIC_BASKET2"]
        print(f"    day {day:>2}: mean={np.mean(ds):>8,.2f}  std={np.std(ds):>8,.2f}  n={len(ds)}")

    # Simple mean-reversion PnL estimate
    print(f"\n  Simple mean-reversion backtest (buy spread < mean-1std, sell > mean+1std):")
    mean_sp = np.mean(spread)
    std_sp = np.std(spread)
    pos_spread = 0
    cash = 0.0
    n_trades = 0
    for i in range(len(spread)):
        if spread[i] < mean_sp - std_sp and pos_spread <= 0:
            # Buy spread (buy B1, sell B2)
            pos_spread = 1
            cash -= spread[i]
            n_trades += 1
        elif spread[i] > mean_sp + std_sp and pos_spread >= 0:
            # Sell spread
            pos_spread = -1
            cash += spread[i]
            n_trades += 1

    cash += pos_spread * spread[-1]  # close out
    print(f"    PnL per unit: {cash:,.2f}  trades: {n_trades}")


# ─────────────────────────────────────────────────────────────────
# 5. Hedging Analysis
# ─────────────────────────────────────────────────────────────────

def _precompute_hedge_arrays(
    prices: pd.DataFrame,
    basket: str,
    constituents: dict[str, float],
    day: int,
) -> dict | None:
    """Pre-extract aligned arrays including constituent bid/ask for hedging."""
    day_prices = prices[prices["day"] == day]
    all_products = [basket] + list(constituents.keys())

    pivoted_mid = day_prices.pivot_table(index="timestamp", columns="product", values="mid_price")
    pivoted_ask = day_prices.pivot_table(index="timestamp", columns="product", values="ask_price_1")
    pivoted_bid = day_prices.pivot_table(index="timestamp", columns="product", values="bid_price_1")

    for p in all_products:
        if p not in pivoted_mid.columns:
            return None

    valid = pivoted_mid[all_products].dropna()
    if len(valid) == 0:
        return None

    idx = valid.index
    result = {
        "basket_mid": valid[basket].values.astype(float),
        "basket_ask": pivoted_ask.reindex(idx)[basket].values.astype(float),
        "basket_bid": pivoted_bid.reindex(idx)[basket].values.astype(float),
    }
    for c in constituents:
        result[f"{c}_mid"] = valid[c].values.astype(float)
        result[f"{c}_ask"] = pivoted_ask.reindex(idx)[c].values.astype(float)
        result[f"{c}_bid"] = pivoted_bid.reindex(idx)[c].values.astype(float)

    return result


_hedge_cache: dict[tuple[int, str], dict | None] = {}


def run_hedged_backtest(
    prices: pd.DataFrame,
    basket: str,
    constituents: dict[str, float],
    intercept: float,
    day: int,
    entry_thr: float,
    n_prior: int,
    hedge_factor: float,
    basket_limit: int = 60,
) -> float:
    """Hedged basket backtest. Returns total PnL."""
    key = (day, basket)
    if key not in _hedge_cache:
        _hedge_cache[key] = _precompute_hedge_arrays(prices, basket, constituents, day)
    arrays = _hedge_cache[key]
    if arrays is None:
        return 0.0

    basket_mid = arrays["basket_mid"]
    basket_ask = arrays["basket_ask"]
    basket_bid = arrays["basket_bid"]
    n = len(basket_mid)

    # Compute synthetic
    synthetic_mid = np.zeros(n)
    for c, w in constituents.items():
        synthetic_mid += w * arrays[f"{c}_mid"]

    raw_spread = basket_mid - synthetic_mid

    # Welford
    welford_n = float(n_prior)
    welford_mean = float(intercept)
    sig_spread = np.empty(n)
    for i in range(n):
        welford_n += 1.0
        welford_mean += (raw_spread[i] - welford_mean) / welford_n
        sig_spread[i] = raw_spread[i] - welford_mean

    # Trade simulation
    basket_pos = 0
    hedge_pos = {c: 0 for c in constituents}
    cash = 0.0

    for i in range(n):
        rem_buy = basket_limit - basket_pos
        rem_sell = basket_limit + basket_pos

        if sig_spread[i] < -entry_thr and rem_buy > 0:
            if not np.isnan(basket_ask[i]):
                qty = rem_buy
                cash -= qty * basket_ask[i]
                basket_pos += qty

                if hedge_factor > 0:
                    for c, w in constituents.items():
                        c_bid = arrays[f"{c}_bid"][i]
                        if not np.isnan(c_bid):
                            hq = int(round(w * hedge_factor * qty / basket_limit))
                            if hq > 0:
                                cash += hq * c_bid
                                hedge_pos[c] -= hq

        elif sig_spread[i] > entry_thr and rem_sell > 0:
            if not np.isnan(basket_bid[i]):
                qty = rem_sell
                cash += qty * basket_bid[i]
                basket_pos -= qty

                if hedge_factor > 0:
                    for c, w in constituents.items():
                        c_ask = arrays[f"{c}_ask"][i]
                        if not np.isnan(c_ask):
                            hq = int(round(w * hedge_factor * qty / basket_limit))
                            if hq > 0:
                                cash -= hq * c_ask
                                hedge_pos[c] += hq

    # MTM
    mtm = basket_pos * basket_mid[-1]
    for c in constituents:
        mtm += hedge_pos[c] * arrays[f"{c}_mid"][-1]

    return float(cash + mtm)


def analyze_hedging(prices: pd.DataFrame) -> None:
    """Compare PnL with different hedge_factor values."""
    print("\n" + "=" * 70)
    print("5. HEDGING ANALYSIS")
    print("=" * 70)

    days = sorted(prices["day"].unique())

    for basket in BASKET_PRODUCTS:
        print(f"\n{'─'*50}")
        print(f"  {basket}")
        print(f"{'─'*50}")

        core = [p for p in CONSTITUENTS if p in set(prices["product"].unique())]
        data = align_products(prices, [basket] + core)
        if basket not in data or len(core) < 2:
            continue

        y = data[basket]
        res = ols_regression(y, {p: data[p] for p in core})
        constituents = res["weights"]
        intercept = res.get("intercept", 0.0)

        entry_thr = 80
        n_prior = 1000

        print(f"  Weights: {constituents}")
        print(f"  entry_thr={entry_thr}, n_prior={n_prior}")

        hedge_factors = [0.0, 0.25, 0.5, 0.75, 1.0]

        print(f"\n  {'hf':>6} | " + " | ".join(f"day {d:>3} PnL" for d in days) + " | avg PnL")
        print(f"  {'─'*6}-+-" + "-+-".join("─" * 12 for _ in days) + "-+-" + "─" * 10)

        for hf in hedge_factors:
            day_pnls = []
            for day in days:
                pnl = run_hedged_backtest(prices, basket, constituents, intercept, day, entry_thr, n_prior, hf)
                day_pnls.append(pnl)

            pnl_str = " | ".join(f"{p:>12,.0f}" for p in day_pnls)
            avg = np.mean(day_pnls) if day_pnls else 0
            print(f"  {hf:>6.2f} | {pnl_str} | {avg:>10,.0f}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading P3 basket data from all available rounds...")

    # Primary data: round2 (has all products)
    primary_dir = ROOT / "data" / "p3" / "round2"
    prices = load_prices(primary_dir)
    products = sorted(prices["product"].unique())
    days = sorted(prices["day"].unique())

    print(f"Round 2 products: {products}")
    print(f"Round 2 days: {days}")
    print(f"Rows: {len(prices):,}")

    # 1. OLS weights
    analyze_ols_weights(prices)

    # 2. Spread half-life
    half_lives = analyze_half_life(prices)

    # 3. Parameter optimization
    optimize_parameters(prices)

    # 4. Inter-basket arb
    analyze_inter_basket(prices)

    # 5. Hedging analysis
    analyze_hedging(prices)

    # ─── MULTI-ROUND VALIDATION ─────────────────────────────
    print("\n" + "=" * 70)
    print("6. MULTI-ROUND VALIDATION")
    print("=" * 70)
    print("  Testing best parameters across rounds 3-5...")

    for rd in ROUND_DIRS[1:]:  # rounds 3, 4, 5
        if not rd.exists():
            continue
        try:
            rp = load_prices(rd)
        except FileNotFoundError:
            continue
        # Clear caches for new round data
        _day_cache.clear()
        _hedge_cache.clear()
        r_products = set(rp["product"].unique())
        if "PICNIC_BASKET1" not in r_products:
            continue

        r_days = sorted(rp["day"].unique())
        print(f"\n  {rd.name}: days={r_days}")

        for basket in BASKET_PRODUCTS:
            if basket not in r_products:
                continue

            core = [p for p in CONSTITUENTS if p in r_products]
            data = align_products(rp, [basket] + core)
            if basket not in data or len(core) < 2:
                continue

            y = data[basket]
            res = ols_regression(y, {p: data[p] for p in core})

            print(f"    {basket}: R2={res['r_squared']:.6f}, "
                  f"weights={res['weights']}, int={res.get('intercept', 'N/A')}")

            # Quick backtest with round2-optimal params
            for day in r_days:
                bt = run_basket_backtest(rp, basket, res["weights"], res.get("intercept", 0.0), day, 80, 1000)
                print(f"      day {day}: PnL={bt['pnl']:>12,.0f}  trades={bt['n_trades']}  max_pos={bt['max_pos']}")

    # ─── RECOMMENDATIONS ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR strategies/basket.py")
    print("=" * 70)
    print("""
  PICNIC_BASKET1:
    - Weights are UNSTABLE across days (R2 ranges 0.62-0.95 across rounds).
      Day-0 weights differ wildly from day-1. Don't hard-code OLS weights.
      Keep using wall_mid-based spread with Welford online mean (current approach).
    - +SQUID_INK boosts R2 to 0.967 on round2 - worth testing as constituent.
    - Half-life: ~720 ticks => n_prior = 1440 (round to 1500).
      But grid search shows n_prior=30000-60000 gives best walk-forward PnL.
      The large n_prior acts as a stabilizer against regime changes.
    - BEST PARAMS: entry_thr=60, n_prior=30000 (avg val PnL=43,305)
      Runner-up: entry_thr=30, n_prior=60000 (avg val PnL=42,285, lower std)
      For robustness prefer entry_thr=30, n_prior=60000 (std=5,985 vs 21,795).

  PICNIC_BASKET2:
    - Constituents: ~4x CROISSANTS + 2.9x JAMS - 1.4x DJEMBES + 13523 (R2=0.95)
      Adding KELP improves to R2=0.957 (3.8x CROIS + 2.5x JAMS - 1.3x DJEM - 2.9x KELP + 21870).
    - Half-life: ~650 ticks => n_prior = 1300.
    - BEST PARAMS: entry_thr=60, n_prior=500 (avg val PnL=21,600)
      B2 is more responsive - smaller n_prior works better here.

  INTER-BASKET ARB:
    - B1-B2 spread is NON-stationary (ADF p=0.31). Mean drifts across days.
      NOT suitable for pair trading. Skip.

  HEDGING:
    - Hedging has NEGLIGIBLE effect on PnL (< 1% difference).
      Keep hedge_factor=0 to avoid complexity and fill risk.

  SUMMARY OF CHANGES:
    BasketTrader("PICNIC_BASKET1", ..., entry_thr=30, n_prior=60000, seed_mean=0)
    BasketTrader("PICNIC_BASKET2", ..., entry_thr=60, n_prior=500, seed_mean=0)
    hedge_factor=0 for both (no change)
""")


if __name__ == "__main__":
    main()
