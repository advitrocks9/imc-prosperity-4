"""Auto-classify Prosperity 4 products from CSV data.

Computes per-product: CV, lag-1 ACF, ADF p-value, Hurst exponent,
excess kurtosis, book structure metrics. Assigns archetype with confidence.

Usage:
    uv run analysis/eda.py data/tutorial/        # classify tutorial products
    uv run analysis/eda.py data/round1/          # classify round 1 products
    uv run analysis/eda.py data/round1/ --json   # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.data_loader import load_prices, load_trades, get_product_mid, get_product_trades


# ─────────────────────────────────────────────────────────────────
# Archetypes
# ─────────────────────────────────────────────────────────────────

ARCHETYPES = {
    1: "STABLE",
    2: "DRIFTING",
    3: "BASKET/ETF",
    4: "OPTIONS",
    5: "CONVERSION",
    6: "SIGNAL-DRIVEN",
}

# Name patterns for fast classification
BASKET_KEYWORDS = {"BASKET", "BUNDLE", "GIFT", "PICNIC", "HAMPER"}
OPTIONS_KEYWORDS = {"VOUCHER", "COUPON", "OPTION", "CALL", "PUT"}
CONVERSION_KEYWORDS = {"MACARON", "ORCHID", "SUGAR", "VANILLA"}


@dataclass
class ProductStats:
    """Statistical profile for a single product."""
    product: str
    n_ticks: int = 0
    mean_price: float = 0.0
    std_price: float = 0.0
    cv: float = 0.0               # coefficient of variation
    price_range_pct: float = 0.0  # (max-min)/mean as percentage
    adf_pvalue: float = 1.0       # ADF test on levels
    lag1_acf: float = 0.0         # lag-1 autocorrelation of returns
    lag2_acf: float = 0.0         # lag-2 autocorrelation of returns
    excess_kurtosis: float = 0.0  # excess kurtosis of returns
    hurst: float = 0.5            # Hurst exponent (0.5=random walk)
    mean_spread: float = 0.0
    spread_cv: float = 0.0
    book_symmetry: float = 0.0    # bid_vol_total / ask_vol_total ratio
    n_trades: int = 0
    mean_trade_size: float = 0.0
    archetype: int = 0
    archetype_name: str = ""
    confidence: str = ""          # high/medium/low
    reasoning: list[str] = field(default_factory=list)


def compute_hurst(ts: np.ndarray, max_lag: int = 20) -> float:
    """Rescaled range (R/S) Hurst exponent estimator.

    H < 0.5 = mean-reverting, H = 0.5 = random walk, H > 0.5 = trending.
    """
    if len(ts) < max_lag * 4:
        return 0.5

    lags = range(2, max_lag + 1)
    rs_values = []

    for lag in lags:
        # Split into non-overlapping chunks
        n_chunks = len(ts) // lag
        if n_chunks < 2:
            continue

        rs_chunk = []
        for i in range(n_chunks):
            chunk = ts[i * lag:(i + 1) * lag]
            mean_chunk = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_chunk)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(chunk, ddof=1)
            if s > 1e-10:
                rs_chunk.append(r / s)

        if rs_chunk:
            rs_values.append((np.log(lag), np.log(np.mean(rs_chunk))))

    if len(rs_values) < 3:
        return 0.5

    log_lags = np.array([v[0] for v in rs_values])
    log_rs = np.array([v[1] for v in rs_values])

    # OLS fit: log(R/S) = H * log(lag) + c
    coeffs = np.polyfit(log_lags, log_rs, 1)
    return float(np.clip(coeffs[0], 0.0, 1.0))


def analyze_product(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    product: str,
    all_products: list[str],
) -> ProductStats:
    """Compute full statistical profile for one product."""
    stats = ProductStats(product=product)

    pdf = get_product_mid(prices, product)
    tdf = get_product_trades(trades, product)

    if pdf.empty:
        stats.reasoning.append("No price data")
        return stats

    mid = pdf["mid_price"].dropna().values
    stats.n_ticks = len(mid)

    if stats.n_ticks < 10:
        stats.reasoning.append(f"Only {stats.n_ticks} ticks - insufficient data")
        return stats

    # Basic stats
    stats.mean_price = float(np.mean(mid))
    stats.std_price = float(np.std(mid, ddof=1))
    stats.cv = stats.std_price / stats.mean_price if stats.mean_price > 0 else 0
    stats.price_range_pct = float((np.max(mid) - np.min(mid)) / stats.mean_price * 100) if stats.mean_price > 0 else 0

    # Returns
    returns = np.diff(mid) / mid[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) > 10:
        # ADF test on levels
        try:
            adf_result = adfuller(mid, maxlag=min(20, len(mid) // 4), regression="c", autolag="AIC")
            stats.adf_pvalue = float(adf_result[1])
        except Exception:
            stats.adf_pvalue = 1.0

        # Autocorrelation of returns
        try:
            acf_vals = acf(returns, nlags=5, fft=True)
            stats.lag1_acf = float(acf_vals[1])
            stats.lag2_acf = float(acf_vals[2])
        except Exception:
            pass

        # Excess kurtosis
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        if std_r > 1e-10:
            stats.excess_kurtosis = float(np.mean(((returns - mean_r) / std_r) ** 4) - 3.0)

        # Hurst exponent on returns
        stats.hurst = compute_hurst(returns)

    # Spread stats
    if "spread" in pdf.columns:
        spreads = pdf["spread"].dropna().values
        if len(spreads) > 0:
            stats.mean_spread = float(np.mean(spreads))
            s_std = float(np.std(spreads, ddof=1))
            stats.spread_cv = s_std / stats.mean_spread if stats.mean_spread > 0 else 0

    # Book symmetry: ratio of total bid volume to total ask volume
    bid_vol_cols = [c for c in pdf.columns if c.startswith("bid_volume")]
    ask_vol_cols = [c for c in pdf.columns if c.startswith("ask_volume")]
    if bid_vol_cols and ask_vol_cols:
        total_bid = pdf[bid_vol_cols].sum(axis=1).mean()
        total_ask = pdf[ask_vol_cols].sum(axis=1).mean()
        if total_ask > 0:
            stats.book_symmetry = float(total_bid / total_ask)

    # Trade stats
    if not tdf.empty:
        stats.n_trades = len(tdf)
        stats.mean_trade_size = float(tdf["quantity"].mean())

    # Classify
    _classify(stats, all_products)
    return stats


def _classify(stats: ProductStats, all_products: list[str]) -> None:
    """Assign archetype using the decision tree from ROUND_PLAYBOOK."""
    name = stats.product.upper()

    # 1. Name-based: conversion
    if any(kw in name for kw in CONVERSION_KEYWORDS):
        stats.archetype = 5
        stats.archetype_name = ARCHETYPES[5]
        stats.confidence = "low"
        stats.reasoning.append("Name matches conversion keyword (confirm via conversionObservations at runtime)")
        return

    # 2. Name-based: options
    if any(kw in name for kw in OPTIONS_KEYWORDS):
        stats.archetype = 4
        stats.archetype_name = ARCHETYPES[4]
        stats.confidence = "high"
        stats.reasoning.append(f"Name contains options keyword")
        return

    # 3. Name-based: basket
    if any(kw in name for kw in BASKET_KEYWORDS):
        stats.archetype = 3
        stats.archetype_name = ARCHETYPES[3]
        stats.confidence = "high"
        stats.reasoning.append(f"Name contains basket keyword")
        return

    # 4. CV-based classification
    if stats.cv < 0.001:
        stats.archetype = 1
        stats.archetype_name = ARCHETYPES[1]
        stats.confidence = "high" if stats.adf_pvalue < 0.01 else "medium"
        stats.reasoning.append(f"CV={stats.cv:.6f} < 0.001 → stationary")
        if stats.adf_pvalue < 0.01:
            stats.reasoning.append(f"ADF p={stats.adf_pvalue:.4f} confirms stationarity")
        return

    # 5. ADF for ambiguous CV range
    if stats.cv < 0.01 and stats.adf_pvalue < 0.01:
        stats.archetype = 1
        stats.archetype_name = ARCHETYPES[1]
        stats.confidence = "medium"
        stats.reasoning.append(f"CV={stats.cv:.6f} ambiguous but ADF p={stats.adf_pvalue:.4f} → stationary")
        return

    # 6. Drifting vs Signal-Driven
    if stats.excess_kurtosis > 5:
        stats.archetype = 6
        stats.archetype_name = ARCHETYPES[6]
        stats.confidence = "medium"
        stats.reasoning.append(f"Kurtosis={stats.excess_kurtosis:.1f} > 5 → spike-driven")
        return

    # 6. Default: drifting
    stats.archetype = 2
    stats.archetype_name = ARCHETYPES[2]

    if stats.adf_pvalue > 0.10 and abs(stats.lag1_acf) > 0.1:
        stats.confidence = "high"
        stats.reasoning.append(f"ADF p={stats.adf_pvalue:.4f} (non-stationary), ACF1={stats.lag1_acf:.3f} (mean-reverting returns)")
    elif stats.adf_pvalue > 0.10:
        stats.confidence = "high"
        stats.reasoning.append(f"ADF p={stats.adf_pvalue:.4f} → non-stationary, drifting")
    else:
        stats.confidence = "medium"
        stats.reasoning.append(f"CV={stats.cv:.6f}, ADF p={stats.adf_pvalue:.4f} → likely drifting")


def check_basket_relationships(
    prices: pd.DataFrame,
    products: list[str],
) -> dict[str, dict]:
    """Check if any product is a linear combination of others.

    Returns dict mapping basket product → {constituents: [...], weights: [...], r_squared: float}.
    """
    results: dict[str, dict] = {}

    if len(products) < 3:
        return results

    # Build mid-price matrix
    product_series: dict[str, np.ndarray] = {}
    for p in products:
        pdf = get_product_mid(prices, p)
        if not pdf.empty:
            # Use first day only for alignment
            day0 = pdf[pdf["day"] == pdf["day"].min()]
            product_series[p] = day0["mid_price"].values

    if len(product_series) < 3:
        return results

    # Align lengths
    min_len = min(len(v) for v in product_series.values())
    for k in product_series:
        product_series[k] = product_series[k][:min_len]

    # For each product, regress on all subsets of 2-4 others
    from itertools import combinations

    for target in products:
        if target not in product_series:
            continue
        y = product_series[target]
        others = [p for p in products if p != target and p in product_series]

        best_r2 = 0.0
        best_result = None

        for n_const in range(2, min(5, len(others) + 1)):
            for combo in combinations(others, n_const):
                X = np.column_stack([product_series[c] for c in combo])
                # OLS with intercept
                X_aug = np.column_stack([X, np.ones(len(X))])
                try:
                    weights, residuals, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                    y_pred = X_aug @ weights
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                except Exception:
                    continue

                if r2 > best_r2:
                    best_r2 = r2
                    best_result = {
                        "constituents": list(combo),
                        "weights": [round(float(w), 4) for w in weights[:-1]],
                        "intercept": round(float(weights[-1]), 4),
                        "r_squared": round(float(r2), 6),
                    }

        if best_r2 > 0.95 and best_result:
            results[target] = best_result

    return results


def run_eda(round_dir: Path | str) -> list[ProductStats]:
    """Run full EDA on a round directory. Returns list of ProductStats."""
    round_dir = Path(round_dir)
    prices = load_prices(round_dir)
    try:
        trades = load_trades(round_dir)
    except FileNotFoundError:
        trades = pd.DataFrame(columns=["day", "timestamp", "buyer", "seller", "symbol", "price", "quantity"])

    products = sorted(prices["product"].unique())

    # Analyze each product
    results = []
    for product in products:
        stats = analyze_product(prices, trades, product, products)
        results.append(stats)

    # Check basket relationships (may reclassify products)
    # Only override products not already classified by name (OPTIONS, CONVERSION)
    name_classified_archetypes = {4, 5}  # OPTIONS, CONVERSION - classified by keyword, don't override
    options_products = {s.product for s in results if s.archetype == 4}
    basket_hits = check_basket_relationships(prices, products)
    for stats in results:
        if stats.archetype in name_classified_archetypes:
            continue  # don't reclassify OPTIONS/CONVERSION as BASKET
        # Skip options underlyings (product name is prefix of an OPTIONS product)
        if any(opt.startswith(stats.product) and opt != stats.product for opt in options_products):
            stats.reasoning.append("Skipped basket override - likely options underlying")
            continue
        if stats.product in basket_hits:
            hit = basket_hits[stats.product]
            if hit["r_squared"] > 0.95:
                stats.archetype = 3
                stats.archetype_name = ARCHETYPES[3]
                stats.confidence = "high"
                stats.reasoning.append(
                    f"OLS R²={hit['r_squared']:.4f} on {hit['constituents']} → basket"
                )

    return results


def format_report(results: list[ProductStats]) -> str:
    """Format EDA results as a readable report."""
    lines = ["# EDA Report", ""]

    # Summary table
    lines.append("## Classification Summary")
    lines.append("")
    lines.append("| Product | Archetype | Confidence | CV | ADF p | ACF1 | Hurst | Kurtosis |")
    lines.append("|---------|-----------|------------|-----|-------|------|-------|----------|")
    for s in results:
        lines.append(
            f"| {s.product} | {s.archetype_name} | {s.confidence} | "
            f"{s.cv:.6f} | {s.adf_pvalue:.4f} | {s.lag1_acf:.3f} | "
            f"{s.hurst:.3f} | {s.excess_kurtosis:.1f} |"
        )
    lines.append("")

    # Detailed per-product
    for s in results:
        lines.append(f"## {s.product} - {s.archetype_name} ({s.confidence})")
        lines.append("")
        lines.append(f"- **Ticks**: {s.n_ticks:,} | **Trades**: {s.n_trades:,}")
        lines.append(f"- **Mean price**: {s.mean_price:.2f} | **Std**: {s.std_price:.2f}")
        lines.append(f"- **Price range**: {s.price_range_pct:.3f}% of mean")
        lines.append(f"- **Spread**: mean={s.mean_spread:.2f}, CV={s.spread_cv:.4f}")
        lines.append(f"- **Book symmetry** (bid/ask vol): {s.book_symmetry:.3f}")
        if s.n_trades > 0:
            lines.append(f"- **Mean trade size**: {s.mean_trade_size:.1f}")
        lines.append(f"- **Reasoning**: {'; '.join(s.reasoning)}")
        lines.append("")

    # Basket relationships
    lines.append("## Basket Relationship Check")
    lines.append("")
    # Re-run for report (fast enough)
    # Already captured in classification, just note it
    basket_found = [s for s in results if s.archetype == 3]
    if basket_found:
        for s in basket_found:
            lines.append(f"- **{s.product}** identified as basket (see reasoning above)")
    else:
        lines.append("- No basket relationships detected (R² > 0.95 threshold)")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prosperity 4 product classifier")
    parser.add_argument("round_dir", type=Path, help="Path to round data directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = run_eda(args.round_dir)

    if args.json:
        print(json.dumps([asdict(s) for s in results], indent=2))
    else:
        print(format_report(results))


if __name__ == "__main__":
    main()
