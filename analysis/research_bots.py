"""Comprehensive bot behavior mining from P3 Round 5 trade data.

Identifies all bot archetypes, fingerprints their strategies, and produces
actionable exploitation recommendations for P4.

Usage:
    uv run analysis/research_bots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.data_loader import load_prices, load_trades, get_product_mid, get_product_trades

ROUND_DIR = Path(__file__).resolve().parent.parent / "data" / "p3" / "round5"
# No SUBMISSION in P3 data - all traders are bots
PLAYER_NAMES = {"SUBMISSION"}

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def build_mid_lookup(prices: pd.DataFrame) -> dict[tuple[int, int, str], float]:
    """Build (day, timestamp, product) -> mid_price lookup for fast joins."""
    lookup: dict[tuple[int, int, str], float] = {}
    for _, row in prices[["day", "timestamp", "product", "mid_price"]].iterrows():
        lookup[(int(row["day"]), int(row["timestamp"]), row["product"])] = float(row["mid_price"])
    return lookup


def get_mid_at(
    mid_lookup: dict[tuple[int, int, str], float],
    day: int,
    timestamp: int,
    product: str,
) -> float | None:
    """Get mid price at exact (day, ts, product), or None."""
    return mid_lookup.get((day, timestamp, product))


def get_future_mid(
    mid_lookup: dict[tuple[int, int, str], float],
    day: int,
    timestamp: int,
    product: str,
    horizon: int = 1000,
) -> float | None:
    """Get mid price at (day, ts + horizon, product). Tries exact match then nearest."""
    exact = mid_lookup.get((day, timestamp + horizon, product))
    if exact is not None:
        return exact
    # Try nearest within +/- 200 of target
    target = timestamp + horizon
    for delta in range(0, 300, 100):
        for sign in [0, 1, -1]:
            candidate = mid_lookup.get((day, target + sign * delta, product))
            if candidate is not None:
                return candidate
    return None


def merge_trades_with_mid(
    trades: pd.DataFrame,
    mid_lookup: dict[tuple[int, int, str], float],
) -> pd.DataFrame:
    """Add mid_price and future_mid columns to trades."""
    mids = []
    future_mids_10 = []  # 10 ticks ahead (1000 ts)
    future_mids_50 = []  # 50 ticks ahead (5000 ts)
    future_mids_100 = []  # 100 ticks ahead (10000 ts)

    for _, row in trades.iterrows():
        day, ts, prod = int(row["day"]), int(row["timestamp"]), row["symbol"]
        mids.append(get_mid_at(mid_lookup, day, ts, prod))
        future_mids_10.append(get_future_mid(mid_lookup, day, ts, prod, 1000))
        future_mids_50.append(get_future_mid(mid_lookup, day, ts, prod, 5000))
        future_mids_100.append(get_future_mid(mid_lookup, day, ts, prod, 10000))

    trades = trades.copy()
    trades["mid_price"] = mids
    trades["future_mid_10"] = future_mids_10
    trades["future_mid_50"] = future_mids_50
    trades["future_mid_100"] = future_mids_100
    return trades


# ─────────────────────────────────────────────────────────────────────
# 1. Bot Census
# ─────────────────────────────────────────────────────────────────────

def bot_census(trades: pd.DataFrame) -> pd.DataFrame:
    """Produce a census table of all bots."""
    all_traders = sorted(
        (set(trades["buyer"].unique()) | set(trades["seller"].unique())) - PLAYER_NAMES
    )

    rows = []
    for name in all_traders:
        buys = trades[trades["buyer"] == name]
        sells = trades[trades["seller"] == name]

        buy_products = set(buys["symbol"].unique())
        sell_products = set(sells["symbol"].unique())
        all_products = sorted(buy_products | sell_products)

        total_buys = len(buys)
        total_sells = len(sells)
        total = total_buys + total_sells

        avg_buy_size = float(buys["quantity"].mean()) if total_buys > 0 else 0
        avg_sell_size = float(sells["quantity"].mean()) if total_sells > 0 else 0

        # Price relative to mid
        buy_vs_mid = []
        if "mid_price" in buys.columns:
            valid = buys.dropna(subset=["mid_price"])
            if len(valid) > 0:
                buy_vs_mid = (valid["price"] - valid["mid_price"]).tolist()

        sell_vs_mid = []
        if "mid_price" in sells.columns:
            valid = sells.dropna(subset=["mid_price"])
            if len(valid) > 0:
                sell_vs_mid = (valid["price"] - valid["mid_price"]).tolist()

        avg_buy_premium = float(np.mean(buy_vs_mid)) if buy_vs_mid else 0
        avg_sell_premium = float(np.mean(sell_vs_mid)) if sell_vs_mid else 0

        rows.append({
            "bot": name,
            "total_trades": total,
            "buys": total_buys,
            "sells": total_sells,
            "buy_sell_ratio": total_buys / total_sells if total_sells > 0 else float("inf"),
            "products": ", ".join(all_products),
            "n_products": len(all_products),
            "avg_buy_size": round(avg_buy_size, 1),
            "avg_sell_size": round(avg_sell_size, 1),
            "avg_buy_premium": round(avg_buy_premium, 2),
            "avg_sell_premium": round(avg_sell_premium, 2),
        })

    return pd.DataFrame(rows).sort_values("total_trades", ascending=False)


# ─────────────────────────────────────────────────────────────────────
# 2. Per-Bot Strategy Fingerprinting
# ─────────────────────────────────────────────────────────────────────

def fingerprint_bot(
    name: str,
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    mid_lookup: dict,
) -> dict:
    """Deep fingerprint of a single bot."""
    buys = trades[trades["buyer"] == name].copy()
    sells = trades[trades["seller"] == name].copy()
    all_trades_mask = (trades["buyer"] == name) | (trades["seller"] == name)
    bot_trades = trades[all_trades_mask].copy()

    if len(bot_trades) < 20:
        return {"bot": name, "insufficient_data": True}

    result: dict = {"bot": name}

    # --- Products traded ---
    buy_prods = sorted(buys["symbol"].unique()) if len(buys) > 0 else []
    sell_prods = sorted(sells["symbol"].unique()) if len(sells) > 0 else []
    result["buy_products"] = buy_prods
    result["sell_products"] = sell_prods

    # --- Timing analysis ---
    timestamps = bot_trades.sort_values(["day", "timestamp"])
    per_day_intervals = []
    for day in timestamps["day"].unique():
        day_ts = timestamps[timestamps["day"] == day]["timestamp"].values
        if len(day_ts) > 1:
            intervals = np.diff(day_ts)
            per_day_intervals.extend(intervals.tolist())

    if per_day_intervals:
        intervals_arr = np.array(per_day_intervals)
        mean_interval = float(np.mean(intervals_arr))
        std_interval = float(np.std(intervals_arr))
        cv = std_interval / mean_interval if mean_interval > 0 else 0

        result["timing"] = {
            "mean_interval": round(mean_interval, 0),
            "std_interval": round(std_interval, 0),
            "cv": round(cv, 2),
            "pattern": "periodic" if cv < 0.3 else ("clustered" if cv > 1.5 else "semi-regular"),
            "min_interval": int(np.min(intervals_arr)),
            "median_interval": int(np.median(intervals_arr)),
        }

        # Check for modular timing (timestamp % N == 0)
        all_ts = bot_trades["timestamp"].values
        mod_patterns = {}
        for n in [100, 200, 500, 1000, 2000, 5000]:
            mods = all_ts % n
            zero_frac = float(np.sum(mods == 0)) / len(mods)
            if zero_frac > 0.8:
                mod_patterns[n] = round(zero_frac, 3)
        if mod_patterns:
            result["timing"]["mod_patterns"] = mod_patterns
    else:
        result["timing"] = {"pattern": "insufficient"}

    # --- Size analysis ---
    buy_sizes = buys["quantity"].values if len(buys) > 0 else np.array([])
    sell_sizes = sells["quantity"].values if len(sells) > 0 else np.array([])
    all_sizes = bot_trades["quantity"].values

    size_dist = dict(pd.Series(all_sizes).value_counts().sort_index().items())
    result["sizes"] = {
        "mean": round(float(np.mean(all_sizes)), 1),
        "std": round(float(np.std(all_sizes)), 1),
        "fixed": float(np.std(all_sizes)) < 0.5,
        "distribution": {int(k): int(v) for k, v in list(size_dist.items())[:10]},
    }

    # --- Direction analysis per product ---
    product_direction = {}
    for prod in sorted(set(buy_prods) | set(sell_prods)):
        prod_buys = buys[buys["symbol"] == prod]
        prod_sells = sells[sells["symbol"] == prod]
        n_buy = len(prod_buys)
        n_sell = len(prod_sells)
        net = n_buy - n_sell
        vol_buy = int(prod_buys["quantity"].sum()) if n_buy > 0 else 0
        vol_sell = int(prod_sells["quantity"].sum()) if n_sell > 0 else 0
        net_vol = vol_buy - vol_sell

        product_direction[prod] = {
            "n_buy": n_buy,
            "n_sell": n_sell,
            "vol_buy": vol_buy,
            "vol_sell": vol_sell,
            "net_vol": net_vol,
            "bias": "buyer" if net_vol > 0 else "seller" if net_vol < 0 else "neutral",
        }
    result["direction"] = product_direction

    # --- Price level analysis ---
    if "mid_price" in bot_trades.columns:
        valid = bot_trades.dropna(subset=["mid_price"])
        if len(valid) > 10:
            # For buys
            buy_valid = buys.dropna(subset=["mid_price"]) if "mid_price" in buys.columns else pd.DataFrame()
            sell_valid = sells.dropna(subset=["mid_price"]) if "mid_price" in sells.columns else pd.DataFrame()

            result["price_level"] = {}
            if len(buy_valid) > 0:
                buy_spread = (buy_valid["price"] - buy_valid["mid_price"]).values
                result["price_level"]["buy_avg_spread"] = round(float(np.mean(buy_spread)), 2)
                result["price_level"]["buy_median_spread"] = round(float(np.median(buy_spread)), 2)
                # What fraction of buys are above mid (aggressive)?
                result["price_level"]["buy_aggressive_frac"] = round(
                    float(np.sum(buy_spread > 0)) / len(buy_spread), 3
                )

            if len(sell_valid) > 0:
                sell_spread = (sell_valid["price"] - sell_valid["mid_price"]).values
                result["price_level"]["sell_avg_spread"] = round(float(np.mean(sell_spread)), 2)
                result["price_level"]["sell_median_spread"] = round(float(np.median(sell_spread)), 2)
                # What fraction of sells are below mid (aggressive)?
                result["price_level"]["sell_aggressive_frac"] = round(
                    float(np.sum(sell_spread < 0)) / len(sell_spread), 3
                )

    # --- Momentum vs mean-reversion ---
    # For each trade, compute price change in the last N ticks before the trade
    # If bot buys when price is rising → momentum; buys when price is falling → mean-reversion
    if "mid_price" in bot_trades.columns:
        momentum_scores = []
        for prod in sorted(set(buy_prods) | set(sell_prods)):
            prod_prices = prices[prices["product"] == prod][["day", "timestamp", "mid_price"]].copy()
            prod_buys = buys[(buys["symbol"] == prod) & buys["mid_price"].notna()]
            prod_sells = sells[(sells["symbol"] == prod) & sells["mid_price"].notna()]

            for _, trade in prod_buys.iterrows():
                day, ts = int(trade["day"]), int(trade["timestamp"])
                # Price 1000 ts ago
                past_mid = get_mid_at(mid_lookup, day, ts - 1000, prod)
                curr_mid = trade["mid_price"]
                if past_mid is not None and curr_mid is not None:
                    price_change = curr_mid - past_mid
                    # Buying: positive score if buying when price rising (momentum)
                    momentum_scores.append(price_change)

            for _, trade in prod_sells.iterrows():
                day, ts = int(trade["day"]), int(trade["timestamp"])
                past_mid = get_mid_at(mid_lookup, day, ts - 1000, prod)
                curr_mid = trade["mid_price"]
                if past_mid is not None and curr_mid is not None:
                    price_change = curr_mid - past_mid
                    # Selling: negative score if selling when price falling (momentum)
                    momentum_scores.append(-price_change)

        if momentum_scores:
            avg_momentum = float(np.mean(momentum_scores))
            result["momentum_score"] = round(avg_momentum, 3)
            result["momentum_label"] = (
                "momentum" if avg_momentum > 0.5
                else "mean_reversion" if avg_momentum < -0.5
                else "neutral"
            )

    return result


# ─────────────────────────────────────────────────────────────────────
# 3. Market Maker Identification
# ─────────────────────────────────────────────────────────────────────

def identify_market_makers(trades: pd.DataFrame, prices: pd.DataFrame) -> list[dict]:
    """Identify bots that behave like market makers."""
    all_traders = sorted(
        (set(trades["buyer"].unique()) | set(trades["seller"].unique())) - PLAYER_NAMES
    )
    mm_candidates = []

    for name in all_traders:
        buys = trades[trades["buyer"] == name]
        sells = trades[trades["seller"] == name]
        total = len(buys) + len(sells)

        if total < 50:
            continue

        # Market maker criteria:
        # 1. Roughly equal buys and sells (ratio 0.5-2.0)
        ratio = len(buys) / len(sells) if len(sells) > 0 else float("inf")
        if not (0.4 <= ratio <= 2.5):
            continue

        # 2. Trades frequently
        bot_trades = trades[(trades["buyer"] == name) | (trades["seller"] == name)]
        per_day = bot_trades.groupby("day").size()
        avg_per_day = float(per_day.mean())

        # 3. Check if they trade near best bid/ask (within spread)
        near_mid_count = 0
        total_checked = 0
        if "mid_price" in bot_trades.columns:
            valid = bot_trades.dropna(subset=["mid_price"])
            total_checked = len(valid)
            if total_checked > 0:
                dist_from_mid = np.abs(valid["price"] - valid["mid_price"])
                near_mid_count = int(np.sum(dist_from_mid <= valid["mid_price"] * 0.005))  # within 0.5%

        products = sorted(set(buys["symbol"].unique()) | set(sells["symbol"].unique()))

        # Per-product symmetry
        prod_symmetry = {}
        for prod in products:
            pb = buys[buys["symbol"] == prod]
            ps = sells[sells["symbol"] == prod]
            if len(pb) > 5 and len(ps) > 5:
                vol_buy = int(pb["quantity"].sum())
                vol_sell = int(ps["quantity"].sum())
                vol_ratio = vol_buy / vol_sell if vol_sell > 0 else float("inf")
                prod_symmetry[prod] = {
                    "vol_buy": vol_buy,
                    "vol_sell": vol_sell,
                    "vol_ratio": round(vol_ratio, 2),
                    "symmetric": 0.5 <= vol_ratio <= 2.0,
                }

        symmetric_products = [p for p, v in prod_symmetry.items() if v.get("symmetric")]

        mm_candidates.append({
            "bot": name,
            "total_trades": total,
            "buy_sell_ratio": round(ratio, 2),
            "avg_trades_per_day": round(avg_per_day, 0),
            "near_mid_frac": round(near_mid_count / total_checked, 3) if total_checked > 0 else 0,
            "products": products,
            "symmetric_products": symmetric_products,
            "product_symmetry": prod_symmetry,
            "is_likely_mm": len(symmetric_products) >= 2 and avg_per_day > 100,
        })

    return sorted(mm_candidates, key=lambda x: -x["total_trades"])


# ─────────────────────────────────────────────────────────────────────
# 4. Informed Trader Identification
# ─────────────────────────────────────────────────────────────────────

def compute_information_ratios(trades: pd.DataFrame) -> list[dict]:
    """Compute each bot's information ratio: avg PnL per trade vs future mid."""
    all_traders = sorted(
        (set(trades["buyer"].unique()) | set(trades["seller"].unique())) - PLAYER_NAMES
    )

    results = []
    for name in all_traders:
        buys = trades[trades["buyer"] == name].copy()
        sells = trades[trades["seller"] == name].copy()

        if len(buys) + len(sells) < 10:
            continue

        pnls_10 = []  # PnL vs 10 ticks later
        pnls_50 = []  # PnL vs 50 ticks later
        pnls_100 = []  # PnL vs 100 ticks later

        # Buys: PnL = future_mid - price (profit if price goes up)
        for horizon, pnl_list in [
            ("future_mid_10", pnls_10),
            ("future_mid_50", pnls_50),
            ("future_mid_100", pnls_100),
        ]:
            if horizon in buys.columns:
                valid = buys.dropna(subset=[horizon])
                if len(valid) > 0:
                    pnl_list.extend((valid[horizon] - valid["price"]).tolist())

            if horizon in sells.columns:
                valid = sells.dropna(subset=[horizon])
                if len(valid) > 0:
                    # Sells: PnL = price - future_mid (profit if price goes down)
                    pnl_list.extend((valid["price"] - valid[horizon]).tolist())

        per_product_ir = {}
        for prod in sorted(set(buys["symbol"].unique()) | set(sells["symbol"].unique())):
            pb = buys[(buys["symbol"] == prod) & buys["future_mid_50"].notna()]
            ps = sells[(sells["symbol"] == prod) & sells["future_mid_50"].notna()]
            prod_pnls = []
            if len(pb) > 0:
                prod_pnls.extend(((pb["future_mid_50"] - pb["price"]) * pb["quantity"]).tolist())
            if len(ps) > 0:
                prod_pnls.extend(((ps["price"] - ps["future_mid_50"]) * ps["quantity"]).tolist())
            if prod_pnls:
                per_product_ir[prod] = {
                    "mean_pnl": round(float(np.mean(prod_pnls)), 2),
                    "total_pnl": round(float(np.sum(prod_pnls)), 0),
                    "n_trades": len(prod_pnls),
                    "win_rate": round(float(np.sum(np.array(prod_pnls) > 0)) / len(prod_pnls), 3),
                }

        results.append({
            "bot": name,
            "total_trades": len(buys) + len(sells),
            "ir_10": round(float(np.mean(pnls_10)), 4) if pnls_10 else None,
            "ir_50": round(float(np.mean(pnls_50)), 4) if pnls_50 else None,
            "ir_100": round(float(np.mean(pnls_100)), 4) if pnls_100 else None,
            "total_pnl_50": round(float(np.sum(pnls_50)), 0) if pnls_50 else None,
            "win_rate_50": round(float(np.sum(np.array(pnls_50) > 0)) / len(pnls_50), 3) if pnls_50 else None,
            "per_product": per_product_ir,
        })

    return sorted(results, key=lambda x: -(x["ir_50"] or 0))


# ─────────────────────────────────────────────────────────────────────
# 5. Bot-vs-Bot vs Bot-vs-Player Analysis
# ─────────────────────────────────────────────────────────────────────

def bot_interaction_analysis(trades: pd.DataFrame) -> dict:
    """Analyze who trades with whom."""
    total = len(trades)

    bot_vs_bot = trades[
        ~trades["buyer"].isin(PLAYER_NAMES) & ~trades["seller"].isin(PLAYER_NAMES)
    ]
    player_involved = trades[
        trades["buyer"].isin(PLAYER_NAMES) | trades["seller"].isin(PLAYER_NAMES)
    ]

    # Since there's no SUBMISSION in P3 data, all trades are bot-vs-bot
    # But let's compute the pair matrix anyway
    pair_counts: dict[tuple[str, str], int] = {}
    for _, row in trades.iterrows():
        pair = (row["buyer"], row["seller"])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Top pairs
    top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:20]

    # Per-product pair analysis
    product_pairs: dict[str, list[tuple[tuple[str, str], int]]] = {}
    for prod in sorted(trades["symbol"].unique()):
        ptrades = trades[trades["symbol"] == prod]
        pp_counts: dict[tuple[str, str], int] = {}
        for _, row in ptrades.iterrows():
            pair = (row["buyer"], row["seller"])
            pp_counts[pair] = pp_counts.get(pair, 0) + 1
        product_pairs[prod] = sorted(pp_counts.items(), key=lambda x: -x[1])[:5]

    return {
        "total_trades": total,
        "bot_vs_bot": len(bot_vs_bot),
        "player_involved": len(player_involved),
        "bot_vs_bot_frac": round(len(bot_vs_bot) / total, 3),
        "top_pairs": [
            {"buyer": p[0], "seller": p[1], "count": c}
            for (p, c) in top_pairs
        ],
        "product_pairs": {
            prod: [{"buyer": p[0], "seller": p[1], "count": c} for (p, c) in pairs]
            for prod, pairs in product_pairs.items()
        },
    }


# ─────────────────────────────────────────────────────────────────────
# 6. Detailed per-bot pattern analysis
# ─────────────────────────────────────────────────────────────────────

def analyze_requote_patterns(
    name: str,
    trades: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict:
    """For market-maker-like bots, check if they requote predictably after events."""
    bot_trades = trades[(trades["buyer"] == name) | (trades["seller"] == name)].copy()
    if len(bot_trades) < 50:
        return {}

    results = {}
    for prod in sorted(bot_trades["symbol"].unique()):
        prod_trades = bot_trades[bot_trades["symbol"] == prod].sort_values(["day", "timestamp"])
        if len(prod_trades) < 20:
            continue

        # Check: after the bot trades, how quickly does it trade again?
        per_day = {}
        for day in prod_trades["day"].unique():
            day_trades = prod_trades[prod_trades["day"] == day]["timestamp"].values
            if len(day_trades) > 1:
                intervals = np.diff(day_trades)
                per_day[int(day)] = {
                    "count": len(day_trades),
                    "mean_interval": round(float(np.mean(intervals)), 0),
                    "min_interval": int(np.min(intervals)),
                }

        # Check if bot trades at the same time every cycle
        all_ts = prod_trades["timestamp"].values
        for period in [100, 200, 500, 1000]:
            mods = all_ts % period
            # Check if concentrated at specific offsets
            unique_mods, counts = np.unique(mods, return_counts=True)
            if len(unique_mods) > 0:
                max_frac = float(np.max(counts)) / len(mods)
                if max_frac > 0.3:
                    top_mod = int(unique_mods[np.argmax(counts)])
                    results[f"{prod}_period_{period}"] = {
                        "top_offset": top_mod,
                        "fraction_at_offset": round(max_frac, 3),
                    }

    return results


def analyze_olivia_detailed(trades: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Deep analysis of Olivia's exact behavior."""
    olivia_buys = trades[trades["buyer"] == "Olivia"]
    olivia_sells = trades[trades["seller"] == "Olivia"]

    results = {"total_buys": len(olivia_buys), "total_sells": len(olivia_sells)}

    # Per product details
    for prod in sorted(set(olivia_buys["symbol"].unique()) | set(olivia_sells["symbol"].unique())):
        pb = olivia_buys[olivia_buys["symbol"] == prod]
        ps = olivia_sells[olivia_sells["symbol"] == prod]

        prod_prices = prices[prices["product"] == prod]

        prod_info = {
            "buys": [],
            "sells": [],
        }

        for _, t in pb.iterrows():
            day = int(t["day"])
            day_prices = prod_prices[prod_prices["day"] == day]["mid_price"]
            day_low = float(day_prices.min()) if len(day_prices) > 0 else None
            day_high = float(day_prices.max()) if len(day_prices) > 0 else None
            day_range = day_high - day_low if day_low and day_high else None

            prod_info["buys"].append({
                "day": day,
                "timestamp": int(t["timestamp"]),
                "price": float(t["price"]),
                "quantity": int(t["quantity"]),
                "day_low": day_low,
                "day_high": day_high,
                "pct_of_range": round((float(t["price"]) - day_low) / day_range, 3) if day_range else None,
            })

        for _, t in ps.iterrows():
            day = int(t["day"])
            day_prices = prod_prices[prod_prices["day"] == day]["mid_price"]
            day_low = float(day_prices.min()) if len(day_prices) > 0 else None
            day_high = float(day_prices.max()) if len(day_prices) > 0 else None
            day_range = day_high - day_low if day_low and day_high else None

            prod_info["sells"].append({
                "day": day,
                "timestamp": int(t["timestamp"]),
                "price": float(t["price"]),
                "quantity": int(t["quantity"]),
                "day_low": day_low,
                "day_high": day_high,
                "pct_of_range": round((float(t["price"]) - day_low) / day_range, 3) if day_range else None,
            })

        results[prod] = prod_info

    return results


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 80)
    print("BOT BEHAVIOR RESEARCH - P3 Round 5")
    print("=" * 80)

    print("\nLoading data...")
    prices = load_prices(ROUND_DIR)
    trades = load_trades(ROUND_DIR)
    print(f"  Trades: {len(trades)} rows, {len(trades['symbol'].unique())} products")
    print(f"  Prices: {len(prices)} rows, {len(prices['product'].unique())} products")
    print(f"  Days: {sorted(trades['day'].unique())}")

    print("\nBuilding mid-price lookup...")
    mid_lookup = build_mid_lookup(prices)
    print(f"  {len(mid_lookup)} entries")

    print("\nMerging trades with mid prices...")
    trades = merge_trades_with_mid(trades, mid_lookup)
    mid_coverage = trades["mid_price"].notna().sum() / len(trades)
    print(f"  Mid price coverage: {mid_coverage:.1%}")

    # ─── 1. Bot Census ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("1. BOT CENSUS")
    print("=" * 80)
    census = bot_census(trades)
    print(census.to_string(index=False))

    # ─── 2. Strategy Fingerprinting ──────────────────────────────────
    print("\n" + "=" * 80)
    print("2. STRATEGY FINGERPRINTING")
    print("=" * 80)

    all_bots = sorted(
        (set(trades["buyer"].unique()) | set(trades["seller"].unique())) - PLAYER_NAMES
    )

    fingerprints = {}
    for name in all_bots:
        fp = fingerprint_bot(name, trades, prices, mid_lookup)
        fingerprints[name] = fp
        if fp.get("insufficient_data"):
            print(f"\n--- {name}: INSUFFICIENT DATA (<20 trades) ---")
            continue

        print(f"\n--- {name} ---")
        print(f"  Timing: {fp.get('timing', {})}")
        print(f"  Sizes: mean={fp['sizes']['mean']}, std={fp['sizes']['std']}, fixed={fp['sizes']['fixed']}")
        if "momentum_score" in fp:
            print(f"  Momentum: score={fp['momentum_score']}, label={fp['momentum_label']}")
        if "price_level" in fp:
            pl = fp["price_level"]
            print(f"  Price level: buy_spread={pl.get('buy_avg_spread', 'N/A')}, "
                  f"sell_spread={pl.get('sell_avg_spread', 'N/A')}, "
                  f"buy_aggr={pl.get('buy_aggressive_frac', 'N/A')}, "
                  f"sell_aggr={pl.get('sell_aggressive_frac', 'N/A')}")

        # Per-product direction summary
        if "direction" in fp:
            biased = {p: d for p, d in fp["direction"].items() if d["bias"] != "neutral"}
            if biased:
                print(f"  Directional biases:")
                for prod, d in sorted(biased.items()):
                    print(f"    {prod}: {d['bias']} (buy_vol={d['vol_buy']}, sell_vol={d['vol_sell']}, net={d['net_vol']})")

    # ─── 3. Market Maker Identification ──────────────────────────────
    print("\n" + "=" * 80)
    print("3. MARKET MAKER IDENTIFICATION")
    print("=" * 80)

    mm_candidates = identify_market_makers(trades, prices)
    for mm in mm_candidates:
        status = "LIKELY MM" if mm["is_likely_mm"] else "possible"
        print(f"\n  {mm['bot']} [{status}]: {mm['total_trades']} trades, "
              f"B/S ratio={mm['buy_sell_ratio']}, "
              f"near_mid={mm['near_mid_frac']:.1%}, "
              f"avg/day={mm['avg_trades_per_day']}")
        if mm["symmetric_products"]:
            print(f"    Symmetric products: {mm['symmetric_products']}")
        for prod, sym in mm.get("product_symmetry", {}).items():
            print(f"    {prod}: vol B/S={sym['vol_ratio']:.2f} ({'sym' if sym['symmetric'] else 'asym'})")

    # Check requote patterns for likely MMs
    for mm in mm_candidates:
        if mm["is_likely_mm"]:
            requote = analyze_requote_patterns(mm["bot"], trades, prices)
            if requote:
                print(f"\n  Requote patterns for {mm['bot']}:")
                for key, val in requote.items():
                    print(f"    {key}: offset={val['top_offset']}, frac={val['fraction_at_offset']}")

    # ─── 4. Informed Trader Identification ───────────────────────────
    print("\n" + "=" * 80)
    print("4. INFORMED TRADER RANKING (by Information Ratio)")
    print("=" * 80)

    ir_results = compute_information_ratios(trades)
    print(f"\n{'Bot':12s} {'Trades':>7s} {'IR@10':>10s} {'IR@50':>10s} {'IR@100':>10s} {'TotPnL@50':>12s} {'WinRate@50':>10s}")
    print("-" * 73)
    for r in ir_results:
        ir10 = f"{r['ir_10']:.4f}" if r["ir_10"] is not None else "N/A"
        ir50 = f"{r['ir_50']:.4f}" if r["ir_50"] is not None else "N/A"
        ir100 = f"{r['ir_100']:.4f}" if r["ir_100"] is not None else "N/A"
        pnl = f"{r['total_pnl_50']:.0f}" if r["total_pnl_50"] is not None else "N/A"
        wr = f"{r['win_rate_50']:.3f}" if r["win_rate_50"] is not None else "N/A"
        print(f"{r['bot']:12s} {r['total_trades']:7d} {ir10:>10s} {ir50:>10s} {ir100:>10s} {pnl:>12s} {wr:>10s}")

    # Per-product breakdown for top informed traders
    print("\n  Per-product IR breakdown for top bots:")
    for r in ir_results[:5]:
        if r.get("per_product"):
            print(f"\n  {r['bot']}:")
            for prod, info in sorted(r["per_product"].items(), key=lambda x: -abs(x[1]["total_pnl"])):
                print(f"    {prod:40s} mean_pnl={info['mean_pnl']:>8.2f}, "
                      f"total_pnl={info['total_pnl']:>10.0f}, "
                      f"n={info['n_trades']:>5d}, "
                      f"win_rate={info['win_rate']:.3f}")

    # ─── 5. Bot-vs-Bot Analysis ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("5. BOT-VS-BOT INTERACTION ANALYSIS")
    print("=" * 80)

    interactions = bot_interaction_analysis(trades)
    print(f"\n  Total trades: {interactions['total_trades']}")
    print(f"  Bot-vs-bot: {interactions['bot_vs_bot']} ({interactions['bot_vs_bot_frac']:.1%})")
    print(f"  Player-involved: {interactions['player_involved']}")

    print("\n  Top 20 trading pairs:")
    for pair in interactions["top_pairs"]:
        print(f"    {pair['buyer']:12s} -> {pair['seller']:12s}: {pair['count']:5d} trades")

    print("\n  Top pairs per product:")
    for prod, pairs in interactions["product_pairs"].items():
        print(f"\n    {prod}:")
        for pair in pairs[:3]:
            print(f"      {pair['buyer']:12s} -> {pair['seller']:12s}: {pair['count']:5d}")

    # ─── Olivia Deep Dive ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OLIVIA DEEP DIVE")
    print("=" * 80)

    olivia_detail = analyze_olivia_detailed(trades, prices)
    print(f"\n  Total: {olivia_detail['total_buys']} buys, {olivia_detail['total_sells']} sells")
    for prod in ["CROISSANTS", "KELP", "SQUID_INK"]:
        if prod in olivia_detail:
            info = olivia_detail[prod]
            print(f"\n  {prod}:")
            for b in info["buys"]:
                print(f"    BUY  day={b['day']} ts={b['timestamp']:6d} price={b['price']:.1f} qty={b['quantity']} "
                      f"day_range=[{b['day_low']:.1f}, {b['day_high']:.1f}] pct_of_range={b.get('pct_of_range', 'N/A')}")
            for s in info["sells"]:
                print(f"    SELL day={s['day']} ts={s['timestamp']:6d} price={s['price']:.1f} qty={s['quantity']} "
                      f"day_range=[{s['day_low']:.1f}, {s['day_high']:.1f}] pct_of_range={s.get('pct_of_range', 'N/A')}")

    # ─── 6. Actionable Patterns ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("6. ACTIONABLE EXPLOITATION STRATEGIES")
    print("=" * 80)

    # Generate data-driven recommendations from fingerprints and IR
    ir_by_bot = {r["bot"]: r for r in ir_results}

    for name in all_bots:
        fp = fingerprints.get(name, {})
        if fp.get("insufficient_data"):
            continue

        ir_data = ir_by_bot.get(name, {})
        direction = fp.get("direction", {})

        # Find strongest directional biases
        strong_buys = [(p, d) for p, d in direction.items() if d["bias"] == "buyer" and abs(d["net_vol"]) > 500]
        strong_sells = [(p, d) for p, d in direction.items() if d["bias"] == "seller" and abs(d["net_vol"]) > 500]

        print(f"\n--- {name} ---")

        # Archetype
        timing = fp.get("timing", {})
        momentum = fp.get("momentum_label", "neutral")
        price_level = fp.get("price_level", {})

        archetype_parts = []
        if name == "Olivia":
            archetype_parts.append("INFORMED TRADER (daily extreme)")
        elif timing.get("pattern") == "periodic":
            archetype_parts.append("periodic trader")
        if momentum == "mean_reversion":
            archetype_parts.append("mean-reversion")
        elif momentum == "momentum":
            archetype_parts.append("momentum-follower")
        if price_level.get("buy_aggressive_frac", 0) > 0.8 and price_level.get("sell_aggressive_frac", 0) > 0.8:
            archetype_parts.append("aggressive taker")
        elif price_level.get("buy_aggressive_frac", 0) < 0.2:
            archetype_parts.append("passive maker")

        if archetype_parts:
            print(f"  Archetype: {', '.join(archetype_parts)}")

        # IR
        ir50 = ir_data.get("ir_50")
        wr50 = ir_data.get("win_rate_50")
        pnl50 = ir_data.get("total_pnl_50")
        if ir50 is not None:
            quality = "PROFITABLE" if ir50 > 0 else "LOSING"
            print(f"  IR@50 ticks: {ir50:.4f} ({quality}), win_rate={wr50:.1%}, total_pnl={pnl50:.0f}")

        # Directional biases
        if strong_buys:
            prods = ", ".join(f"{p} (+{d['net_vol']})" for p, d in sorted(strong_buys, key=lambda x: -x[1]["net_vol"]))
            print(f"  NET BUYER: {prods}")
        if strong_sells:
            prods = ", ".join(f"{p} ({d['net_vol']})" for p, d in sorted(strong_sells, key=lambda x: x[1]["net_vol"]))
            print(f"  NET SELLER: {prods}")

        # Timing
        if timing.get("mean_interval"):
            print(f"  Timing: {timing['pattern']}, mean_interval={timing['mean_interval']:.0f}ts, "
                  f"median={timing.get('median_interval', 'N/A')}ts")
        if timing.get("mod_patterns"):
            for n, frac in timing["mod_patterns"].items():
                print(f"  ALL trades at ts % {n} == 0 ({frac:.1%})")

        # Exploitation strategy
        print(f"  EXPLOIT:")
        if name == "Olivia":
            print(f"    - Copy-trade: When Olivia buys → daily low is in (go long). When sells → daily high is in (go short).")
            print(f"    - Her buys are at pct_of_range ~1-16%, sells at ~80-99% - almost always within 2% of the true extreme.")
            print(f"    - Fixed sizes: 3 CROISSANTS, 13-15 KELP, 14-15 SQUID_INK - use size to identify her trades.")
        elif ir50 is not None and ir50 < -1:
            print(f"    - FADE this bot: {name} loses {abs(ir50):.2f} per trade at 50-tick horizon.")
            print(f"    - When {name} buys, lean short. When {name} sells, lean long.")
            if strong_sells:
                for p, d in strong_sells:
                    print(f"    - {p}: {name} is a consistent seller - place bids to catch their sells.")
        elif ir50 is not None and ir50 > 1:
            print(f"    - FOLLOW this bot: {name} makes {ir50:.2f} per trade.")
            print(f"    - Copy {name}'s direction - they have real information.")
        elif strong_buys and not strong_sells:
            print(f"    - Reliable demand-side liquidity. Sell to {name} on their strong-buy products.")
        elif strong_sells and not strong_buys:
            print(f"    - Reliable supply. Buy from {name} when they sell aggressively.")
        else:
            print(f"    - Symmetric/balanced. Front-run by quoting tighter spreads.")

    # ─── RECOMMENDATIONS ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR P4")
    print("=" * 80)

    # Rank bots by exploitability: high IR magnitude + directional bias + volume
    print("\n## Bots Ranked by Exploitability\n")
    ranked = []
    for r in ir_results:
        fp = fingerprints.get(r["bot"], {})
        direction = fp.get("direction", {})
        max_bias = max(
            (abs(d["net_vol"]) for d in direction.values()),
            default=0,
        )
        # Score: abs(IR) * log(trades) * (1 + directional_bias/10000)
        score = abs(r["ir_50"] or 0) * np.log1p(r["total_trades"]) * (1 + max_bias / 10000)
        ranked.append((r["bot"], score, r))

    ranked.sort(key=lambda x: -x[1])
    for rank, (name, score, r) in enumerate(ranked, 1):
        ir50 = r["ir_50"]
        action = "FOLLOW" if (ir50 or 0) > 0.5 else "FADE" if (ir50 or 0) < -0.5 else "SPREAD"
        print(f"  {rank}. {name:12s} score={score:8.1f}  IR@50={ir50 or 0:+.4f}  action={action}  trades={r['total_trades']}")

    print("""
## Signals to Add to Strategy Code

1. OliviaTracker (exists) - copy-trade at daily extremes, fixed sizes
2. LosingBotFader - when Gary/Pablo/Penelope/Paris trade, lean opposite
   - Gary:     IR@50=-2.55, loses on every product, 100% aggressive taker
   - Pablo:    IR@50=-2.31, systematic seller, momentum-follower (sells into drops)
   - Penelope: IR@50=-2.22, aggressive taker, near-symmetric but still loses
   - Paris:    IR@50=-1.43, universal MM, tiny sizes, consistently negative IR
   → Combined these 4 account for 39K trades. Fading them is high-frequency alpha.
3. CharlieFollower - Charlie has IR@50=+2.62, wins 76.6% of trades
   - Mean-reversion strategy (momentum_score=-0.82)
   - Strongest on RESIN (+100K PnL, 94.8% win rate)
   → When Charlie buys, join the bid. When Charlie sells, join the ask.
4. CaesarCamillaFlow - track the Caesar(buy food)/Camilla(sell food) dynamic
   - Caesar buys JAMS (+7841 net vol), CROISSANTS (+5260)
   - Camilla sells JAMS (-7495 net vol), CROISSANTS (-4620)
   → These two are matched counterparties. Net flow direction signals fair value.

## Key Quantitative Findings

- ALL bots trade at ts % 100 == 0 (100% of trades). Game engine tick = 100.
- Charlie is the BEST bot: +45K PnL, +2.62 IR, mean-reversion, 76.6% win rate
- Gary is the WORST bot:  -3.9K PnL, -2.55 IR, aggressive taker, 23.1% win rate
- Olivia is rare (20 trades/3 days) but has 80% win rate and best per-trade IR (+16.4)
- Caesar self-trades on VOLCANIC_ROCK (1173 trades) - he fills his own orders
- Voucher market is dominated by Camilla(buy) vs Caesar(sell) - they are the book
- Food market is dominated by Caesar(buy) vs Camilla/Paris(sell)
- KELP/SQUID_INK primary pairs: Charlie <-> Paris (makes + passive MM)

## Names May Change in P4 - Match by Behavior
- "Olivia-like":  <20 trades/day, daily extremes, fixed size per product
- "Charlie-like": Mean-reversion MM, balanced, size 3-4, high win-rate
- "Caesar-like":  Trades everything, huge volume, directional food vs vouchers
- "Paris-like":   Universal presence, tiny sizes (1-2), tight spreads, LOSES money
- "Gary-like":    3-product specialist, large sizes (7-9), aggressive taker, LOSES badly
- "Gina-like":    3-product, systematic net seller, momentum-follower
""")
    print(f"\nTotal bot trades analyzed: {len(trades)}")


if __name__ == "__main__":
    main()
