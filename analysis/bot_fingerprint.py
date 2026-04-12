"""Bot behavior pattern detection from trade logs.

Identifies bot archetypes: Olivia (informed, size=15 at daily extremes),
market makers (symmetric, high-frequency), momentum bots, etc.

Usage:
    uv run analysis/bot_fingerprint.py data/tutorial/
    uv run analysis/bot_fingerprint.py data/round1/ --json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.data_loader import load_prices, load_trades, get_product_mid, get_product_trades


@dataclass
class BotProfile:
    """Fingerprint of a detected bot archetype."""
    product: str
    bot_label: str           # "olivia-like", "market-maker", "momentum", "unknown"
    buyer_name: str = ""     # if buyer/seller IDs visible
    seller_name: str = ""
    trade_count: int = 0
    typical_size: float = 0.0
    size_std: float = 0.0
    buys_at_extreme: bool = False    # buys near daily low
    sells_at_extreme: bool = False   # sells near daily high
    timing_pattern: str = ""  # "periodic", "clustered", "random"
    mean_interval: float = 0.0
    confidence: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class TradeFlowStats:
    """Aggregate trade flow statistics for a product."""
    product: str
    total_trades: int = 0
    total_volume: int = 0
    buy_volume: int = 0       # trades at or above mid
    sell_volume: int = 0      # trades below mid
    vwap: float = 0.0
    ofi: float = 0.0         # order flow imbalance (buy-sell)/total
    size_distribution: dict = field(default_factory=dict)  # size → count
    hourly_pattern: list[int] = field(default_factory=list)  # trades per 100k-tick block


def detect_olivia(
    tdf: pd.DataFrame,
    pdf: pd.DataFrame,
    product: str,
) -> BotProfile | None:
    """Detect Olivia-like informed bot: buys at daily low, sells at daily high.

    Works with or without buyer/seller IDs.
    """
    if tdf.empty or pdf.empty:
        return None

    profile = BotProfile(product=product, bot_label="olivia-like")

    # If buyer/seller IDs exist, look for specific names
    has_ids = tdf["buyer"].notna().any() and (tdf["buyer"] != "").any()

    if has_ids:
        # Direct ID check
        for name in ["Olivia", "olivia"]:
            buys = tdf[tdf["buyer"] == name]
            sells = tdf[tdf["seller"] == name]
            if len(buys) > 0 or len(sells) > 0:
                profile.buyer_name = name
                profile.seller_name = name
                profile.trade_count = len(buys) + len(sells)
                if len(buys) > 0:
                    profile.typical_size = float(buys["quantity"].median())
                profile.buys_at_extreme = True
                profile.sells_at_extreme = True
                profile.confidence = "high"
                profile.notes.append(f"Olivia detected by ID: {len(buys)} buys, {len(sells)} sells")
                return profile

    # No IDs - detect by behavior pattern:
    # Olivia buys at daily low, sells at daily high, with a consistent lot size.
    # Skip stable products (tiny range means everything is "near" the extreme).
    mid_values = pdf["mid_price"].values
    overall_cv = float(np.std(mid_values) / np.mean(mid_values)) if np.mean(mid_values) > 0 else 0
    if overall_cv < 0.001:
        return None  # stable product - no meaningful extremes

    size_extreme_counts: dict[int, dict[str, int]] = {}  # size → {"low": n, "high": n}

    for day in pdf["day"].unique():
        day_prices = pdf[pdf["day"] == day]["mid_price"].values
        if len(day_prices) < 100:
            continue

        day_low = float(np.min(day_prices))
        day_high = float(np.max(day_prices))
        price_range = day_high - day_low
        if price_range < 5:
            continue

        day_trades = tdf[tdf["day"] == day] if "day" in tdf.columns else tdf

        # Bottom/top 10% of daily range
        low_thresh = day_low + price_range * 0.10
        high_thresh = day_high - price_range * 0.10

        near_low = day_trades[day_trades["price"] <= low_thresh]
        near_high = day_trades[day_trades["price"] >= high_thresh]

        for _, t in near_low.iterrows():
            sz = int(t["quantity"])
            size_extreme_counts.setdefault(sz, {"low": 0, "high": 0})
            size_extreme_counts[sz]["low"] += 1
        for _, t in near_high.iterrows():
            sz = int(t["quantity"])
            size_extreme_counts.setdefault(sz, {"low": 0, "high": 0})
            size_extreme_counts[sz]["high"] += 1

    # Find a size that consistently appears at BOTH extremes across days
    total_trades = len(tdf)
    for size, counts in sorted(size_extreme_counts.items(), key=lambda x: -(x[1]["low"] + x[1]["high"])):
        low_n = counts["low"]
        high_n = counts["high"]
        extreme_total = low_n + high_n

        # Require: appears at both extremes, and extreme trades are disproportionate
        # vs total trades of that size
        size_total = int((tdf["quantity"] == size).sum())
        extreme_ratio = extreme_total / size_total if size_total > 0 else 0

        if low_n >= 3 and high_n >= 3 and extreme_ratio > 0.4:
            profile.typical_size = float(size)
            profile.trade_count = extreme_total
            profile.buys_at_extreme = True
            profile.sells_at_extreme = True
            profile.confidence = "medium"
            profile.notes.append(
                f"size={size}: {low_n}x near daily low, {high_n}x near daily high "
                f"({extreme_ratio:.0%} of all size-{size} trades)"
            )
            break  # report only the strongest signal

    if profile.trade_count > 0:
        return profile
    return None


def analyze_size_distribution(tdf: pd.DataFrame) -> dict[int, int]:
    """Get trade size frequency distribution."""
    if tdf.empty:
        return {}
    return dict(tdf["quantity"].value_counts().sort_index().items())


def analyze_timing(tdf: pd.DataFrame) -> tuple[str, float]:
    """Classify timing pattern: periodic, clustered, or random.

    Returns (pattern_label, mean_interval).
    """
    if len(tdf) < 5:
        return "insufficient", 0.0

    timestamps = tdf["timestamp"].sort_values().values
    intervals = np.diff(timestamps)

    if len(intervals) < 3:
        return "insufficient", float(np.mean(intervals))

    mean_int = float(np.mean(intervals))
    cv = float(np.std(intervals) / mean_int) if mean_int > 0 else 0

    # CV < 0.3 → periodic, CV > 1.5 → clustered, else random
    if cv < 0.3:
        return "periodic", mean_int
    elif cv > 1.5:
        return "clustered", mean_int
    else:
        return "random", mean_int


def compute_trade_flow(
    tdf: pd.DataFrame,
    pdf: pd.DataFrame,
    product: str,
) -> TradeFlowStats:
    """Compute aggregate trade flow statistics."""
    stats = TradeFlowStats(product=product)

    ptdf = get_product_trades(tdf, product) if "symbol" in tdf.columns else tdf
    ppdf = get_product_mid(pdf, product) if "product" in pdf.columns else pdf

    if ptdf.empty:
        return stats

    stats.total_trades = len(ptdf)
    stats.total_volume = int(ptdf["quantity"].sum())
    stats.vwap = float((ptdf["price"] * ptdf["quantity"]).sum() / stats.total_volume) if stats.total_volume > 0 else 0

    # Classify buys vs sells using mid_price at nearest timestamp
    if not ppdf.empty:
        # Merge trade prices with nearest mid
        for _, trade in ptdf.iterrows():
            ts = trade["timestamp"]
            nearest = ppdf.iloc[(ppdf["timestamp"] - ts).abs().argsort()[:1]]
            if not nearest.empty:
                mid = nearest["mid_price"].values[0]
                vol = int(trade["quantity"])
                if trade["price"] >= mid:
                    stats.buy_volume += vol
                else:
                    stats.sell_volume += vol

    if stats.total_volume > 0:
        stats.ofi = (stats.buy_volume - stats.sell_volume) / stats.total_volume

    stats.size_distribution = analyze_size_distribution(ptdf)

    # Hourly pattern (blocks of 100k timestamps = ~1 hour in game time)
    if "timestamp" in ptdf.columns:
        blocks = (ptdf["timestamp"] // 100_000).astype(int)
        block_counts = blocks.value_counts().sort_index()
        stats.hourly_pattern = [int(block_counts.get(i, 0)) for i in range(10)]

    return stats


def run_fingerprint(round_dir: Path | str) -> tuple[list[BotProfile], list[TradeFlowStats]]:
    """Run full bot fingerprinting on a round directory."""
    round_dir = Path(round_dir)
    prices = load_prices(round_dir)
    try:
        trades = load_trades(round_dir)
    except FileNotFoundError:
        return [], []

    products = sorted(prices["product"].unique())

    bots: list[BotProfile] = []
    flows: list[TradeFlowStats] = []

    for product in products:
        pdf = get_product_mid(prices, product)
        tdf = get_product_trades(trades, product)

        # Olivia detection
        olivia = detect_olivia(tdf, pdf, product)
        if olivia:
            bots.append(olivia)

        # Trade flow
        flow = compute_trade_flow(trades, prices, product)
        flows.append(flow)

    return bots, flows


def format_report(bots: list[BotProfile], flows: list[TradeFlowStats]) -> str:
    """Format fingerprinting results as readable report."""
    lines = ["# Bot Fingerprint Report", ""]

    # Olivia detection
    lines.append("## Informed Bot Detection (Olivia-like)")
    lines.append("")
    if bots:
        for b in bots:
            lines.append(f"### {b.product} - {b.bot_label} ({b.confidence})")
            lines.append(f"- Trades: {b.trade_count}, typical size: {b.typical_size}")
            lines.append(f"- Buys at low: {b.buys_at_extreme}, sells at high: {b.sells_at_extreme}")
            if b.buyer_name:
                lines.append(f"- ID: {b.buyer_name}")
            for note in b.notes:
                lines.append(f"- {note}")
            lines.append("")
    else:
        lines.append("No Olivia-like bots detected.")
        lines.append("")

    # Trade flow
    lines.append("## Trade Flow Statistics")
    lines.append("")
    lines.append("| Product | Trades | Volume | Buy Vol | Sell Vol | OFI | VWAP | Top Sizes |")
    lines.append("|---------|--------|--------|---------|----------|-----|------|-----------|")
    for f in flows:
        top_sizes = sorted(f.size_distribution.items(), key=lambda x: -x[1])[:3]
        sizes_str = ", ".join(f"{s}×{c}" for s, c in top_sizes)
        lines.append(
            f"| {f.product} | {f.total_trades} | {f.total_volume} | "
            f"{f.buy_volume} | {f.sell_volume} | {f.ofi:+.3f} | "
            f"{f.vwap:.1f} | {sizes_str} |"
        )
    lines.append("")

    # Timing analysis per product
    lines.append("## Trade Timing Patterns")
    lines.append("")
    lines.append("| Product | Hourly Distribution (per 100k ticks) |")
    lines.append("|---------|---------------------------------------|")
    for f in flows:
        dist = " ".join(str(x) for x in f.hourly_pattern) if f.hourly_pattern else "N/A"
        lines.append(f"| {f.product} | {dist} |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bot behavior fingerprinting")
    parser.add_argument("round_dir", type=Path, help="Path to round data directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    bots, flows = run_fingerprint(args.round_dir)

    if args.json:
        print(json.dumps({
            "bots": [asdict(b) for b in bots],
            "flows": [asdict(f) for f in flows],
        }, indent=2))
    else:
        print(format_report(bots, flows))


if __name__ == "__main__":
    main()
