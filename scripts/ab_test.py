"""A/B test: compare market-follow vs FV-based quoting on any drifting product.

Runs both strategies on the same data, reports PnL side by side.
Use this on Round 1 data to decide which approach to use for new drifting products.

Usage:
    uv run scripts/ab_test.py data/round1/ KELP --limit 50
    uv run scripts/ab_test.py data/tutorial/ TOMATOES --limit 80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.data_loader import load_prices, get_product_mid
from analysis.parameter_search import SimTick, load_sim_ticks, simulate_pnl


def market_follow_strategy(tick: SimTick, pos: int, params: dict, state: dict):
    """Best_bid+1 / best_ask-1 with soft limit tapering."""
    limit = int(params["limit"])
    soft_ratio = 0.6

    bid_price = int(tick.best_bid) + 1
    ask_price = int(tick.best_ask) - 1
    if bid_price >= ask_price:
        return [], state

    rem_buy = limit - pos
    rem_sell = limit + pos

    buy_size = rem_buy
    sell_size = rem_sell

    if pos > limit * soft_ratio and rem_buy > 0:
        excess = (pos - limit * soft_ratio) / (limit * (1 - soft_ratio))
        buy_size = max(1, int(rem_buy * (1 - excess * 0.8)))
    elif pos < -limit * soft_ratio and rem_sell > 0:
        excess = (-pos - limit * soft_ratio) / (limit * (1 - soft_ratio))
        sell_size = max(1, int(rem_sell * (1 - excess * 0.8)))

    orders = []
    if buy_size > 0:
        orders.append((bid_price, buy_size))
    if sell_size > 0:
        orders.append((ask_price, -sell_size))
    return orders, state


def fv_based_strategy(tick: SimTick, pos: int, params: dict, state: dict):
    """Wall_mid FV with TAKE/CLEAR/MAKE."""
    limit = int(params["limit"])
    make_spread = int(params.get("make_spread", 3))
    take_edge = float(params.get("take_edge", 2.0))
    skew_sens = float(params.get("skew_sens", 2.0))

    fv = tick.wall_mid
    orders = []
    rem_buy = limit - pos
    rem_sell = limit + pos

    # TAKE
    if tick.best_ask < fv - take_edge and rem_buy > 0:
        qty = min(rem_buy, int(tick.ask_volume_1))
        orders.append((int(tick.best_ask), qty))
        rem_buy -= qty
        pos += qty
    if tick.best_bid > fv + take_edge and rem_sell > 0:
        qty = min(rem_sell, int(tick.bid_volume_1))
        orders.append((int(tick.best_bid), -qty))
        rem_sell -= qty
        pos -= qty

    # CLEAR
    if abs(pos) >= 0.5 * limit and pos != 0:
        if pos > 0 and rem_sell > 0:
            clear_qty = min(abs(pos), rem_sell)
            orders.append((int(fv) - 1, -clear_qty))
            rem_sell -= clear_qty
        elif pos < 0 and rem_buy > 0:
            clear_qty = min(abs(pos), rem_buy)
            orders.append((int(fv) + 1, clear_qty))
            rem_buy -= clear_qty

    # MAKE
    skew = round(skew_sens * pos / limit)
    bid_p = int(round(fv)) - make_spread - skew
    ask_p = int(round(fv)) + make_spread - skew

    if rem_buy > 0:
        orders.append((bid_p, rem_buy))
    if rem_sell > 0:
        orders.append((ask_p, -rem_sell))

    return orders, state


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B: market-follow vs FV-based")
    parser.add_argument("round_dir", type=Path, help="Path to round data directory")
    parser.add_argument("product", type=str, help="Product to test")
    parser.add_argument("--limit", type=int, default=80, help="Position limit")
    args = parser.parse_args()

    prices = load_prices(args.round_dir)
    days = sorted(prices[prices["product"] == args.product]["day"].unique())

    if not days:
        print(f"No data for {args.product} in {args.round_dir}")
        sys.exit(1)

    print(f"=== A/B Test: {args.product} (limit={args.limit}) ===")
    print(f"Days available: {days}")
    print()

    params_mf = {"limit": args.limit}
    params_fv = {"limit": args.limit, "make_spread": 3, "take_edge": 2.0, "skew_sens": 2.0}

    # Also test FV with different spreads
    fv_spreads = [2, 3, 4]

    for day in days:
        ticks = load_sim_ticks(prices, args.product, day)
        if not ticks:
            continue

        pnl_mf = simulate_pnl(ticks, market_follow_strategy, params_mf, args.limit)

        print(f"Day {day} ({len(ticks)} ticks):")
        print(f"  Market-follow (bid+1/ask-1):  {pnl_mf:>10,.0f}")

        for spread in fv_spreads:
            p = {**params_fv, "make_spread": spread}
            pnl_fv = simulate_pnl(ticks, fv_based_strategy, p, args.limit)
            delta = pnl_fv - pnl_mf
            pct = (delta / abs(pnl_mf) * 100) if pnl_mf != 0 else 0
            marker = " <-- WINNER" if pnl_fv > pnl_mf else ""
            print(f"  FV-based (spread={spread}):       {pnl_fv:>10,.0f}  ({delta:+,.0f}, {pct:+.1f}%){marker}")

    print()
    print("Recommendation: use whichever has higher PnL on day -1 (validation day).")
    print("If market-follow wins, use inline _tomatoes()-style logic.")
    print("If FV-based wins, use DriftingTrader skeleton with calibrated params.")


if __name__ == "__main__":
    main()
