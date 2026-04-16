"""Test skeleton strategies against P3 CSV data.

Constructs real OrderDepth objects from CSV rows and feeds them through
actual strategy classes tick-by-tick. Reports PnL, errors, and traderData usage.

Usage:
    uv run scripts/skeleton_test.py basket data/p3/round2/
    uv run scripts/skeleton_test.py options data/p3/round3/
    uv run scripts/skeleton_test.py conversion data/p3/round4/
    uv run scripts/skeleton_test.py drifting data/p3/round1/ --product KELP
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datamodel import Order, OrderDepth, ConversionObservation, Trade
from analysis.data_loader import load_prices, load_trades


def construct_order_depth(row: pd.Series) -> OrderDepth:
    """Build an OrderDepth from a CSV row with 3 levels per side."""
    od = OrderDepth()
    for i in range(1, 4):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and bv > 0:
            od.buy_orders[int(bp)] = int(bv)
        if pd.notna(ap) and pd.notna(av) and av > 0:
            od.sell_orders[int(ap)] = -int(av)  # sell_orders are negative
    return od


def construct_trades(tdf: pd.DataFrame, ts: int, product: str) -> list[Trade]:
    """Build Trade objects for a given timestamp and product."""
    mask = (tdf["timestamp"] == ts) & (tdf["symbol"] == product)
    trades = []
    for _, r in tdf[mask].iterrows():
        trades.append(Trade(
            symbol=product,
            price=int(r["price"]),
            quantity=int(r["quantity"]),
            buyer=str(r.get("buyer", "")),
            seller=str(r.get("seller", "")),
            timestamp=ts,
        ))
    return trades


def simple_pnl(orders: list[Order], od: OrderDepth, pos: int, limit: int) -> tuple[float, int]:
    """Optimistic fill model. Returns (cash_delta, new_pos)."""
    cash = 0.0
    best_bid = max(od.buy_orders) if od.buy_orders else None
    best_ask = min(od.sell_orders) if od.sell_orders else None

    for order in orders:
        qty = order.quantity
        # Enforce position limits
        if qty > 0:
            qty = min(qty, limit - pos)
        elif qty < 0:
            qty = max(qty, -(limit + pos))
        if qty == 0:
            continue

        # Aggressive fill (crossing spread)
        if qty > 0 and best_ask is not None and order.price >= best_ask:
            fill_qty = min(qty, abs(od.sell_orders.get(best_ask, 0)))
            if fill_qty > 0:
                cash -= fill_qty * best_ask
                pos += fill_qty
        elif qty < 0 and best_bid is not None and order.price <= best_bid:
            fill_qty = max(qty, -od.buy_orders.get(best_bid, 0))
            if fill_qty < 0:
                cash -= fill_qty * best_bid
                pos += fill_qty
        else:
            # Passive fill - 20% chance, at posted price
            if np.random.random() < 0.20:
                fill_qty = max(1, abs(qty) // 4)
                if qty > 0:
                    fill_qty = min(fill_qty, limit - pos)
                    cash -= fill_qty * order.price
                    pos += fill_qty
                else:
                    fill_qty = min(fill_qty, limit + pos)
                    cash += fill_qty * order.price
                    pos -= fill_qty

    return cash, pos


def get_ticks(prices: pd.DataFrame, day: int) -> list[tuple[int, dict[str, pd.Series]]]:
    """Group price data by timestamp, return list of (timestamp, {product: row})."""
    day_data = prices[prices["day"] == day].sort_values("timestamp")
    ticks = []
    for ts, group in day_data.groupby("timestamp"):
        product_rows = {}
        for _, row in group.iterrows():
            product_rows[row["product"]] = row
        ticks.append((int(ts), product_rows))
    return ticks


# ─── Strategy-specific test runners ────────────────────────────────

def test_basket(round_dir: Path) -> None:
    """Test BasketTrader on PICNIC_BASKET1."""
    from strategies.basket import BasketTrader
    from analysis.eda import check_basket_relationships

    prices = load_prices(round_dir)
    products = sorted(prices["product"].unique())
    days = sorted(prices["day"].unique())

    # Find basket constituents via OLS
    basket_hits = check_basket_relationships(prices, products)
    basket_product = None
    for p, hit in basket_hits.items():
        if "BASKET" in p.upper() or "PICNIC" in p.upper():
            basket_product = p
            break

    if not basket_product:
        # Try any product with R² > 0.95
        for p, hit in basket_hits.items():
            if hit["r_squared"] > 0.95:
                basket_product = p
                break

    if not basket_product:
        print("No basket product found in this round")
        return

    hit = basket_hits[basket_product]
    constituents = dict(zip(hit["constituents"], hit["weights"]))
    print(f"Basket: {basket_product}")
    print(f"Constituents: {constituents}")
    print(f"R²: {hit['r_squared']:.6f}, intercept: {hit['intercept']:.2f}")
    print()

    trader = BasketTrader(
        basket_product=basket_product,
        constituents=constituents,
        basket_limit=60,
        entry_thr=50.0,  # start conservative, tune later
        n_prior=1000,
        seed_mean=hit["intercept"],
    )

    # Test on each day
    for day in days:
        ticks = get_ticks(prices, day)
        td: dict = {}
        pos: dict[str, int] = {}
        cash = 0.0
        n_trades = 0
        errors = 0

        for ts, product_rows in ticks:
            order_depths: dict[str, OrderDepth] = {}
            for p, row in product_rows.items():
                order_depths[p] = construct_order_depth(row)

            try:
                result, td = trader.run(order_depths, pos, td)
                for p, orders in result.items():
                    if orders:
                        od = order_depths.get(p)
                        if od:
                            limit = 60 if p == basket_product else 300
                            c, new_pos = simple_pnl(orders, od, pos.get(p, 0), limit)
                            cash += c
                            pos[p] = new_pos
                            n_trades += len(orders)
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  ERROR at ts={ts}: {e}")

        # Mark to market
        mtm = 0.0
        last_rows = ticks[-1][1] if ticks else {}
        for p, position in pos.items():
            if p in last_rows:
                mid = last_rows[p].get("mid_price", 0)
                mtm += position * mid

        total = cash + mtm
        td_size = len(json.dumps(td, separators=(",", ":")))
        print(f"Day {day}: PnL={total:>12,.0f}  trades={n_trades:>5}  errors={errors}  "
              f"positions={dict((k,v) for k,v in pos.items() if v != 0)}  td={td_size} chars")

    # Tune entry_thr
    print("\n--- Entry threshold sweep ---")
    for thr in [30, 50, 80, 100, 150, 200]:
        trader_t = BasketTrader(
            basket_product=basket_product,
            constituents=constituents,
            basket_limit=60,
            entry_thr=thr,
            n_prior=1000,
            seed_mean=hit["intercept"],
        )
        day = days[-1]  # validation day
        ticks = get_ticks(prices, day)
        td_t: dict = {}
        pos_t: dict[str, int] = {}
        cash_t = 0.0

        for ts, product_rows in ticks:
            order_depths = {p: construct_order_depth(r) for p, r in product_rows.items()}
            try:
                result, td_t = trader_t.run(order_depths, pos_t, td_t)
                for p, orders in result.items():
                    od = order_depths.get(p)
                    if od and orders:
                        limit = 60 if p == basket_product else 300
                        c, new_pos = simple_pnl(orders, od, pos_t.get(p, 0), limit)
                        cash_t += c
                        pos_t[p] = new_pos
            except Exception:
                pass

        mtm_t = sum(pos_t.get(p, 0) * ticks[-1][1].get(p, pd.Series({"mid_price": 0})).get("mid_price", 0) for p in pos_t)
        print(f"  entry_thr={thr:>3}: PnL={cash_t + mtm_t:>12,.0f}")


def test_options(round_dir: Path) -> None:
    """Test OptionsTrader on VOLCANIC_ROCK_VOUCHER_*."""
    from strategies.options import OptionsTrader

    prices = load_prices(round_dir)
    products = sorted(prices["product"].unique())
    days = sorted(prices["day"].unique())

    # Find options products
    underlying = None
    option_products: dict[int, str] = {}
    for p in products:
        if "VOUCHER" in p.upper():
            # Extract strike from name (e.g., VOLCANIC_ROCK_VOUCHER_10000 → 10000)
            parts = p.split("_")
            try:
                strike = int(parts[-1])
                option_products[strike] = p
            except ValueError:
                pass
        elif "VOLCANIC" in p.upper() and "VOUCHER" not in p.upper():
            underlying = p

    if not underlying or len(option_products) < 2:
        print(f"Insufficient options data: underlying={underlying}, options={option_products}")
        return

    print(f"Underlying: {underlying}")
    print(f"Options: {option_products}")
    print()

    trader = OptionsTrader(
        underlying=underlying,
        option_products=option_products,
        underlying_limit=80,
        option_limit=200,
        tte=None,  # auto-detect
        open_thr=0.5,
        refit_every=500,
        min_vega=0.7,
    )

    for day in days:
        ticks = get_ticks(prices, day)
        td: dict = {}
        pos: dict[str, int] = {}
        cash = 0.0
        n_trades = 0
        errors = 0
        tte_detected = None
        smile_fitted = False

        for ts, product_rows in ticks:
            order_depths = {p: construct_order_depth(r) for p, r in product_rows.items()}

            try:
                result, td = trader.run(order_depths, pos, td, ts)
                for p, orders in result.items():
                    if orders:
                        od = order_depths.get(p)
                        if od:
                            limit = 80 if p == underlying else 200
                            c, new_pos = simple_pnl(orders, od, pos.get(p, 0), limit)
                            cash += c
                            pos[p] = new_pos
                            n_trades += len(orders)

                if td.get("tte") and tte_detected is None:
                    tte_detected = td["tte"]
                if td.get("smile"):
                    smile_fitted = True
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  ERROR at ts={ts}: {e}")

        # Mark to market
        mtm = 0.0
        last_rows = ticks[-1][1] if ticks else {}
        for p, position in pos.items():
            if p in last_rows:
                mtm += position * last_rows[p].get("mid_price", 0)

        total = cash + mtm
        td_size = len(json.dumps(td, separators=(",", ":")))

        print(f"Day {day}: PnL={total:>12,.0f}  trades={n_trades:>5}  errors={errors}  td={td_size} chars")
        if tte_detected:
            print(f"  TTE detected: {tte_detected:.6f} years = {tte_detected*365:.1f} days")
        if smile_fitted:
            smile = td.get("smile", {})
            print(f"  Smile: a={smile.get('a', '?')}, b={smile.get('b', '?')}, c={smile.get('c', '?')}")
        opt_pos = {k: v for k, v in pos.items() if v != 0}
        if opt_pos:
            print(f"  Positions: {opt_pos}")


def test_conversion(round_dir: Path) -> None:
    """Test ConversionTrader on MAGNIFICENT_MACARONS."""
    from strategies.conversion import ConversionTrader

    prices = load_prices(round_dir)
    products = sorted(prices["product"].unique())
    days = sorted(prices["day"].unique())

    # Find conversion product
    conv_product = None
    for p in products:
        if "MACARON" in p.upper() or "ORCHID" in p.upper():
            conv_product = p
            break

    if not conv_product:
        print("No conversion product found")
        return

    # Load observations
    obs_files = sorted(round_dir.glob("observations_round_*_day_*.csv"))
    if not obs_files:
        print("No observations CSV found - cannot test conversion")
        return

    obs_dfs = []
    for f in obs_files:
        parts = f.stem.split("_")
        day_idx = parts.index("day") + 1
        day = int(parts[day_idx])
        odf = pd.read_csv(f)  # comma-separated
        odf["day"] = day
        obs_dfs.append(odf)
    obs_df = pd.concat(obs_dfs, ignore_index=True)

    print(f"Conversion product: {conv_product}")
    print(f"Observations: {len(obs_df)} rows, columns: {list(obs_df.columns)}")
    print()

    trader = ConversionTrader(
        product=conv_product,
        limit=75,
        conv_limit=10,
        min_edge=0.5,
    )

    for day in days:
        ticks = get_ticks(prices, day)
        day_obs = obs_df[obs_df["day"] == day]
        pos = 0
        cash = 0.0
        n_trades = 0
        n_conversions = 0
        errors = 0
        edges: list[float] = []

        for ts, product_rows in ticks:
            if conv_product not in product_rows:
                continue

            od = construct_order_depth(product_rows[conv_product])

            # Find matching observation
            obs_row = day_obs[day_obs["timestamp"] == ts]
            if obs_row.empty:
                continue

            r = obs_row.iloc[0]
            obs = ConversionObservation(
                bidPrice=float(r["bidPrice"]),
                askPrice=float(r["askPrice"]),
                transportFees=float(r["transportFees"]),
                exportTariff=float(r["exportTariff"]),
                importTariff=float(r["importTariff"]),
                sugarPrice=float(r.get("sugarPrice", 0)),
                sunlightIndex=float(r.get("sunlightIndex", 0)),
            )

            try:
                orders, conversions = trader.run(od, obs, pos)
                if orders:
                    c, pos = simple_pnl(orders, od, pos, 75)
                    cash += c
                    n_trades += len(orders)

                # Apply conversions (import at cost)
                if conversions > 0:
                    import_cost = obs.askPrice + obs.importTariff + obs.transportFees
                    cash -= conversions * import_cost
                    pos += conversions
                    n_conversions += conversions

                # Track edge
                import_cost = obs.askPrice + obs.importTariff + obs.transportFees
                if od.sell_orders:
                    local_ask = min(od.sell_orders)
                    edges.append(local_ask - import_cost)

            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  ERROR at ts={ts}: {e}")

        # Mark to market
        last_mid = ticks[-1][1].get(conv_product, pd.Series({"mid_price": 0})).get("mid_price", 0) if ticks else 0
        mtm = pos * last_mid
        total = cash + mtm

        mean_edge = np.mean(edges) if edges else 0
        pos_edge = sum(1 for e in edges if e > 0) / len(edges) * 100 if edges else 0
        print(f"Day {day}: PnL={total:>10,.0f}  sells={n_trades}  conversions={n_conversions}  "
              f"errors={errors}  final_pos={pos}  mean_edge={mean_edge:.2f}  edge>0={pos_edge:.0f}%")


def test_drifting(round_dir: Path, product: str = "KELP") -> None:
    """Test DriftingTrader on a drifting product with both modes."""
    from strategies.drifting import DriftingTrader

    prices = load_prices(round_dir)
    days = sorted(prices[prices["product"] == product]["day"].unique())

    if not days:
        print(f"No data for {product}")
        return

    try:
        trades = load_trades(round_dir)
    except FileNotFoundError:
        trades = pd.DataFrame(columns=["day", "timestamp", "buyer", "seller", "symbol", "price", "quantity"])

    print(f"Product: {product}, days: {list(days)}")
    print()

    configs = [
        ("FV spread=2", {"product": product, "limit": 50, "make_spread": 2, "ar_beta": -0.229, "use_olivia": True}),
        ("FV spread=3", {"product": product, "limit": 50, "make_spread": 3, "ar_beta": -0.229, "use_olivia": True}),
        ("FV + follow", {"product": product, "limit": 50, "make_spread": 2, "ar_beta": -0.229, "use_olivia": True, "olivia_mode": "follow"}),
    ]

    for label, kwargs in configs:
        trader = DriftingTrader(**kwargs)
        day = days[-1]  # validation day
        ticks = get_ticks(prices, day)
        td: dict = {}
        pos = 0
        cash = 0.0
        n_trades = 0
        errors = 0

        for ts, product_rows in ticks:
            if product not in product_rows:
                continue
            od = construct_order_depth(product_rows[product])
            market_trades = construct_trades(trades, ts, product)

            try:
                orders, td = trader.run(od, pos, td, market_trades, ts)
                if orders:
                    c, pos = simple_pnl(orders, od, pos, 50)
                    cash += c
                    n_trades += len(orders)
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  ERROR at ts={ts}: {e}")

        last_rows = ticks[-1][1] if ticks else {}
        mid = last_rows.get(product, pd.Series({"mid_price": 0})).get("mid_price", 0) if ticks else 0
        total = cash + pos * mid
        td_size = len(json.dumps(td, separators=(",", ":")))
        print(f"  {label:20s}: PnL={total:>12,.0f}  trades={n_trades:>5}  errors={errors}  td={td_size} chars")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test skeleton strategies on P3 data")
    parser.add_argument("archetype", choices=["basket", "options", "conversion", "drifting", "all"])
    parser.add_argument("round_dir", type=Path, help="Path to P3 round data")
    parser.add_argument("--product", type=str, default="KELP", help="Product for drifting test")
    args = parser.parse_args()

    if args.archetype == "basket" or args.archetype == "all":
        print("=" * 60)
        print("BASKET TRADER TEST")
        print("=" * 60)
        test_basket(args.round_dir)
        print()

    if args.archetype == "options" or args.archetype == "all":
        print("=" * 60)
        print("OPTIONS TRADER TEST")
        print("=" * 60)
        test_options(args.round_dir)
        print()

    if args.archetype == "conversion" or args.archetype == "all":
        print("=" * 60)
        print("CONVERSION TRADER TEST")
        print("=" * 60)
        test_conversion(args.round_dir)
        print()

    if args.archetype == "drifting" or args.archetype == "all":
        print("=" * 60)
        print("DRIFTING TRADER TEST")
        print("=" * 60)
        test_drifting(args.round_dir, args.product)
        print()


if __name__ == "__main__":
    main()
