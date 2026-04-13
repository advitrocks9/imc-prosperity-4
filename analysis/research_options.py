"""Options volatility surface research for IMC Prosperity 4.

Investigates:
1. TTE decay shape (linear vs exponential within and between days)
2. Grid search over open_thr, min_vega, trade_qty
3. IV skew analysis across strikes and time
4. Position limit utilization and optimal trade sizing

Usage:
    uv run analysis/research_options.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.data_loader import load_prices
from datamodel import Order, OrderDepth
from strategies._base import bs_call, bs_vega, find_iv, detect_tte

# ── Constants ──────────────────────────────────────────────────────
UNDERLYING = "VOLCANIC_ROCK"
OPTION_PRODUCTS: dict[int, str] = {
    9500: "VOLCANIC_ROCK_VOUCHER_9500",
    9750: "VOLCANIC_ROCK_VOUCHER_9750",
    10000: "VOLCANIC_ROCK_VOUCHER_10000",
    10250: "VOLCANIC_ROCK_VOUCHER_10250",
    10500: "VOLCANIC_ROCK_VOUCHER_10500",
}
ROUND3_DIR = ROOT / "data" / "p3" / "round3"
ROUND5_DIR = ROOT / "data" / "p3" / "round5"
UND_LIMIT = 80
OPT_LIMIT = 200
RELEVANT_PRODUCTS = {UNDERLYING} | set(OPTION_PRODUCTS.values())

SEPARATOR = "=" * 72


# ── Data helpers (pre-build once, reuse many times) ───────────────

def construct_order_depth(row: pd.Series) -> OrderDepth:
    """Build OrderDepth from CSV row."""
    od = OrderDepth()
    for i in range(1, 4):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and bv > 0:
            od.buy_orders[int(bp)] = int(bv)
        if pd.notna(ap) and pd.notna(av) and av > 0:
            od.sell_orders[int(ap)] = -int(av)
    return od


TickData = list[tuple[int, dict[str, OrderDepth], dict[str, float]]]


def prebuild_ticks(prices: pd.DataFrame, day: int) -> TickData:
    """Pre-build OrderDepth objects and mids for a day. Called once per day."""
    day_data = prices[(prices["day"] == day) & prices["product"].isin(RELEVANT_PRODUCTS)]
    day_data = day_data.sort_values("timestamp")

    ticks: TickData = []
    for ts, group in day_data.groupby("timestamp"):
        ods: dict[str, OrderDepth] = {}
        mids: dict[str, float] = {}
        for _, row in group.iterrows():
            p = row["product"]
            ods[p] = construct_order_depth(row)
            mid = row.get("mid_price")
            if pd.notna(mid):
                mids[p] = float(mid)
        ticks.append((int(ts), ods, mids))
    return ticks


def pivot_mids(df: pd.DataFrame) -> pd.DataFrame:
    """Create wide DataFrame: (day, timestamp) x product -> mid_price."""
    pivot = df.pivot_table(
        index=["day", "timestamp"], columns="product",
        values="mid_price", aggfunc="first",
    ).reset_index()
    pivot.sort_values(["day", "timestamp"], inplace=True)
    return pivot


# ── Simple fill model ─────────────────────────────────────────────

def simple_pnl(
    orders: list[Order], od: OrderDepth, pos: int, limit: int,
) -> tuple[float, int]:
    """Deterministic aggressive-only fill model (no passive noise)."""
    cash = 0.0
    best_bid = max(od.buy_orders) if od.buy_orders else None
    best_ask = min(od.sell_orders) if od.sell_orders else None

    for order in orders:
        qty = order.quantity
        if qty > 0:
            qty = min(qty, limit - pos)
        elif qty < 0:
            qty = max(qty, -(limit + pos))
        if qty == 0:
            continue

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

    return cash, pos


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. TTE DECAY SHAPE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyze_tte_decay(df: pd.DataFrame) -> None:
    print(SEPARATOR)
    print("1. TTE DECAY SHAPE ANALYSIS")
    print(SEPARATOR)

    pivot = pivot_mids(df)
    days = sorted(pivot["day"].unique())
    strikes = sorted(OPTION_PRODUCTS.keys())

    all_tv: list[dict] = []

    for day in days:
        day_data = pivot[pivot["day"] == day].copy()
        first_spot = day_data[UNDERLYING].dropna().iloc[0]
        atm_strike = min(strikes, key=lambda k: abs(k - first_spot))
        atm_sym = OPTION_PRODUCTS[atm_strike]
        print(f"\nDay {day}: spot~{first_spot:.0f}, ATM strike={atm_strike}")

        for _, row in day_data.iterrows():
            spot = row.get(UNDERLYING)
            opt_mid = row.get(atm_sym)
            ts = row["timestamp"]
            if pd.isna(spot) or pd.isna(opt_mid):
                continue
            intrinsic = max(0.0, spot - atm_strike)
            time_value = opt_mid - intrinsic
            all_tv.append({
                "day": day, "timestamp": ts, "spot": spot,
                "atm_strike": atm_strike, "opt_mid": opt_mid,
                "intrinsic": intrinsic, "time_value": time_value,
            })

    tv_df = pd.DataFrame(all_tv)
    tv_df["global_time"] = tv_df["day"] * 1_000_000 + tv_df["timestamp"]
    gt = tv_df["global_time"]
    tv_df["global_time_norm"] = (gt - gt.min()) / (gt.max() - gt.min())

    for day in days:
        dd = tv_df[tv_df["day"] == day]
        print(f"  Day {day}: TV start={dd['time_value'].iloc[0]:.2f}, "
              f"end={dd['time_value'].iloc[-1]:.2f}, "
              f"mean={dd['time_value'].mean():.2f}, "
              f"std={dd['time_value'].std():.2f}")

    # Within-day fits
    print("\n--- Within-day decay fits ---")
    for day in days:
        dd = tv_df[tv_df["day"] == day]
        t = dd["timestamp"].values.astype(float)
        tv = dd["time_value"].values
        if len(t) < 100 or tv.mean() < 1.0:
            print(f"  Day {day}: insufficient data or near-zero TV")
            continue
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-9)

        # Linear
        lin_c = np.polyfit(t_norm, tv, 1)
        lin_pred = np.polyval(lin_c, t_norm)
        sst = np.sum((tv - tv.mean()) ** 2)
        lin_r2 = 1 - np.sum((tv - lin_pred) ** 2) / (sst + 1e-9)

        # Exponential
        try:
            def _exp(x, a, b, c):
                return a * np.exp(-b * x) + c
            popt, _ = curve_fit(_exp, t_norm, tv,
                                p0=[tv.max() - tv.min(), 1.0, tv.min()], maxfev=5000)
            exp_pred = _exp(t_norm, *popt)
            exp_r2 = 1 - np.sum((tv - exp_pred) ** 2) / (sst + 1e-9)
            exp_ok = True
        except Exception:
            exp_r2 = -1.0
            exp_ok = False

        better = "EXPONENTIAL" if exp_ok and exp_r2 > lin_r2 else "LINEAR"
        print(f"  Day {day}: Linear R2={lin_r2:.6f} (slope={lin_c[0]:.4f}), "
              f"Exp R2={exp_r2:.6f} -> {better}")

    # Cross-day fit
    print("\n--- Cross-day decay fit ---")
    t_g = tv_df["global_time_norm"].values
    tv_all = tv_df["time_value"].values
    if len(t_g) > 100 and tv_all.mean() > 1.0:
        sst = np.sum((tv_all - tv_all.mean()) ** 2)
        lin_c = np.polyfit(t_g, tv_all, 1)
        lin_r2 = 1 - np.sum((tv_all - np.polyval(lin_c, t_g)) ** 2) / (sst + 1e-9)
        try:
            def _exp_g(x, a, b, c):
                return a * np.exp(-b * x) + c
            popt, _ = curve_fit(_exp_g, t_g, tv_all,
                                p0=[tv_all.max() - tv_all.min(), 2.0, tv_all.min()],
                                maxfev=5000)
            exp_r2 = 1 - np.sum((tv_all - _exp_g(t_g, *popt)) ** 2) / (sst + 1e-9)
            exp_ok = True
        except Exception:
            exp_r2 = -1.0
            exp_ok = False
        better = "EXPONENTIAL" if exp_ok and exp_r2 > lin_r2 else "LINEAR"
        print(f"  Cross-day: Linear R2={lin_r2:.6f}, Exp R2={exp_r2:.6f} -> {better}")

    # TTE detection consistency
    print("\n--- TTE detection per day (sampled every 1000 ticks) ---")
    for day in days:
        dd = tv_df[tv_df["day"] == day]
        samples = dd.iloc[::1000]
        ttes = []
        for _, row in samples.iterrows():
            tte = detect_tte(row["spot"], row["opt_mid"])
            if tte is not None:
                ttes.append(tte * 365)
        if ttes:
            print(f"  Day {day}: TTE(days) mean={np.mean(ttes):.2f}, "
                  f"std={np.std(ttes):.2f}, range=[{min(ttes):.2f}, {max(ttes):.2f}]")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. GRID SEARCH (inline strategy to avoid import overhead)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_options_sim(
    ticks: TickData,
    open_thr: float,
    min_vega: float,
    trade_qty: int,
    refit_every: int = 500,
) -> dict:
    """Run the OptionsTrader logic inline on pre-built tick data."""
    td: dict = {}
    pos: dict[str, int] = {}
    cash = 0.0
    n_trades = 0
    errors = 0
    max_pos: dict[str, int] = {}
    min_pos: dict[str, int] = {}
    limit_hits = 0

    for ts, order_depths, _ in ticks:
        try:
            # --- Inline OptionsTrader.run() ---
            result: dict[str, list[Order]] = {}
            und_od = order_depths.get(UNDERLYING)
            if not und_od or not und_od.buy_orders or not und_od.sell_orders:
                continue

            spot = (max(und_od.buy_orders) + min(und_od.sell_orders)) / 2.0

            # TTE
            tte = td.get("tte")
            if tte is None:
                strikes_by_atm = sorted(OPTION_PRODUCTS.keys(), key=lambda k: abs(k - spot))
                for strike in strikes_by_atm:
                    opt_sym = OPTION_PRODUCTS[strike]
                    opt_od = order_depths.get(opt_sym)
                    if opt_od and opt_od.buy_orders and opt_od.sell_orders:
                        opt_mid = (max(opt_od.buy_orders) + min(opt_od.sell_orders)) / 2.0
                        tte = detect_tte(spot, opt_mid)
                        if tte is not None:
                            break
                if tte is None:
                    tte = 7 / 365.0
                td["tte"] = tte

            # Collect IVs
            iv_data: list[tuple[float, float]] = []
            option_mids: dict[int, float] = {}
            for strike, opt_sym in OPTION_PRODUCTS.items():
                opt_od = order_depths.get(opt_sym)
                if not opt_od or not opt_od.buy_orders or not opt_od.sell_orders:
                    continue
                opt_mid = (max(opt_od.buy_orders) + min(opt_od.sell_orders)) / 2.0
                option_mids[strike] = opt_mid
                iv = find_iv(spot, float(strike), tte, 0, opt_mid)
                if iv is not None and 0.01 <= iv <= 2.0:
                    m_t = math.log(strike / spot) / math.sqrt(tte) if tte > 0 else 0
                    iv_data.append((m_t, iv))

            if len(iv_data) < 3:
                continue

            # Smile
            smile = td.get("smile")
            tc = td.get("tc", 0) + 1
            td["tc"] = tc
            if smile is None or tc % refit_every == 0:
                m_arr = np.array([d[0] for d in iv_data])
                iv_arr = np.array([d[1] for d in iv_data])
                try:
                    coeffs = np.polyfit(m_arr, iv_arr, 2)
                    smile = {"a": float(coeffs[0]), "b": float(coeffs[1]), "c": float(coeffs[2])}
                    td["smile"] = smile
                except Exception:
                    if smile is None:
                        continue

            a, b, c = smile["a"], smile["b"], smile["c"]
            total_delta = 0.0

            for strike, opt_sym in OPTION_PRODUCTS.items():
                if strike not in option_mids:
                    continue
                opt_mid = option_mids[strike]
                m_t = math.log(strike / spot) / math.sqrt(tte) if tte > 0 else 0
                smile_iv = max(0.01, a * m_t ** 2 + b * m_t + c)
                fair_price, fair_delta = bs_call(spot, float(strike), tte, 0, smile_iv)
                vega = bs_vega(spot, float(strike), tte, smile_iv)
                deviation = opt_mid - fair_price
                if vega < min_vega:
                    continue
                opt_od = order_depths.get(opt_sym)
                if not opt_od:
                    continue
                opt_pos = pos.get(opt_sym, 0)
                norm_dev = deviation / vega if vega > 0 else 0

                if norm_dev > open_thr:
                    rem_sell = OPT_LIMIT + opt_pos
                    if rem_sell > 0 and opt_od.buy_orders:
                        qty = min(rem_sell, trade_qty)
                        result.setdefault(opt_sym, []).append(
                            Order(opt_sym, max(opt_od.buy_orders), -qty))
                elif norm_dev < -open_thr:
                    rem_buy = OPT_LIMIT - opt_pos
                    if rem_buy > 0 and opt_od.sell_orders:
                        qty = min(rem_buy, trade_qty)
                        result.setdefault(opt_sym, []).append(
                            Order(opt_sym, min(opt_od.sell_orders), qty))

                total_delta += opt_pos * fair_delta

            # Delta hedge
            und_pos = pos.get(UNDERLYING, 0)
            target_und = -round(total_delta)
            hedge_qty = int(target_und - und_pos)
            if hedge_qty != 0 and und_od.buy_orders and und_od.sell_orders:
                if hedge_qty > 0:
                    rem = UND_LIMIT - und_pos
                    qty = min(hedge_qty, rem)
                    if qty > 0:
                        result.setdefault(UNDERLYING, []).append(
                            Order(UNDERLYING, min(und_od.sell_orders), qty))
                else:
                    rem = UND_LIMIT + und_pos
                    qty = min(-hedge_qty, rem)
                    if qty > 0:
                        result.setdefault(UNDERLYING, []).append(
                            Order(UNDERLYING, max(und_od.buy_orders), -qty))

            # --- Fill orders ---
            for p, orders in result.items():
                if orders:
                    od = order_depths.get(p)
                    if od:
                        limit = UND_LIMIT if p == UNDERLYING else OPT_LIMIT
                        c_delta, new_pos = simple_pnl(orders, od, pos.get(p, 0), limit)
                        cash += c_delta
                        pos[p] = new_pos
                        n_trades += len(orders)

        except Exception:
            errors += 1

        # Track positions
        for p, v in pos.items():
            max_pos[p] = max(max_pos.get(p, 0), v)
            min_pos[p] = min(min_pos.get(p, 0), v)
            limit = UND_LIMIT if p == UNDERLYING else OPT_LIMIT
            if abs(v) >= limit:
                limit_hits += 1

    # Mark to market using last tick mids
    mtm = 0.0
    if ticks:
        _, last_ods, last_mids = ticks[-1]
        for p, position in pos.items():
            if p in last_mids:
                mtm += position * last_mids[p]

    return {
        "pnl": cash + mtm,
        "cash": cash,
        "mtm": mtm,
        "n_trades": n_trades,
        "errors": errors,
        "max_pos": max_pos,
        "min_pos": min_pos,
        "limit_hits": limit_hits,
        "final_pos": dict(pos),
    }


def grid_search(prices: pd.DataFrame) -> list[dict]:
    print(f"\n{SEPARATOR}")
    print("2. GRID SEARCH: open_thr x min_vega x trade_qty")
    print(SEPARATOR)

    days = sorted(prices[prices["product"] == UNDERLYING]["day"].unique())
    train_day = days[0]
    val_days = days[1:]

    open_thrs = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    min_vegas = [0.3, 0.5, 0.7, 1.0, 1.5]
    trade_qtys = [3, 5, 8, 10]

    print(f"Train day: {train_day}, Validation days: {val_days}")
    total = len(open_thrs) * len(min_vegas) * len(trade_qtys)
    print(f"Grid: {len(open_thrs)} x {len(min_vegas)} x {len(trade_qtys)} = {total} combos\n")

    # Pre-build tick data once per day
    print("Pre-building tick data...")
    t0 = time.time()
    tick_cache: dict[int, TickData] = {}
    for d in days:
        tick_cache[d] = prebuild_ticks(prices, d)
    print(f"  Built in {time.time() - t0:.1f}s "
          f"({sum(len(v) for v in tick_cache.values())} total ticks)\n")

    results: list[dict] = []
    done = 0
    t0 = time.time()

    for ot in open_thrs:
        for mv in min_vegas:
            for tq in trade_qtys:
                done += 1
                train_res = run_options_sim(tick_cache[train_day], ot, mv, tq)
                val_pnls = [
                    run_options_sim(tick_cache[vd], ot, mv, tq)["pnl"]
                    for vd in val_days
                ]
                avg_val = np.mean(val_pnls)
                min_val = min(val_pnls)

                results.append({
                    "open_thr": ot, "min_vega": mv, "trade_qty": tq,
                    "train_pnl": train_res["pnl"],
                    "val_avg_pnl": avg_val, "val_min_pnl": min_val,
                    "val_pnls": val_pnls,
                    "train_trades": train_res["n_trades"],
                    "train_errors": train_res["errors"],
                    "train_limit_hits": train_res["limit_hits"],
                })

                if done % 28 == 0 or done == total:
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total - done)
                    print(f"  [{done}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining  "
                          f"last: ot={ot}, mv={mv}, tq={tq} -> "
                          f"train={train_res['pnl']:>10,.0f}, val_avg={avg_val:>10,.0f}")

    results.sort(key=lambda x: x["val_avg_pnl"], reverse=True)

    print(f"\n--- Top 15 by validation average PnL ---")
    print(f"{'open_thr':>8} {'min_vega':>8} {'trade_q':>7} {'train':>10} "
          f"{'val_avg':>10} {'val_min':>10} {'trades':>6} {'lim_hit':>7}")
    for r in results[:15]:
        print(f"{r['open_thr']:>8.1f} {r['min_vega']:>8.1f} {r['trade_qty']:>7d} "
              f"{r['train_pnl']:>10,.0f} {r['val_avg_pnl']:>10,.0f} "
              f"{r['val_min_pnl']:>10,.0f} {r['train_trades']:>6d} "
              f"{r['train_limit_hits']:>7d}")

    print(f"\n--- Bottom 5 ---")
    for r in results[-5:]:
        print(f"  ot={r['open_thr']}, mv={r['min_vega']}, tq={r['trade_qty']} -> "
              f"val_avg={r['val_avg_pnl']:>10,.0f}")

    best = results[0]
    print(f"\n*** BEST: open_thr={best['open_thr']}, "
          f"min_vega={best['min_vega']}, trade_qty={best['trade_qty']} ***")
    print(f"    Train PnL: {best['train_pnl']:,.0f}")
    print(f"    Val Avg PnL: {best['val_avg_pnl']:,.0f}")
    print(f"    Val Min PnL: {best['val_min_pnl']:,.0f}")
    print(f"    Val PnLs: {[f'{p:,.0f}' for p in best['val_pnls']]}")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. VOL SKEW ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyze_vol_skew(df: pd.DataFrame) -> None:
    print(f"\n{SEPARATOR}")
    print("3. VOLATILITY SKEW ANALYSIS")
    print(SEPARATOR)

    pivot = pivot_mids(df)
    days = sorted(pivot["day"].unique())
    strikes = sorted(OPTION_PRODUCTS.keys())

    # Detect TTE per day
    day_ttes: dict[int, float] = {}
    for day in days:
        dd = pivot[pivot["day"] == day]
        first_row = dd.iloc[0]
        spot = first_row[UNDERLYING]
        if pd.isna(spot):
            continue
        atm_strike = min(strikes, key=lambda k: abs(k - spot))
        atm_sym = OPTION_PRODUCTS[atm_strike]
        atm_mid = first_row.get(atm_sym)
        if pd.notna(atm_mid) and atm_mid > 0:
            tte = detect_tte(spot, atm_mid)
            if tte is not None:
                day_ttes[day] = tte
                print(f"Day {day}: TTE={tte*365:.2f}d")

    # Compute IVs (sample every 200 ticks for speed)
    iv_records: list[dict] = []
    sample_every = 200

    for day in days:
        tte = day_ttes.get(day)
        if tte is None:
            continue
        dd = pivot[pivot["day"] == day]
        for idx, (_, row) in enumerate(dd.iterrows()):
            if idx % sample_every != 0:
                continue
            spot = row[UNDERLYING]
            if pd.isna(spot):
                continue
            ts = row["timestamp"]
            for strike in strikes:
                opt_sym = OPTION_PRODUCTS[strike]
                opt_mid = row.get(opt_sym)
                if pd.isna(opt_mid) or opt_mid <= 0:
                    continue
                iv = find_iv(spot, float(strike), tte, 0, opt_mid)
                if iv is not None and 0.01 <= iv <= 2.0:
                    iv_records.append({
                        "day": day, "timestamp": ts, "strike": strike,
                        "spot": spot, "iv": iv, "opt_mid": opt_mid,
                        "moneyness": (strike - spot) / spot,
                        "log_moneyness": math.log(strike / spot) / math.sqrt(tte),
                    })

    iv_df = pd.DataFrame(iv_records)
    if iv_df.empty:
        print("No IV data computed!")
        return

    print(f"\nComputed {len(iv_df)} IV observations")

    # Average IV per strike per day
    print("\n--- Average IV by strike and day ---")
    pivot_iv = iv_df.pivot_table(index="strike", columns="day", values="iv", aggfunc="mean")
    print(pivot_iv.round(4).to_string())

    # Overall averages
    print("\n--- Average IV by strike (all days) ---")
    avg_iv = iv_df.groupby("strike")["iv"].agg(["mean", "std", "min", "max", "count"])
    print(avg_iv.round(4).to_string())

    # Skew metric
    print("\n--- Skew metrics (IV_low_K - IV_high_K) ---")
    for day in days:
        dd = iv_df[iv_df["day"] == day]
        if dd.empty:
            continue
        avg_by_strike = dd.groupby("strike")["iv"].mean()
        if len(avg_by_strike) >= 2:
            low_iv = avg_by_strike.iloc[0]
            high_iv = avg_by_strike.iloc[-1]
            skew = low_iv - high_iv
            atm_iv = avg_by_strike.get(10000, avg_by_strike.median())
            print(f"  Day {day}: skew = {skew:+.4f} "
                  f"({skew/atm_iv*100:+.1f}% of ATM), ATM IV={atm_iv:.4f}")

    # Skew stability
    print("\n--- Skew profile correlation across days ---")
    day_profiles = {}
    for day in days:
        dd = iv_df[iv_df["day"] == day]
        prof = dd.groupby("strike")["iv"].mean()
        if len(prof) >= 3:
            day_profiles[day] = prof

    dk = sorted(day_profiles.keys())
    for i in range(len(dk)):
        for j in range(i + 1, len(dk)):
            common = day_profiles[dk[i]].index.intersection(day_profiles[dk[j]].index)
            if len(common) >= 3:
                corr = np.corrcoef(
                    day_profiles[dk[i]].loc[common].values,
                    day_profiles[dk[j]].loc[common].values,
                )[0, 1]
                print(f"  Day {dk[i]} vs Day {dk[j]}: r = {corr:.4f}")

    # Within-day skew evolution
    print("\n--- Skew evolution within day (first 20% vs last 20%) ---")
    for day in days:
        dd = iv_df[iv_df["day"] == day]
        if dd.empty:
            continue
        max_ts = dd["timestamp"].max()
        early = dd[dd["timestamp"] < max_ts * 0.2].groupby("strike")["iv"].mean()
        late = dd[dd["timestamp"] > max_ts * 0.8].groupby("strike")["iv"].mean()
        if len(early) >= 2 and len(late) >= 2:
            e_s = early.iloc[0] - early.iloc[-1]
            l_s = late.iloc[0] - late.iloc[-1]
            print(f"  Day {day}: early_skew={e_s:+.4f}, late_skew={l_s:+.4f}, "
                  f"delta={l_s - e_s:+.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. POSITION LIMIT UTILIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyze_positions(prices: pd.DataFrame) -> None:
    print(f"\n{SEPARATOR}")
    print("4. POSITION LIMIT UTILIZATION")
    print(SEPARATOR)

    days = sorted(prices[prices["product"] == UNDERLYING]["day"].unique())

    # Pre-build ticks once
    tick_cache: dict[int, TickData] = {}
    for d in days:
        tick_cache[d] = prebuild_ticks(prices, d)

    # For each trade_qty, show position profiles
    for tq in [3, 5, 8, 10]:
        print(f"\n--- trade_qty={tq}, open_thr=0.5, min_vega=0.7 ---")
        for day in days:
            res = run_options_sim(tick_cache[day], open_thr=0.5, min_vega=0.7, trade_qty=tq)
            print(f"  Day {day}: PnL={res['pnl']:>10,.0f}  trades={res['n_trades']:>5}  "
                  f"limit_hits={res['limit_hits']}")
            for p in sorted(res["max_pos"].keys()):
                limit = UND_LIMIT if p == UNDERLYING else OPT_LIMIT
                mx = res["max_pos"].get(p, 0)
                mn = res["min_pos"].get(p, 0)
                pct = max(abs(mx), abs(mn)) / limit * 100
                if mx != 0 or mn != 0:
                    print(f"    {p:>35s}: [{mn:>4d}, {mx:>4d}] "
                          f"({pct:.0f}% of {limit})")

    # Summary table
    print(f"\n--- trade_qty impact summary (validation days {days[1:]}) ---")
    val_days = days[1:] if len(days) > 1 else days
    print(f"{'tq':>4} {'avg_PnL':>12} {'min_PnL':>12} {'max_PnL':>12} {'avg_trades':>10}")
    for tq in [3, 5, 8, 10]:
        pnls = []
        trades_list = []
        for day in val_days:
            res = run_options_sim(tick_cache[day], 0.5, 0.7, tq)
            pnls.append(res["pnl"])
            trades_list.append(res["n_trades"])
        print(f"{tq:>4d} {np.mean(pnls):>12,.0f} {min(pnls):>12,.0f} "
              f"{max(pnls):>12,.0f} {np.mean(trades_list):>10.0f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    t_start = time.time()
    print("Loading Round 3 data...")
    prices_r3 = load_prices(ROUND3_DIR)
    df_r3 = prices_r3[prices_r3["product"].isin(RELEVANT_PRODUCTS)].copy()
    print(f"Loaded: {len(df_r3):,} rows, "
          f"days={sorted(df_r3['day'].unique())}, "
          f"products={sorted(df_r3['product'].unique())}")

    # 1. TTE Decay
    analyze_tte_decay(df_r3)

    # 2. Grid Search
    grid_results = grid_search(prices_r3)

    # 3. Vol Skew
    analyze_vol_skew(df_r3)

    # 4. Position Limits
    analyze_positions(prices_r3)

    # ── Round 5 out-of-sample ─────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("ROUND 5 OUT-OF-SAMPLE VALIDATION")
    print(SEPARATOR)

    try:
        prices_r5 = load_prices(ROUND5_DIR)
        if UNDERLYING in set(prices_r5["product"].unique()):
            days_r5 = sorted(prices_r5["day"].unique())
            print(f"Round 5 days: {days_r5}")

            r5_tick_cache = {d: prebuild_ticks(prices_r5, d) for d in days_r5}

            best = grid_results[0]
            ot, mv, tq = best["open_thr"], best["min_vega"], best["trade_qty"]
            print(f"\nBest params (ot={ot}, mv={mv}, tq={tq}) on Round 5:")
            for day in days_r5:
                res = run_options_sim(r5_tick_cache[day], ot, mv, tq)
                print(f"  Day {day}: PnL={res['pnl']:>10,.0f}  trades={res['n_trades']:>5}")

            print(f"\nDefaults (ot=0.5, mv=0.7, tq=5) on Round 5:")
            for day in days_r5:
                res = run_options_sim(r5_tick_cache[day], 0.5, 0.7, 5)
                print(f"  Day {day}: PnL={res['pnl']:>10,.0f}  trades={res['n_trades']:>5}")
        else:
            print("VOLCANIC_ROCK not in Round 5 data")
    except FileNotFoundError:
        print("Round 5 data not available")

    # ── Recommendation ────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("RECOMMENDATION")
    print(SEPARATOR)
    best = grid_results[0]
    current_defaults = {"open_thr": 0.5, "min_vega": 0.7, "trade_qty": 5}
    print(f"""
Walk-forward grid search results (train=day 0, validate=days 1,2):

  Parameter    Current    Optimal
  ---------    -------    -------
  open_thr     {current_defaults['open_thr']:<10} {best['open_thr']}
  min_vega     {current_defaults['min_vega']:<10} {best['min_vega']}
  trade_qty    {current_defaults['trade_qty']:<10} {best['trade_qty']}

  Validation avg PnL: {best['val_avg_pnl']:>12,.0f}
  Validation min PnL: {best['val_min_pnl']:>12,.0f}
  Train PnL:          {best['train_pnl']:>12,.0f}
""")

    elapsed = time.time() - t_start
    print(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
