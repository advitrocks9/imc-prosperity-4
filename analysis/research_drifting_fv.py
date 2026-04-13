"""Research: compare FV estimators for drifting product market maker.

Tests microprice, OFI, vol-adaptive spread, and multi-lag MR against
the baseline wall_mid FV on P3 KELP, SQUID_INK (R1), and CROISSANTS (R2).

Usage:
    uv run analysis/research_drifting_fv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.parameter_search import SimTick, load_sim_ticks, simulate_pnl
from analysis.data_loader import load_prices


# ── Constants ────────────────────────────────────────────────────────
POSITION_LIMIT = 50  # P3 KELP/SQUID_INK limit
CROISSANTS_LIMIT = 250  # P3 CROISSANTS limit

PRODUCTS = [
    ("KELP", "data/p3/round1", [-2, -1, 0], POSITION_LIMIT),
    ("SQUID_INK", "data/p3/round1", [-2, -1, 0], POSITION_LIMIT),
    ("CROISSANTS", "data/p3/round2", [-1, 0, 1], CROISSANTS_LIMIT),
]


# ── Strategy Helpers ─────────────────────────────────────────────────

def _mm_orders(
    fv: float,
    spread: int,
    pos: int,
    limit: int,
    skew_sens: float = 2.0,
) -> list[tuple[int, int]]:
    """Generate simple MM bid/ask around fair value with inventory skew."""
    skew = -pos / limit * skew_sens
    bid_price = int(round(fv - spread + skew))
    ask_price = int(round(fv + spread + skew))
    if bid_price >= ask_price:
        ask_price = bid_price + 1

    rem_buy = limit - pos
    rem_sell = limit + pos
    orders: list[tuple[int, int]] = []
    if rem_buy > 0:
        orders.append((bid_price, rem_buy))
    if rem_sell > 0:
        orders.append((ask_price, -rem_sell))
    return orders


# ── Strategies ───────────────────────────────────────────────────────

def baseline_wall_mid(
    tick: SimTick, pos: int, params: dict, state: dict,
) -> tuple[list[tuple[int, int]], dict]:
    """Baseline: FV = wall_mid, fixed spread."""
    spread = int(params.get("spread", 2))
    fv = tick.wall_mid
    state["fv"] = fv
    return _mm_orders(fv, spread, pos, params["limit"]), state


def microprice_strategy(
    tick: SimTick, pos: int, params: dict, state: dict,
) -> tuple[list[tuple[int, int]], dict]:
    """FV = microprice (volume-imbalance-weighted mid)."""
    spread = int(params.get("spread", 2))
    bv = tick.bid_volume_1
    av = tick.ask_volume_1
    total = bv + av
    if total > 0:
        fv = tick.best_bid * av / total + tick.best_ask * bv / total
    else:
        fv = tick.wall_mid
    state["fv"] = fv
    return _mm_orders(fv, spread, pos, params["limit"]), state


def micro_wallmid_blend(
    tick: SimTick, pos: int, params: dict, state: dict,
) -> tuple[list[tuple[int, int]], dict]:
    """FV = 0.5 * microprice + 0.5 * wall_mid."""
    spread = int(params.get("spread", 2))
    bv = tick.bid_volume_1
    av = tick.ask_volume_1
    total = bv + av
    if total > 0:
        micro = tick.best_bid * av / total + tick.best_ask * bv / total
    else:
        micro = tick.wall_mid
    fv = 0.5 * micro + 0.5 * tick.wall_mid
    state["fv"] = fv
    return _mm_orders(fv, spread, pos, params["limit"]), state


def ofi_strategy(
    tick: SimTick, pos: int, params: dict, state: dict,
) -> tuple[list[tuple[int, int]], dict]:
    """FV = wall_mid + ofi_sensitivity * EMA(order flow imbalance)."""
    spread = int(params.get("spread", 2))
    sensitivity = float(params.get("ofi_sensitivity", 0.3))
    alpha = float(params.get("ofi_alpha", 0.1))

    prev_bv = state.get("prev_bv", tick.bid_volume_1)
    prev_av = state.get("prev_av", tick.ask_volume_1)
    ofi_ema = state.get("ofi_ema", 0.0)

    ofi = (tick.bid_volume_1 - prev_bv) - (tick.ask_volume_1 - prev_av)
    ofi_ema = alpha * ofi + (1 - alpha) * ofi_ema

    fv = tick.wall_mid + sensitivity * ofi_ema

    state["prev_bv"] = tick.bid_volume_1
    state["prev_av"] = tick.ask_volume_1
    state["ofi_ema"] = ofi_ema
    state["fv"] = fv
    return _mm_orders(fv, spread, pos, params["limit"]), state


def vol_adaptive_strategy(
    tick: SimTick, pos: int, params: dict, state: dict,
) -> tuple[list[tuple[int, int]], dict]:
    """FV = wall_mid, spread = adaptive based on realized vol."""
    base_spread = int(params.get("base_spread", 2))
    window = int(params.get("vol_window", 20))

    mids: list[float] = state.get("mids", [])
    mids.append(tick.mid_price)
    if len(mids) > window + 1:
        mids = mids[-(window + 1):]
    state["mids"] = mids

    fv = tick.wall_mid
    state["fv"] = fv

    if len(mids) >= 3:
        returns = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
        vol = float(np.std(returns))
        # Compute a rolling median vol for normalization
        vol_history: list[float] = state.get("vol_history", [])
        vol_history.append(vol)
        if len(vol_history) > 200:
            vol_history = vol_history[-200:]
        state["vol_history"] = vol_history
        median_vol = float(np.median(vol_history)) if vol_history else vol
        if median_vol > 1e-6:
            ratio = vol / median_vol
            adaptive_spread = max(1, min(5, int(round(base_spread * ratio))))
        else:
            adaptive_spread = base_spread
    else:
        adaptive_spread = base_spread

    return _mm_orders(fv, adaptive_spread, pos, params["limit"]), state


def dual_ema_strategy(
    tick: SimTick, pos: int, params: dict, state: dict,
) -> tuple[list[tuple[int, int]], dict]:
    """FV = 0.7 * fast_EMA(wall_mid) + 0.3 * slow_EMA(wall_mid)."""
    spread = int(params.get("spread", 2))
    fast_span = int(params.get("fast_span", 5))
    slow_span = int(params.get("slow_span", 50))

    alpha_fast = 2.0 / (fast_span + 1)
    alpha_slow = 2.0 / (slow_span + 1)

    fast_ema = state.get("fast_ema", tick.wall_mid)
    slow_ema = state.get("slow_ema", tick.wall_mid)

    fast_ema = alpha_fast * tick.wall_mid + (1 - alpha_fast) * fast_ema
    slow_ema = alpha_slow * tick.wall_mid + (1 - alpha_slow) * slow_ema

    fv = 0.7 * fast_ema + 0.3 * slow_ema

    state["fast_ema"] = fast_ema
    state["slow_ema"] = slow_ema
    state["fv"] = fv
    return _mm_orders(fv, spread, pos, params["limit"]), state


def ar2_strategy(
    tick: SimTick, pos: int, params: dict, state: dict,
) -> tuple[list[tuple[int, int]], dict]:
    """FV = wall_mid with AR(2) mean reversion correction."""
    spread = int(params.get("spread", 2))
    beta1 = float(params.get("beta1", -0.23))
    beta2 = float(params.get("beta2", -0.1))

    fv = tick.wall_mid
    last_fv = state.get("fv")
    last_last_fv = state.get("prev_fv")

    if last_fv is not None:
        correction = beta1 * (last_fv - fv)
        if last_last_fv is not None:
            correction += beta2 * (last_last_fv - last_fv)
        fv = fv + correction

    state["prev_fv"] = state.get("fv")
    state["fv"] = tick.wall_mid  # store raw for next AR calc
    return _mm_orders(fv, spread, pos, params["limit"]), state


def microprice_ofi_combo(
    tick: SimTick, pos: int, params: dict, state: dict,
) -> tuple[list[tuple[int, int]], dict]:
    """FV = 0.5 * microprice + 0.5 * wall_mid + OFI shift. Best of both."""
    spread = int(params.get("spread", 2))
    sensitivity = float(params.get("ofi_sensitivity", 0.3))
    alpha = float(params.get("ofi_alpha", 0.1))

    # Microprice
    bv = tick.bid_volume_1
    av = tick.ask_volume_1
    total = bv + av
    if total > 0:
        micro = tick.best_bid * av / total + tick.best_ask * bv / total
    else:
        micro = tick.wall_mid

    # OFI
    prev_bv = state.get("prev_bv", bv)
    prev_av = state.get("prev_av", av)
    ofi_ema = state.get("ofi_ema", 0.0)
    ofi = (bv - prev_bv) - (av - prev_av)
    ofi_ema = alpha * ofi + (1 - alpha) * ofi_ema

    fv = 0.5 * micro + 0.5 * tick.wall_mid + sensitivity * ofi_ema

    state["prev_bv"] = bv
    state["prev_av"] = av
    state["ofi_ema"] = ofi_ema
    state["fv"] = fv
    return _mm_orders(fv, spread, pos, params["limit"]), state


# ── Runner ───────────────────────────────────────────────────────────

def run_all() -> None:
    """Run every estimator on every product-day and print comparison table."""

    # Default spread per product - CROISSANTS has 1-tick spread so needs spread=1
    PRODUCT_SPREAD: dict[str, int] = {
        "KELP": 2,
        "SQUID_INK": 2,
        "CROISSANTS": 1,
    }

    strategies: dict[str, tuple] = {
        # name: (strategy_fn, extra_params)
        "Baseline(WM)": (baseline_wall_mid, {}),
        "Microprice": (microprice_strategy, {}),
        "Micro+WM": (micro_wallmid_blend, {}),
        "OFI(0.1)": (ofi_strategy, {"ofi_sensitivity": 0.1}),
        "OFI(0.3)": (ofi_strategy, {"ofi_sensitivity": 0.3}),
        "OFI(0.5)": (ofi_strategy, {"ofi_sensitivity": 0.5}),
        "OFI(1.0)": (ofi_strategy, {"ofi_sensitivity": 1.0}),
        "VolAdapt": (vol_adaptive_strategy, {}),  # base_spread set per product
        "DualEMA": (dual_ema_strategy, {"fast_span": 5, "slow_span": 50}),
        "AR(2)": (ar2_strategy, {"beta1": -0.23, "beta2": -0.1}),
        "Micro+OFI": (microprice_ofi_combo, {"ofi_sensitivity": 0.3}),
    }

    # Collect results: {(product, day): {strategy_name: pnl}}
    results: dict[tuple[str, int], dict[str, float]] = {}

    for product, data_path, days, limit in PRODUCTS:
        print(f"\n{'='*60}")
        print(f"Loading {product} from {data_path} ...")
        prices = load_prices(data_path)

        for day in days:
            ticks = load_sim_ticks(prices, product, day)
            if not ticks:
                print(f"  WARNING: No ticks for {product} day {day}")
                continue

            key = (product, day)
            results[key] = {}

            default_spread = PRODUCT_SPREAD.get(product, 2)
            for name, (strategy_fn, extra_params) in strategies.items():
                params = {"spread": default_spread, "base_spread": default_spread,
                          "limit": limit, **extra_params}
                pnl = simulate_pnl(ticks, strategy_fn, params, limit)
                results[key][name] = pnl

            # Print per-day results
            base_pnl = results[key]["Baseline(WM)"]
            print(f"\n  {product} day={day}  (baseline={base_pnl:,.0f})")
            for name in strategies:
                pnl = results[key][name]
                delta = pnl - base_pnl
                sign = "+" if delta >= 0 else ""
                print(f"    {name:<16s}  {pnl:>10,.0f}  ({sign}{delta:,.0f})")

    # ── Summary table ────────────────────────────────────────────
    strat_names = list(strategies.keys())
    print("\n\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)

    # Header
    header = f"{'Product':<14s} {'Day':>4s}"
    for name in strat_names:
        header += f"  {name:>14s}"
    print(header)
    print("-" * len(header))

    # Rows
    totals: dict[str, float] = {n: 0.0 for n in strat_names}
    for (product, day), day_results in sorted(results.items()):
        row = f"{product:<14s} {day:>4d}"
        for name in strat_names:
            pnl = day_results.get(name, 0.0)
            totals[name] += pnl
            row += f"  {pnl:>14,.0f}"
        print(row)

    # Total row
    print("-" * len(header))
    total_row = f"{'TOTAL':<14s} {'':>4s}"
    for name in strat_names:
        total_row += f"  {totals[name]:>14,.0f}"
    print(total_row)

    # Delta from baseline
    base_total = totals["Baseline(WM)"]
    delta_row = f"{'DELTA vs BASE':<14s} {'':>4s}"
    for name in strat_names:
        d = totals[name] - base_total
        sign = "+" if d >= 0 else ""
        delta_row += f"  {sign}{d:>13,.0f}"
    print(delta_row)

    # Pct improvement
    pct_row = f"{'% IMPROVE':<14s} {'':>4s}"
    for name in strat_names:
        if abs(base_total) > 1e-6:
            pct = (totals[name] - base_total) / abs(base_total) * 100
        else:
            pct = 0.0
        pct_row += f"  {pct:>13.1f}%"
    print(pct_row)

    # ── Per-product consistency check ────────────────────────────
    print("\n\n" + "=" * 80)
    print("CONSISTENCY CHECK (avg PnL delta vs baseline, per product)")
    print("=" * 80)
    for product, _, days, _ in PRODUCTS:
        print(f"\n  {product}:")
        for name in strat_names:
            if name == "Baseline(WM)":
                continue
            deltas_for_product = []
            wins = 0
            for day in days:
                key = (product, day)
                if key in results:
                    d = results[key][name] - results[key]["Baseline(WM)"]
                    deltas_for_product.append(d)
                    if d > 0:
                        wins += 1
            if deltas_for_product:
                avg_d = sum(deltas_for_product) / len(deltas_for_product)
                sign = "+" if avg_d >= 0 else ""
                print(f"    {name:<16s}  avg delta: {sign}{avg_d:>8,.0f}  wins: {wins}/{len(deltas_for_product)}")

    # ── Recommendation ───────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Find best strategy by total PnL
    best_name = max(strat_names, key=lambda n: totals[n])
    best_total = totals[best_name]
    improvement = best_total - base_total

    print(f"\n  Best overall: {best_name}")
    print(f"  Total PnL: {best_total:,.0f} (baseline: {base_total:,.0f})")
    print(f"  Improvement: +{improvement:,.0f} ({improvement/abs(base_total)*100:.1f}%)")

    # Check consistency: strategy must win on majority of product-days
    all_keys = sorted(results.keys())
    best_wins = sum(
        1 for k in all_keys
        if results[k][best_name] > results[k]["Baseline(WM)"]
    )
    print(f"  Wins: {best_wins}/{len(all_keys)} product-days")

    # Find best *consistent* strategy (wins most product-days, tiebreak by total)
    consistency_scores = {}
    for name in strat_names:
        if name == "Baseline(WM)":
            continue
        wins = sum(1 for k in all_keys if results[k][name] > results[k]["Baseline(WM)"])
        consistency_scores[name] = (wins, totals[name])

    most_consistent = max(consistency_scores, key=lambda n: consistency_scores[n])
    mc_wins, mc_total = consistency_scores[most_consistent]
    mc_improvement = mc_total - base_total

    if most_consistent != best_name:
        print(f"\n  Most consistent: {most_consistent}")
        print(f"  Total PnL: {mc_total:,.0f}")
        print(f"  Improvement: +{mc_improvement:,.0f} ({mc_improvement/abs(base_total)*100:.1f}%)")
        print(f"  Wins: {mc_wins}/{len(all_keys)} product-days")

    # ── Spread sweep for KELP+SQUID_INK only (drifting products) ──
    print("\n\n" + "=" * 80)
    print("SPREAD SWEEP - KELP + SQUID_INK only (top 3 strategies, spread 1-4)")
    print("=" * 80)

    drifting_products = [p for p in PRODUCTS if p[0] in ("KELP", "SQUID_INK")]

    # Rank by KELP+SQUID_INK total only
    kelp_squid_totals: dict[str, float] = {n: 0.0 for n in strat_names}
    for (product, day), day_results in results.items():
        if product in ("KELP", "SQUID_INK"):
            for name in strat_names:
                kelp_squid_totals[name] += day_results.get(name, 0.0)

    ranked = sorted(
        [(n, kelp_squid_totals[n]) for n in strat_names if n != "Baseline(WM)"],
        key=lambda x: -x[1],
    )[:5]

    print(f"\n  KELP+SQUID_INK baseline total: {kelp_squid_totals['Baseline(WM)']:,.0f}")
    for name, t in ranked:
        d = t - kelp_squid_totals["Baseline(WM)"]
        print(f"  {name:<16s}: {t:>12,.0f}  (delta: {d:+,.0f})")

    for name, _ in ranked[:3]:
        strategy_fn, extra_params = strategies[name]
        print(f"\n  {name}:")
        for spread_val in [1, 2, 3, 4]:
            total_pnl = 0.0
            for product, data_path, days, limit in drifting_products:
                prices = load_prices(data_path)
                for day in days:
                    ticks = load_sim_ticks(prices, product, day)
                    if not ticks:
                        continue
                    params = {"spread": spread_val, "base_spread": spread_val,
                              "limit": limit, **extra_params}
                    pnl = simulate_pnl(ticks, strategy_fn, params, limit)
                    total_pnl += pnl
            print(f"    spread={spread_val}:  total PnL = {total_pnl:>12,.0f}")

    # ── Also test spread=1 baseline for KELP+SQUID_INK ──────────
    print("\n\n" + "=" * 80)
    print("SPREAD=1 TEST - all strategies on KELP + SQUID_INK")
    print("=" * 80)
    spread1_totals: dict[str, float] = {n: 0.0 for n in strat_names}
    for product, data_path, days, limit in drifting_products:
        prices = load_prices(data_path)
        for day in days:
            ticks = load_sim_ticks(prices, product, day)
            if not ticks:
                continue
            for name, (strategy_fn, extra_params) in strategies.items():
                params = {"spread": 1, "base_spread": 1, "limit": limit, **extra_params}
                pnl = simulate_pnl(ticks, strategy_fn, params, limit)
                spread1_totals[name] += pnl

    base1 = spread1_totals["Baseline(WM)"]
    print(f"\n  Baseline (spread=1): {base1:,.0f}")
    for name in strat_names:
        if name == "Baseline(WM)":
            continue
        t = spread1_totals[name]
        d = t - base1
        pct = d / abs(base1) * 100 if abs(base1) > 1e-6 else 0.0
        print(f"  {name:<16s}: {t:>12,.0f}  (delta: {d:>+10,.0f}, {pct:>+6.1f}%)")


if __name__ == "__main__":
    run_all()
