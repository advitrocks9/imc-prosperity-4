"""Walk-forward grid search for strategy parameter optimization.

Train on day N, validate on day N+1. Checks landscape smoothness
to detect overfitting (sharp PnL peaks = overfit, smooth = robust).

Usage:
    # Use programmatically - see search() function
    uv run analysis/parameter_search.py  # runs demo on tutorial data
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from itertools import product as itertools_product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.data_loader import load_prices, get_product_mid


@dataclass
class SearchResult:
    """Result of a parameter search."""
    product: str
    param_names: list[str]
    best_params: dict[str, Any]
    best_train_pnl: float
    best_val_pnl: float
    overfit_ratio: float      # val_pnl / train_pnl - closer to 1.0 is better
    landscape_smoothness: float  # 0-1, higher = smoother = more robust
    all_results: list[dict] = field(default_factory=list)  # for heatmap generation
    notes: list[str] = field(default_factory=list)


@dataclass
class SimTick:
    """One tick of price data for the strategy simulator."""
    timestamp: int
    mid_price: float
    best_bid: float
    best_ask: float
    bid_volume_1: float
    ask_volume_1: float
    wall_mid: float


def load_sim_ticks(prices: pd.DataFrame, product_name: str, day: int) -> list[SimTick]:
    """Extract simulation ticks for one product-day."""
    pdf = get_product_mid(prices, product_name)
    day_data = pdf[pdf["day"] == day].sort_values("timestamp")

    ticks = []
    for _, row in day_data.iterrows():
        ticks.append(SimTick(
            timestamp=int(row["timestamp"]),
            mid_price=float(row["mid_price"]),
            best_bid=float(row.get("best_bid", row["mid_price"] - 5)),
            best_ask=float(row.get("best_ask", row["mid_price"] + 5)),
            bid_volume_1=float(row.get("bid_volume_1", 10)),
            ask_volume_1=float(row.get("ask_volume_1", 10)),
            wall_mid=float(row.get("wall_mid", row["mid_price"])),
        ))
    return ticks


StrategyFn = Callable[[SimTick, int, dict[str, Any], dict], tuple[list[tuple[int, int]], dict]]
"""Strategy function signature:
    Args: tick, current_position, params, state_dict
    Returns: (orders as [(price, qty)], updated_state_dict)
    Positive qty = buy, negative = sell.
"""


def simulate_pnl(
    ticks: list[SimTick],
    strategy: StrategyFn,
    params: dict[str, Any],
    position_limit: int,
) -> float:
    """Simple PnL simulator. Fills orders at their posted price (optimistic).

    This is a rough approximation - the real matching engine is different.
    Good enough for relative parameter comparison.
    """
    pos = 0
    cash = 0.0
    state: dict = {}

    for tick in ticks:
        orders, state = strategy(tick, pos, params, state)

        for price, qty in orders:
            # Clip to position limits
            if qty > 0:
                qty = min(qty, position_limit - pos)
            elif qty < 0:
                qty = max(qty, -(position_limit + pos))

            if qty == 0:
                continue

            # Simple fill model: fill if price is competitive
            if qty > 0 and price >= tick.best_ask:
                # Buy fills at ask
                fill_price = tick.best_ask
                fill_qty = min(qty, int(tick.ask_volume_1))
            elif qty < 0 and price <= tick.best_bid:
                # Sell fills at bid
                fill_price = tick.best_bid
                fill_qty = max(qty, -int(tick.bid_volume_1))
            elif qty > 0 and price >= tick.best_bid:
                # Passive buy - assume partial fill
                fill_price = price
                fill_qty = max(1, qty // 4)
            elif qty < 0 and price <= tick.best_ask:
                # Passive sell - assume partial fill
                fill_price = price
                fill_qty = min(-1, qty // 4)
            else:
                continue

            pos += fill_qty
            cash -= fill_qty * fill_price

    # Mark to market at last mid
    if ticks:
        cash += pos * ticks[-1].mid_price

    return cash


def landscape_smoothness(pnl_grid: np.ndarray) -> float:
    """Measure how smooth the PnL landscape is (0=spiky, 1=perfectly smooth).

    Uses normalized Laplacian: a smooth surface has small second derivatives.
    """
    if pnl_grid.size <= 1:
        return 1.0

    # Flatten for 1D case
    if pnl_grid.ndim == 1:
        if len(pnl_grid) < 3:
            return 1.0
        laplacian = np.abs(np.diff(pnl_grid, n=2))
        pnl_range = np.ptp(pnl_grid)
        if pnl_range < 1e-6:
            return 1.0
        return float(1.0 - np.clip(np.mean(laplacian) / pnl_range, 0, 1))

    # 2D: compute discrete Laplacian
    if pnl_grid.shape[0] < 3 or pnl_grid.shape[1] < 3:
        return 1.0

    laplacian = np.zeros_like(pnl_grid[1:-1, 1:-1])
    laplacian += pnl_grid[:-2, 1:-1]  # up
    laplacian += pnl_grid[2:, 1:-1]   # down
    laplacian += pnl_grid[1:-1, :-2]  # left
    laplacian += pnl_grid[1:-1, 2:]   # right
    laplacian -= 4 * pnl_grid[1:-1, 1:-1]

    pnl_range = np.ptp(pnl_grid)
    if pnl_range < 1e-6:
        return 1.0
    return float(1.0 - np.clip(np.mean(np.abs(laplacian)) / pnl_range, 0, 1))


def search(
    prices: pd.DataFrame,
    product_name: str,
    strategy: StrategyFn,
    param_grid: dict[str, list[Any]],
    position_limit: int,
    train_day: int | None = None,
    val_day: int | None = None,
) -> SearchResult:
    """Run walk-forward grid search.

    If train_day/val_day not specified, uses first two days found in data.
    """
    days = sorted(prices[prices["product"] == product_name]["day"].unique())
    if train_day is None:
        train_day = days[0]
    if val_day is None:
        val_day = days[1] if len(days) > 1 else days[0]

    train_ticks = load_sim_ticks(prices, product_name, train_day)
    val_ticks = load_sim_ticks(prices, product_name, val_day)

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    all_results = []
    best_train_pnl = -float("inf")
    best_params: dict[str, Any] = {}
    best_val_pnl = 0.0

    # Grid search
    for combo in itertools_product(*param_values):
        params = dict(zip(param_names, combo))
        train_pnl = simulate_pnl(train_ticks, strategy, params, position_limit)
        val_pnl = simulate_pnl(val_ticks, strategy, params, position_limit)

        result_entry = {**params, "train_pnl": round(train_pnl, 2), "val_pnl": round(val_pnl, 2)}
        all_results.append(result_entry)

        if train_pnl > best_train_pnl:
            best_train_pnl = train_pnl
            best_params = params
            best_val_pnl = val_pnl

    # Overfit ratio
    overfit_ratio = best_val_pnl / best_train_pnl if best_train_pnl > 0 else 0

    # Landscape smoothness (use val PnL for robustness check)
    if len(param_names) == 1:
        pnl_array = np.array([r["val_pnl"] for r in all_results])
        smooth = landscape_smoothness(pnl_array)
    elif len(param_names) == 2:
        # Build 2D grid
        v0 = sorted(set(r[param_names[0]] for r in all_results))
        v1 = sorted(set(r[param_names[1]] for r in all_results))
        grid = np.zeros((len(v0), len(v1)))
        idx0 = {v: i for i, v in enumerate(v0)}
        idx1 = {v: i for i, v in enumerate(v1)}
        for r in all_results:
            grid[idx0[r[param_names[0]]], idx1[r[param_names[1]]]] = r["val_pnl"]
        smooth = landscape_smoothness(grid)
    else:
        # Higher-dimensional: use 1D slice through best params
        pnl_array = np.array([r["val_pnl"] for r in all_results])
        smooth = landscape_smoothness(pnl_array)

    notes = []
    if overfit_ratio < 0.5:
        notes.append("WARNING: val PnL < 50% of train PnL - likely overfit")
    if smooth < 0.5:
        notes.append("WARNING: PnL landscape is spiky - parameters may be fragile")

    return SearchResult(
        product=product_name,
        param_names=param_names,
        best_params=best_params,
        best_train_pnl=round(best_train_pnl, 2),
        best_val_pnl=round(best_val_pnl, 2),
        overfit_ratio=round(overfit_ratio, 4),
        landscape_smoothness=round(smooth, 4),
        all_results=all_results,
        notes=notes,
    )


def format_report(result: SearchResult) -> str:
    """Format search results as readable report."""
    lines = [
        f"# Parameter Search: {result.product}",
        "",
        f"**Best params**: {result.best_params}",
        f"**Train PnL**: {result.best_train_pnl:,.0f}",
        f"**Val PnL**: {result.best_val_pnl:,.0f}",
        f"**Overfit ratio**: {result.overfit_ratio:.2f} (1.0 = perfect generalization)",
        f"**Landscape smoothness**: {result.landscape_smoothness:.2f} (1.0 = smooth)",
        "",
    ]

    for note in result.notes:
        lines.append(f"- {note}")

    if result.notes:
        lines.append("")

    # Top 10 by val PnL
    lines.append("## Top 10 Configurations (by validation PnL)")
    lines.append("")
    top = sorted(result.all_results, key=lambda x: -x["val_pnl"])[:10]
    header = " | ".join(result.param_names + ["Train PnL", "Val PnL"])
    lines.append(f"| {header} |")
    lines.append("|" + "|".join(["---"] * (len(result.param_names) + 2)) + "|")
    for r in top:
        vals = [str(r[p]) for p in result.param_names] + [f"{r['train_pnl']:,.0f}", f"{r['val_pnl']:,.0f}"]
        lines.append(f"| {' | '.join(vals)} |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Demo: run parameter search on tutorial TOMATOES with a simple MM strategy."""
    from analysis.data_loader import load_round

    prices, _ = load_round(0)

    def simple_mm(tick: SimTick, pos: int, params: dict, state: dict):
        spread = int(params["spread"])
        mid = tick.wall_mid
        orders = [
            (int(mid - spread), max(1, 80 - pos)),
            (int(mid + spread), -max(1, 80 + pos)),
        ]
        return orders, state

    result = search(
        prices=prices,
        product_name="TOMATOES",
        strategy=simple_mm,
        param_grid={"spread": list(range(1, 8))},
        position_limit=80,
    )

    print(format_report(result))


if __name__ == "__main__":
    main()
