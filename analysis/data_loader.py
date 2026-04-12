"""Load Prosperity 4 round CSVs into usable DataFrames.

Usage:
    from analysis.data_loader import load_prices, load_trades, load_round
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# CSV columns ─────────────────────────────────────────────────────
PRICE_COLS = [
    "day", "timestamp", "product",
    "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2",
    "bid_price_3", "bid_volume_3",
    "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2",
    "ask_price_3", "ask_volume_3",
    "mid_price", "profit_and_loss",
]

TRADE_COLS = [
    "timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity",
]


def load_prices(round_dir: Path | str) -> pd.DataFrame:
    """Load all prices_round_*_day_*.csv from a round directory.

    Returns DataFrame indexed by (day, timestamp, product) with book levels
    and a computed 'wall_mid' column.
    """
    round_dir = Path(round_dir)
    files = sorted(round_dir.glob("prices_round_*_day_*.csv"))
    if not files:
        raise FileNotFoundError(f"No price CSVs in {round_dir}")

    dfs = [pd.read_csv(f, sep=";") for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Compute wall_mid: midpoint of worst bid and worst ask
    worst_bid = df[["bid_price_1", "bid_price_2", "bid_price_3"]].min(axis=1)
    worst_ask = df[["ask_price_1", "ask_price_2", "ask_price_3"]].max(axis=1)
    df["wall_mid"] = (worst_bid + worst_ask) / 2.0

    # Best bid/ask for spread calculations
    df["best_bid"] = df["bid_price_1"]
    df["best_ask"] = df["ask_price_1"]
    df["spread"] = df["best_ask"] - df["best_bid"]

    # Sort for deterministic ordering
    df.sort_values(["day", "timestamp", "product"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_trades(round_dir: Path | str) -> pd.DataFrame:
    """Load all trades_round_*_day_*.csv from a round directory.

    Returns DataFrame with columns: day, timestamp, buyer, seller, symbol, price, quantity.
    The 'day' is extracted from the filename (trades CSVs have no day column).
    """
    round_dir = Path(round_dir)
    files = sorted(round_dir.glob("trades_round_*_day_*.csv"))
    if not files:
        raise FileNotFoundError(f"No trade CSVs in {round_dir}")

    dfs = []
    for f in files:
        # Extract day from filename: trades_round_0_day_-1.csv → -1
        parts = f.stem.split("_")
        day_idx = parts.index("day") + 1
        day = int(parts[day_idx].split("_")[0])  # handle trades_round_0_day_-1_nn
        tdf = pd.read_csv(f, sep=";")
        tdf["day"] = day
        dfs.append(tdf)

    df = pd.concat(dfs, ignore_index=True)
    df.sort_values(["day", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_round(round_num: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: load prices + trades for a round.

    Round 0 = tutorial, Round 1-5 = competition rounds.
    """
    if round_num == 0:
        round_dir = DATA_DIR / "tutorial"
    else:
        round_dir = DATA_DIR / f"round{round_num}"

    prices = load_prices(round_dir)
    try:
        trades = load_trades(round_dir)
    except FileNotFoundError:
        trades = pd.DataFrame(columns=["day", "timestamp", "buyer", "seller", "symbol", "price", "quantity"])

    return prices, trades


def get_product_mid(prices: pd.DataFrame, product: str) -> pd.DataFrame:
    """Extract time-series for a single product.

    Returns DataFrame with columns: day, timestamp, mid_price, wall_mid, best_bid, best_ask, spread.
    """
    mask = prices["product"] == product
    cols = ["day", "timestamp", "mid_price", "wall_mid", "best_bid", "best_ask", "spread",
            "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2",
            "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2"]
    return prices.loc[mask, [c for c in cols if c in prices.columns]].reset_index(drop=True)


def get_product_trades(trades: pd.DataFrame, product: str) -> pd.DataFrame:
    """Extract trades for a single product."""
    return trades.loc[trades["symbol"] == product].reset_index(drop=True)
