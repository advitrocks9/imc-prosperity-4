from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/round4")
WINDOW = 200


def load_prices() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in (1, 2, 3):
        path = DATA_DIR / f"prices_round_4_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        df["day"] = day
        if "mid_price" not in df.columns or df["mid_price"].isna().any():
            df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
        df = df[["day", "timestamp", "product", "mid_price"]].copy()
        df = df.sort_values(["product", "timestamp"]).reset_index(drop=True)
        df["rolling_low_200"] = (
            df.groupby("product")["mid_price"]
            .transform(lambda s: s.rolling(window=WINDOW, min_periods=1).min())
        )
        df["rolling_high_200"] = (
            df.groupby("product")["mid_price"]
            .transform(lambda s: s.rolling(window=WINDOW, min_periods=1).max())
        )
        df["is_rolling_low"] = df["mid_price"] <= df["rolling_low_200"]
        df["is_rolling_high"] = df["mid_price"] >= df["rolling_high_200"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_trades() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in (1, 2, 3):
        path = DATA_DIR / f"trades_round_4_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        df["day"] = day
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def mark_trade_events(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    merged = trades.merge(
        prices,
        left_on=["day", "timestamp", "symbol"],
        right_on=["day", "timestamp", "product"],
        how="left",
        validate="many_to_one",
    )
    if merged[["mid_price", "rolling_low_200", "rolling_high_200"]].isna().any().any():
        missing = merged[merged["mid_price"].isna()][["day", "timestamp", "symbol"]]
        raise ValueError(f"Missing price rows for trades:\n{missing.head(10).to_string(index=False)}")

    buy_events = merged.loc[merged["buyer"].astype(str).str.startswith("Mark ")].copy()
    buy_events["mark"] = buy_events["buyer"]
    buy_events["side"] = "BUY"
    buy_events["is_extrema"] = buy_events["is_rolling_low"]

    sell_events = merged.loc[merged["seller"].astype(str).str.startswith("Mark ")].copy()
    sell_events["mark"] = sell_events["seller"]
    sell_events["side"] = "SELL"
    sell_events["is_extrema"] = sell_events["is_rolling_high"]

    events = pd.concat([buy_events, sell_events], ignore_index=True)
    return events[["mark", "symbol", "side", "is_extrema"]].copy()


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        events.groupby(["mark", "symbol", "side"])["is_extrema"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped["pct"] = grouped["mean"] * 100.0

    pivot = grouped.pivot_table(
        index=["mark", "symbol"],
        columns="side",
        values=["pct", "count"],
        fill_value=0.0,
    )
    pivot.columns = [f"{metric.lower()}_{side.lower()}" for metric, side in pivot.columns]
    summary = pivot.reset_index()

    for col in ("pct_buy", "pct_sell"):
        if col not in summary.columns:
            summary[col] = 0.0
    for col in ("count_buy", "count_sell"):
        if col not in summary.columns:
            summary[col] = 0

    summary["count_buy"] = summary["count_buy"].astype(int)
    summary["count_sell"] = summary["count_sell"].astype(int)
    summary["qualifies"] = (
        (summary["count_buy"] > 0)
        & (summary["count_sell"] > 0)
        & (summary["pct_buy"] > 60.0)
        & (summary["pct_sell"] > 60.0)
    )
    return summary.sort_values(["mark", "symbol"]).reset_index(drop=True)


def print_report(summary: pd.DataFrame) -> None:
    print("# Rolling-extrema scanner")
    print(f"Window: {WINDOW} ticks, reset per day/product, inclusive of current tick.")
    print()
    print("| Mark | Product | Extrema buy % | Extrema sell % | Buy n | Sell n | Qualifies |")
    print("|---|---|---:|---:|---:|---:|---|")
    for row in summary.itertuples(index=False):
        qualifies = "YES" if row.qualifies else ""
        print(
            f"| {row.mark} | {row.symbol} | {row.pct_buy:.1f}% | {row.pct_sell:.1f}% | "
            f"{row.count_buy} | {row.count_sell} | {qualifies} |"
        )

    qualifiers = summary.loc[summary["qualifies"]]
    print()
    if qualifiers.empty:
        print("Qualified candidates (>60% on both buy-low and sell-high): none")
    else:
        print("Qualified candidates (>60% on both buy-low and sell-high):")
        for row in qualifiers.itertuples(index=False):
            print(
                f"- {row.mark} on {row.symbol}: "
                f"buy {row.pct_buy:.1f}% ({row.count_buy}), "
                f"sell {row.pct_sell:.1f}% ({row.count_sell})"
            )


def main() -> None:
    prices = load_prices()
    trades = load_trades()
    events = mark_trade_events(trades, prices)
    summary = summarize(events)
    print_report(summary)


if __name__ == "__main__":
    main()
