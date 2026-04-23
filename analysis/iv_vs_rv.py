from __future__ import annotations

import csv
import io
import math
import sys
from contextlib import redirect_stderr
from pathlib import Path
from statistics import fmean

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategies.bs_options import implied_vol

DATA_DIR = Path("data/round4")
REPORT_PATH = Path("analysis/output/iv_vs_rv.md")
DAYS = (1, 2, 3)
TRADING_DAYS_PER_YEAR = 252.0
TICKS_PER_DAY = 10_000
RV_SUBSAMPLE_STEP = 5


def option_tte_years(day: int, timestamp: int) -> float:
    days_remaining = max((6.0 - float(day)) - (timestamp / 1_000_000.0), 1.0 / 24.0)
    return days_remaining / 365.0


def underlying_mid(row: dict[str, str]) -> float:
    return float(row["mid_price"])


def voucher_mid(row: dict[str, str]) -> float:
    bids: list[tuple[int, int]] = []
    asks: list[tuple[int, int]] = []
    for level in (1, 2, 3):
        bid_price = row.get(f"bid_price_{level}", "")
        bid_volume = row.get(f"bid_volume_{level}", "")
        ask_price = row.get(f"ask_price_{level}", "")
        ask_volume = row.get(f"ask_volume_{level}", "")
        if bid_price and bid_volume:
            bids.append((int(float(bid_price)), int(float(bid_volume))))
        if ask_price and ask_volume:
            asks.append((int(float(ask_price)), int(float(ask_volume))))

    if bids and asks:
        best_bid, bid_volume = max(bids, key=lambda item: item[0])
        best_ask, ask_volume = min(asks, key=lambda item: item[0])
        if bid_volume > 0 and ask_volume > 0:
            return (
                (best_bid * bid_volume) + (best_ask * ask_volume)
            ) / float(bid_volume + ask_volume)
        return (best_bid + best_ask) / 2.0
    return float(row["mid_price"])


def load_day(day: int) -> dict[str, list[dict[str, str]]]:
    by_product: dict[str, list[dict[str, str]]] = {}
    path = DATA_DIR / f"prices_round_4_day_{day}.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            by_product.setdefault(row["product"], []).append(row)
    return by_product


def subsampled_daily_rv_annualized(mids: list[float]) -> float:
    log_prices = [math.log(price) for price in mids]
    path_variances: list[float] = []
    for offset in range(RV_SUBSAMPLE_STEP):
        sampled = log_prices[offset::RV_SUBSAMPLE_STEP]
        if len(sampled) < 2:
            continue
        realized_var = 0.0
        for idx in range(1, len(sampled)):
            ret = sampled[idx] - sampled[idx - 1]
            realized_var += ret * ret
        path_variances.append(realized_var)
    return math.sqrt(fmean(path_variances) * TRADING_DAYS_PER_YEAR)


def safe_implied_vol(price: float, spot: float, strike: int, tte_years: float) -> float | None:
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        iv = implied_vol(price, spot, float(strike), tte_years, 0.0, "call")
    if not math.isfinite(iv) or iv <= 0:
        return None
    return iv


def build_analysis() -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    regime_lines: list[str] = []

    for day in DAYS:
        day_rows = load_day(day)
        spot_rows = day_rows["VELVETFRUIT_EXTRACT"]
        spot_by_ts = {int(row["timestamp"]): underlying_mid(row) for row in spot_rows}
        rv = subsampled_daily_rv_annualized([underlying_mid(row) for row in spot_rows])

        symbols = sorted(
            (product for product in day_rows if product.startswith("VEV_")),
            key=lambda product: int(product.split("_", 1)[1]),
        )
        positives: list[int] = []

        for symbol in symbols:
            strike = int(symbol.split("_", 1)[1])
            ivs: list[float] = []
            valid_count = 0
            for row in day_rows[symbol]:
                timestamp = int(row["timestamp"])
                spot = spot_by_ts[timestamp]
                option_mid = voucher_mid(row)
                tte_years = option_tte_years(day, timestamp)
                iv = safe_implied_vol(option_mid, spot, strike, tte_years)
                if iv is None:
                    continue
                valid_count += 1
                ivs.append(iv)

            avg_iv = fmean(ivs) if ivs else float("nan")
            gap = rv - avg_iv if ivs else float("nan")
            if ivs and gap > 0:
                positives.append(strike)

            rows.append(
                {
                    "day": day,
                    "strike": strike,
                    "rv": rv,
                    "iv": avg_iv,
                    "gap": gap,
                    "rv_gt_iv": bool(ivs and gap > 0),
                    "valid": valid_count,
                    "total": len(day_rows[symbol]),
                }
            )

        if positives:
            regime_lines.append(
                f"- Day {day}: RV exceeded IV on {', '.join(strike_label(strike) for strike in positives)}."
            )
        else:
            regime_lines.append(f"- Day {day}: no strike had daily RV above daily IV.")

    return rows, regime_lines


def strike_label(strike: int) -> str:
    return f"VEV_{strike}"


def render_report(rows: list[dict[str, object]], regime_lines: list[str]) -> str:
    lines = [
        "# W1.5 Gamma",
        "",
        "## 1. IV vs RV Table",
        "",
        "| Day | Strike | RV | IV | RV-IV | RV>IV | Valid IV ticks |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        iv_text = "n/a" if math.isnan(row["iv"]) else f"{row['iv']:.4f}"
        gap_text = "n/a" if math.isnan(row["gap"]) else f"{row['gap']:+.4f}"
        lines.append(
            "| "
            f"{row['day']} | {row['strike']} | {row['rv']:.4f} | {iv_text} | {gap_text} | "
            f"{'yes' if row['rv_gt_iv'] else 'no'} | {row['valid']}/{row['total']} |"
        )

    lines.extend(
        [
            "",
            "## 1.5 RV Estimator Note",
            "",
            "RV uses 5-tick subsampled realized variance on `VELVETFRUIT_EXTRACT` mids: "
            "for offsets 0..4, take every 5th log-price, sum squared returns on each path, average "
            "the 5 path variances, then annualize with 252 trading days. This is more microstructure-robust "
            "than naive 1-tick RV because bid-ask bounce alternates quote mids at the touch and inflates "
            "high-frequency squared returns.",
            "",
            "TTE is inferred from the day file plus timestamp: day 1 starts near 5 calendar days to expiry, "
            "day 2 near 4, day 3 near 3, with linear intraday decay. IV uses a quote-derived voucher mid "
            "(best bid/ask, volume-weighted when both are present) because the raw CSV midpoint produces "
            "frequent no-arbitrage violations on deep ITM names.",
            "",
            "## 2. Regime Triggers Identified",
            "",
            *regime_lines,
            "",
            "## 3. Gamma Scalper Implementation + Integration Gate Description",
            "",
            "Pending strategy build step.",
            "",
            "## 4. 3-Day Backtest Delta vs v24",
            "",
            "Pending strategy build step.",
            "",
            "## 5. Gate Verdict and Recommendation",
            "",
            "Pending strategy decision.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rows, regime_lines = build_analysis()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(rows, regime_lines))
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
