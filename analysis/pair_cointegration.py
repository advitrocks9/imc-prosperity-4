"""
Pair-cointegration scan.

Loads three days of price files, scores every product pair on correlation and
residual stationarity, and writes a compact markdown summary of the top pairs.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path
from statistics import mean

DATA_DIR = Path("data/round4")
OUTPUT_PATH = Path("analysis/output/pair_cointegration.md")
DAYS = (1, 2, 3)
TOP_K = 10
VOUCHER_PREFIX = "VEV_"
WINDOW = 200


def load_mid_prices() -> dict[str, list[float]]:
    by_product: dict[str, list[float]] = defaultdict(list)
    for day in DAYS:
        path = DATA_DIR / f"prices_round_4_day_{day}.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            for row in reader:
                by_product[row["product"]].append(float(row["mid_price"]))
    return dict(sorted(by_product.items()))


def variance(xs: list[float]) -> float:
    if not xs:
        return 0.0
    mu = mean(xs)
    return sum((x - mu) * (x - mu) for x in xs) / len(xs)


def covariance(xs: list[float], ys: list[float]) -> float:
    if not xs or not ys or len(xs) != len(ys):
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)


def correlation(xs: list[float], ys: list[float]) -> float:
    vx = variance(xs)
    vy = variance(ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    return covariance(xs, ys) / math.sqrt(vx * vy)


def ols_fit(y: list[float], x: list[float]) -> tuple[float, float]:
    x_var = variance(x)
    y_mean = mean(y)
    if x_var <= 1e-12:
        return 0.0, y_mean
    beta = covariance(x, y) / x_var
    alpha = y_mean - beta * mean(x)
    return beta, alpha


def ar1_with_intercept(y: list[float], x: list[float]) -> tuple[float, float, float]:
    n = len(x)
    if n < 3:
        return 0.0, 0.0, float("nan")

    sx = sum(x)
    sy = sum(y)
    sxx = sum(v * v for v in x)
    sxy = sum(a * b for a, b in zip(x, y))

    det = n * sxx - sx * sx
    if abs(det) <= 1e-12:
        intercept = sy / n
        return intercept, 0.0, float("nan")

    intercept = (sy * sxx - sx * sxy) / det
    slope = (n * sxy - sx * sy) / det

    residuals = [yt - (intercept + slope * xt) for xt, yt in zip(x, y)]
    dof = n - 2
    if dof <= 0:
        return intercept, slope, float("nan")

    sse = sum(err * err for err in residuals)
    sigma2 = sse / dof
    slope_var = sigma2 * n / det
    if slope_var <= 1e-18:
        return intercept, slope, float("nan")
    t_stat = slope / math.sqrt(slope_var)
    return intercept, slope, t_stat


def fit_adf_like(residuals: list[float]) -> tuple[float, float, bool]:
    if len(residuals) < 5:
        return 0.0, float("nan"), False
    lagged = residuals[:-1]
    delta = [residuals[i + 1] - residuals[i] for i in range(len(residuals) - 1)]
    _, coeff, t_stat = ar1_with_intercept(delta, lagged)
    passes = coeff < 0.0 and t_stat < -2.9
    return coeff, t_stat, passes


def ou_half_life(residuals: list[float]) -> float:
    if len(residuals) < 3:
        return float("inf")
    prev = residuals[:-1]
    curr = residuals[1:]
    _, theta, _ = ar1_with_intercept(curr, prev)
    if theta <= 0.0 or theta >= 1.0:
        return float("inf")
    return -math.log(2.0) / math.log(theta)


def lag1_autocorr(xs: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    return correlation(xs[:-1], xs[1:])


def rolling_z_stability(values: list[float], window: int = WINDOW) -> float:
    if len(values) < window:
        return 0.0

    buf: deque[float] = deque()
    running_sum = 0.0
    running_sq = 0.0
    z_values: list[float] = []

    for value in values:
        buf.append(value)
        running_sum += value
        running_sq += value * value
        if len(buf) > window:
            old = buf.popleft()
            running_sum -= old
            running_sq -= old * old
        if len(buf) < window:
            continue
        window_mean = running_sum / window
        variance_val = (running_sq - (running_sum * running_sum) / window) / max(window - 1, 1)
        if variance_val <= 1e-12:
            continue
        z_values.append((value - window_mean) / math.sqrt(variance_val))

    if len(z_values) < 2:
        return 0.0

    z_mean_abs = sum(abs(v) for v in z_values) / len(z_values)
    z_std = math.sqrt(variance(z_values))
    return 1.0 / (1.0 + abs(z_mean_abs) + abs(z_std - 1.0))


def stationarity_score(t_stat: float, z_score_stability: float) -> float:
    if not math.isfinite(t_stat):
        return 0.0
    adf_strength = max(0.0, (-t_stat - 2.9) / 2.9)
    return adf_strength + z_score_stability


def is_voucher(product: str) -> bool:
    return product.startswith(VOUCHER_PREFIX)


def analyze_pairs(by_product: dict[str, list[float]]) -> list[dict[str, float | str | bool]]:
    rows: list[dict[str, float | str | bool]] = []
    products = sorted(by_product)
    for product_a, product_b in combinations(products, 2):
        a = by_product[product_a]
        b = by_product[product_b]
        n = min(len(a), len(b))
        if n < WINDOW + 5:
            continue
        a = a[:n]
        b = b[:n]
        corr = correlation(a, b)
        beta, alpha = ols_fit(a, b)
        residuals = [av - (beta * bv + alpha) for av, bv in zip(a, b)]
        adf_coeff, adf_t, adf_pass = fit_adf_like(residuals)
        half_life = ou_half_life(residuals)
        residual_changes = [residuals[i + 1] - residuals[i] for i in range(len(residuals) - 1)]
        diff_autocorr = lag1_autocorr(residual_changes)
        z_stability = rolling_z_stability(residuals)
        stat_score = stationarity_score(adf_t, z_stability)
        combined = abs(corr) * stat_score
        voucher_pair = is_voucher(product_a) and is_voucher(product_b)
        tradeable = (not voucher_pair) and adf_pass and half_life < 500.0 and diff_autocorr < -0.05

        rows.append(
            {
                "pair": f"{product_a} vs {product_b}",
                "product_a": product_a,
                "product_b": product_b,
                "corr": corr,
                "beta": beta,
                "alpha": alpha,
                "adf_coeff": adf_coeff,
                "adf_t": adf_t,
                "adf_pass": adf_pass,
                "half_life": half_life,
                "lag1_diff_autocorr": diff_autocorr,
                "z_stability": z_stability,
                "stationarity_score": stat_score,
                "combined_score": combined,
                "voucher_pair": voucher_pair,
                "tradeable": tradeable,
            }
        )

    rows.sort(key=lambda row: float(row["combined_score"]), reverse=True)
    return rows


def format_num(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "inf"
    return f"{value:.{digits}f}"


def write_top10_markdown(rows: list[dict[str, float | str | bool]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    top_rows = rows[:TOP_K]
    lines = [
        "# W1.2 Pair Cointegration",
        "",
        "## 1. Cointegration scores table top-10 pairs",
        "",
        "| Rank | Pair | Corr | Beta | ADF t | Half-life | Lag-1 dResid AC | Score | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for idx, row in enumerate(top_rows, start=1):
        notes: list[str] = []
        if bool(row["voucher_pair"]):
            notes.append("voucher-only; non-tradeable")
        if bool(row["adf_pass"]):
            notes.append("ADF pass")
        if bool(row["tradeable"]):
            notes.append("gate pass")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row["pair"]),
                    format_num(float(row["corr"])),
                    format_num(float(row["beta"])),
                    format_num(float(row["adf_t"])),
                    format_num(float(row["half_life"]), 1),
                    format_num(float(row["lag1_diff_autocorr"])),
                    format_num(float(row["combined_score"])),
                    ", ".join(notes) if notes else "-",
                ]
            )
            + " |"
        )
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="ascii")


def print_summary(rows: list[dict[str, float | str | bool]]) -> None:
    print(f"Loaded {len(rows)} scored pairs across days {DAYS}.")
    print()
    print("Top 10 pairs by |corr| x stationarity_score:")
    for idx, row in enumerate(rows[:TOP_K], start=1):
        flags: list[str] = []
        if bool(row["voucher_pair"]):
            flags.append("voucher-only")
        if bool(row["tradeable"]):
            flags.append("tradeable")
        elif bool(row["adf_pass"]):
            flags.append("ADF-pass")
        print(
            f"{idx:2d}. {row['pair']}"
            f" | corr={float(row['corr']):.4f}"
            f" | beta={float(row['beta']):.4f}"
            f" | adf_t={float(row['adf_t']):.3f}"
            f" | half_life={format_num(float(row['half_life']), 1)}"
            f" | lag1_dresid={float(row['lag1_diff_autocorr']):.3f}"
            f" | score={float(row['combined_score']):.3f}"
            f" | {'; '.join(flags) if flags else 'no gate'}"
        )

    tradeable = [row for row in rows if bool(row["tradeable"])]
    print()
    if tradeable:
        best = tradeable[0]
        print(
            "Best non-voucher gate-passing pair:"
            f" {best['pair']} | beta={float(best['beta']):.4f}"
            f" | half_life={format_num(float(best['half_life']), 1)}"
            f" | lag1_dresid={float(best['lag1_diff_autocorr']):.3f}"
            f" | adf_t={float(best['adf_t']):.3f}"
        )
    else:
        print("No non-voucher pair passed the trade gate.")


def main() -> None:
    by_product = load_mid_prices()
    rows = analyze_pairs(by_product)
    write_top10_markdown(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
