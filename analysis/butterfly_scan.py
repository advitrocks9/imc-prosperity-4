from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROUND4_DAYS = (1, 2, 3)
ALL_STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
CENTERS = (4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000)
DELTAS = (100, 200, 500)
THRESHOLDS = (0.3, 0.5, 0.7)


def resolve_price_files(data_root: Path) -> tuple[str, list[Path]]:
    primary = [data_root / "round4" / f"prices_round_4_day_{day}.csv" for day in ROUND4_DAYS]
    if all(path.exists() for path in primary):
        return "data/round4", primary

    fallback = [data_root / "p3" / "round4" / f"prices_round_4_day_{day}.csv" for day in ROUND4_DAYS]
    if all(path.exists() for path in fallback):
        return "data/p3/round4", fallback

    return "", []


def valid_triples() -> list[tuple[int, int, int]]:
    strikes = set(ALL_STRIKES)
    triples: list[tuple[int, int, int]] = []
    for center in CENTERS:
        for delta in DELTAS:
            low = center - delta
            high = center + delta
            if low in strikes and high in strikes:
                triples.append((low, center, high))
    return triples


def load_day(path: Path) -> dict[int, dict[str, dict[str, str]]]:
    by_timestamp: dict[int, dict[str, dict[str, str]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            product = row["product"]
            if not product.startswith("VEV_"):
                continue
            timestamp = int(row["timestamp"])
            by_timestamp.setdefault(timestamp, {})[product] = row
    return by_timestamp


def butterfly_value(rows: dict[str, dict[str, str]], triple: tuple[int, int, int]) -> float | None:
    low, center, high = (rows.get(f"VEV_{strike}") for strike in triple)
    if low is None or center is None or high is None:
        return None
    return float(low["mid_price"]) - 2.0 * float(center["mid_price"]) + float(high["mid_price"])


def executable_cost(rows: dict[str, dict[str, str]], triple: tuple[int, int, int]) -> float | None:
    low, center, high = (rows.get(f"VEV_{strike}") for strike in triple)
    if low is None or center is None or high is None:
        return None
    if not low["ask_price_1"] or not center["bid_price_1"] or not high["ask_price_1"]:
        return None
    return float(low["ask_price_1"]) + float(high["ask_price_1"]) - 2.0 * float(center["bid_price_1"])


def summarize(paths: list[Path]) -> dict[str, object]:
    triples = valid_triples()
    by_triple: dict[tuple[int, int, int], dict[str, object]] = {
        triple: {
            "total_ticks": 0,
            "negative_ticks": 0,
            "negative_values": [],
            "threshold_hits": {str(threshold): 0 for threshold in THRESHOLDS},
            "successes": 0,
            "waits": [],
            "exec_costs_on_negative": [],
        }
        for triple in triples
    }

    for path in paths:
        day_rows = load_day(path)
        timestamps = sorted(day_rows)
        values_by_triple = {
            triple: [butterfly_value(day_rows[ts], triple) for ts in timestamps]
            for triple in triples
        }
        for triple in triples:
            stats = by_triple[triple]
            values = values_by_triple[triple]
            for index, value in enumerate(values):
                if value is None:
                    continue
                stats["total_ticks"] += 1
                if value >= 0:
                    continue
                stats["negative_ticks"] += 1
                stats["negative_values"].append(value)
                for threshold in THRESHOLDS:
                    if value < -threshold:
                        stats["threshold_hits"][str(threshold)] += 1
                exec_cost = executable_cost(day_rows[timestamps[index]], triple)
                if exec_cost is not None:
                    stats["exec_costs_on_negative"].append(exec_cost)
                for future in range(index + 1, len(values)):
                    next_value = values[future]
                    if next_value is not None and next_value >= 0:
                        stats["successes"] += 1
                        stats["waits"].append(future - index)
                        break

    rows: list[dict[str, object]] = []
    for triple in triples:
        stats = by_triple[triple]
        total_ticks = int(stats["total_ticks"])
        negative_ticks = int(stats["negative_ticks"])
        negative_values = list(stats["negative_values"])
        success_rate = (int(stats["successes"]) / negative_ticks) if negative_ticks else 0.0
        avg_negative = mean(negative_values) if negative_values else None
        row = {
            "triple": triple,
            "total_ticks": total_ticks,
            "negative_ticks": negative_ticks,
            "negative_pct": (100.0 * negative_ticks / total_ticks) if total_ticks else 0.0,
            "avg_negative": avg_negative,
            "success_rate": success_rate,
            "avg_wait_ticks": mean(stats["waits"]) if stats["waits"] else None,
            "pnl_score": (negative_ticks / total_ticks) * abs(avg_negative) * success_rate if total_ticks and avg_negative is not None else 0.0,
            "threshold_hits": stats["threshold_hits"],
            "avg_exec_cost_negative": mean(stats["exec_costs_on_negative"]) if stats["exec_costs_on_negative"] else None,
        }
        rows.append(row)

    rows.sort(key=lambda row: (row["pnl_score"], row["negative_ticks"]), reverse=True)
    return {"triples": rows}


def render_markdown(source: str, summary: dict[str, object]) -> str:
    def fmt(value: float | None, digits: int) -> str:
        if value is None:
            return "n/a"
        return f"{value:.{digits}f}"

    lines = [
        f"Source: `{source}`",
        "",
        "| Triple | Neg ticks | Neg % | Avg neg | Conv. success | Avg wait (ticks) | Score | Hits < -0.3/-0.5/-0.7 | Avg exec cost on neg |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["triples"]:
        threshold_hits = row["threshold_hits"]
        lines.append(
            "| "
            f"{row['triple'][0]}-{row['triple'][1]}-{row['triple'][2]} | "
            f"{row['negative_ticks']} | "
            f"{row['negative_pct']:.3f}% | "
            f"{fmt(row['avg_negative'], 3)} | "
            f"{100.0 * row['success_rate']:.1f}% | "
            f"{fmt(row['avg_wait_ticks'], 2)} | "
            f"{row['pnl_score']:.6f} | "
            f"{threshold_hits['0.3']}/{threshold_hits['0.5']}/{threshold_hits['0.7']} | "
            f"{fmt(row['avg_exec_cost_negative'], 2)} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan round-4 voucher butterflies on historical mids.")
    parser.add_argument("--data-root", default="data", help="Repository data root")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")
    args = parser.parse_args()

    source, paths = resolve_price_files(Path(args.data_root))
    if not paths:
        payload = {
            "source": None,
            "error": "No round-4 voucher price data found in data/round4 or data/p3/round4.",
        }
        print(json.dumps(payload, indent=2, sort_keys=True) if args.json else payload["error"])
        return 0

    summary = summarize(paths)
    if args.json:
        print(json.dumps({"source": source, **summary}, indent=2, sort_keys=True))
    else:
        print(render_markdown(source, summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
