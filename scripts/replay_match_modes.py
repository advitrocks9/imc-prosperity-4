#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from replay_backtest import parse_limits, run_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("portal_log_path")
    parser.add_argument("algorithm")
    parser.add_argument("--limits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limits = parse_limits(args.limits)
    none_summary = run_replay(args.portal_log_path, args.algorithm, match_mode="none", limits=limits, save_json=False)
    all_summary = run_replay(args.portal_log_path, args.algorithm, match_mode="all", limits=limits, save_json=False)

    algo_name = Path(args.algorithm).stem
    log_name = Path(args.portal_log_path).name

    print(f"log:  {log_name}")
    print(f"algo: {algo_name}")
    print(f"{'product':24} {'none':>10} {'all':>10} {'delta':>10}")

    products = sorted(set(none_summary["per_product_pnl"]) | set(all_summary["per_product_pnl"]))
    for product in products:
        none_pnl = none_summary["per_product_pnl"].get(product, 0)
        all_pnl = all_summary["per_product_pnl"].get(product, 0)
        print(f"{product:24} {none_pnl:>10,} {all_pnl:>10,} {all_pnl - none_pnl:>10,}")

    total_none = none_summary["total_pnl"]
    total_all = all_summary["total_pnl"]
    print(f"{'TOTAL':24} {total_none:>10,} {total_all:>10,} {total_all - total_none:>10,}")


if __name__ == "__main__":
    main()
