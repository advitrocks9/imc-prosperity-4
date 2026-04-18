"""Bellman-Ford currency arbitrage solver.

Finds the most profitable currency conversion sequence given an N×N exchange
rate matrix and a maximum number of intermediate trades.

Uses DP in log-space: maximizing product of rates = maximizing sum of log(rates).

Usage:
    uv run manual/fx_arb.py                  # run P2 R2 example
    # Or import and call solve() with your own rates
"""
from __future__ import annotations

import math


def solve(
    rates: list[list[float]],
    max_trades: int = 5,
    start: int = 0,
) -> tuple[float, list[int]]:
    """Find the most profitable currency conversion sequence.

    Args:
        rates: N×N matrix where rates[i][j] = units of j per 1 unit of i
        max_trades: maximum number of intermediate trades (hops)
        start: starting/ending currency index

    Returns:
        (profit_multiplier, path) where path is [start, c1, c2, ..., start]
        profit_multiplier > 1.0 means profitable.
    """
    n = len(rates)
    NEG_INF = float("-inf")

    # dp[k][v] = best log-gain to reach currency v in exactly k trades from start
    dp = [[NEG_INF] * n for _ in range(max_trades + 1)]
    parent = [[(-1, -1)] * n for _ in range(max_trades + 1)]
    dp[0][start] = 0.0

    for k in range(max_trades):
        for u in range(n):
            if dp[k][u] == NEG_INF:
                continue
            for v in range(n):
                if rates[u][v] <= 0:
                    continue
                gain = dp[k][u] + math.log(rates[u][v])
                if gain > dp[k + 1][v]:
                    dp[k + 1][v] = gain
                    parent[k + 1][v] = (k, u)

    # Find best return to start across all hop counts
    best_gain = NEG_INF
    best_k = 0
    for k in range(1, max_trades + 1):
        if dp[k][start] > best_gain:
            best_gain = dp[k][start]
            best_k = k

    # Reconstruct path
    path = [start]
    node = start
    k = best_k
    while k > 0:
        prev_k, prev_node = parent[k][node]
        path.append(prev_node)
        node = prev_node
        k = prev_k
    path.reverse()

    return math.exp(best_gain), path


def solve_all_starts(
    rates: list[list[float]],
    max_trades: int = 5,
) -> tuple[float, list[int], int]:
    """Try all starting currencies, return the best."""
    best_mult = 0.0
    best_path: list[int] = []
    best_start = 0

    for start in range(len(rates)):
        mult, path = solve(rates, max_trades, start)
        if mult > best_mult:
            best_mult = mult
            best_path = path
            best_start = start

    return best_mult, best_path, best_start


def format_path(path: list[int], currency_names: list[str] | None = None) -> str:
    """Format a path as a readable string."""
    if currency_names:
        return " → ".join(currency_names[i] for i in path)
    return " → ".join(str(i) for i in path)


def main() -> None:
    """Run P2 R2 example: Shell, Pizza, Wasabi, Snowball."""
    currencies = ["Shell", "Pizza", "Wasabi", "Snowball"]

    rates = [
        [1.00, 1.41, 0.61, 2.08],
        [0.71, 1.00, 0.48, 1.52],
        [1.56, 2.05, 1.00, 3.26],
        [0.46, 0.64, 0.30, 1.00],
    ]

    print("=" * 50)
    print("P2 R2 Currency Arbitrage")
    print("=" * 50)
    print()

    for max_t in range(1, 7):
        mult, path = solve(rates, max_trades=max_t, start=0)
        pct = (mult - 1) * 100
        path_str = format_path(path, currencies)
        print(f"  Max {max_t} trades: {pct:+.2f}%  {path_str}")

    print()
    mult, path, start = solve_all_starts(rates, max_trades=5)
    print(f"Best from any start: {format_path(path, currencies)}")
    print(f"Return: {mult:.6f}x ({(mult-1)*100:+.2f}%)")


if __name__ == "__main__":
    main()
