"""Two-bid reserve price auction optimizer.

You submit two bids (p_low, p_high). Sellers with reserve price R accept
the lowest of your bids that exceeds R. You resell at fixed price V.
Supports optional coordination penalty (average-bid games).

Usage:
    uv run manual/auction.py              # run P2 R1 example
    # Or import solve() with your own CDF
"""
from __future__ import annotations

from typing import Callable


def solve(
    V: float,
    cdf: Callable[[float], float],
    lo: int,
    hi: int,
    n_sellers: int = 1,
    avg_high: float | None = None,
    penalty_power: float = 1.0,
) -> tuple[tuple[int, int], float]:
    """Find optimal (p_low, p_high) by grid search.

    Args:
        V: resale price (revenue per unit)
        cdf: CDF of seller reserve prices - F(r) = P(R ≤ r)
        lo: minimum bid value
        hi: maximum bid value
        n_sellers: number of sellers (scales expected profit)
        avg_high: if set, applies coordination penalty on p_high < avg_high
        penalty_power: exponent for the penalty (1=linear, 3=cubic)

    Returns:
        ((p_low, p_high), expected_total_profit)
    """
    best_val = float("-inf")
    best_pair = (lo, lo)

    for p_low in range(lo, hi + 1):
        f_low = cdf(p_low)
        for p_high in range(p_low, hi + 1):
            f_high = cdf(p_high)

            # Expected profit per seller
            ev_low = (V - p_low) * f_low
            ev_high = (V - p_high) * (f_high - f_low)

            # Coordination penalty on the high bid
            if avg_high is not None and p_high < avg_high and p_high < V:
                penalty = ((V - avg_high) / (V - p_high)) ** penalty_power
                ev_high *= penalty

            ev = (ev_low + ev_high) * n_sellers

            if ev > best_val:
                best_val = ev
                best_pair = (p_low, p_high)

    return best_pair, best_val


def solve_fine(
    V: float,
    cdf: Callable[[float], float],
    lo: float,
    hi: float,
    step: float = 0.1,
    n_sellers: int = 1,
    avg_high: float | None = None,
    penalty_power: float = 1.0,
) -> tuple[tuple[float, float], float]:
    """Fine-grained search with fractional step (for non-integer bid spaces)."""
    best_val = float("-inf")
    best_pair = (lo, lo)

    p_low = lo
    while p_low <= hi:
        f_low = cdf(p_low)
        p_high = p_low
        while p_high <= hi:
            f_high = cdf(p_high)

            ev_low = (V - p_low) * f_low
            ev_high = (V - p_high) * (f_high - f_low)

            if avg_high is not None and p_high < avg_high and p_high < V:
                penalty = ((V - avg_high) / (V - p_high)) ** penalty_power
                ev_high *= penalty

            ev = (ev_low + ev_high) * n_sellers

            if ev > best_val:
                best_val = ev
                best_pair = (round(p_low, 2), round(p_high, 2))

            p_high += step
        p_low += step

    return best_pair, best_val


def triangular_cdf(lo: float, hi: float) -> Callable[[float], float]:
    """CDF for triangular distribution on [lo, hi] with mode at hi.

    f(r) = 2(r - lo) / (hi - lo)^2, F(r) = (r - lo)^2 / (hi - lo)^2
    """
    width_sq = (hi - lo) ** 2

    def cdf(r: float) -> float:
        if r <= lo:
            return 0.0
        if r >= hi:
            return 1.0
        return (r - lo) ** 2 / width_sq

    return cdf


def uniform_cdf(lo: float, hi: float) -> Callable[[float], float]:
    """CDF for uniform distribution on [lo, hi]."""
    width = hi - lo

    def cdf(r: float) -> float:
        if r <= lo:
            return 0.0
        if r >= hi:
            return 1.0
        return (r - lo) / width

    return cdf


def bimodal_cdf(
    lo1: float, hi1: float,
    lo2: float, hi2: float,
    weight1: float = 0.5,
) -> Callable[[float], float]:
    """CDF for bimodal uniform: weight1 on [lo1, hi1], (1-weight1) on [lo2, hi2]."""
    w2 = 1.0 - weight1

    def cdf(r: float) -> float:
        p = 0.0
        # First cluster
        if r >= hi1:
            p += weight1
        elif r > lo1:
            p += weight1 * (r - lo1) / (hi1 - lo1)
        # Second cluster
        if r >= hi2:
            p += w2
        elif r > lo2:
            p += w2 * (r - lo2) / (hi2 - lo2)
        return p

    return cdf


def main() -> None:
    print("=" * 50)
    print("P2 R1: Triangular [900, 1000], V=1000, N=5000")
    print("=" * 50)

    cdf_tri = triangular_cdf(900, 1000)
    pair, ev = solve(V=1000, cdf=cdf_tri, lo=900, hi=1000, n_sellers=5000)
    print(f"  Optimal bids: ({pair[0]}, {pair[1]})")
    print(f"  Expected profit: {ev:,.0f}")
    print()

    print("=" * 50)
    print("P2 R4: Same + coordination penalty (avg_high=978)")
    print("=" * 50)

    pair2, ev2 = solve(V=1000, cdf=cdf_tri, lo=900, hi=1000, n_sellers=5000,
                       avg_high=978, penalty_power=1.0)
    print(f"  Optimal bids: ({pair2[0]}, {pair2[1]})")
    print(f"  Expected profit: {ev2:,.0f}")
    print()

    print("=" * 50)
    print("P3 R3: Bimodal [160,200]∪[250,320], V=320, N=5000, cubic penalty")
    print("=" * 50)

    cdf_bi = bimodal_cdf(160, 200, 250, 320, weight1=0.5)
    pair3, ev3 = solve(V=320, cdf=cdf_bi, lo=160, hi=320, n_sellers=5000)
    print(f"  No penalty:      ({pair3[0]}, {pair3[1]})  EV={ev3:,.0f}")

    pair3c, ev3c = solve(V=320, cdf=cdf_bi, lo=160, hi=320, n_sellers=5000,
                         avg_high=285, penalty_power=3.0)
    print(f"  Cubic avg=285:   ({pair3c[0]}, {pair3c[1]})  EV={ev3c:,.0f}")


if __name__ == "__main__":
    main()
