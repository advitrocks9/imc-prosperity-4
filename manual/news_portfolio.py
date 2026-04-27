"""Quadratic-fee portfolio optimizer for news-based manual challenges.

Given return estimates per product and a quadratic fee structure,
finds the integer allocation that maximizes expected profit.

Uses KKT conditions with bisection for the Lagrange multiplier - no cvxpy.

Usage:
    uv run manual/news_portfolio.py          # run P2 R5 example
    # Or import solve() with your own returns
"""
from __future__ import annotations

import math


def solve(
    product_names: list[str],
    returns: list[float],
    capital: float = 750_000,
    budget: int = 100,
    fee_coeff: float = 90.0,
    kelly_fraction: float = 1.0,
) -> dict:
    """Solve news portfolio allocation.

    Maximizes: Σ_i [capital/100 * r_i * π_i - fee_coeff * π_i²]
    Subject to: Σ_i |π_i| ≤ budget, π_i ∈ Z

    Args:
        product_names: names for each product
        returns: estimated return per product (can be negative = short)
        capital: total capital in seashells
        budget: max sum of |π_i|
        fee_coeff: quadratic fee coefficient per unit squared
        kelly_fraction: confidence scaling (0.5 = half Kelly, 1.0 = full)

    Returns:
        dict with allocations, per-product EV, and total EV
    """
    n = len(returns)
    scale = capital / 100.0  # scale factor for returns

    # Apply Kelly fraction to returns (shrink toward 0 when uncertain)
    r = [ret * kelly_fraction for ret in returns]

    # KKT: optimal continuous allocation for product i is:
    #   π_i = (scale * r_i - λ * sign(r_i)) / (2 * fee_coeff)  if |scale * r_i| > λ
    #   π_i = 0  otherwise
    #
    # Find λ via bisection such that Σ|π_i| = budget

    def compute_allocations(lam: float) -> list[float]:
        allocs = []
        for r_i in r:
            marginal = scale * r_i
            if abs(marginal) > lam:
                sign = 1.0 if r_i > 0 else -1.0
                pi_i = (marginal - lam * sign) / (2 * fee_coeff)
                allocs.append(pi_i)
            else:
                allocs.append(0.0)
        return allocs

    def total_abs(lam: float) -> float:
        return sum(abs(p) for p in compute_allocations(lam))

    # Bisect for λ where total allocation = budget
    lo_lam = 0.0
    hi_lam = scale * max(abs(r_i) for r_i in r) + 1.0

    # Check if unconstrained solution already fits
    if total_abs(0.0) <= budget:
        lam_star = 0.0
    else:
        for _ in range(200):
            mid = (lo_lam + hi_lam) / 2
            if total_abs(mid) > budget:
                lo_lam = mid
            else:
                hi_lam = mid
        lam_star = (lo_lam + hi_lam) / 2

    # Continuous optimal
    pi_cont = compute_allocations(lam_star)

    # Round to integers
    pi_int = [round(p) for p in pi_cont]

    # Fix budget violation from rounding
    while sum(abs(p) for p in pi_int) > budget:
        # Reduce the position with smallest absolute value
        nonzero = [i for i, p in enumerate(pi_int) if p != 0]
        if not nonzero:
            break
        idx = min(nonzero, key=lambda i: abs(pi_int[i]))
        if pi_int[idx] > 0:
            pi_int[idx] -= 1
        else:
            pi_int[idx] += 1

    # Compute expected profits
    total_ev = 0.0
    details = []
    for i in range(n):
        pi_i = pi_int[i]
        gross = scale * returns[i] * pi_i
        fee = fee_coeff * pi_i * pi_i
        net = gross - fee
        total_ev += net
        details.append({
            "product": product_names[i],
            "return_est": returns[i],
            "allocation": pi_i,
            "gross_profit": round(gross, 0),
            "fee": round(fee, 0),
            "net_profit": round(net, 0),
        })

    # Sort by absolute allocation
    details.sort(key=lambda x: -abs(x["allocation"]))

    return {
        "allocations": {d["product"]: d["allocation"] for d in details},
        "total_budget_used": sum(abs(p) for p in pi_int),
        "total_expected_profit": round(total_ev, 0),
        "details": details,
        "lambda_star": round(lam_star, 2),
    }


def format_report(result: dict) -> str:
    """Format portfolio result as readable report."""
    lines = ["# News Portfolio Allocation", ""]
    lines.append(f"**Total EV**: {result['total_expected_profit']:,.0f}")
    lines.append(f"**Budget used**: {result['total_budget_used']}/{100}")
    lines.append(f"**Lambda (shadow price)**: {result['lambda_star']}")
    lines.append("")

    lines.append("| Product | Return | Alloc | Gross | Fee | Net |")
    lines.append("|---------|--------|-------|-------|-----|-----|")
    for d in result["details"]:
        lines.append(
            f"| {d['product']} | {d['return_est']:+.2f} | {d['allocation']:+d} | "
            f"{d['gross_profit']:+,.0f} | {d['fee']:,.0f} | {d['net_profit']:+,.0f} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Run P2 R5 example."""
    print("=" * 60)
    print("P2 R5 News Portfolio (Iceberg newspaper)")
    print("=" * 60)
    print()

    # Sentiment-to-return estimates (from P2 R5 analysis)
    products = [
        "Serum", "Blankets", "PS6", "Earrings", "Sculptures",
        "Sleds", "Refrigerators", "Lamps", "Chocolate",
    ]
    returns = [
        -0.40,   # Serum: strong negative (shortage story)
        -0.25,   # Blankets: moderate negative
        +0.15,   # PS6: moderate positive
        +0.10,   # Earrings: slight positive
        +0.10,   # Sculptures: slight positive
        -0.08,   # Sleds: slight negative
        +0.05,   # Refrigerators: weak positive
        +0.05,   # Lamps: weak positive
        -0.05,   # Chocolate: weak negative
    ]

    # Full Kelly
    result = solve(products, returns, capital=750_000, budget=100, fee_coeff=90)
    print(format_report(result))

    # Half Kelly (conservative)
    print("--- Half Kelly (conservative) ---")
    result_half = solve(products, returns, capital=750_000, budget=100,
                        fee_coeff=90, kelly_fraction=0.5)
    print(format_report(result_half))


if __name__ == "__main__":
    main()
