"""Expedition allocation optimizer - Nash equilibrium + behavioral prior.

Given a grid of spots with multipliers and pre-placed hunters, compute the
optimal 1-2 expedition allocation that maximizes expected payoff under a
blended Nash/behavioral model of where other players will go.

Usage:
    uv run manual/allocation.py             # run P2 R3 example
    # Or import solve() with your own grid
"""
from __future__ import annotations

import numpy as np


def nash_equilibrium(
    multipliers: list[float],
    hunters: list[float],
    base_treasure: float = 7500.0,
) -> tuple[float, np.ndarray]:
    """Compute Nash equilibrium shares via bisection on π*.

    In Nash equilibrium, all chosen spots yield equal expected payoff π*.
    For each spot i with share s_i > 0: base * M_i / (H_i + 100*s_i) = π*
    Rearranging: s_i = (base * M_i / π* - H_i) / 100 (if positive)

    Returns:
        (pi_star, shares) where shares sum to ~1.0
    """
    M = np.array(multipliers, dtype=float)
    H = np.array(hunters, dtype=float)

    def total_share(pi_star: float) -> float:
        shares = np.maximum(0, (base_treasure * M / pi_star - H) / 100)
        return float(shares.sum())

    # Bisect to find π* where shares sum to 1.0
    lo = 1.0
    hi = float(base_treasure * M.max())

    for _ in range(200):
        mid = (lo + hi) / 2
        if total_share(mid) > 1.0:
            lo = mid
        else:
            hi = mid

    pi_star = (lo + hi) / 2
    shares = np.maximum(0, (base_treasure * M / pi_star - H) / 100)

    # Normalize to exactly 1.0
    s_sum = shares.sum()
    if s_sum > 0:
        shares /= s_sum

    return pi_star, shares


def behavioral_shares(
    multipliers: list[float],
    hunters: list[float],
    exponent: float = 3.0,
    hunter_offset: float = 4.0,
) -> np.ndarray:
    """Behavioral prior: players disproportionately pick high M/(H+c) spots.

    Salience ∝ (M/(H+c))^exponent. Higher exponent = stronger bias toward
    "obviously good" spots (which are therefore overcrowded).
    """
    M = np.array(multipliers, dtype=float)
    H = np.array(hunters, dtype=float)
    salience = (M / (H + hunter_offset)) ** exponent
    return salience / salience.sum()


def solve(
    multipliers: list[float],
    hunters: list[float],
    base_treasure: float = 7500.0,
    fee_2nd: float = 25_000.0,
    fee_3rd: float = 75_000.0,
    nash_weight: float = 0.7,
    behavioral_exponent: float = 3.0,
    spot_names: list[str] | None = None,
) -> dict:
    """Solve expedition allocation game.

    Args:
        multipliers: M_i for each spot
        hunters: H_i (pre-placed hunters) for each spot
        base_treasure: payoff = base * M / (H + 100*share)
        fee_2nd: fee for 2nd expedition
        fee_3rd: fee for 3rd expedition
        nash_weight: weight for Nash prior (1 - nash_weight = behavioral weight)
        behavioral_exponent: exponent for salience model
        spot_names: optional names for spots

    Returns:
        dict with recommendations, EVs, and analysis
    """
    n = len(multipliers)
    M = np.array(multipliers, dtype=float)
    H = np.array(hunters, dtype=float)

    if spot_names is None:
        spot_names = [f"Spot_{i}" for i in range(n)]

    # Compute Nash and behavioral priors
    pi_star, shares_nash = nash_equilibrium(multipliers, hunters, base_treasure)
    shares_behav = behavioral_shares(multipliers, hunters, behavioral_exponent)

    # Blend
    shares = nash_weight * shares_nash + (1 - nash_weight) * shares_behav
    shares /= shares.sum()

    # Expected value per spot under blended prior
    ev = base_treasure * M / (H + 100 * shares)

    # Rank spots
    ranking = np.argsort(-ev)

    # Best single expedition
    best_1 = int(ranking[0])
    ev_1 = float(ev[best_1])

    # Best pair of expeditions
    best_2a = int(ranking[0])
    best_2b = int(ranking[1])
    ev_2 = float(ev[best_2a]) + float(ev[best_2b]) - fee_2nd

    # Best triple
    best_3c = int(ranking[2])
    ev_3 = ev_2 + float(ev[best_3c]) - fee_3rd

    # Recommendation
    if ev_3 > ev_2 and ev_3 > ev_1:
        rec_spots = [best_2a, best_2b, best_3c]
        rec_ev = ev_3
        rec_n = 3
    elif ev_2 > ev_1:
        rec_spots = [best_2a, best_2b]
        rec_ev = ev_2
        rec_n = 2
    else:
        rec_spots = [best_1]
        rec_ev = ev_1
        rec_n = 1

    return {
        "recommendation": {
            "spots": [spot_names[i] for i in rec_spots],
            "spot_indices": rec_spots,
            "n_expeditions": rec_n,
            "expected_profit": round(rec_ev, 0),
        },
        "spot_analysis": [
            {
                "name": spot_names[i],
                "index": i,
                "multiplier": float(M[i]),
                "hunters": float(H[i]),
                "nash_share_pct": round(float(shares_nash[i]) * 100, 2),
                "behavioral_share_pct": round(float(shares_behav[i]) * 100, 2),
                "blended_share_pct": round(float(shares[i]) * 100, 2),
                "expected_value": round(float(ev[i]), 0),
            }
            for i in ranking
        ],
        "ev_1_expedition": round(ev_1, 0),
        "ev_2_expeditions": round(ev_2, 0),
        "ev_3_expeditions": round(ev_3, 0),
        "nash_pi_star": round(pi_star, 2),
    }


def format_report(result: dict) -> str:
    """Format allocation result as readable report."""
    lines = ["# Expedition Allocation Report", ""]

    rec = result["recommendation"]
    lines.append(f"## Recommendation: {rec['n_expeditions']} expedition(s)")
    lines.append(f"**Spots**: {', '.join(rec['spots'])}")
    lines.append(f"**Expected profit**: {rec['expected_profit']:,.0f}")
    lines.append("")

    lines.append(f"Nash equilibrium payoff: {result['nash_pi_star']:,.0f}")
    lines.append(f"EV with 1 expedition: {result['ev_1_expedition']:,.0f}")
    lines.append(f"EV with 2 expeditions: {result['ev_2_expeditions']:,.0f}")
    lines.append(f"EV with 3 expeditions: {result['ev_3_expeditions']:,.0f}")
    lines.append("")

    lines.append("## Full Ranking")
    lines.append("")
    lines.append("| Spot | M | H | Nash% | Behav% | Blend% | EV |")
    lines.append("|------|---|---|-------|--------|--------|----|")
    for s in result["spot_analysis"]:
        lines.append(
            f"| {s['name']} | {s['multiplier']:.0f} | {s['hunters']:.0f} | "
            f"{s['nash_share_pct']:.1f} | {s['behavioral_share_pct']:.1f} | "
            f"{s['blended_share_pct']:.1f} | {s['expected_value']:,.0f} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Run P2 R3 example: 5×5 grid of expedition spots."""
    # P2 R3 data (flattened from 5×5 grid)
    multipliers = [
        24, 70, 41, 21, 60,
        47, 82, 87, 80, 35,
        73, 89, 100, 90, 17,
        77, 83, 85, 79, 55,
        12, 27, 52, 15, 30,
    ]
    hunters = [
        2, 4, 3, 2, 4,
        3, 5, 5, 5, 3,
        4, 5, 8, 7, 2,
        5, 5, 5, 5, 4,
        2, 3, 4, 2, 3,
    ]
    names = [f"({r},{c}) M={multipliers[r*5+c]}"
             for r in range(5) for c in range(5)]

    print("=" * 60)
    print("P2 R3 Expedition Allocation (5×5 grid)")
    print("=" * 60)
    print()

    result = solve(
        multipliers=multipliers,
        hunters=hunters,
        base_treasure=7500,
        fee_2nd=25_000,
        fee_3rd=75_000,
        nash_weight=0.7,
        behavioral_exponent=3.0,
        spot_names=names,
    )

    print(format_report(result))


if __name__ == "__main__":
    main()
