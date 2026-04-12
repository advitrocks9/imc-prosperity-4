"""P3 → P4 data correlation tester.

Tests if Prosperity 3 price data correlates with Prosperity 4 data.
P1→P2 had R²=0.99, and exploiting this won 2nd place in P2.
If R² > 0.9 for any product pair, the P3 data becomes a look-ahead signal.

Usage:
    uv run analysis/correlation.py data/round1/ path/to/p3_data/
    uv run analysis/correlation.py data/round1/ path/to/p3_data/ --json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.data_loader import load_prices, get_product_mid


@dataclass
class CorrelationResult:
    """Correlation test result for a product pair."""
    p4_product: str
    p3_product: str
    n_aligned: int          # number of aligned ticks
    r_squared: float        # coefficient of determination
    pearson_r: float        # Pearson correlation
    scale_factor: float     # P4_price ≈ scale * P3_price + offset
    offset: float
    residual_std: float     # std of P4 - predicted
    exploitable: bool       # R² > 0.9
    notes: str = ""


def align_series(
    s1: np.ndarray,
    s2: np.ndarray,
    max_shift: int = 50,
) -> tuple[int, float]:
    """Find the optimal lag shift between two series using cross-correlation.

    Returns (best_shift, best_r²). Positive shift means s2 is delayed.
    """
    n = min(len(s1), len(s2))
    if n < 100:
        return 0, 0.0

    s1 = s1[:n]
    s2 = s2[:n]

    # Normalize
    s1_norm = (s1 - np.mean(s1)) / (np.std(s1) + 1e-10)
    s2_norm = (s2 - np.mean(s2)) / (np.std(s2) + 1e-10)

    best_shift = 0
    best_r2 = 0.0

    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            a = s1_norm[shift:]
            b = s2_norm[:len(a)]
        else:
            b = s2_norm[-shift:]
            a = s1_norm[:len(b)]

        if len(a) < 100:
            continue

        r = np.corrcoef(a, b)[0, 1]
        r2 = r ** 2
        if r2 > best_r2:
            best_r2 = r2
            best_shift = shift

    return best_shift, best_r2


def test_correlation(
    p4_mid: np.ndarray,
    p3_mid: np.ndarray,
    p4_name: str,
    p3_name: str,
) -> CorrelationResult:
    """Test if P3 product data predicts P4 product data."""
    n = min(len(p4_mid), len(p3_mid))

    if n < 50:
        return CorrelationResult(
            p4_product=p4_name, p3_product=p3_name,
            n_aligned=n, r_squared=0, pearson_r=0,
            scale_factor=0, offset=0, residual_std=0,
            exploitable=False, notes="Too few aligned ticks"
        )

    # Try direct alignment first (same tick indices)
    y = p4_mid[:n]
    x = p3_mid[:n]

    # Find best shift
    best_shift, shift_r2 = align_series(y, x)

    # Apply shift
    if best_shift >= 0:
        y_aligned = y[best_shift:]
        x_aligned = x[:len(y_aligned)]
    else:
        x_aligned = x[-best_shift:]
        y_aligned = y[:len(x_aligned)]

    n_aligned = len(y_aligned)

    # OLS: P4 = scale * P3 + offset
    X = np.column_stack([x_aligned, np.ones(n_aligned)])
    try:
        coeffs, residuals, _, _ = np.linalg.lstsq(X, y_aligned, rcond=None)
        scale = float(coeffs[0])
        offset = float(coeffs[1])
    except Exception:
        return CorrelationResult(
            p4_product=p4_name, p3_product=p3_name,
            n_aligned=n_aligned, r_squared=0, pearson_r=0,
            scale_factor=0, offset=0, residual_std=0,
            exploitable=False, notes="OLS failed"
        )

    y_pred = X @ coeffs
    ss_res = np.sum((y_aligned - y_pred) ** 2)
    ss_tot = np.sum((y_aligned - np.mean(y_aligned)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    pearson = float(np.corrcoef(y_aligned, x_aligned)[0, 1])
    resid_std = float(np.std(y_aligned - y_pred))

    notes_parts = []
    if best_shift != 0:
        notes_parts.append(f"Optimal shift={best_shift} ticks")
    if r2 > 0.99:
        notes_parts.append("NEAR-PERFECT MATCH - strong look-ahead signal")
    elif r2 > 0.9:
        notes_parts.append("STRONG CORRELATION - exploitable with noise model")

    return CorrelationResult(
        p4_product=p4_name,
        p3_product=p3_name,
        n_aligned=n_aligned,
        r_squared=round(float(r2), 6),
        pearson_r=round(pearson, 6),
        scale_factor=round(scale, 6),
        offset=round(offset, 4),
        residual_std=round(resid_std, 4),
        exploitable=r2 > 0.9,
        notes="; ".join(notes_parts),
    )


def run_correlation(
    p4_dir: Path | str,
    p3_dir: Path | str,
    product_map: dict[str, str] | None = None,
) -> list[CorrelationResult]:
    """Test all P4 products against P3 data.

    If product_map is None, tries to match by name and by brute-force
    all-pairs correlation.

    Args:
        p4_dir: Path to P4 round data
        p3_dir: Path to P3 round data
        product_map: Optional {p4_product: p3_product} mapping
    """
    p4_prices = load_prices(p4_dir)
    p3_prices = load_prices(p3_dir)

    p4_products = sorted(p4_prices["product"].unique())
    p3_products = sorted(p3_prices["product"].unique())

    results: list[CorrelationResult] = []

    if product_map:
        # Test specified pairs
        for p4_name, p3_name in product_map.items():
            p4_mid = get_product_mid(p4_prices, p4_name)["mid_price"].values
            p3_mid = get_product_mid(p3_prices, p3_name)["mid_price"].values
            results.append(test_correlation(p4_mid, p3_mid, p4_name, p3_name))
    else:
        # Brute-force: test all P4 × P3 pairs
        for p4_name in p4_products:
            p4_mid = get_product_mid(p4_prices, p4_name)["mid_price"].values
            best_result: CorrelationResult | None = None

            for p3_name in p3_products:
                p3_mid = get_product_mid(p3_prices, p3_name)["mid_price"].values
                result = test_correlation(p4_mid, p3_mid, p4_name, p3_name)

                if best_result is None or result.r_squared > best_result.r_squared:
                    best_result = result

            if best_result is not None:
                results.append(best_result)

    return results


def format_report(results: list[CorrelationResult]) -> str:
    """Format correlation test results as readable report."""
    lines = ["# P3 → P4 Correlation Report", ""]

    # Summary
    exploitable = [r for r in results if r.exploitable]
    if exploitable:
        lines.append(f"## ⚠ EXPLOITABLE CORRELATIONS FOUND: {len(exploitable)}")
        lines.append("")
        for r in exploitable:
            lines.append(f"- **{r.p4_product}** ↔ **{r.p3_product}**: R²={r.r_squared:.4f}")
        lines.append("")
    else:
        lines.append("## No exploitable correlations (R² > 0.9) found")
        lines.append("")

    # Full table
    lines.append("| P4 Product | P3 Match | R² | Pearson r | Scale | Offset | Residual σ | Notes |")
    lines.append("|------------|----------|-----|-----------|-------|--------|------------|-------|")
    for r in sorted(results, key=lambda x: -x.r_squared):
        lines.append(
            f"| {r.p4_product} | {r.p3_product} | {r.r_squared:.4f} | "
            f"{r.pearson_r:.4f} | {r.scale_factor:.4f} | {r.offset:.1f} | "
            f"{r.residual_std:.2f} | {r.notes} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P3→P4 data correlation tester")
    parser.add_argument("p4_dir", type=Path, help="Path to P4 round data")
    parser.add_argument("p3_dir", type=Path, help="Path to P3 round data")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--map", type=str, default=None,
        help='Product mapping as JSON: \'{"P4_NAME": "P3_NAME"}\''
    )
    args = parser.parse_args()

    product_map = json.loads(args.map) if args.map else None
    results = run_correlation(args.p4_dir, args.p3_dir, product_map)

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print(format_report(results))


if __name__ == "__main__":
    main()
