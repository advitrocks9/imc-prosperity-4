"""Automated round-opening pipeline.

When a new round opens, run this to:
1. Load CSV data from data/roundN/
2. Run EDA (product classification)
3. Run bot fingerprinting
4. Output analysis report to analysis/output/roundN.md

Usage:
    uv run scripts/round_open.py 1            # Round 1
    uv run scripts/round_open.py 0            # Tutorial (re-analyze)
    uv run scripts/round_open.py 1 --p3-dir data/p3/round5/  # With P3 correlation test
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.data_loader import load_prices, load_trades, DATA_DIR
from analysis.eda import run_eda, format_report as eda_report
from analysis.bot_fingerprint import run_fingerprint, format_report as bot_report


def resolve_round_dir(round_num: int) -> Path:
    """Get the data directory for a round."""
    if round_num == 0:
        return DATA_DIR / "tutorial"
    return DATA_DIR / f"round{round_num}"


def run_pipeline(
    round_num: int,
    p3_dir: Path | None = None,
) -> str:
    """Run full analysis pipeline. Returns markdown report."""
    round_dir = resolve_round_dir(round_num)

    if not round_dir.exists():
        return f"# ERROR\n\nRound directory not found: {round_dir}\n"

    sections: list[str] = []
    sections.append(f"# Round {round_num} Analysis Report")
    sections.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    sections.append("")

    # ── Step 1: EDA ──────────────────────────────────────────────
    print(f"[1/3] Running EDA on {round_dir}...")
    try:
        eda_results = run_eda(round_dir)
        sections.append(eda_report(eda_results))

        # Quick summary for console
        for s in eda_results:
            print(f"  {s.product}: {s.archetype_name} ({s.confidence}) - CV={s.cv:.6f}, ADF p={s.adf_pvalue:.4f}")
    except Exception as e:
        sections.append(f"## EDA\n\nERROR: {e}\n")
        print(f"  EDA failed: {e}")

    # ── Step 2: Bot fingerprinting ───────────────────────────────
    print(f"[2/3] Running bot fingerprinting...")
    try:
        bots, flows = run_fingerprint(round_dir)
        sections.append(bot_report(bots, flows))

        if bots:
            for b in bots:
                print(f"  DETECTED: {b.product} has {b.bot_label} bot (confidence={b.confidence})")
        else:
            print("  No informed bots detected")
    except Exception as e:
        sections.append(f"## Bot Fingerprint\n\nERROR: {e}\n")
        print(f"  Bot fingerprinting failed: {e}")

    # ── Step 3: P3 correlation (if P3 data provided) ─────────────
    if p3_dir and p3_dir.exists():
        print(f"[3/3] Running P3→P4 correlation test against {p3_dir}...")
        try:
            from analysis.correlation import run_correlation, format_report as corr_report
            corr_results = run_correlation(round_dir, p3_dir)
            sections.append(corr_report(corr_results))

            exploitable = [r for r in corr_results if r.exploitable]
            if exploitable:
                for r in exploitable:
                    print(f"  EXPLOITABLE: {r.p4_product} ↔ {r.p3_product} R²={r.r_squared:.4f}")
            else:
                print("  No exploitable correlations found")
        except Exception as e:
            sections.append(f"## P3→P4 Correlation\n\nERROR: {e}\n")
            print(f"  Correlation test failed: {e}")
    else:
        print("[3/3] Skipping P3 correlation (no P3 data directory provided)")
        sections.append("## P3→P4 Correlation\n\nSkipped - no P3 data provided. "
                       "Run with `--p3-dir` when P3 data is available.\n")

    # ── Strategy recommendations ─────────────────────────────────
    sections.append("## Recommended Strategy Assignments")
    sections.append("")
    try:
        archetype_to_strategy = {
            "STABLE": "strategies/stable.py - hardcode FV, quote inside L1",
            "DRIFTING": "strategies/drifting.py - wall_mid + VWAP blend, AR correction, market-follow",
            "BASKET/ETF": "strategies/basket.py - Welford premium tracking, z-score entry",
            "OPTIONS": "strategies/options.py - BS pricer, vol smile fit, IV scalp",
            "CONVERSION": "strategies/conversion.py - cross-exchange arb, taker bot detection",
            "SIGNAL-DRIVEN": "strategies/signal.py - Olivia copy-trade, EMA z-score",
        }

        for s in eda_results:
            rec = archetype_to_strategy.get(s.archetype_name, "manual classification needed")
            sections.append(f"- **{s.product}** ({s.archetype_name}): {rec}")

        sections.append("")
    except Exception:
        pass

    # ── Generated skeleton code ────────────────────────────────────
    sections.append("## Generated SKELETON_PRODUCTS Code")
    sections.append("")
    sections.append("Copy-paste into `trader.py` SKELETON_PRODUCTS dict:")
    sections.append("")
    sections.append("```python")
    sections.append("SKELETON_PRODUCTS: dict = {")

    archetype_to_code = {
        "STABLE": lambda s: (
            f'    "{s.product}": (StableTrader, {{"product": "{s.product}", '
            f'"fv": {int(round(s.mean_price))}, "limit": 50, "make_spread": 7}}),'
        ),
        "DRIFTING": lambda s: (
            f'    "{s.product}": (DriftingTrader, {{"product": "{s.product}", '
            f'"limit": 50, "make_spread": 2, "ar_beta": -0.229, "use_olivia": True}}),'
            f'  # Run ab_test.py first! If market-follow wins, add fv_mode="follow"'
        ),
        "BASKET/ETF": lambda s: (
            f'    # "{s.product}": (BasketTrader, {{"basket_product": "{s.product}", '
            f'"constituents": {{}}, "basket_limit": 60, "entry_thr": 80.0}}),  # FILL IN constituents'
        ),
        "OPTIONS": lambda s: (
            f'    # "{s.product}": (OptionsTrader, {{"underlying": "???", '
            f'"option_products": {{}}, "option_limit": 200}}),  # FILL IN underlying + strikes'
        ),
        "CONVERSION": lambda s: (
            f'    "{s.product}": (ConversionTrader, {{"product": "{s.product}", '
            f'"limit": 75, "conv_limit": 10, "min_edge": 0.5}}),'
        ),
        "SIGNAL-DRIVEN": lambda s: (
            f'    "{s.product}": (SignalTrader, {{"product": "{s.product}", '
            f'"limit": 50, "z_buy": 5.0, "z_sell": 5.0, "use_olivia": True}}),'
        ),
    }

    try:
        # Skip tutorial products (already have inline handlers)
        tutorial_products = {"EMERALDS", "TOMATOES"}
        for s in eda_results:
            if s.product in tutorial_products:
                sections.append(f"    # {s.product}: handled inline (tutorial product)")
                continue
            gen = archetype_to_code.get(s.archetype_name)
            if gen:
                sections.append(gen(s))
            else:
                sections.append(f'    # "{s.product}": ???  # Unknown archetype: {s.archetype_name}')
    except Exception:
        sections.append("    # Code generation failed - classify manually")

    sections.append("}")
    sections.append("```")
    sections.append("")

    # ── Action items ─────────────────────────────────────────────
    sections.append("## Immediate Action Items")
    sections.append("")
    sections.append("1. Review classifications above - override any low-confidence ones manually")
    sections.append("2. Copy SKELETON_PRODUCTS code into trader.py, fill in TODOs")
    sections.append("3. Submit stub trader immediately (zero orders for new products)")
    sections.append("4. Run `uv run scripts/ab_test.py` on drifting products to pick strategy")
    sections.append("5. Calibrate parameters using `analysis/parameter_search.py`")
    sections.append("6. Check Discord for community intel on new products")
    sections.append("7. If any P3 correlation is exploitable (R² > 0.9), prioritize that signal")
    sections.append("")

    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Round opening analysis pipeline")
    parser.add_argument("round", type=int, help="Round number (0=tutorial, 1-5=competition)")
    parser.add_argument("--p3-dir", type=Path, default=None, help="Path to P3 round data for correlation test")
    parser.add_argument("--no-save", action="store_true", help="Print report only, don't save to disk")
    args = parser.parse_args()

    report = run_pipeline(args.round, args.p3_dir)

    # Save report
    if not args.no_save:
        label = "tutorial" if args.round == 0 else f"round{args.round}"
        output_dir = ROOT / "analysis" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{label}.md"
        output_file.write_text(report)
        print(f"\nReport saved to {output_file}")
    else:
        print("\n" + "=" * 60)
        print(report)


if __name__ == "__main__":
    main()
