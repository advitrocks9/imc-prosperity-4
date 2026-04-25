"""Regime-detect-and-lock stable trader.

First N ticks: observe wall_mid, accumulate EMA. During warmup, run
StableTrader at a "neutral" FV (configurable). After N ticks, examine
observed mean and LOCK in a static FV from a discrete bin set:

- mean < low_thresh  → fv_low      (preview-like, asymmetric short)
- low <= mean < high → fv_mid      (middle ground)
- mean >= high       → fv_high     (historical-like)

Once locked, runs as StableTrader for remainder.

This addresses the problem that v6 (FV=9950) is overfit to portal preview
seed (mean=9979); historical seeds (mean ~9990) prefer FV=9970-10000.
"""
import math
from datamodel import Order, OrderDepth
from strategies._base import wall_mid


class RegimeLockTrader:
    def __init__(
        self,
        product: str,
        limit: int,
        warmup_ticks: int = 500,
        ema_halflife: float = 200,
        warmup_fv: int = 9970,           # neutral FV during warmup
        warmup_make_spread: int = 10,
        # Bin thresholds + locked FVs
        low_thresh: float = 9985.0,
        high_thresh: float = 9995.0,
        fv_low: int = 9950,              # for mean < low (preview-like)
        fv_mid: int = 9970,              # for low <= mean < high
        fv_high: int = 10000,            # for mean >= high (historical-like)
        make_spread: int = 10,
    ) -> None:
        self.product = product
        self.limit = limit
        self.warmup_ticks = warmup_ticks
        self.alpha = 1.0 - math.exp(-math.log(2) / ema_halflife)
        self.warmup_fv = warmup_fv
        self.warmup_make_spread = warmup_make_spread
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.fv_low = fv_low
        self.fv_mid = fv_mid
        self.fv_high = fv_high
        self.make_spread = make_spread

    def run(self, od: OrderDepth, pos: int, td: dict) -> tuple[list[Order], dict]:
        # Update EMA on wall_mid
        wm = wall_mid(od, fallback=td.get("ema"))
        if wm is None:
            ema = td.get("ema", float(self.warmup_fv))
        else:
            prev_ema = td.get("ema", float(wm))
            ema = self.alpha * wm + (1 - self.alpha) * prev_ema
        td["ema"] = round(ema, 2)
        ticks_seen = td.get("ticks_seen", 0) + 1
        td["ticks_seen"] = ticks_seen

        # Determine FV
        if ticks_seen < self.warmup_ticks:
            fv = self.warmup_fv
            sp = self.warmup_make_spread
        else:
            # Lock once
            if td.get("locked_fv") is None:
                if ema < self.low_thresh:
                    td["locked_fv"] = self.fv_low
                elif ema < self.high_thresh:
                    td["locked_fv"] = self.fv_mid
                else:
                    td["locked_fv"] = self.fv_high
            fv = td["locked_fv"]
            sp = self.make_spread

        orders: list[Order] = []
        rem_buy = self.limit - pos
        rem_sell = self.limit + pos

        # TAKE: sweep any ask below FV
        for ask in sorted(od.sell_orders):
            if ask >= fv or rem_buy <= 0:
                break
            qty = min(abs(od.sell_orders[ask]), rem_buy)
            orders.append(Order(self.product, ask, qty))
            rem_buy -= qty

        # TAKE: sweep any bid above FV
        for bid in sorted(od.buy_orders, reverse=True):
            if bid <= fv or rem_sell <= 0:
                break
            qty = min(od.buy_orders[bid], rem_sell)
            orders.append(Order(self.product, bid, -qty))
            rem_sell -= qty

        # MAKE
        bid_price = fv - sp
        ask_price = fv + sp

        if rem_buy > 0:
            orders.append(Order(self.product, bid_price, rem_buy))
        if rem_sell > 0:
            orders.append(Order(self.product, ask_price, -rem_sell))

        return orders, td
