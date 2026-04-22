"""Passive MAKE-only voucher market maker for VEV_5300 / VEV_5400 / VEV_5500.

Theory: liquid voucher strikes have decent bot trade flow (121-267 historical
trades per strike per 30K ticks). Avoiding TAKE bid-ask cost = MAKE-only quotes
inside the smile. We lose adverse-selection risk but gain spread capture.

Strategy:
- Each tick, compute per-strike fair value from smile fit (or just wall_mid).
- MAKE-bid at fair - half_spread; MAKE-ask at fair + half_spread.
- Skip TAKE entirely. Skip strikes with bad book or adverse-selection signal.
- Optional position-skew: lean quotes against inventory.

This is the lightest-weight voucher strategy. If it makes money on portal,
we have new edge. If it loses, we abandon.
"""
import math
from datamodel import Order, OrderDepth
from strategies._base import wall_mid


class VoucherMakeMM:
    """Passive MAKE quotes on liquid voucher strikes."""

    def __init__(
        self,
        strikes: list[int],
        option_products: dict[int, str],
        option_limit: int = 300,
        half_spread: int = 1,
        max_pos_per_strike: int = 50,
        skew_sens: float = 0.0,
    ) -> None:
        self.strikes = strikes
        self.option_products = option_products
        self.option_limit = option_limit
        self.half_spread = half_spread
        self.max_pos_per_strike = max_pos_per_strike
        self.skew_sens = skew_sens

    def run(
        self,
        order_depths: dict,
        position: dict,
    ) -> dict[str, list[Order]]:
        result: dict[str, list[Order]] = {}

        for strike in self.strikes:
            sym = self.option_products.get(strike)
            if sym is None or sym not in order_depths:
                continue
            od = order_depths[sym]
            mid = wall_mid(od)
            if mid is None or mid <= 0:
                continue

            pos = position.get(sym, 0)
            cap = self.max_pos_per_strike

            # Inventory skew: lean quotes against position
            skew = -self.skew_sens * (pos / cap) if cap > 0 else 0.0
            fair = mid + skew

            bid_price = int(round(fair - self.half_spread))
            ask_price = int(round(fair + self.half_spread))

            # Sanity: don't quote at 0 or below
            if bid_price < 1:
                bid_price = 1

            # Capacity within hard limit and per-strike cap
            rem_buy = min(self.option_limit, cap) - pos
            rem_sell = min(self.option_limit, cap) + pos

            orders: list[Order] = []
            if rem_buy > 0 and bid_price > 0:
                orders.append(Order(sym, bid_price, rem_buy))
            if rem_sell > 0:
                orders.append(Order(sym, ask_price, -rem_sell))
            if orders:
                result[sym] = orders

        return result
