"""Anchored-EMA stable trader.

Like StableTrader but FV adapts to observed mid via slow EMA, anchored to
a default. Useful when the live round may open at a slightly different
mid than the historical mean.

FV(t) = anchor_weight * default_fv + (1 - anchor_weight) * EMA(wall_mid)
"""
import math
from datamodel import Order, OrderDepth
from strategies._base import wall_mid


class AnchoredStableTrader:
    """Stable MM with EMA-anchored FV.

    Args:
        product: product symbol
        default_fv: prior on FV (e.g., 5250)
        limit: position limit
        make_spread: distance from FV for passive quotes
        ema_halflife: EMA halflife in ticks (longer = slower)
        anchor_weight: weight on default_fv vs EMA (1.0 = pure StableTrader,
                       0.0 = pure DriftingTrader)
    """

    def __init__(
        self,
        product: str,
        default_fv: int,
        limit: int,
        make_spread: int = 2,
        ema_halflife: float = 2000,
        anchor_weight: float = 0.7,
    ) -> None:
        self.product = product
        self.default_fv = default_fv
        self.limit = limit
        self.make_spread = make_spread
        self.alpha = 1.0 - math.exp(-math.log(2) / ema_halflife)
        self.anchor_weight = anchor_weight

    def run(self, od: OrderDepth, pos: int, td: dict) -> tuple[list[Order], dict]:
        # Update EMA. Cold-start: seed from first wall_mid (not default_fv) to
        # avoid warm-up bleed when default_fv is mismatched to live regime.
        wm = wall_mid(od, fallback=td.get("ema"))
        if wm is None:
            ema = td.get("ema", float(self.default_fv))
        else:
            prev_ema = td.get("ema", float(wm))
            ema = self.alpha * wm + (1 - self.alpha) * prev_ema
        td["ema"] = round(ema, 2)

        fv = self.anchor_weight * self.default_fv + (1 - self.anchor_weight) * ema
        # Round to nearest 0.5 for limit-order purposes; keep float for compare
        fv_int = round(fv)

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

        # MAKE: post at FV ± make_spread
        bid_price = fv_int - self.make_spread
        ask_price = fv_int + self.make_spread

        if rem_buy > 0:
            orders.append(Order(self.product, bid_price, rem_buy))
        if rem_sell > 0:
            orders.append(Order(self.product, ask_price, -rem_sell))

        return orders, td
