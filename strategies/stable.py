"""Stable product market maker (Archetype 1: EMERALDS / AMETHYSTS / RESIN).

FV is constant (hardcoded to a round number). Quote inside bot L1.
Take any mispricing. No inventory management needed - FV is certain.

Parameters to calibrate: make_spread (1-3 ticks from FV).
"""
from datamodel import Order, OrderDepth
from strategies._base import wall_mid


class StableTrader:
    """Pure passive market maker for stable-price products.

    Args:
        product: product symbol
        fv: constant fair value (e.g., 10_000 for EMERALDS)
        limit: position limit
        make_spread: distance from FV for passive quotes (default 7 = inside bot L1 at ±8)
    """

    def __init__(
        self,
        product: str,
        fv: int,
        limit: int,
        make_spread: int = 7,
        make_bid_spread: int | None = None,
        make_ask_spread: int | None = None,
    ) -> None:
        self.product = product
        self.fv = fv
        self.limit = limit
        self.make_spread = make_spread
        self.make_bid_spread = make_bid_spread if make_bid_spread is not None else make_spread
        self.make_ask_spread = make_ask_spread if make_ask_spread is not None else make_spread

    def run(self, od: OrderDepth, pos: int) -> list[Order]:
        orders: list[Order] = []
        rem_buy = self.limit - pos
        rem_sell = self.limit + pos

        # TAKE: sweep any ask below FV
        for ask in sorted(od.sell_orders):
            if ask >= self.fv or rem_buy <= 0:
                break
            qty = min(abs(od.sell_orders[ask]), rem_buy)
            orders.append(Order(self.product, ask, qty))
            rem_buy -= qty

        # TAKE: sweep any bid above FV
        for bid in sorted(od.buy_orders, reverse=True):
            if bid <= self.fv or rem_sell <= 0:
                break
            qty = min(od.buy_orders[bid], rem_sell)
            orders.append(Order(self.product, bid, -qty))
            rem_sell -= qty

        # MAKE: post at FV - make_bid_spread / FV + make_ask_spread
        bid_price = self.fv - self.make_bid_spread
        ask_price = self.fv + self.make_ask_spread

        if rem_buy > 0:
            orders.append(Order(self.product, bid_price, rem_buy))
        if rem_sell > 0:
            orders.append(Order(self.product, ask_price, -rem_sell))

        return orders
