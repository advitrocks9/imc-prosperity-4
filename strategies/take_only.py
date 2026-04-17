"""TAKE-only stable trader. No MAKE quotes ever posted.

For products where MAKE fills are adverse on the portal (live competition fills
our stale quotes against us), TAKE-only avoids the issue entirely.
"""
from datamodel import Order, OrderDepth


class TakeOnlyTrader:
    """Take asks below FV, hit bids above FV. No MAKE."""

    def __init__(self, product: str, fv: int, limit: int, take_edge: float = 0.0) -> None:
        self.product = product
        self.fv = float(fv)
        self.limit = limit
        self.take_edge = take_edge

    def run(self, od: OrderDepth, pos: int) -> list[Order]:
        orders: list[Order] = []
        rem_buy = self.limit - pos
        rem_sell = self.limit + pos

        for ask in sorted(od.sell_orders):
            if ask >= self.fv - self.take_edge or rem_buy <= 0:
                break
            qty = min(abs(od.sell_orders[ask]), rem_buy)
            orders.append(Order(self.product, ask, qty))
            rem_buy -= qty

        for bid in sorted(od.buy_orders, reverse=True):
            if bid <= self.fv + self.take_edge or rem_sell <= 0:
                break
            qty = min(od.buy_orders[bid], rem_sell)
            orders.append(Order(self.product, bid, -qty))
            rem_sell -= qty

        return orders
