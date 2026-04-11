"""Stable MM with multi-level MAKE depth ladder.

Posts MAKE quotes at multiple price levels on each side, splitting the
remaining position-limit budget. Captures bots walking the book.

Each level: (offset, weight). Offset = ticks from FV. Weight = fraction
of remaining capacity at that level. Weights need not sum to 1; remainder
goes to the deepest level.

Example: ask_levels = [(8, 0.5), (12, 0.3), (16, 0.2)]
- 50% of rem_sell at FV+8
- 30% at FV+12
- 20% at FV+16
"""
from datamodel import Order, OrderDepth


class StableLadderTrader:
    def __init__(
        self,
        product: str,
        fv: int,
        limit: int,
        bid_levels: list[tuple[int, float]],
        ask_levels: list[tuple[int, float]],
    ) -> None:
        self.product = product
        self.fv = fv
        self.limit = limit
        self.bid_levels = bid_levels
        self.ask_levels = ask_levels

    def run(self, od: OrderDepth, pos: int) -> list[Order]:
        orders: list[Order] = []
        rem_buy = self.limit - pos
        rem_sell = self.limit + pos

        # TAKE side
        for ask in sorted(od.sell_orders):
            if ask >= self.fv or rem_buy <= 0:
                break
            qty = min(abs(od.sell_orders[ask]), rem_buy)
            orders.append(Order(self.product, ask, qty))
            rem_buy -= qty

        for bid in sorted(od.buy_orders, reverse=True):
            if bid <= self.fv or rem_sell <= 0:
                break
            qty = min(od.buy_orders[bid], rem_sell)
            orders.append(Order(self.product, bid, -qty))
            rem_sell -= qty

        # MAKE: split rem_buy across bid levels
        if rem_buy > 0:
            allocated = 0
            n = len(self.bid_levels)
            for i, (offset, weight) in enumerate(self.bid_levels):
                if i == n - 1:
                    qty = rem_buy - allocated
                else:
                    qty = round(rem_buy * weight)
                if qty > 0:
                    orders.append(Order(self.product, self.fv - offset, qty))
                    allocated += qty

        if rem_sell > 0:
            allocated = 0
            n = len(self.ask_levels)
            for i, (offset, weight) in enumerate(self.ask_levels):
                if i == n - 1:
                    qty = rem_sell - allocated
                else:
                    qty = round(rem_sell * weight)
                if qty > 0:
                    orders.append(Order(self.product, self.fv + offset, -qty))
                    allocated += qty

        return orders
