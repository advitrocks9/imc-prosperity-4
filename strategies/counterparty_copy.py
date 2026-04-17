from datamodel import Listing, Observation, Order, OrderDepth, Trade, TradingState


class CounterpartyCopyTrader:
    """
    Copy-trade signals from named counterparties.

    Per tick:
      1. Inspect state.market_trades for trades where buyer in COPY_BUYERS or seller in COPY_SELLERS.
      2. If a copy-buyer is BUYING product P -> issue BUY at top-of-book ask (TAKE).
      3. If a copy-seller is SELLING product P -> issue SELL at top-of-book bid (TAKE).
      4. Size: COPY_SIZE per signal, capped at position limit.
      5. Decay: hold for COPY_HOLD_TICKS, then close (sell longs, buy shorts).
    """

    def __init__(
        self,
        buyers: set[str] | None = None,
        sellers: set[str] | None = None,
        products: set[str] | None = None,
        size: int = 50,
        position_max: int = 200,
        hold_ticks: int = 1000,
        take_only: bool = True,
    ) -> None:
        self.buyers = buyers or {"Mark 67"}
        self.sellers = sellers or {"Mark 49"}
        self.products = products or {"VELVETFRUIT_EXTRACT"}
        self.size = size
        self.position_max = position_max
        self.hold_ticks = hold_ticks
        self.take_only = take_only

    def generate_orders(
        self,
        state: TradingState,
        trader_state: dict | None,
        existing_orders: dict[str, list[Order]] | None = None,
    ) -> tuple[dict[str, list[Order]], dict]:
        orders_by_product: dict[str, list[Order]] = {}
        next_state = dict(trader_state or {})
        existing_orders = existing_orders or {}

        for product in self.products:
            order_depth = state.order_depths.get(product)
            if order_depth is None:
                continue

            product_state = dict(next_state.get(product, {}))
            target = int(product_state.get("t", 0))
            expiry = int(product_state.get("e", -1))

            buy_signal = False
            sell_signal = False
            for trade in state.market_trades.get(product, []):
                if trade.buyer in self.buyers:
                    buy_signal = True
                if trade.seller in self.sellers:
                    sell_signal = True

            if buy_signal and not sell_signal:
                target = min(self.position_max, target + self.size)
                expiry = state.timestamp + self.hold_ticks
            elif sell_signal and not buy_signal:
                target = max(-self.position_max, target - self.size)
                expiry = state.timestamp + self.hold_ticks
            elif expiry >= 0 and state.timestamp >= expiry:
                target = 0
                expiry = -1

            effective_pos = state.position.get(product, 0) + sum(
                order.quantity for order in existing_orders.get(product, [])
            )
            delta = target - effective_pos
            product_orders: list[Order] = []

            if delta > 0:
                best_ask = self._best_ask(order_depth)
                buy_capacity = max(0, self.position_max - effective_pos)
                qty = min(delta, buy_capacity)
                if best_ask is not None and qty > 0:
                    product_orders.append(Order(product, best_ask, qty))
            elif delta < 0:
                best_bid = self._best_bid(order_depth)
                sell_capacity = max(0, self.position_max + effective_pos)
                qty = min(-delta, sell_capacity)
                if best_bid is not None and qty > 0:
                    product_orders.append(Order(product, best_bid, -qty))

            if product_orders:
                orders_by_product[product] = product_orders

            if target != 0 or expiry >= 0:
                next_state[product] = {"t": target, "e": expiry}
            else:
                next_state.pop(product, None)

        return orders_by_product, next_state

    def _best_ask(self, order_depth: OrderDepth) -> int | None:
        if not self.take_only or not order_depth.sell_orders:
            return min(order_depth.sell_orders) if order_depth.sell_orders else None
        return min(order_depth.sell_orders)

    def _best_bid(self, order_depth: OrderDepth) -> int | None:
        if not self.take_only or not order_depth.buy_orders:
            return max(order_depth.buy_orders) if order_depth.buy_orders else None
        return max(order_depth.buy_orders)


if __name__ == "__main__":
    depth = OrderDepth()
    depth.buy_orders = {99: 20}
    depth.sell_orders = {101: -20}

    state = TradingState(
        traderData="",
        timestamp=100,
        listings={
            "VELVETFRUIT_EXTRACT": Listing(
                "VELVETFRUIT_EXTRACT", "VELVETFRUIT_EXTRACT", 1
            )
        },
        order_depths={"VELVETFRUIT_EXTRACT": depth},
        own_trades={},
        market_trades={
            "VELVETFRUIT_EXTRACT": [
                Trade(
                    symbol="VELVETFRUIT_EXTRACT",
                    price=101,
                    quantity=15,
                    buyer="Mark 67",
                    seller="Other",
                    timestamp=100,
                )
            ]
        },
        position={"VELVETFRUIT_EXTRACT": 0},
        observations=Observation({}, {}),
    )

    trader = CounterpartyCopyTrader()
    orders, _ = trader.generate_orders(state, {}, {})
    vel_orders = orders.get("VELVETFRUIT_EXTRACT", [])

    if any(order.quantity > 0 for order in vel_orders):
        print("PASS")
    else:
        print("FAIL")
