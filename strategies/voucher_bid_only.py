from datamodel import Order, TradingState


class VoucherBidOnlyMM:
    def __init__(self, symbols: list[str], half_spread: int, position_cap: int) -> None:
        self.symbols = list(symbols)
        self.half_spread = int(half_spread)
        self.position_cap = int(position_cap)

    def run(self, state: TradingState) -> list[Order]:
        orders: list[Order] = []
        for symbol in self.symbols:
            order_depth = state.order_depths.get(symbol)
            if order_depth is None:
                continue

            best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

            wall_mid: int | None = None
            if best_bid is not None and best_ask is not None:
                wall_mid = (best_bid + best_ask) // 2
            elif best_bid is not None:
                wall_mid = best_bid
            elif best_ask is not None:
                wall_mid = best_ask
            if wall_mid is None:
                continue

            position = int(state.position.get(symbol, 0))
            if position >= self.position_cap:
                continue

            quantity = min(self.position_cap - position, 20)
            if quantity <= 0:
                continue

            bid_price = max(0, wall_mid - self.half_spread)
            orders.append(Order(symbol, bid_price, quantity))
        return orders
