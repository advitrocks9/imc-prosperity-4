from datamodel import Order, OrderDepth, TradingState

BUTTERFLY_TRIPLETS = (
    (4500, 5000, 5500),
    (5000, 5100, 5200),
    (5100, 5200, 5300),
    (5200, 5300, 5400),
    (5300, 5400, 5500),
)


class VoucherButterflyTrader:
    """TAKE-only voucher butterfly arbitrage on selected strike triplets."""

    def __init__(
        self,
        threshold: float = 1.0,
        max_value: float = 5.0,
        position_max: int = 200,
        size: int = 1,
        triplets: tuple[tuple[int, int, int], ...] = BUTTERFLY_TRIPLETS,
    ) -> None:
        self.threshold = float(threshold)
        self.max_value = float(max_value)
        self.position_max = int(position_max)
        self.size = int(size)
        self.triplets = tuple(triplets)

    def _mid(self, order_depth: OrderDepth) -> float | None:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0

    def _planned_position(
        self,
        positions: dict[str, int],
        pending_positions: dict[str, int],
        symbol: str,
    ) -> int:
        if symbol not in pending_positions:
            pending_positions[symbol] = positions.get(symbol, 0)
        return pending_positions[symbol]

    def _top_buy_liquidity(
        self,
        symbol: str,
        order_depth: OrderDepth,
        buy_usage: dict[tuple[str, int], int],
    ) -> tuple[int, int] | None:
        if not order_depth.sell_orders:
            return None
        best_ask = min(order_depth.sell_orders)
        avail = abs(order_depth.sell_orders[best_ask]) - buy_usage.get((symbol, best_ask), 0)
        if avail <= 0:
            return None
        return best_ask, avail

    def _top_sell_liquidity(
        self,
        symbol: str,
        order_depth: OrderDepth,
        sell_usage: dict[tuple[str, int], int],
    ) -> tuple[int, int] | None:
        if not order_depth.buy_orders:
            return None
        best_bid = max(order_depth.buy_orders)
        avail = order_depth.buy_orders[best_bid] - sell_usage.get((symbol, best_bid), 0)
        if avail <= 0:
            return None
        return best_bid, avail

    def run(
        self,
        state: TradingState,
        existing_orders: dict[str, list[Order]] | None = None,
    ) -> dict[str, list[Order]]:
        results: dict[str, list[Order]] = {}
        pending_positions: dict[str, int] = {}
        buy_usage: dict[tuple[str, int], int] = {}
        sell_usage: dict[tuple[str, int], int] = {}

        for symbol, planned in (existing_orders or {}).items():
            for order in planned:
                if order.quantity == 0:
                    continue
                pending_positions[symbol] = self._planned_position(state.position, pending_positions, symbol) + order.quantity
                usage_key = (symbol, int(order.price))
                if order.quantity > 0:
                    buy_usage[usage_key] = buy_usage.get(usage_key, 0) + int(order.quantity)
                else:
                    sell_usage[usage_key] = sell_usage.get(usage_key, 0) + int(-order.quantity)

        for low_strike, mid_strike, high_strike in self.triplets:
            low_symbol = f"VEV_{low_strike}"
            mid_symbol = f"VEV_{mid_strike}"
            high_symbol = f"VEV_{high_strike}"
            symbols = (low_symbol, mid_symbol, high_symbol)

            depths = tuple(state.order_depths.get(symbol) for symbol in symbols)
            if any(depth is None for depth in depths):
                continue

            low_depth, mid_depth, high_depth = depths
            low_mid = self._mid(low_depth)
            mid_mid = self._mid(mid_depth)
            high_mid = self._mid(high_depth)
            if low_mid is None or mid_mid is None or high_mid is None:
                continue

            butterfly_mid = low_mid + high_mid - 2.0 * mid_mid
            if butterfly_mid < -self.threshold:
                legs = (
                    (low_symbol, self.size),
                    (mid_symbol, -2 * self.size),
                    (high_symbol, self.size),
                )
            elif butterfly_mid > self.max_value:
                legs = (
                    (low_symbol, -self.size),
                    (mid_symbol, 2 * self.size),
                    (high_symbol, -self.size),
                )
            else:
                continue

            blocked = False
            leg_details: list[tuple[str, int, int]] = []
            for symbol, quantity in legs:
                depth = state.order_depths.get(symbol)
                next_position = self._planned_position(state.position, pending_positions, symbol) + quantity
                if abs(next_position) > self.position_max:
                    blocked = True
                    break

                if quantity > 0:
                    quote = self._top_buy_liquidity(symbol, depth, buy_usage)
                    if quote is None or quote[1] < quantity:
                        blocked = True
                        break
                    leg_details.append((symbol, quote[0], quantity))
                else:
                    quote = self._top_sell_liquidity(symbol, depth, sell_usage)
                    if quote is None or quote[1] < -quantity:
                        blocked = True
                        break
                    leg_details.append((symbol, quote[0], quantity))

            if blocked:
                continue

            for symbol, price, quantity in leg_details:
                results.setdefault(symbol, []).append(Order(symbol, int(price), int(quantity)))
                pending_positions[symbol] = self._planned_position(state.position, pending_positions, symbol) + quantity
                usage_key = (symbol, int(price))
                if quantity > 0:
                    buy_usage[usage_key] = buy_usage.get(usage_key, 0) + quantity
                else:
                    sell_usage[usage_key] = sell_usage.get(usage_key, 0) + (-quantity)

        return results
