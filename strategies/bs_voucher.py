from __future__ import annotations

import math
from typing import TYPE_CHECKING

if __package__ in {None, ""}:
    _sys = __import__("sys")
    _pathlib = __import__("pathlib")
    _sys.path.append(str(_pathlib.Path(__file__).resolve().parents[1]))

if TYPE_CHECKING:
    from datamodel import Order, OrderDepth, TradingState

UNDERLYING_SYMBOL = "VELVETFRUIT_EXTRACT"
POSITION_CAP_PER_STRIKE = 50
DEFAULT_STRIKE_THRESHOLDS = {
    4000: 3,
    4500: 3,
    5000: 3,
    5100: 3,
    5200: 3,
    5300: 3,
    5400: 3,
    5500: 3,
    6000: 3,
    6500: 3,
}


def _strike_from_symbol(symbol: str) -> int | None:
    if not symbol.startswith("VEV_"):
        return None
    try:
        return int(symbol.split("_", 1)[1])
    except ValueError:
        return None


def _best_mid(order_depth: OrderDepth) -> float | None:
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0


def _voucher_mid(order_depth: OrderDepth) -> float | None:
    return _best_mid(order_depth)


def _tte_years(timestamp: int) -> float:
    # R4 spans 4 -> 3 trading days over the round.
    days_remaining = max(4.0 - (timestamp / 1_000_000.0), 3.0)
    return days_remaining / 252.0


class BSVoucherTrader:
    def __init__(
        self,
        threshold: int = 3,
        anchor_strike: str = "VEV_5200",
        strike_thresholds: dict[int, int] | None = None,
    ) -> None:
        self.threshold = int(threshold)
        self.anchor_strike = anchor_strike
        self.position_cap = POSITION_CAP_PER_STRIKE
        self.last_anchor_iv: float | None = None
        thresholds = DEFAULT_STRIKE_THRESHOLDS.copy()
        if strike_thresholds is None:
            thresholds = {strike: self.threshold for strike in thresholds}
        else:
            for strike, value in strike_thresholds.items():
                thresholds[int(strike)] = int(value)
        self.strike_thresholds = thresholds

    def _anchor_iv(self, state: TradingState) -> tuple[float, float] | None:
        from strategies._base import find_iv

        underlying_depth = state.order_depths.get(UNDERLYING_SYMBOL)
        anchor_depth = state.order_depths.get(self.anchor_strike)
        anchor_k = _strike_from_symbol(self.anchor_strike)
        if underlying_depth is None or anchor_depth is None or anchor_k is None:
            return None

        spot = _best_mid(underlying_depth)
        anchor_mid = _voucher_mid(anchor_depth)
        tte = _tte_years(state.timestamp)
        if spot is None or anchor_mid is None or spot <= 0 or anchor_mid <= 0 or tte <= 0:
            return None

        iv = find_iv(spot, float(anchor_k), tte, 0.0, anchor_mid)
        if iv is None or not math.isfinite(iv) or iv <= 0:
            return None

        self.last_anchor_iv = iv
        return spot, iv

    def _buy_order(self, symbol: str, order_depth: OrderDepth, position: int) -> Order | None:
        from datamodel import Order

        if not order_depth.sell_orders:
            return None
        best_ask = min(order_depth.sell_orders)
        buy_capacity = self.position_cap - position
        if buy_capacity <= 0:
            return None
        ask_volume = max(0, -order_depth.sell_orders[best_ask])
        quantity = min(buy_capacity, ask_volume)
        if quantity <= 0:
            return None
        return Order(symbol, int(best_ask), int(quantity))

    def _sell_order(self, symbol: str, order_depth: OrderDepth, position: int) -> Order | None:
        from datamodel import Order

        if not order_depth.buy_orders:
            return None
        best_bid = max(order_depth.buy_orders)
        sell_capacity = self.position_cap + position
        if sell_capacity <= 0:
            return None
        bid_volume = max(0, order_depth.buy_orders[best_bid])
        quantity = min(sell_capacity, bid_volume)
        if quantity <= 0:
            return None
        return Order(symbol, int(best_bid), -int(quantity))

    def run(self, state: TradingState) -> dict[str, list[Order]]:
        from strategies._base import bs_call

        anchor = self._anchor_iv(state)
        if anchor is None:
            return {}

        spot, flat_iv = anchor
        tte = _tte_years(state.timestamp)
        results: dict[str, list[Order]] = {}

        voucher_symbols = sorted(
            (symbol for symbol in state.order_depths if symbol.startswith("VEV_")),
            key=lambda symbol: (_strike_from_symbol(symbol) or 0, symbol),
        )

        for symbol in voucher_symbols:
            if symbol == self.anchor_strike:
                continue

            strike = _strike_from_symbol(symbol)
            order_depth = state.order_depths.get(symbol)
            if strike is None or order_depth is None:
                continue

            mid = _voucher_mid(order_depth)
            if mid is None:
                continue

            theoretical, _ = bs_call(spot, float(strike), tte, 0.0, flat_iv)
            position = state.position.get(symbol, 0)
            threshold = self.strike_thresholds.get(strike, self.threshold)

            order: Order | None = None
            if mid > theoretical + threshold:
                order = self._sell_order(symbol, order_depth, position)
            elif mid < theoretical - threshold:
                order = self._buy_order(symbol, order_depth, position)

            if order is not None:
                results[symbol] = [order]

        return results
