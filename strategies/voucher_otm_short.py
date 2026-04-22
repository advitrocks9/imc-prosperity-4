from __future__ import annotations

import math
from statistics import NormalDist
from typing import TYPE_CHECKING

from datamodel import Order, OrderDepth

if TYPE_CHECKING:
    from datamodel import TradingState


class VoucherOTMShort:
    """Short far-OTM vouchers when market bid is far above BS theoretical."""

    UNDERLYING = "VELVETFRUIT_EXTRACT"
    SYMBOLS = ("VEV_6000", "VEV_6500")
    POSITION_LIMIT = 300

    def __init__(self, factor: float, n: int) -> None:
        self.factor = float(factor)
        self.n = int(n)
        self.cumulative_pnl_estimate = 0.0
        self.last_theoretical: dict[str, float] = {}

    @staticmethod
    def _strike(symbol: str) -> int:
        return int(symbol.split("_", 1)[1])

    @staticmethod
    def _tte_years(timestamp: int) -> float:
        days_remaining = max(4.0 - (timestamp / 1_000_000.0), 3.0)
        return days_remaining / 252.0

    @staticmethod
    def _spot(order_depth: OrderDepth) -> float | None:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0

    @staticmethod
    def _bs_call(spot: float, strike: float, tte: float, sigma: float, rate: float = 0.0) -> float:
        if spot <= 0.0 or strike <= 0.0 or tte <= 0.0 or sigma <= 0.0:
            return max(0.0, spot - strike)
        sqrt_t = math.sqrt(tte)
        vol_term = sigma * sqrt_t
        d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * tte) / vol_term
        d2 = d1 - vol_term
        normal = NormalDist()
        return spot * normal.cdf(d1) - strike * math.exp(-rate * tte) * normal.cdf(d2)

    def run(
        self,
        order_depths: dict[str, OrderDepth],
        positions: dict[str, int],
        timestamp: int = 0,
        sigma: float = 0.293,
    ) -> dict[str, list[Order]]:
        result: dict[str, list[Order]] = {}
        underlying_depth = order_depths.get(self.UNDERLYING)
        if underlying_depth is None:
            return result

        spot = self._spot(underlying_depth)
        tte = self._tte_years(timestamp)
        if spot is None or tte <= 0.0:
            return result

        for symbol in self.SYMBOLS:
            order_depth = order_depths.get(symbol)
            if order_depth is None or not order_depth.buy_orders:
                continue

            theoretical = self._bs_call(spot, float(self._strike(symbol)), tte, sigma)
            self.last_theoretical[symbol] = theoretical

            best_bid = max(order_depth.buy_orders)
            bid_size = max(0, order_depth.buy_orders[best_bid])
            sell_capacity = self.POSITION_LIMIT + int(positions.get(symbol, 0))
            if sell_capacity <= 0:
                continue
            if best_bid <= theoretical * self.factor:
                continue

            qty = min(bid_size, self.n, sell_capacity)
            if qty <= 0:
                continue

            result[symbol] = [Order(symbol, int(best_bid), -int(qty))]
            self.cumulative_pnl_estimate += (float(best_bid) - theoretical) * qty

        return result
