"""Fixed-IV voucher TAKE-only trader (no MAKE, no delta hedge).

For each strike, hardcoded IV from per-day calibration. At each tick:
- compute theo = bs_call(spot, K, tte, 0, IV[K])
- if best_ask < theo - take_edge → TAKE buy
- if best_bid > theo + take_edge → TAKE sell
- mean-revert via sign-flip closes (handled naturally as dev changes sign)

No delta hedge; voucher net delta is small enough at the position limits.
The HYDROGEL/VELVETFRUIT MM continues independently and absorbs any net
exposure via its own MM dynamics.
"""
import math
from datamodel import Order, OrderDepth
from strategies._base import bs_call


class VoucherTakeTrader:
    """TAKE-only voucher trader with fixed per-strike IV.

    Args:
        underlying: symbol of underlying (for spot)
        strike_iv: dict {strike: iv} - calibrated IVs to use as fair value
        option_products: dict {strike: voucher_symbol}
        option_limit: position limit per voucher
        take_edge: minimum |market - theo| in price units to trigger
        trade_qty: per-tick order size per strike
        tte_func: callable(timestamp) → tte_years
    """

    def __init__(
        self,
        underlying: str,
        strike_iv: dict[int, float],
        option_products: dict[int, str],
        option_limit: int = 300,
        take_edge: float = 2.0,
        trade_qty: int = 10,
        tte_func=None,
    ) -> None:
        self.underlying = underlying
        self.strike_iv = strike_iv
        self.option_products = option_products
        self.option_limit = option_limit
        self.take_edge = take_edge
        self.trade_qty = trade_qty
        self.tte_func = tte_func or (lambda ts: 5.0 / 365.0)

    def run(
        self,
        order_depths: dict[str, OrderDepth],
        positions: dict[str, int],
        timestamp: int = 0,
    ) -> dict[str, list[Order]]:
        result: dict[str, list[Order]] = {}

        und_od = order_depths.get(self.underlying)
        if not und_od or not und_od.buy_orders or not und_od.sell_orders:
            return result
        spot = (max(und_od.buy_orders) + min(und_od.sell_orders)) / 2.0
        tte = self.tte_func(timestamp)

        for strike, sym in self.option_products.items():
            iv = self.strike_iv.get(strike)
            if iv is None:
                continue
            od = order_depths.get(sym)
            if not od:
                continue

            theo, _ = bs_call(spot, float(strike), tte, 0.0, iv)
            pos = positions.get(sym, 0)
            orders: list[Order] = []

            # TAKE buy - ask is below fair by enough
            if od.sell_orders:
                best_ask = min(od.sell_orders)
                if best_ask < theo - self.take_edge:
                    rem_buy = self.option_limit - pos
                    if rem_buy > 0:
                        avail = abs(od.sell_orders[best_ask])
                        qty = min(rem_buy, self.trade_qty, avail)
                        if qty > 0:
                            orders.append(Order(sym, best_ask, qty))

            # TAKE sell - bid is above fair by enough
            if od.buy_orders:
                best_bid = max(od.buy_orders)
                if best_bid > theo + self.take_edge:
                    rem_sell = self.option_limit + pos
                    if rem_sell > 0:
                        avail = od.buy_orders[best_bid]
                        qty = min(rem_sell, self.trade_qty, avail)
                        if qty > 0:
                            orders.append(Order(sym, best_bid, -qty))

            if orders:
                result[sym] = orders
        return result
