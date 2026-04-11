"""Adaptive market maker for HYDROGEL_PACK.

R3+probe1 lesson: hardcoded FV=10000 is wrong for any seed not centered there.
R4 portal preview seed had HYD mean=10033 → static FV=10000 lost -12K extrap.
Fix: use EMA of wall_mid as FV. Default fv stays as the cold-start fallback.
"""
from datamodel import Order, OrderDepth

HYD_FV = 10_000
HYD_MAKE_SPREAD = 5
HYD_TAKE_EDGE = 1.5
HYD_POSITION_MAX = 200
HYD_INV_SOFT_CAP = 100
HYD_EMA_ALPHA = 0.1
HYD_FV_BAND = 50  # max ticks the adaptive FV can drift from the cold-start fallback


class HydrogelMMTrader:
    def __init__(
        self,
        product: str,
        fv: int = HYD_FV,
        limit: int = HYD_POSITION_MAX,
        make_spread: int = HYD_MAKE_SPREAD,
        take_edge: float = HYD_TAKE_EDGE,
        inv_soft_cap: int = HYD_INV_SOFT_CAP,
        ema_alpha: float = HYD_EMA_ALPHA,
        fv_band: int = HYD_FV_BAND,
    ) -> None:
        self.product = product
        self.cold_fv = int(fv)
        self.limit = int(limit)
        self.make_spread = int(make_spread)
        self.take_edge = float(take_edge)
        self.inv_soft_cap = int(inv_soft_cap)
        self.ema_alpha = float(ema_alpha)
        self.fv_band = int(fv_band)
        self.ema_fv: float | None = None

    def _wall_mid(self, od: OrderDepth) -> float | None:
        if not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders) + min(od.sell_orders)) / 2.0

    def _update_fv(self, wall_mid: float) -> int:
        if self.ema_fv is None:
            self.ema_fv = wall_mid
        else:
            self.ema_fv = (1.0 - self.ema_alpha) * self.ema_fv + self.ema_alpha * wall_mid
        clamped = max(self.cold_fv - self.fv_band, min(self.cold_fv + self.fv_band, self.ema_fv))
        return int(round(clamped))

    def _take(self, od: OrderDepth, fv: int, rem_buy: int, rem_sell: int) -> tuple[list[Order], int, int]:
        orders: list[Order] = []
        for ask in sorted(od.sell_orders):
            if ask >= fv - self.take_edge or rem_buy <= 0:
                break
            qty = min(-od.sell_orders[ask], rem_buy)
            orders.append(Order(self.product, int(ask), qty))
            rem_buy -= qty
        for bid in sorted(od.buy_orders, reverse=True):
            if bid <= fv + self.take_edge or rem_sell <= 0:
                break
            qty = min(od.buy_orders[bid], rem_sell)
            orders.append(Order(self.product, int(bid), -qty))
            rem_sell -= qty
        return orders, rem_buy, rem_sell

    def _quotes(self, fv: int, pos: int) -> tuple[int, int]:
        bid_price = fv - self.make_spread
        ask_price = fv + self.make_spread
        if pos > self.inv_soft_cap:
            bid_price -= 1
            ask_price -= 1
        elif pos < -self.inv_soft_cap:
            bid_price += 1
            ask_price += 1
        return int(bid_price), int(ask_price)

    def run(self, od: OrderDepth, pos: int) -> list[Order]:
        wall_mid = self._wall_mid(od)
        fv = self._update_fv(wall_mid) if wall_mid is not None else self.cold_fv
        rem_buy = self.limit - pos
        rem_sell = self.limit + pos
        orders, rem_buy, rem_sell = self._take(od, fv, rem_buy, rem_sell)
        bid_price, ask_price = self._quotes(fv, pos)
        if rem_buy > 0:
            orders.append(Order(self.product, bid_price, rem_buy))
        if rem_sell > 0:
            orders.append(Order(self.product, ask_price, -rem_sell))
        return orders
