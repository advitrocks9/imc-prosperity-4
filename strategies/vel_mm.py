"""Adaptive market maker for VELVETFRUIT_EXTRACT."""
from datamodel import Order, OrderDepth

VEL_FV = 5_250
MAKE_SPREAD = 2
TAKE_EDGE = 1.0
POSITION_MAX = 200
INV_SOFT_CAP = 100
EMA_ALPHA = 0.1
FV_BAND = 50


class VelMmTrader:
    def __init__(
        self,
        product: str,
        fv: int = VEL_FV,
        limit: int = POSITION_MAX,
        make_spread: int = MAKE_SPREAD,
        take_edge: float = TAKE_EDGE,
        inv_soft_cap: int = INV_SOFT_CAP,
        ema_alpha: float = EMA_ALPHA,
        fv_band: int = FV_BAND,
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
