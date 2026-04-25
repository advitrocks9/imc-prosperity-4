"""Regime-adaptive stable MM. Anchored at the portal-validated peak FV but
shifts by (observed_mean - peak_mean) to preserve the asymmetric MM offset
when the scored seed has a different mean than the portal preview.

Mechanism: peak_fv was found empirically on a portal preview where live
mean = peak_mean. The asymmetric MM mechanic profits from MAKE-ask sitting
at peak_mean + (peak_fv + sp - peak_mean) = a specific offset relative to
the live mean. If scored seed shifts the mean, we shift FV by the same
amount to preserve the offset.

FV(t) = peak_fv + (ema_mid(t) - peak_mean) * adapt_strength
"""
import math
from datamodel import Order, OrderDepth
from strategies._base import wall_mid


class RegimeAdaptiveStable:
    """Stable MM with regime-shift insurance via offset-preserving FV.

    Args:
        product: product symbol
        peak_fv: portal-validated optimal static FV
        peak_mean: observed live mean at which peak_fv is optimal
        limit: position limit
        make_spread: MAKE half-spread
        ema_halflife: EMA halflife for tracking live mean
        adapt_strength: 0.0 = pure static (=peak_fv), 1.0 = full shift, 0.5 = half-shift
        max_shift: cap on |fv - peak_fv| (absolute ticks). Prevents runaway.
        warmup_ticks: ticks to ignore EMA (use peak_fv) until EMA settles
    """

    def __init__(
        self,
        product: str,
        peak_fv: int,
        peak_mean: float,
        limit: int,
        make_spread: int = 10,
        ema_halflife: float = 500,
        adapt_strength: float = 1.0,
        max_shift: int = 30,
        warmup_ticks: int = 200,
    ) -> None:
        self.product = product
        self.peak_fv = peak_fv
        self.peak_mean = peak_mean
        self.limit = limit
        self.make_spread = make_spread
        self.alpha = 1.0 - math.exp(-math.log(2) / ema_halflife)
        self.adapt_strength = adapt_strength
        self.max_shift = max_shift
        self.warmup_ticks = warmup_ticks

    def run(self, od: OrderDepth, pos: int, td: dict) -> tuple[list[Order], dict]:
        wm = wall_mid(od, fallback=td.get("ema"))
        if wm is None:
            ema = td.get("ema", float(self.peak_mean))
        else:
            prev_ema = td.get("ema", float(wm))
            ema = self.alpha * wm + (1 - self.alpha) * prev_ema
        td["ema"] = round(ema, 2)
        ticks_seen = td.get("ticks_seen", 0) + 1
        td["ticks_seen"] = ticks_seen

        if ticks_seen < self.warmup_ticks:
            fv_float = float(self.peak_fv)
        else:
            shift = (ema - self.peak_mean) * self.adapt_strength
            shift = max(-self.max_shift, min(self.max_shift, shift))
            fv_float = self.peak_fv + shift
        fv_int = round(fv_float)

        orders: list[Order] = []
        rem_buy = self.limit - pos
        rem_sell = self.limit + pos

        for ask in sorted(od.sell_orders):
            if ask >= fv_int or rem_buy <= 0:
                break
            qty = min(abs(od.sell_orders[ask]), rem_buy)
            orders.append(Order(self.product, ask, qty))
            rem_buy -= qty

        for bid in sorted(od.buy_orders, reverse=True):
            if bid <= fv_int or rem_sell <= 0:
                break
            qty = min(od.buy_orders[bid], rem_sell)
            orders.append(Order(self.product, bid, -qty))
            rem_sell -= qty

        bid_price = fv_int - self.make_spread
        ask_price = fv_int + self.make_spread

        if rem_buy > 0:
            orders.append(Order(self.product, bid_price, rem_buy))
        if rem_sell > 0:
            orders.append(Order(self.product, ask_price, -rem_sell))

        return orders, td
