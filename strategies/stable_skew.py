"""Stable MM with explicit inventory skew + clear phase.

Same as StableTrader but with three phases:
- TAKE: lift any ask < FV-take_edge, hit any bid > FV+take_edge
- CLEAR: aggressive unwind when |pos| ≥ clear_thresh × limit
- MAKE: post bid/ask at FV ± make_spread, skewed by inventory
"""
from datamodel import Order, OrderDepth
from strategies._base import build_orders


class StableSkewTrader:
    """Stable MM with skewed quotes and inventory-clearing."""

    def __init__(
        self,
        product: str,
        fv: int,
        limit: int,
        make_spread: int = 4,
        take_edge: float = 1.0,
        skew_sens: float = 2.0,
        clear_thresh: float = 0.5,
        adverse_vol: int = 999,  # disabled by default for stable products
    ) -> None:
        self.product = product
        self.fv = float(fv)
        self.limit = limit
        self.make_spread = make_spread
        self.take_edge = take_edge
        self.skew_sens = skew_sens
        self.clear_thresh = clear_thresh
        self.adverse_vol = adverse_vol

    def run(self, od: OrderDepth, pos: int) -> list[Order]:
        if not od.buy_orders or not od.sell_orders:
            return []
        return build_orders(
            product=self.product,
            od=od,
            pos=pos,
            limit=self.limit,
            fv=self.fv,
            take_edge=self.take_edge,
            make_spread=self.make_spread,
            skew_sens=self.skew_sens,
            adverse_vol=self.adverse_vol,
            clear_thresh=self.clear_thresh,
        )
