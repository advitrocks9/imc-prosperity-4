"""Drifting product market maker (Archetype 2: TOMATOES / KELP / STARFRUIT).

FV = 0.5 * wall_mid + 0.5 * vwap_top3, with optional AR(1) correction.
Three-phase orders: TAKE / CLEAR / MAKE with inventory skew.

Parameters to calibrate: make_spread (2-4), ar_beta (-0.1 to -0.35),
take_edge (1-3), skew_sens (1-3).
"""
from datamodel import Order, OrderDepth, Trade
from strategies._base import wall_mid, vwap_mid, build_orders, OliviaTrackerNoID


class DriftingTrader:
    """Market maker for non-stationary products with mean-reverting returns.

    Args:
        product: product symbol
        limit: position limit
        make_spread: passive quote distance from FV
        take_edge: minimum mispricing to cross the spread
        ar_beta: AR(1) reversion coefficient (negative = mean-reverting)
        skew_sens: inventory skew sensitivity
        adverse_vol: skip fills ≥ this size (informed bot filter)
        clear_thresh: fraction of limit to start aggressive clearing
        use_olivia: enable no-ID Olivia tracking
        olivia_mode: "widen" (default, widen spread) or "follow" (aggressive directional)
        fv_mode: "fv" (wall_mid+VWAP, default) or "follow" (best_bid+1/best_ask-1)
    """

    def __init__(
        self,
        product: str,
        limit: int,
        make_spread: int = 3,
        take_edge: float = 2.0,
        ar_beta: float = 0.0,
        skew_sens: float = 2.0,
        adverse_vol: int = 15,
        clear_thresh: float = 0.5,
        use_olivia: bool = False,
        olivia_mode: str = "widen",
        fv_mode: str = "fv",
        soft_limit_ratio: float = 0.6,
    ) -> None:
        self.product = product
        self.limit = limit
        self.make_spread = make_spread
        self.take_edge = take_edge
        self.ar_beta = ar_beta
        self.skew_sens = skew_sens
        self.adverse_vol = adverse_vol
        self.clear_thresh = clear_thresh
        self.use_olivia = use_olivia
        self.olivia_mode = olivia_mode
        self.fv_mode = fv_mode
        self.soft_limit_ratio = soft_limit_ratio

    def run(
        self,
        od: OrderDepth,
        pos: int,
        td: dict,
        market_trades: list[Trade] | None = None,
        ts: int = 0,
    ) -> tuple[list[Order], dict]:
        """Generate orders for one tick.

        Args:
            od: current order book
            pos: current position
            td: product-specific traderData dict (mutated and returned)
            market_trades: market trades this tick (for Olivia detection)
            ts: current timestamp

        Returns:
            (orders, updated_td)
        """
        if not od.buy_orders or not od.sell_orders:
            return [], td

        # ── Market-follow mode (best_bid+1/best_ask-1) ──────────
        if self.fv_mode == "follow":
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            bid_price = best_bid + 1
            ask_price = best_ask - 1
            if bid_price >= ask_price:
                return [], td

            rem_buy = self.limit - pos
            rem_sell = self.limit + pos
            buy_size = rem_buy
            sell_size = rem_sell

            if pos > self.limit * self.soft_limit_ratio and rem_buy > 0:
                excess = (pos - self.limit * self.soft_limit_ratio) / (
                    self.limit * (1 - self.soft_limit_ratio)
                )
                buy_size = max(1, int(rem_buy * (1 - excess * 0.8)))
            elif pos < -self.limit * self.soft_limit_ratio and rem_sell > 0:
                excess = (-pos - self.limit * self.soft_limit_ratio) / (
                    self.limit * (1 - self.soft_limit_ratio)
                )
                sell_size = max(1, int(rem_sell * (1 - excess * 0.8)))

            orders: list[Order] = []
            if buy_size > 0:
                orders.append(Order(self.product, bid_price, buy_size))
            if sell_size > 0:
                orders.append(Order(self.product, ask_price, -sell_size))
            return orders, td

        # ── FV-based mode (wall_mid + VWAP blend) ───────────────
        wm = wall_mid(od, fallback=td.get("wm"))
        vm = vwap_mid(od, n=3)

        if wm is not None and vm is not None:
            fv = 0.5 * wm + 0.5 * vm
        elif wm is not None:
            fv = wm
        elif vm is not None:
            fv = vm
        else:
            fv = td.get("fv")
            if fv is None:
                return [], td

        # AR(1) correction
        if self.ar_beta != 0.0:
            last_fv = td.get("fv")
            if last_fv is not None:
                fv = fv + self.ar_beta * (last_fv - fv)

        if wm is not None:
            td["wm"] = round(wm, 1)
        td["fv"] = round(fv, 2)

        # Olivia tracking
        active_make_spread = self.make_spread
        olivia_sig = 0
        if self.use_olivia and market_trades is not None:
            olivia = OliviaTrackerNoID.from_dict(td.get("ot", {}))
            olivia.update(self.product, market_trades, fv, ts)
            olivia_sig = olivia.signal(self.product, ts)
            td["ot"] = olivia.to_dict()

            if olivia_sig != 0 and self.olivia_mode == "widen":
                active_make_spread = max(self.make_spread + 2, 5)

        # Olivia "follow" mode: aggressive directional orders before passive MM
        pre_orders: list[Order] = []
        if olivia_sig != 0 and self.olivia_mode == "follow":
            best_bid = max(od.buy_orders) if od.buy_orders else None
            best_ask = min(od.sell_orders) if od.sell_orders else None
            follow_qty = min(self.limit // 4, 20)
            if olivia_sig == 1 and best_ask is not None:
                rem = self.limit - pos
                qty = min(follow_qty, rem)
                if qty > 0:
                    pre_orders.append(Order(self.product, best_ask, qty))
                    pos += qty
            elif olivia_sig == -1 and best_bid is not None:
                rem = self.limit + pos
                qty = min(follow_qty, rem)
                if qty > 0:
                    pre_orders.append(Order(self.product, best_bid, -qty))
                    pos -= qty

        orders = build_orders(
            product=self.product,
            od=od,
            pos=pos,
            limit=self.limit,
            fv=fv,
            take_edge=self.take_edge,
            make_spread=active_make_spread,
            skew_sens=self.skew_sens,
            adverse_vol=self.adverse_vol,
            clear_thresh=self.clear_thresh,
        )

        return pre_orders + orders, td
