"""Signal-driven strategy (Archetype 6: SQUID_INK / DIVING_GEAR).

Uses dual-EMA z-score for spike detection and Olivia copy-trading.
Spikes revert - sell on up-spikes, buy on down-spikes.

Parameters to calibrate: z_buy/z_sell (5-15), alpha_fast (0.3), alpha_slow (0.05).
"""
from datamodel import Order, OrderDepth, Trade
from strategies._base import DualEMAZScore, OliviaTrackerNoID, build_orders, wall_mid


class SignalTrader:
    """Spike reversal + informed bot copy-trading.

    Args:
        product: product symbol
        limit: position limit
        z_buy: z-score threshold to open long (buy on down-spike)
        z_sell: z-score threshold to open short (sell on up-spike)
        alpha_fast: fast EMA alpha
        alpha_slow: slow EMA alpha
        make_spread: passive quote distance when not signaling
        use_olivia: enable no-ID Olivia tracking
        olivia_spread_widen: extra spread ticks when Olivia is active
    """

    def __init__(
        self,
        product: str,
        limit: int = 80,
        z_buy: float = 5.0,
        z_sell: float = 5.0,
        alpha_fast: float = 0.3,
        alpha_slow: float = 0.05,
        make_spread: int = 3,
        use_olivia: bool = True,
        olivia_spread_widen: int = 2,
    ) -> None:
        self.product = product
        self.limit = limit
        self.z_buy = z_buy
        self.z_sell = z_sell
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.make_spread = make_spread
        self.use_olivia = use_olivia
        self.olivia_spread_widen = olivia_spread_widen

    def run(
        self,
        od: OrderDepth,
        pos: int,
        td: dict,
        market_trades: list[Trade] | None = None,
        ts: int = 0,
    ) -> tuple[list[Order], dict]:
        """Generate orders for one tick.

        Returns:
            (orders, updated_td)
        """
        if not od.buy_orders or not od.sell_orders:
            return [], td

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid = (best_bid + best_ask) / 2.0

        # Load/create z-score tracker
        zscore_data = td.get("zs")
        if zscore_data:
            zs = DualEMAZScore.from_dict(zscore_data)
        else:
            zs = DualEMAZScore(self.alpha_fast, self.alpha_slow)

        z = zs.update(mid)
        td["zs"] = zs.to_dict()

        # Olivia tracking
        olivia_sig = 0
        if self.use_olivia and market_trades is not None:
            olivia = OliviaTrackerNoID.from_dict(td.get("ot", {}))
            olivia.update(self.product, market_trades, mid, ts)
            olivia_sig = olivia.signal(self.product, ts)
            td["ot"] = olivia.to_dict()

        orders: list[Order] = []
        rem_buy = self.limit - pos
        rem_sell = self.limit + pos

        # Signal-based aggressive orders
        if z is not None:
            if z > self.z_sell and rem_sell > 0:
                # Price spiked up → sell (bet on reversion)
                qty = min(rem_sell, self.limit // 4)
                orders.append(Order(self.product, best_bid, -qty))
                rem_sell -= qty
            elif z < -self.z_buy and rem_buy > 0:
                # Price spiked down → buy (bet on reversion)
                qty = min(rem_buy, self.limit // 4)
                orders.append(Order(self.product, best_ask, qty))
                rem_buy -= qty

        # Olivia copy-trade (directional, additive to z-score)
        if olivia_sig == 1 and rem_buy > 0:
            qty = min(rem_buy, 10)
            orders.append(Order(self.product, best_ask, qty))
            rem_buy -= qty
        elif olivia_sig == -1 and rem_sell > 0:
            qty = min(rem_sell, 10)
            orders.append(Order(self.product, best_bid, -qty))
            rem_sell -= qty

        # Passive market-making (widen spread when Olivia active)
        spread = self.make_spread
        if olivia_sig != 0:
            spread += self.olivia_spread_widen

        # Use wall_mid for FV if available, else simple mid
        fv = wall_mid(od, fallback=mid) or mid

        bid_price = int(round(fv)) - spread
        ask_price = int(round(fv)) + spread

        # Don't cross the spread
        if bid_price < ask_price:
            if rem_buy > 0:
                orders.append(Order(self.product, bid_price, rem_buy))
            if rem_sell > 0:
                orders.append(Order(self.product, ask_price, -rem_sell))

        return orders, td
