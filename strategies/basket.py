"""Basket/ETF spread arbitrage (Archetype 3: PICNIC_BASKET / GIFT_BASKET).

Trades the basket against its synthetic NAV (sum of constituent prices × weights).
Uses Welford online mean to track the basket premium and trades when the
spread deviates significantly.

Parameters to calibrate: entry_thr (30 for wide-spread baskets, 60 for tight),
hedge_factor (0 is optimal), n_prior (60,000 for slow-reverting, 500 for fast).
"""
from datamodel import Order, OrderDepth
from strategies._base import BasketSpreadTracker, wall_mid


class BasketTrader:
    """Spread arbitrage between a basket product and its constituents.

    Args:
        basket_product: the basket symbol (e.g., "PICNIC_BASKET")
        constituents: {product_name: weight} for NAV calculation
        basket_limit: position limit for the basket product
        entry_thr: spread threshold for opening positions (30 for B1-like, 60 for B2-like)
        n_prior: Welford prior count (higher = more stable mean estimate)
        seed_mean: initial mean premium estimate (from offline analysis)
        hedge_factor: 0 = basket-only, 0.5 = half-hedge with constituents
        constituent_limits: {product: limit} for hedging (ignored if hedge_factor=0)
        reserve_ratio: fraction of constituent limit to keep free for future arb (default 0.4)
    """

    def __init__(
        self,
        basket_product: str,
        constituents: dict[str, float],
        basket_limit: int = 60,
        entry_thr: float = 30.0,
        n_prior: int = 60_000,
        seed_mean: float = 0.0,
        hedge_factor: float = 0.0,
        constituent_limits: dict[str, int] | None = None,
        reserve_ratio: float = 0.4,
    ) -> None:
        self.basket_product = basket_product
        self.constituents = constituents
        self.basket_limit = basket_limit
        self.entry_thr = entry_thr
        self.hedge_factor = hedge_factor
        self.constituent_limits = constituent_limits or {}
        self.reserve_ratio = reserve_ratio
        self._n_prior = n_prior
        self._seed_mean = seed_mean

    def run(
        self,
        order_depths: dict[str, OrderDepth],
        positions: dict[str, int],
        td: dict,
    ) -> tuple[dict[str, list[Order]], dict]:
        """Generate orders for basket and optionally constituents.

        Returns:
            ({product: [orders]}, updated_td)
        """
        result: dict[str, list[Order]] = {}

        basket_od = order_depths.get(self.basket_product)
        if not basket_od or not basket_od.buy_orders or not basket_od.sell_orders:
            return result, td

        # Use wall_mid for lower noise (max dev 0.49 vs 4.49 for best_mid)
        basket_wm = wall_mid(basket_od)
        if basket_wm is None:
            return result, td
        basket_mid = basket_wm

        # Compute synthetic NAV from constituents
        synthetic_mid = 0.0
        for product, weight in self.constituents.items():
            cod = order_depths.get(product)
            if not cod:
                return result, td  # can't compute NAV without all constituents
            cwm = wall_mid(cod)
            if cwm is None:
                return result, td
            synthetic_mid += weight * cwm

        # Load/create tracker
        tracker = BasketSpreadTracker.from_dict(td.get("bst", {
            "n": self._n_prior, "mp": self._seed_mean, "et": self.entry_thr,
        }))
        tracker.entry_thr = self.entry_thr  # allow runtime override

        sig = tracker.signal(basket_mid, synthetic_mid)
        td["bst"] = tracker.to_dict()
        td["spread"] = round(basket_mid - synthetic_mid, 2)

        basket_pos = positions.get(self.basket_product, 0)
        rem_buy = self.basket_limit - basket_pos
        rem_sell = self.basket_limit + basket_pos

        basket_orders: list[Order] = []

        if sig == 1 and rem_buy > 0:
            # Basket cheap → buy basket
            best_ask = min(basket_od.sell_orders)
            basket_orders.append(Order(self.basket_product, best_ask, rem_buy))
        elif sig == -1 and rem_sell > 0:
            # Basket expensive → sell basket
            best_bid = max(basket_od.buy_orders)
            basket_orders.append(Order(self.basket_product, best_bid, -rem_sell))

        if basket_orders:
            result[self.basket_product] = basket_orders

        # Optional: half-hedge with constituents
        if self.hedge_factor > 0 and sig != 0:
            for product, weight in self.constituents.items():
                cod = order_depths.get(product)
                if not cod or not cod.buy_orders or not cod.sell_orders:
                    continue
                climit = self.constituent_limits.get(product, 0)
                if climit <= 0:
                    continue

                cpos = positions.get(product, 0)
                # Reserve capacity for future arb trades
                usable_limit = int(climit * (1 - self.reserve_ratio))
                # Hedge in opposite direction of basket trade
                if sig == 1:
                    # Bought basket → sell constituents
                    available = max(0, usable_limit + cpos)
                    hedge_qty = min(int(weight * self.hedge_factor), available)
                    if hedge_qty > 0:
                        result.setdefault(product, []).append(
                            Order(product, max(cod.buy_orders), -hedge_qty)
                        )
                elif sig == -1:
                    # Sold basket → buy constituents
                    available = max(0, usable_limit - cpos)
                    hedge_qty = min(int(weight * self.hedge_factor), available)
                    if hedge_qty > 0:
                        result.setdefault(product, []).append(
                            Order(product, min(cod.sell_orders), hedge_qty)
                        )

        return result, td
