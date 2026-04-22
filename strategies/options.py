"""Options IV scalping engine (Archetype 4: VOLCANIC_ROCK_VOUCHERS / COCONUT_COUPON).

Fits a quadratic vol smile across strikes, then trades individual options
when their market IV deviates from the smile-predicted IV. Delta-hedges
on the underlying.

Parameters to calibrate: tte (detect from data), open_thr (0.5), refit_every (1000).
"""
import math
import numpy as np
from datamodel import Order, OrderDepth
from strategies._base import bs_call, bs_vega, find_iv, bs_delta, detect_tte, best_mid


class OptionsTrader:
    """IV mean-reversion scalper with delta hedging.

    Args:
        underlying: underlying product symbol
        option_products: {strike: option_symbol} mapping
        underlying_limit: position limit for underlying
        option_limit: position limit per option
        tte: time to expiry in years (auto-detected if None)
        open_thr: normalized deviation threshold to open a position
        close_thr: normalized deviation threshold to close
        refit_every: refit smile every N ticks
        min_vega: minimum vega to consider an option tradeable
        trade_qty: order size per signal (higher = faster position build)
    """

    def __init__(
        self,
        underlying: str,
        option_products: dict[int, str],
        underlying_limit: int = 80,
        option_limit: int = 200,
        tte: float | None = None,
        open_thr: float = 0.5,
        close_thr: float = 0.0,
        refit_every: int = 1000,
        min_vega: float = 0.7,
        trade_qty: int = 5,
    ) -> None:
        self.underlying = underlying
        self.option_products = option_products
        self.underlying_limit = underlying_limit
        self.option_limit = option_limit
        self.tte_init = tte
        self.open_thr = open_thr
        self.close_thr = close_thr
        self.refit_every = refit_every
        self.min_vega = min_vega
        self.trade_qty = trade_qty

    def run(
        self,
        order_depths: dict[str, OrderDepth],
        positions: dict[str, int],
        td: dict,
        timestamp: int = 0,
    ) -> tuple[dict[str, list[Order]], dict]:
        """Generate orders for options and underlying.

        Returns:
            ({product: [orders]}, updated_td)
        """
        result: dict[str, list[Order]] = {}

        # Get underlying mid
        und_od = order_depths.get(self.underlying)
        if not und_od or not und_od.buy_orders or not und_od.sell_orders:
            return result, td

        spot = (max(und_od.buy_orders) + min(und_od.sell_orders)) / 2.0

        # Detect or load TTE (use closest-to-ATM option for accuracy)
        tte = td.get("tte", self.tte_init)
        if tte is None:
            strikes_by_atm = sorted(self.option_products.keys(), key=lambda k: abs(k - spot))
            for strike in strikes_by_atm:
                opt_sym = self.option_products[strike]
                opt_od = order_depths.get(opt_sym)
                if opt_od and opt_od.buy_orders and opt_od.sell_orders:
                    opt_mid = (max(opt_od.buy_orders) + min(opt_od.sell_orders)) / 2.0
                    tte = detect_tte(spot, opt_mid)
                    if tte is not None:
                        break
            if tte is None:
                tte = 7 / 365.0
            td["tte"] = tte

        # Collect IVs for smile fitting
        iv_data: list[tuple[float, float]] = []
        option_mids: dict[int, float] = {}

        for strike, opt_sym in self.option_products.items():
            opt_od = order_depths.get(opt_sym)
            if not opt_od or not opt_od.buy_orders or not opt_od.sell_orders:
                continue
            opt_mid = (max(opt_od.buy_orders) + min(opt_od.sell_orders)) / 2.0
            option_mids[strike] = opt_mid

            iv = find_iv(spot, float(strike), tte, 0, opt_mid)
            if iv is not None and 0.01 <= iv <= 2.0:
                m_t = math.log(strike / spot) / math.sqrt(tte) if tte > 0 else 0
                iv_data.append((m_t, iv))

        if len(iv_data) < 3:
            return result, td

        # Fit quadratic vol smile: iv = a*m² + b*m + c
        smile = td.get("smile")
        tick_count = td.get("tc", 0) + 1
        td["tc"] = tick_count

        if smile is None or tick_count % self.refit_every == 0:
            moneyness = np.array([d[0] for d in iv_data])
            ivs = np.array([d[1] for d in iv_data])
            try:
                coeffs = np.polyfit(moneyness, ivs, 2)
                smile = {"a": round(float(coeffs[0]), 6),
                         "b": round(float(coeffs[1]), 6),
                         "c": round(float(coeffs[2]), 6)}
                td["smile"] = smile
            except Exception:
                if smile is None:
                    return result, td

        a, b, c = smile["a"], smile["b"], smile["c"]

        # Trade options where market IV deviates from smile
        total_delta = 0.0

        for strike, opt_sym in self.option_products.items():
            if strike not in option_mids:
                continue

            opt_mid = option_mids[strike]
            m_t = math.log(strike / spot) / math.sqrt(tte) if tte > 0 else 0
            smile_iv = a * m_t ** 2 + b * m_t + c
            smile_iv = max(0.01, smile_iv)

            fair_price, fair_delta = bs_call(spot, float(strike), tte, 0, smile_iv)
            vega = bs_vega(spot, float(strike), tte, smile_iv)
            deviation = opt_mid - fair_price

            if vega < self.min_vega:
                continue

            opt_od = order_depths.get(opt_sym)
            if not opt_od:
                continue

            opt_pos = positions.get(opt_sym, 0)
            opt_orders: list[Order] = []

            norm_dev = deviation / vega if vega > 0 else 0

            if norm_dev > self.open_thr:
                # Option overpriced → sell
                rem_sell = self.option_limit + opt_pos
                if rem_sell > 0 and opt_od.buy_orders:
                    qty = min(rem_sell, self.trade_qty)
                    opt_orders.append(Order(opt_sym, max(opt_od.buy_orders), -qty))
            elif norm_dev < -self.open_thr:
                # Option underpriced → buy
                rem_buy = self.option_limit - opt_pos
                if rem_buy > 0 and opt_od.sell_orders:
                    qty = min(rem_buy, self.trade_qty)
                    opt_orders.append(Order(opt_sym, min(opt_od.sell_orders), qty))

            if opt_orders:
                result[opt_sym] = opt_orders

            total_delta += opt_pos * fair_delta

        # Delta-hedge on underlying
        und_pos = positions.get(self.underlying, 0)
        target_und = -round(total_delta)
        hedge_qty = int(target_und - und_pos)

        if hedge_qty != 0 and und_od.buy_orders and und_od.sell_orders:
            und_orders: list[Order] = []
            if hedge_qty > 0:
                rem = self.underlying_limit - und_pos
                qty = min(hedge_qty, rem)
                if qty > 0:
                    und_orders.append(Order(self.underlying, min(und_od.sell_orders), qty))
            else:
                rem = self.underlying_limit + und_pos
                qty = min(-hedge_qty, rem)
                if qty > 0:
                    und_orders.append(Order(self.underlying, max(und_od.buy_orders), -qty))
            if und_orders:
                result[self.underlying] = und_orders

        return result, td
