"""Voucher z-score IV scalper (Signal A from E_002 research).

Hypothesis: ATM IV (smile intercept c) is mean-reverting around an EMA. When
c_t deviates >k sigma from EMA_c, vol is regime-elevated/depressed → trade
vega-neutral position in the direction of mean-reversion.

Architecture:
1. Each tick, refit quadratic smile across liquid strikes:
       iv(m) = a*m^2 + b*m + c    where m = log(K/S)/sqrt(T)
2. Track EMA of c (alpha=0.05) and rolling std of (c - EMA_c).
3. z = (c_t - EMA_c) / rolling_std.
4. If |z| > open_thr: target sign = -sign(z), vega-weight per strike.
5. If |z| < close_thr: flatten.
6. Optional: delta hedge every N ticks via underlying.

Key differences vs OptionsTrader:
- OptionsTrader trades each strike vs its own smile-fit theo (cross-sectional).
- This trades a TIME-SERIES signal on the smile LEVEL (vol regime change).
"""
import math
from datamodel import Order, OrderDepth
from strategies._base import bs_call, bs_vega, bs_delta, find_iv, wall_mid


def _fit_quadratic_smile(
    moneyness: list[float], ivs: list[float],
) -> tuple[float, float, float, float] | None:
    """OLS fit iv = a·m² + b·m + c. Returns (a, b, c, r2) or None."""
    n = len(moneyness)
    if n < 3:
        return None
    sx = sum(moneyness)
    sx2 = sum(m * m for m in moneyness)
    sx3 = sum(m ** 3 for m in moneyness)
    sx4 = sum(m ** 4 for m in moneyness)
    sy = sum(ivs)
    sxy = sum(m * v for m, v in zip(moneyness, ivs))
    sx2y = sum(m * m * v for m, v in zip(moneyness, ivs))
    # Normal equations 3x3
    A = [
        [sx4, sx3, sx2],
        [sx3, sx2, sx],
        [sx2, sx, n],
    ]
    rhs = [sx2y, sxy, sy]
    sol = _solve3(A, rhs)
    if sol is None:
        return None
    a, b, c = sol
    yhat = [a * m * m + b * m + c for m in moneyness]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ivs, yhat))
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for y in ivs)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return a, b, c, r2


def _solve3(A: list[list[float]], b: list[float]) -> tuple[float, float, float] | None:
    """Cramer's rule on 3x3."""
    def det3(M):
        return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
              - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
              + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    D = det3(A)
    if abs(D) < 1e-12:
        return None
    Mx = [row[:] for row in A]
    for i in range(3):
        Mx[i][0] = b[i]
    My = [row[:] for row in A]
    for i in range(3):
        My[i][1] = b[i]
    Mz = [row[:] for row in A]
    for i in range(3):
        Mz[i][2] = b[i]
    return det3(Mx) / D, det3(My) / D, det3(Mz) / D


class VoucherZIVTrader:
    """Z-score-based ATM IV mean reversion.

    Args:
        underlying: underlying product symbol
        option_products: {strike: option_symbol}
        option_limit: per-strike voucher limit (default 300)
        underlying_limit: underlying limit (default 200)
        open_thr: |z| threshold to open position (default 1.5)
        close_thr: |z| threshold to flatten (default 0.5)
        ema_alpha: EMA alpha for c (default 0.05, halflife ~14 ticks)
        std_window: rolling-std window (default 100)
        vega_target: target vega per strike (default 20)
        max_pos_per_strike: hard cap per strike (default 25)
        min_vega: skip strikes with vega below this (default 0.5)
        min_r2: skip if smile fit R² below (default 0.85)
        hedge_every: hedge cadence in ticks (default 50; 0 = disabled)
        hedge_threshold: |net delta| to trigger hedge (default 30)
        hedge_fraction: fraction of net delta to hedge (default 0.5)
        tte_func: callable(timestamp) -> tte_years
    """

    def __init__(
        self,
        underlying: str,
        option_products: dict[int, str],
        option_limit: int = 300,
        underlying_limit: int = 200,
        open_thr: float = 1.5,
        close_thr: float = 0.5,
        ema_alpha: float = 0.05,
        std_window: int = 100,
        vega_target: float = 20.0,
        max_pos_per_strike: int = 25,
        min_vega: float = 0.5,
        min_r2: float = 0.85,
        hedge_every: int = 50,
        hedge_threshold: int = 30,
        hedge_fraction: float = 0.5,
        tte_func=None,
    ) -> None:
        self.underlying = underlying
        self.option_products = option_products
        self.option_limit = option_limit
        self.underlying_limit = underlying_limit
        self.open_thr = open_thr
        self.close_thr = close_thr
        self.ema_alpha = ema_alpha
        self.std_window = std_window
        self.vega_target = vega_target
        self.max_pos_per_strike = max_pos_per_strike
        self.min_vega = min_vega
        self.min_r2 = min_r2
        self.hedge_every = hedge_every
        self.hedge_threshold = hedge_threshold
        self.hedge_fraction = hedge_fraction
        self.tte_func = tte_func or (lambda ts: 5.0 / 365.0)

    def _solve_per_strike_iv(
        self, spot: float, T: float, order_depths: dict
    ) -> dict[int, float]:
        """For each strike, solve IV from market mid. Returns dict {K: iv}."""
        ivs: dict[int, float] = {}
        for strike, sym in self.option_products.items():
            od = order_depths.get(sym)
            if od is None:
                continue
            mid = wall_mid(od)
            if mid is None or mid <= 0:
                continue
            iv = find_iv(spot, float(strike), T, 0.0, mid)
            if iv is not None and 0.05 <= iv <= 2.0:
                ivs[strike] = iv
        return ivs

    def run(
        self,
        order_depths: dict,
        position: dict,
        td: dict,
        timestamp: int,
    ) -> tuple[dict[str, list[Order]], dict]:
        result: dict[str, list[Order]] = {}

        # 1. Spot
        u_od = order_depths.get(self.underlying)
        if u_od is None:
            return result, td
        spot = wall_mid(u_od)
        if spot is None:
            return result, td

        # 2. TTE
        T = self.tte_func(timestamp)
        if T <= 0:
            return result, td

        # 3. Per-strike IV from market mid
        ivs = self._solve_per_strike_iv(spot, T, order_depths)
        if len(ivs) < 3:
            return result, td

        # 4. Fit smile
        sqrtT = math.sqrt(T)
        moneyness = [math.log(k / spot) / sqrtT for k in ivs.keys()]
        iv_list = list(ivs.values())
        fit = _fit_quadratic_smile(moneyness, iv_list)
        if fit is None:
            return result, td
        a, b, c, r2 = fit

        # 5. Update EMA + rolling std of (c - EMA)
        ema_c = td.get("ema_c", c)
        ema_c = self.ema_alpha * c + (1.0 - self.ema_alpha) * ema_c
        td["ema_c"] = ema_c
        dev = c - ema_c
        # Rolling deviations buffer
        devs: list = td.get("dev_buf", [])
        devs.append(dev)
        if len(devs) > self.std_window:
            devs = devs[-self.std_window:]
        td["dev_buf"] = devs
        if len(devs) < 20:
            # Not enough history yet; just refit and wait
            td["c"] = c
            td["r2"] = r2
            return result, td

        mean_dev = sum(devs) / len(devs)
        var = sum((d - mean_dev) ** 2 for d in devs) / max(1, len(devs) - 1)
        std = math.sqrt(var) if var > 0 else 1e-9
        z = dev / std if std > 0 else 0.0
        td["c"] = c
        td["r2"] = r2
        td["z"] = z

        # 6. Decide target direction
        if r2 < self.min_r2:
            # Smile fit is bad; reduce risk by flattening
            target_dir = 0
            mode = "no_fit"
        elif z > self.open_thr:
            target_dir = -1   # vol elevated → short vouchers (sell vega)
            mode = "short_vol"
        elif z < -self.open_thr:
            target_dir = +1   # vol depressed → long vouchers
            mode = "long_vol"
        elif abs(z) < self.close_thr:
            target_dir = 0
            mode = "flatten"
        else:
            # Hold - don't change position
            target_dir = None
            mode = "hold"
        td["mode"] = mode

        # 7. Per-strike target position (vega-weighted)
        targets: dict[int, int] = {}
        if target_dir is not None:
            for strike in self.option_products.keys():
                if target_dir == 0:
                    targets[strike] = 0
                    continue
                iv_k = ivs.get(strike)
                if iv_k is None:
                    targets[strike] = 0
                    continue
                vega = bs_vega(spot, float(strike), T, iv_k)
                if vega < self.min_vega:
                    targets[strike] = 0
                    continue
                qty = self.vega_target / max(vega, self.min_vega)
                qty = min(qty, float(self.max_pos_per_strike))
                targets[strike] = int(round(target_dir * qty))

        # 8. Generate orders to TAKE toward target
        if target_dir is not None:
            for strike, sym in self.option_products.items():
                target = targets.get(strike, 0)
                cur = position.get(sym, 0)
                delta_pos = target - cur
                if delta_pos == 0:
                    continue
                od = order_depths.get(sym)
                if od is None:
                    continue
                orders: list[Order] = []
                rem = abs(delta_pos)
                if delta_pos > 0:
                    # Buy: lift asks
                    for ask in sorted(od.sell_orders.keys()):
                        if rem <= 0:
                            break
                        avail = abs(od.sell_orders[ask])
                        qty = min(avail, rem)
                        orders.append(Order(sym, int(ask), qty))
                        rem -= qty
                else:
                    # Sell: hit bids
                    for bid in sorted(od.buy_orders.keys(), reverse=True):
                        if rem <= 0:
                            break
                        avail = od.buy_orders[bid]
                        qty = min(avail, rem)
                        orders.append(Order(sym, int(bid), -qty))
                        rem -= qty
                if orders:
                    result[sym] = orders

        # 9. Delta hedge via underlying
        if self.hedge_every > 0 and timestamp % self.hedge_every == 0:
            net_delta = 0.0
            for strike, sym in self.option_products.items():
                pos_k = position.get(sym, 0)
                if pos_k == 0:
                    continue
                iv_k = ivs.get(strike)
                if iv_k is None:
                    continue
                d = bs_delta(spot, float(strike), T, 0.0, iv_k)
                net_delta += pos_k * d
            if abs(net_delta) > self.hedge_threshold:
                hedge_qty = -int(round(net_delta * self.hedge_fraction))
                u_pos = position.get(self.underlying, 0)
                # Cap to limit
                if hedge_qty > 0:
                    hedge_qty = min(hedge_qty, self.underlying_limit - u_pos)
                else:
                    hedge_qty = max(hedge_qty, -self.underlying_limit - u_pos)
                if hedge_qty != 0:
                    if hedge_qty > 0:
                        # Buy underlying
                        rem = hedge_qty
                        u_orders: list[Order] = []
                        for ask in sorted(u_od.sell_orders.keys()):
                            if rem <= 0:
                                break
                            avail = abs(u_od.sell_orders[ask])
                            qty = min(avail, rem)
                            u_orders.append(Order(self.underlying, int(ask), qty))
                            rem -= qty
                        if u_orders:
                            existing = result.get(self.underlying, [])
                            result[self.underlying] = existing + u_orders
                    else:
                        rem = abs(hedge_qty)
                        u_orders = []
                        for bid in sorted(u_od.buy_orders.keys(), reverse=True):
                            if rem <= 0:
                                break
                            avail = u_od.buy_orders[bid]
                            qty = min(avail, rem)
                            u_orders.append(Order(self.underlying, int(bid), -qty))
                            rem -= qty
                        if u_orders:
                            existing = result.get(self.underlying, [])
                            result[self.underlying] = existing + u_orders

        return result, td
