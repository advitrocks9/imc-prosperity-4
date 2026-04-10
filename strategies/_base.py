"""
Shared components for all Prosperity 4 strategies.

Every strategy file imports from here. Nothing in this file is pseudocode.
All implementations are competition-ready (stdlib + numpy/pandas/jsonpickle only).

Components:
  1. wall_mid / vwap_mid - fair value estimators
  2. EMA / RingBuffer / DualEMAZScore - time-series primitives
  3. OliviaTracker / OliviaTrackerNoID - informed bot detection
  4. build_orders - three-phase order generation (TAKE/CLEAR/MAKE)
  5. StateManager - traderData JSON wrapper with budget guard
  6. Logger - jmerle-visualizer compatible logging
  7. Black-Scholes suite - options pricing
  8. BasketSpreadTracker - Welford online premium estimator
  9. conversion_arb - cross-exchange arb calculator
"""
import json
import math
from statistics import NormalDist
from datamodel import Order, OrderDepth, ConversionObservation, TradingState, Trade


# ─────────────────────────────────────────────────────────────────
# 1. Fair Value Estimators
# ─────────────────────────────────────────────────────────────────

def wall_mid(order_depth: OrderDepth, fallback: float | None = None) -> float | None:
    """Midpoint of WORST bid and WORST ask (deepest liquidity walls).

    Empirically the best single FV estimator across all P2/P3 products.
    """
    bids = order_depth.buy_orders
    asks = order_depth.sell_orders
    if not bids or not asks:
        return fallback
    return (min(bids.keys()) + max(asks.keys())) / 2.0


def vwap_mid(order_depth: OrderDepth, n: int = 3) -> float | None:
    """Volume-weighted mid across top-N levels on each side.

    n=3 is optimal per community analysis. Complements wall_mid.
    """
    bids = sorted(order_depth.buy_orders.items(), reverse=True)[:n]
    asks = sorted(order_depth.sell_orders.items())[:n]
    if not bids or not asks:
        return None
    bid_vol = sum(v for _, v in bids)
    ask_vol = sum(-v for _, v in asks)  # sell_orders values are negative
    if bid_vol <= 0 or ask_vol <= 0:
        return None
    bid_vwap = sum(p * v for p, v in bids) / bid_vol
    ask_vwap = sum(p * (-v) for p, v in asks) / ask_vol
    return (bid_vwap + ask_vwap) / 2.0


def best_mid(order_depth: OrderDepth) -> float | None:
    """Simple mid = (best_bid + best_ask) / 2."""
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0


# ─────────────────────────────────────────────────────────────────
# 2. Time-Series Primitives
# ─────────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average with clean cold-start and serialization."""

    def __init__(self, alpha: float, init_val: float | None = None) -> None:
        self.alpha = alpha
        self._val: float | None = init_val

    def update(self, x: float) -> float:
        if self._val is None:
            self._val = x
        else:
            self._val = self.alpha * x + (1.0 - self.alpha) * self._val
        return self._val

    def value(self) -> float | None:
        return self._val

    def to_dict(self) -> dict:
        return {"a": self.alpha, "v": self._val}

    @classmethod
    def from_dict(cls, d: dict) -> "EMA":
        return cls(d["a"], d.get("v"))


def alpha_from_halflife(halflife_ticks: float) -> float:
    """Convert half-life in ticks to EMA alpha. halflife=5 → alpha≈0.129."""
    return 1.0 - math.exp(-math.log(2) / halflife_ticks)


class RingBuffer:
    """O(1) fixed-size circular buffer with running sum and sum-of-squares."""

    def __init__(self, maxlen: int) -> None:
        self._maxlen = maxlen
        self._buf: list[float] = [0.0] * maxlen
        self._i: int = 0
        self._n: int = 0
        self._s: float = 0.0
        self._sq: float = 0.0

    def append(self, x: float) -> None:
        if self._n == self._maxlen:
            old = self._buf[self._i]
            self._s -= old
            self._sq -= old * old
        else:
            self._n += 1
        self._buf[self._i] = x
        self._s += x
        self._sq += x * x
        self._i = (self._i + 1) % self._maxlen

    def mean(self) -> float | None:
        return self._s / self._n if self._n > 0 else None

    def std(self) -> float | None:
        if self._n < 2:
            return None
        variance = (self._sq - self._s * self._s / self._n) / (self._n - 1)
        return math.sqrt(max(variance, 0.0))

    def full(self) -> bool:
        return self._n == self._maxlen

    def __len__(self) -> int:
        return self._n

    def to_dict(self) -> dict:
        return {
            "ml": self._maxlen, "buf": self._buf,
            "i": self._i, "n": self._n,
            "s": round(self._s, 4), "sq": round(self._sq, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RingBuffer":
        obj = cls(d["ml"])
        obj._buf = d["buf"]
        obj._i = d["i"]
        obj._n = d["n"]
        obj._s = d["s"]
        obj._sq = d["sq"]
        return obj


class DualEMAZScore:
    """Fast EMA deviation z-scored by slow EMA stddev. For signal-driven products."""

    def __init__(self, alpha_fast: float = 0.3, alpha_slow: float = 0.05) -> None:
        self.fast = EMA(alpha_fast)
        self.slow = EMA(alpha_slow)
        self.std_slow = EMA(alpha_slow)

    def update(self, x: float) -> float | None:
        fast_val = self.fast.update(x)
        slow_val = self.slow.update(x)
        dev = abs(x - slow_val)
        std_val = self.std_slow.update(dev)
        if std_val is None or std_val < 1e-8:
            return None
        return (fast_val - slow_val) / std_val

    def to_dict(self) -> dict:
        return {"f": self.fast.to_dict(), "s": self.slow.to_dict(), "sd": self.std_slow.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "DualEMAZScore":
        obj = cls.__new__(cls)
        obj.fast = EMA.from_dict(d["f"])
        obj.slow = EMA.from_dict(d["s"])
        obj.std_slow = EMA.from_dict(d["sd"])
        return obj


# ─────────────────────────────────────────────────────────────────
# 3. Olivia Tracker (informed bot detection)
# ─────────────────────────────────────────────────────────────────

class OliviaTracker:
    """Copy-trade the informed bot when trader IDs are visible (Round 5+)."""

    INFORMED_IDS: set[str] = {"Olivia"}
    SIGNAL_TTL: int = 500

    def __init__(self) -> None:
        self.buy_ts: dict[str, int] = {}
        self.sell_ts: dict[str, int] = {}

    def update(self, product: str, trades: list[Trade], ts: int) -> None:
        for t in trades:
            if t.buyer in self.INFORMED_IDS:
                self.buy_ts[product] = ts
            if t.seller in self.INFORMED_IDS:
                self.sell_ts[product] = ts

    def signal(self, product: str, ts: int) -> int:
        """Returns +1 (bullish), -1 (bearish), or 0 (no signal)."""
        buy_age = ts - self.buy_ts.get(product, -999_999)
        sell_age = ts - self.sell_ts.get(product, -999_999)
        if buy_age <= self.SIGNAL_TTL and sell_age <= self.SIGNAL_TTL:
            return 1 if self.buy_ts[product] > self.sell_ts[product] else -1
        if buy_age <= self.SIGNAL_TTL:
            return 1
        if sell_age <= self.SIGNAL_TTL:
            return -1
        return 0

    def to_dict(self) -> dict:
        return {"bt": self.buy_ts, "st": self.sell_ts}

    @classmethod
    def from_dict(cls, d: dict) -> "OliviaTracker":
        obj = cls()
        obj.buy_ts = d.get("bt", {})
        obj.sell_ts = d.get("st", {})
        return obj


class OliviaTrackerNoID:
    """Proxy Olivia detection for Rounds 1-4 (no trader IDs).

    Detects by matching trade.quantity in TRADE_SIZES at daily price extremes.
    P3 data shows Olivia uses different sizes per product (3, 14, 15).
    """

    EXTREME_MARGIN: float = 1.5
    SIGNAL_TTL: int = 500

    def __init__(self, trade_sizes: set[int] | None = None) -> None:
        self.trade_sizes: set[int] = trade_sizes or {3, 14, 15}
        self.buy_ts: dict[str, int] = {}
        self.sell_ts: dict[str, int] = {}
        self.daily_low: dict[str, float] = {}
        self.daily_high: dict[str, float] = {}

    def update(self, product: str, trades: list[Trade], mid: float, ts: int) -> None:
        if ts == 0:
            self.daily_low[product] = mid
            self.daily_high[product] = mid

        low = self.daily_low.get(product, mid)
        high = self.daily_high.get(product, mid)
        self.daily_low[product] = min(low, mid)
        self.daily_high[product] = max(high, mid)

        for t in trades:
            if t.quantity in self.trade_sizes:
                if t.price <= self.daily_low[product] + self.EXTREME_MARGIN:
                    self.buy_ts[product] = ts
                elif t.price >= self.daily_high[product] - self.EXTREME_MARGIN:
                    self.sell_ts[product] = ts

    def signal(self, product: str, ts: int) -> int:
        buy_age = ts - self.buy_ts.get(product, -999_999)
        sell_age = ts - self.sell_ts.get(product, -999_999)
        if buy_age <= self.SIGNAL_TTL and sell_age <= self.SIGNAL_TTL:
            return 1 if self.buy_ts[product] > self.sell_ts[product] else -1
        if buy_age <= self.SIGNAL_TTL:
            return 1
        if sell_age <= self.SIGNAL_TTL:
            return -1
        return 0

    def to_dict(self) -> dict:
        return {"bt": self.buy_ts, "st": self.sell_ts, "dl": self.daily_low, "dh": self.daily_high,
                "ts": list(self.trade_sizes)}

    @classmethod
    def from_dict(cls, d: dict) -> "OliviaTrackerNoID":
        obj = cls(trade_sizes=set(d.get("ts", [3, 14, 15])))
        obj.buy_ts = d.get("bt", {})
        obj.sell_ts = d.get("st", {})
        obj.daily_low = d.get("dl", {})
        obj.daily_high = d.get("dh", {})
        return obj


# ─────────────────────────────────────────────────────────────────
# 4. Three-Phase Order Builder (TAKE / CLEAR / MAKE)
# ─────────────────────────────────────────────────────────────────

def build_orders(
    product: str,
    od: OrderDepth,
    pos: int,
    limit: int,
    fv: float,
    take_edge: float = 2.0,
    make_spread: int = 3,
    skew_sens: float = 2.0,
    adverse_vol: int = 15,
    clear_thresh: float = 0.5,
) -> list[Order]:
    """Position-aware TAKE / CLEAR / MAKE order generator.

    take_edge: only take when price deviates from FV by more than this.
    make_spread: passive quote distance from FV.
    skew_sens: how aggressively to skew quotes per unit of (pos/limit).
    adverse_vol: skip fills from lots this size or larger (informed-bot filter).
    clear_thresh: fraction of limit at which to start aggressive unwinding.
    """
    orders: list[Order] = []
    rem_buy = limit - pos
    rem_sell = limit + pos

    # Phase 1: TAKE - cross the spread on clear mispricings
    for ask in sorted(od.sell_orders):
        if ask >= fv - take_edge or rem_buy <= 0:
            break
        vol = -od.sell_orders[ask]
        if vol >= adverse_vol:
            continue
        qty = min(vol, rem_buy)
        orders.append(Order(product, ask, qty))
        rem_buy -= qty
        pos += qty

    for bid in sorted(od.buy_orders, reverse=True):
        if bid <= fv + take_edge or rem_sell <= 0:
            break
        vol = od.buy_orders[bid]
        if vol >= adverse_vol:
            continue
        qty = min(vol, rem_sell)
        orders.append(Order(product, bid, -qty))
        rem_sell -= qty
        pos -= qty

    # Phase 2: CLEAR - reduce inventory aggressively when near limit
    if abs(pos) >= clear_thresh * limit and pos != 0:
        if pos > 0 and rem_sell > 0:
            clear_price = int(fv) - 1
            clear_qty = min(abs(pos), rem_sell)
            orders.append(Order(product, clear_price, -clear_qty))
            rem_sell -= clear_qty
        elif pos < 0 and rem_buy > 0:
            clear_price = int(fv) + 1
            clear_qty = min(abs(pos), rem_buy)
            orders.append(Order(product, clear_price, clear_qty))
            rem_buy -= clear_qty

    # Phase 3: MAKE - passive quotes with inventory skew
    skew = round(skew_sens * pos / limit)
    bid_p = int(round(fv)) - make_spread - skew
    ask_p = int(round(fv)) + make_spread - skew

    if rem_buy > 0:
        orders.append(Order(product, bid_p, rem_buy))
    if rem_sell > 0:
        orders.append(Order(product, ask_p, -rem_sell))

    return orders


# ─────────────────────────────────────────────────────────────────
# 5. State Manager (traderData wrapper)
# ─────────────────────────────────────────────────────────────────

class StateManager:
    """Thread-local traderData wrapper with corruption guard and budget check.

    Budget: 50,000 chars hard limit. Use short keys (2-4 chars).
    """

    MAX_LEN: int = 49_000

    def __init__(self, trader_data_str: str) -> None:
        try:
            self._state: dict = json.loads(trader_data_str) if trader_data_str else {}
        except Exception:
            self._state = {}

    def get(self, key: str, default=None):
        return self._state.get(key, default)

    def set(self, key: str, value) -> None:
        self._state[key] = value

    def update(self, d: dict) -> None:
        self._state.update(d)

    def dump(self) -> str:
        try:
            s = json.dumps(self._state, separators=(",", ":"))
        except Exception:
            return "{}"
        # Drop largest keys until under budget
        while len(s) > self.MAX_LEN and self._state:
            largest_key = max(self._state, key=lambda k: len(json.dumps(self._state[k])))
            del self._state[largest_key]
            try:
                s = json.dumps(self._state, separators=(",", ":"))
            except Exception:
                return "{}"
        return s


# ─────────────────────────────────────────────────────────────────
# 6. Logger (jmerle-visualizer compatible)
# ─────────────────────────────────────────────────────────────────

class ProsperityEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


class Logger:
    """Accumulates log lines and emits one JSON blob per tick for the visualizer."""

    def __init__(self) -> None:
        self.logs: str = ""
        self.max_log_length: int = 3750

    def print(self, *objects, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[str, list[Order]],
              conversions: int, td: str) -> None:
        base_length = len(self._to_json([
            self._compress_state(state, ""),
            self._compress_orders(orders),
            conversions, "", "",
        ]))
        max_item = (self.max_log_length - base_length) // 3
        print(self._to_json([
            self._compress_state(state, self._truncate(state.traderData, max_item)),
            self._compress_orders(orders),
            conversions,
            self._truncate(td, max_item),
            self._truncate(self.logs, max_item),
        ]))
        self.logs = ""

    def _compress_state(self, state: TradingState, trader_data: str) -> list:
        return [
            state.timestamp, trader_data,
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
            {s: [od.buy_orders, od.sell_orders] for s, od in state.order_depths.items()},
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
             for arr in state.own_trades.values() for t in arr],
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
             for arr in state.market_trades.values() for t in arr],
            state.position,
            [state.observations.plainValueObservations,
             {p: [o.bidPrice, o.askPrice, o.transportFees,
                  o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
              for p, o in state.observations.conversionObservations.items()}],
        ]

    def _compress_orders(self, orders: dict[str, list[Order]]) -> list:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def _to_json(self, value) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def _truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[:max_length - 3] + "..."


logger = Logger()


# ─────────────────────────────────────────────────────────────────
# 7. Black-Scholes Suite
# ─────────────────────────────────────────────────────────────────

_N = NormalDist()


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Returns (call_price, delta). Handles T≤0 and sigma≤0 gracefully."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0)
        delta = 1.0 if S > K else 0.0
        return intrinsic, delta
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    price = S * _N.cdf(d1) - K * math.exp(-r * T) * _N.cdf(d2)
    delta = _N.cdf(d1)
    return price, delta


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Returns (put_price, put_delta) via put-call parity."""
    call_price, call_delta = bs_call(S, K, T, r, sigma)
    put_price = call_price - S + K * math.exp(-r * T)
    put_delta = call_delta - 1.0
    return put_price, put_delta


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    """d(price)/d(sigma). Newton-Raphson denominator for find_iv."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    return S * _N.pdf(d1) * math.sqrt(T)


def find_iv(
    S: float, K: float, T: float, r: float, target: float,
    tol: float = 1e-6, max_iter: int = 50,
) -> float | None:
    """Newton-Raphson with bisection fallback. Returns IV or None if unsolvable."""
    if T <= 0 or target <= 0:
        return None

    # Brenner-Subrahmanyam ATM approximation as seed
    sigma = math.sqrt(2 * math.pi / T) * (target / S)
    sigma = max(1e-6, min(sigma, 5.0))

    for _ in range(max_iter):
        price, _ = bs_call(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, sigma)
        if vega < 1e-10:
            break
        sigma -= (price - target) / vega
        sigma = max(1e-6, min(sigma, 5.0))
        if abs(price - target) < tol:
            return sigma

    # Bisection fallback
    lo, hi = 1e-6, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if bs_call(S, K, T, r, mid)[0] < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            return (lo + hi) / 2.0
    return None


def bs_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Convenience: just the call delta."""
    _, delta = bs_call(S, K, T, r, sigma)
    return delta


def detect_tte(
    spot: float, atm_price: float, r: float = 0.0, target_iv: float = 0.20,
) -> float | None:
    """Find TTE where ATM IV is closest to target_iv (default 0.20).

    Scans half-day increments 0.5..30. Picks the TTE whose IV is nearest
    to target_iv, avoiding the trap of short-TTE always being "valid."
    """
    best_tte: float | None = None
    best_diff = float("inf")
    for half_d in range(1, 61):
        tte = half_d / (2 * 365.0)
        iv = find_iv(spot, spot, tte, r, atm_price)
        if iv is not None and 0.05 <= iv <= 1.5:
            diff = abs(iv - target_iv)
            if diff < best_diff:
                best_diff = diff
                best_tte = tte
    return best_tte


# ─────────────────────────────────────────────────────────────────
# 8. Basket Spread Tracker
# ─────────────────────────────────────────────────────────────────

class BasketSpreadTracker:
    """Welford online mean for basket premium with z-score entry signal.

    n_prior: effective prior observations (larger = more resistant to early drift).
    seed_mean: offline-computed mean premium from day-0 data.
    entry_thr: raw spread deviation threshold for opening positions.
    """

    def __init__(self, n_prior: int = 60_000, seed_mean: float = 0.0,
                 entry_thr: float = 80.0) -> None:
        self.n = n_prior
        self.mean_premium = seed_mean
        self.entry_thr = entry_thr

    def update(self, basket_mid: float, synthetic_mid: float) -> float:
        """Returns signal_spread = raw_spread - mean_premium."""
        raw_spread = basket_mid - synthetic_mid
        self.n += 1
        self.mean_premium += (raw_spread - self.mean_premium) / self.n
        return raw_spread - self.mean_premium

    def signal(self, basket_mid: float, synthetic_mid: float) -> int:
        """+1 = basket cheap (buy), -1 = basket expensive (sell), 0 = flat."""
        sig_spread = self.update(basket_mid, synthetic_mid)
        if sig_spread < -self.entry_thr:
            return 1
        if sig_spread > self.entry_thr:
            return -1
        return 0

    def to_dict(self) -> dict:
        return {"n": self.n, "mp": round(self.mean_premium, 4), "et": self.entry_thr}

    @classmethod
    def from_dict(cls, d: dict) -> "BasketSpreadTracker":
        obj = cls.__new__(cls)
        obj.n = d.get("n", 60_000)
        obj.mean_premium = d.get("mp", 0.0)
        obj.entry_thr = d.get("et", 80.0)
        return obj


# ─────────────────────────────────────────────────────────────────
# 9. Conversion Arb Calculator
# ─────────────────────────────────────────────────────────────────

def conversion_arb(
    product: str,
    obs: ConversionObservation,
    local_od: OrderDepth,
    pos: int,
    limit: int = 75,
    conv_limit: int = 10,
    min_edge: float = 0.2,
) -> tuple[list[Order], int]:
    """Sell locally above import_cost, convert short position each tick.

    Returns (orders, conversions_requested).
    Never hold long inventory - storage costs bleed.
    """
    orders: list[Order] = []

    import_cost = obs.askPrice + obs.importTariff + obs.transportFees

    # Sell at floor(foreignBid + 0.5) - undercuts local ask, matches hidden
    # taker bot fill price. P3 research: Paris buys at/above ask every ~77
    # ticks; posting here gets us filled first.
    local_sell_price = math.floor(obs.bidPrice + 0.5)

    rem_sell = limit + pos
    edge = local_sell_price - import_cost

    if edge >= min_edge and rem_sell > 0:
        qty = min(rem_sell, conv_limit)
        orders.append(Order(product, local_sell_price, -qty))

    # Convert entire short position back each tick
    conversions = max(min(-pos, conv_limit), 0)

    return orders, conversions
