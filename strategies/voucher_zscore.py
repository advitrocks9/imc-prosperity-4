import math
import statistics
from collections import deque

if __package__ in {None, ""}:
    _sys = __import__("sys")
    _pathlib = __import__("pathlib")
    _sys.path.append(str(_pathlib.Path(__file__).resolve().parents[1]))

from datamodel import Observation, Order, OrderDepth, TradingState
from strategies._base import best_mid, bs_call, bs_vega, find_iv

VOUCHER_ZSCORE_WINDOW = 100
VOUCHER_ZSCORE_THR_OPEN = 0.7
VOUCHER_ZSCORE_THR_CLOSE = 0.0
VOUCHER_ZSCORE_VEGA_FLOOR = 0.5
VOUCHER_POSITION_MAX = 100
VOUCHER_MIN_TICKS_FOR_ZSCORE = 30
VOUCHER_ZSCORE_LONG_IV_CAP = 0.20
VOUCHER_ZSCORE_TAKE_EDGE = 0.5
VOUCHER_ZSCORE_MIN_STRIKE = 5300
VOUCHER_PER_STRIKE_CONFIG: dict[int, dict] | None = None
VOUCHER_MAKE_ENABLE = True
VOUCHER_MAKE_SELL_ONLY = False
VOUCHER_MAKE_EDGE = 0
VOUCHER_MAKE_SIZE = 20
VOUCHER_MAKE_LAYERS = ((0, 20),)
VOUCHER_MAKE_Z_GATE = 999.0
VOUCHER_MAKE_TIME_LIMIT = 10000
VOUCHER_DD_CAP_PER_STRIKE = -999999.0
VOUCHER_DUST_STRIKES = (6000, 6500)
VOUCHER_DUST_BID = 0
VOUCHER_DUST_ASK = 1
VOUCHER_DUST_SIZE = 100
VOUCHER_ZSCORE_FAIR_IV_CAP = {
    5100: 0.2272,
    5200: 0.2334,
    5300: 0.2300,
    5400: 0.2150,
    5500: 0.2400,
}
VOUCHER_ZSCORE_TRADE_IV_CAP = {
    5300: 0.2400,
    5400: 0.2250,
}
UNDERLYING_SYMBOL = "VELVETFRUIT_EXTRACT"
LIQUID_VOUCHER_STRIKES = (4500, 5000, 5100, 5200, 5300, 5400, 5500)


def _strike_from_symbol(symbol: str) -> int | None:
    if not symbol.startswith("VEV_"):
        return None
    try:
        return int(symbol.split("_", 1)[1])
    except ValueError:
        return None


def _voucher_wall_mid(order_depth: OrderDepth) -> float | None:
    bids = order_depth.buy_orders
    asks = order_depth.sell_orders

    if bids and asks:
        bid_px = max(bids)
        ask_px = min(asks)
        bid_vol = max(0, bids[bid_px])
        ask_vol = max(0, -asks[ask_px])
        if bid_vol > 0 and ask_vol > 0:
            return (bid_vol * bid_px + ask_vol * ask_px) / (bid_vol + ask_vol)
        return (bid_px + ask_px) / 2.0

    if bids:
        return (min(bids) + max(bids)) / 2.0
    if asks:
        return (min(asks) + max(asks)) / 2.0
    return None


def _tte_years(timestamp: int) -> float:
    # R4 timestamps advance in 100-unit steps over 10,000 ticks: 0..999_900.
    days_remaining = max(4.0 - (timestamp / 1_000_000.0), 3.0)
    return days_remaining / 252.0


class VoucherZScoreTrader:
    """Per-strike rolling-IV z-score voucher market-making without delta hedge."""

    def __init__(self) -> None:
        self.window = VOUCHER_ZSCORE_WINDOW
        self.open_thr = VOUCHER_ZSCORE_THR_OPEN
        self.close_thr = VOUCHER_ZSCORE_THR_CLOSE
        self.vega_floor = VOUCHER_ZSCORE_VEGA_FLOOR
        self.position_max = VOUCHER_POSITION_MAX
        self.min_ticks_for_zscore = VOUCHER_MIN_TICKS_FOR_ZSCORE
        self.long_iv_cap = VOUCHER_ZSCORE_LONG_IV_CAP
        self.take_edge = VOUCHER_ZSCORE_TAKE_EDGE
        self.min_strike = VOUCHER_ZSCORE_MIN_STRIKE
        self.make_enable = VOUCHER_MAKE_ENABLE
        self.make_sell_only = VOUCHER_MAKE_SELL_ONLY
        self.make_edge = VOUCHER_MAKE_EDGE
        self.make_size = VOUCHER_MAKE_SIZE
        self.make_layers = VOUCHER_MAKE_LAYERS
        self.make_z_gate = VOUCHER_MAKE_Z_GATE
        self.make_time_limit = VOUCHER_MAKE_TIME_LIMIT
        self.dd_cap_per_strike = VOUCHER_DD_CAP_PER_STRIKE
        self.dust_strikes = tuple(VOUCHER_DUST_STRIKES)
        self.dust_bid = VOUCHER_DUST_BID
        self.dust_ask = VOUCHER_DUST_ASK
        self.dust_size = VOUCHER_DUST_SIZE
        self.fair_iv_cap = dict(VOUCHER_ZSCORE_FAIR_IV_CAP)
        self.trade_iv_cap = dict(VOUCHER_ZSCORE_TRADE_IV_CAP)
        self.iv_buffers: dict[str, deque[float]] = {}
        self.last_z_scores: dict[str, float | None] = {}
        self.strike_pnl: dict[str, float] = {}
        self.strike_positions: dict[str, int] = {}
        self.strike_avg_cost: dict[str, float] = {}
        self.make_disabled_strikes: set[str] = set()
        self.trade_cursor: dict[str, dict[str, int]] = {}

    def load_state(self, raw_state: dict[str, object] | None) -> None:
        self.iv_buffers = {}
        self.strike_pnl = {}
        self.strike_positions = {}
        self.strike_avg_cost = {}
        self.make_disabled_strikes = set()
        self.trade_cursor = {}

        state = raw_state or {}
        if isinstance(state, dict) and "iv_buffers" in state:
            buffer_state = state.get("iv_buffers", {})
        else:
            buffer_state = state

        for symbol, values in buffer_state.items():
            if not isinstance(values, list):
                continue
            buf = deque(maxlen=self.window)
            for value in values[-self.window:]:
                if isinstance(value, (int, float)) and math.isfinite(value):
                    buf.append(float(value))
            self.iv_buffers[symbol] = buf
        if not isinstance(state, dict):
            return

        pnl_state = state.get("strike_pnl", {})
        if isinstance(pnl_state, dict):
            for strike_key, value in pnl_state.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self.strike_pnl[str(strike_key)] = float(value)

        pos_state = state.get("strike_positions", {})
        if isinstance(pos_state, dict):
            for strike_key, value in pos_state.items():
                if isinstance(value, int):
                    self.strike_positions[str(strike_key)] = int(value)

        avg_cost_state = state.get("strike_avg_cost", {})
        if isinstance(avg_cost_state, dict):
            for strike_key, value in avg_cost_state.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self.strike_avg_cost[str(strike_key)] = float(value)

        disabled_state = state.get("make_disabled_strikes", [])
        if isinstance(disabled_state, list):
            self.make_disabled_strikes = {str(value) for value in disabled_state}

        cursor_state = state.get("trade_cursor", {})
        if isinstance(cursor_state, dict):
            for symbol, cursor in cursor_state.items():
                if not isinstance(cursor, dict):
                    continue
                timestamp = cursor.get("timestamp")
                count = cursor.get("count")
                if isinstance(timestamp, int) and isinstance(count, int):
                    self.trade_cursor[str(symbol)] = {
                        "timestamp": int(timestamp),
                        "count": int(count),
                    }

    def dump_state(self) -> dict[str, object]:
        return {
            "iv_buffers": {symbol: list(buf) for symbol, buf in self.iv_buffers.items() if buf},
            "strike_pnl": self.strike_pnl,
            "strike_positions": self.strike_positions,
            "strike_avg_cost": self.strike_avg_cost,
            "make_disabled_strikes": sorted(self.make_disabled_strikes),
            "trade_cursor": self.trade_cursor,
        }

    def _buffer_for_symbol(self, symbol: str) -> deque[float]:
        buffer = self.iv_buffers.get(symbol)
        if buffer is None or buffer.maxlen != self.window:
            existing = list(buffer) if buffer is not None else []
            buffer = deque(existing[-self.window:], maxlen=self.window)
            self.iv_buffers[symbol] = buffer
        return buffer

    def _planned_position(
        self,
        state: TradingState,
        pending_positions: dict[str, int],
        symbol: str,
    ) -> int:
        if symbol not in pending_positions:
            pending_positions[symbol] = state.position.get(symbol, 0)
        return pending_positions[symbol]

    def _append_order(
        self,
        results: dict[str, list[Order]],
        pending_positions: dict[str, int],
        state: TradingState,
        symbol: str,
        price: int,
        quantity: int,
    ) -> None:
        if quantity == 0:
            return
        results.setdefault(symbol, []).append(Order(symbol, int(price), int(quantity)))
        pending_positions[symbol] = self._planned_position(state, pending_positions, symbol) + int(quantity)

    def _strike_key(self, strike: int) -> str:
        return str(int(strike))

    def _update_trade_cursor(self, symbol: str, trades: list) -> list:
        cursor = self.trade_cursor.get(symbol, {})
        last_ts = cursor.get("timestamp", -1)
        last_count = cursor.get("count", 0)
        trades_by_time = sorted(
            trades,
            key=lambda trade: (
                trade.timestamp,
                trade.price,
                trade.quantity,
                trade.buyer or "",
                trade.seller or "",
            ),
        )

        ts_counts: dict[int, int] = {}
        new_trades = []
        max_ts = last_ts
        max_count = last_count
        for trade in trades_by_time:
            ts = int(trade.timestamp)
            ts_counts[ts] = ts_counts.get(ts, 0) + 1
            if ts < last_ts:
                continue
            if ts == last_ts and ts_counts[ts] <= last_count:
                continue
            new_trades.append(trade)
            if ts > max_ts:
                max_ts = ts
            if ts == max_ts:
                max_count = ts_counts[ts]

        if max_ts >= 0:
            self.trade_cursor[symbol] = {"timestamp": max_ts, "count": max_count}
        return new_trades

    def _apply_fill_to_realized_pnl(self, strike: int, signed_qty: int, price: int) -> None:
        if signed_qty == 0:
            return

        strike_key = self._strike_key(strike)
        position = self.strike_positions.get(strike_key, 0)
        avg_cost = self.strike_avg_cost.get(strike_key, 0.0)
        realized_pnl = self.strike_pnl.get(strike_key, 0.0)

        if position == 0 or (position > 0) == (signed_qty > 0):
            new_position = position + signed_qty
            total_size = abs(position) + abs(signed_qty)
            avg_cost = (
                ((abs(position) * avg_cost) + (abs(signed_qty) * float(price))) / total_size
                if total_size > 0
                else 0.0
            )
        else:
            close_qty = min(abs(position), abs(signed_qty))
            realized_pnl += close_qty * (float(price) - avg_cost) * (1.0 if position > 0 else -1.0)
            new_position = position + signed_qty
            if new_position == 0:
                avg_cost = 0.0
            elif (new_position > 0) == (position > 0):
                pass
            else:
                avg_cost = float(price)

        self.strike_positions[strike_key] = new_position
        self.strike_avg_cost[strike_key] = avg_cost
        self.strike_pnl[strike_key] = realized_pnl
        if realized_pnl < self.dd_cap_per_strike:
            self.make_disabled_strikes.add(strike_key)

    def _update_strike_pnl_from_fills(self, state: TradingState) -> None:
        for symbol, trades in state.own_trades.items():
            strike = _strike_from_symbol(symbol)
            if strike is None or not trades:
                continue
            for trade in self._update_trade_cursor(symbol, trades):
                signed_qty = 0
                if trade.buyer == "SUBMISSION":
                    signed_qty = abs(int(trade.quantity))
                elif trade.seller == "SUBMISSION":
                    signed_qty = -abs(int(trade.quantity))
                self._apply_fill_to_realized_pnl(strike, signed_qty, int(trade.price))

    def _make_allowed(self, strike: int, state: TradingState, z_score: float | None) -> bool:
        if (state.timestamp // 100) >= self.make_time_limit:
            return False
        if z_score is not None and abs(z_score) >= self.make_z_gate:
            return False
        if self._strike_key(strike) in self.make_disabled_strikes:
            return False
        return True

    def _quote_side_order(self, planned_position: int, sell_only: bool = False) -> tuple[str, ...]:
        if sell_only:
            return ("sell",)
        if planned_position < 0:
            return ("buy", "sell")
        return ("sell", "buy")

    def _strike_make_config(self, strike: int) -> dict[str, int]:
        if VOUCHER_PER_STRIKE_CONFIG is None:
            return {
                "edge": int(self.make_edge),
                "size": int(self.make_size),
                "pos": int(self.position_max),
            }

        raw_config = VOUCHER_PER_STRIKE_CONFIG.get(int(strike), {})
        config = raw_config if isinstance(raw_config, dict) else {}
        return {
            "edge": int(config.get("edge", VOUCHER_MAKE_EDGE)),
            "size": int(config.get("size", VOUCHER_MAKE_SIZE)),
            "pos": int(config.get("pos", self.position_max)),
        }

    def _normalized_make_layers(self, strike: int) -> tuple[tuple[int, int], ...]:
        strike_config = self._strike_make_config(strike)
        if VOUCHER_PER_STRIKE_CONFIG is not None:
            size = int(strike_config["size"])
            if size < 1:
                return ()
            return ((int(strike_config["edge"]), size),)

        if self.make_layers is None:
            size = int(strike_config["size"])
            if size < 1:
                return ()
            return ((int(strike_config["edge"]), size),)

        normalized: list[tuple[int, int]] = []
        for layer in self.make_layers:
            if not isinstance(layer, (tuple, list)) or len(layer) != 2:
                continue
            edge, size = layer
            if not isinstance(edge, (int, float)) or not isinstance(size, (int, float)):
                continue
            size_int = int(size)
            if size_int < 1:
                continue
            normalized.append((int(edge), size_int))
        return tuple(normalized)

    def _strike_make_position_limit(self, strike: int) -> int:
        position_limit = int(self._strike_make_config(strike)["pos"])
        if position_limit < 1:
            return int(self.position_max)
        return position_limit

    def _post_make_quotes(
        self,
        state: TradingState,
        results: dict[str, list[Order]],
        pending_positions: dict[str, int],
        strike: int,
        symbol: str,
        planned_position: int,
        bs_price: float,
    ) -> None:
        layers = self._normalized_make_layers(strike)
        if not layers:
            return

        position_limit = self._strike_make_position_limit(strike)
        remaining_bid_capacity = max(0, position_limit - planned_position)
        remaining_ask_capacity = max(0, planned_position + position_limit)

        for edge, size in layers:
            if remaining_ask_capacity >= size:
                ask_price = int(bs_price + edge)
                self._append_order(results, pending_positions, state, symbol, ask_price, -size)
                remaining_ask_capacity -= size

            if self.make_sell_only:
                continue

            if remaining_bid_capacity >= size:
                bid_price = int(bs_price - edge)
                self._append_order(results, pending_positions, state, symbol, bid_price, size)
                remaining_bid_capacity -= size

    def _post_dust_quotes(
        self,
        state: TradingState,
        results: dict[str, list[Order]],
        pending_positions: dict[str, int],
        symbol: str,
    ) -> None:
        planned_position = self._planned_position(state, pending_positions, symbol)
        for side in self._quote_side_order(planned_position, sell_only=self.make_sell_only):
            planned_position = self._planned_position(state, pending_positions, symbol)
            if side == "buy":
                qty = min(self.dust_size, self.position_max - planned_position)
                if qty >= 1:
                    self._append_order(results, pending_positions, state, symbol, self.dust_bid, qty)
            else:
                qty = min(self.dust_size, planned_position + self.position_max)
                if qty >= 1:
                    self._append_order(results, pending_positions, state, symbol, self.dust_ask, -qty)

    def _target_position(
        self,
        z_score: float,
        iv: float,
        vega: float,
        max_vega: float,
        current_position: int,
    ) -> int:
        open_thr = min(self.open_thr, 0.5)
        abs_z = abs(z_score)
        if abs_z <= self.close_thr:
            return 0

        z_scale = max(1.5, 3.0 * open_thr)
        z_weight = min(abs_z / z_scale, 1.0)
        vega_reference = max(self.vega_floor * max_vega, 1e-9)
        vega_weight = min(max(vega_reference / vega, 0.0), 1.0)
        target_size = int(round(self.position_max * z_weight * vega_weight))
        if target_size <= 0:
            return 0

        if z_score > 0:
            if current_position >= 0 and abs_z < open_thr:
                return 0
            return -target_size

        if iv > self.long_iv_cap:
            return 0
        if current_position <= 0 and abs_z < open_thr:
            return 0
        return target_size

    def _target_order(
        self,
        symbol: str,
        order_depth: OrderDepth,
        current_position: int,
        target_position: int,
        fair_price: float,
    ) -> Order | None:
        target = max(-self.position_max, min(self.position_max, target_position))
        delta = target - current_position
        max_rebalance = max(10, self.position_max // 3)
        if delta > max_rebalance:
            delta = max_rebalance
        elif delta < -max_rebalance:
            delta = -max_rebalance

        if delta > 0:
            if not order_depth.sell_orders:
                return None
            best_ask = min(order_depth.sell_orders)
            if fair_price - best_ask <= self.take_edge:
                return None
            buy_qty = min(delta, self.position_max - current_position)
            buy_qty = min(buy_qty, abs(order_depth.sell_orders[best_ask]))
            if buy_qty >= 1:
                return Order(symbol, int(best_ask), int(buy_qty))
            return None

        if delta < 0:
            if not order_depth.buy_orders:
                return None
            best_bid = max(order_depth.buy_orders)
            if best_bid - fair_price <= self.take_edge:
                return None
            sell_qty = min(-delta, current_position + self.position_max)
            sell_qty = min(sell_qty, order_depth.buy_orders[best_bid])
            if sell_qty >= 1:
                return Order(symbol, int(best_bid), -int(sell_qty))
        return None

    def run(self, state: TradingState) -> list[tuple[str, list[Order]]]:
        results_by_symbol: dict[str, list[Order]] = {}
        pending_positions: dict[str, int] = {}
        self.last_z_scores = {}
        self._update_strike_pnl_from_fills(state)

        underlying_depth = state.order_depths.get(UNDERLYING_SYMBOL)
        if underlying_depth is None:
            return []

        spot = best_mid(underlying_depth)
        if spot is None or spot <= 0:
            return []

        tte_years = _tte_years(state.timestamp)
        voucher_symbols = sorted(
            (symbol for symbol in state.order_depths if symbol.startswith("VEV_")),
            key=lambda symbol: _strike_from_symbol(symbol) or 0,
        )
        if not voucher_symbols:
            return []

        per_symbol: dict[str, tuple[int, float, float, OrderDepth]] = {}
        max_vega = 0.0

        for symbol in voucher_symbols:
            strike = _strike_from_symbol(symbol)
            order_depth = state.order_depths.get(symbol)
            if strike is None or order_depth is None:
                continue
            if strike < self.min_strike and strike not in LIQUID_VOUCHER_STRIKES:
                continue

            voucher_mid = _voucher_wall_mid(order_depth)
            if voucher_mid is None or voucher_mid <= 0:
                continue

            iv = find_iv(spot, float(strike), tte_years, 0.0, voucher_mid)
            if iv is None or not math.isfinite(iv) or iv <= 0:
                continue

            vega = bs_vega(spot, float(strike), tte_years, iv)
            if not math.isfinite(vega) or vega <= 0:
                continue

            per_symbol[symbol] = (strike, iv, vega, order_depth)
            if strike >= self.min_strike:
                max_vega = max(max_vega, vega)

        rolling_means: dict[str, float] = {}

        for symbol in voucher_symbols:
            info = per_symbol.get(symbol)
            self.last_z_scores[symbol] = None
            if info is None:
                continue

            strike, iv, vega, order_depth = info
            buffer = self._buffer_for_symbol(symbol)
            buffer.append(iv)
            mean = statistics.fmean(buffer)
            rolling_means[symbol] = mean
            if strike < self.min_strike or max_vega <= 0:
                continue
            if len(buffer) < self.min_ticks_for_zscore:
                continue

            std = statistics.stdev(buffer)
            if std <= 0:
                continue

            z_score = (iv - mean) / std
            self.last_z_scores[symbol] = z_score
            pricing_iv = min(mean, self.fair_iv_cap.get(strike, mean))
            fair_price, _ = bs_call(spot, float(strike), tte_years, 0.0, pricing_iv)

            if iv > self.trade_iv_cap.get(strike, float("inf")):
                continue

            if vega < self.vega_floor * max_vega:
                continue

            current_position = state.position.get(symbol, 0)
            target = self._target_position(z_score, iv, vega, max_vega, current_position)
            if target == current_position:
                continue

            order = self._target_order(
                symbol,
                order_depth,
                current_position,
                target,
                fair_price,
            )
            if order is not None and abs(order.quantity) >= 1:
                self._append_order(
                    results_by_symbol,
                    pending_positions,
                    state,
                    symbol,
                    order.price,
                    order.quantity,
                )

        if self.make_enable:
            for symbol in voucher_symbols:
                info = per_symbol.get(symbol)
                if info is None:
                    continue
                strike, _, _, order_depth = info
                if strike not in LIQUID_VOUCHER_STRIKES:
                    continue
                rolling_mean_iv = rolling_means.get(symbol)
                if rolling_mean_iv is None or not math.isfinite(rolling_mean_iv) or rolling_mean_iv <= 0:
                    continue

                market_mid = best_mid(order_depth)
                if market_mid is None:
                    continue

                bs_price, _ = bs_call(spot, float(strike), tte_years, 0.0, rolling_mean_iv)
                if bs_price <= 0 or abs(market_mid - bs_price) > 10:
                    continue
                if not self._make_allowed(strike, state, self.last_z_scores.get(symbol)):
                    continue

                planned_position = self._planned_position(state, pending_positions, symbol)
                self._post_make_quotes(
                    state,
                    results_by_symbol,
                    pending_positions,
                    strike,
                    symbol,
                    planned_position,
                    bs_price,
                )

        for strike in self.dust_strikes:
            symbol = f"VEV_{strike}"
            if symbol not in state.order_depths:
                continue
            self._post_dust_quotes(
                state,
                results_by_symbol,
                pending_positions,
                symbol,
            )

        results: list[tuple[str, list[Order]]] = []
        for symbol in voucher_symbols:
            orders = results_by_symbol.get(symbol)
            if orders:
                results.append((symbol, orders))
        return results


def _build_depth(bid: int, ask: int, bid_vol: int = 20, ask_vol: int = 20) -> OrderDepth:
    order_depth = OrderDepth()
    order_depth.buy_orders = {bid: bid_vol}
    order_depth.sell_orders = {ask: -ask_vol}
    return order_depth


if __name__ == "__main__":
    trader = VoucherZScoreTrader()
    positions: dict[str, int] = {}
    strikes = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
    base_iv = 0.24

    for tick in range(50):
        timestamp = tick * 200
        spot = 5_250.0 + 15.0 * math.sin(tick / 8.0)
        order_depths: dict[str, OrderDepth] = {
            UNDERLYING_SYMBOL: _build_depth(int(spot - 2), int(spot + 2), 30, 30),
        }

        for strike in strikes:
            iv_shift = 0.1 * math.sin((tick / 5.0) + (strike / 700.0))
            price = bs_call(spot, float(strike), _tte_years(timestamp), 0.0, base_iv + iv_shift)[0]
            mid = max(1.0, price)
            bid = max(1, int(math.floor(mid)))
            ask = max(bid + 1, int(math.ceil(mid)))
            order_depths[f"VEV_{strike}"] = _build_depth(bid, ask)

        state = TradingState(
            traderData="",
            timestamp=timestamp,
            listings={},
            order_depths=order_depths,
            own_trades={},
            market_trades={},
            position=positions.copy(),
            observations=Observation({}, {}),
        )
        orders = trader.run(state)
        for symbol, symbol_orders in orders:
            for order in symbol_orders:
                positions[symbol] = positions.get(symbol, 0) + order.quantity

        z_parts = []
        for symbol in ("VEV_5000", "VEV_5200", "VEV_5400"):
            z_score = trader.last_z_scores.get(symbol)
            z_parts.append(f"{symbol}={z_score:.2f}" if z_score is not None else f"{symbol}=None")
        print(f"tick={tick:02d} z[{', '.join(z_parts)}] orders={orders}")
