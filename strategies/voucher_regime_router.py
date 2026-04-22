from __future__ import annotations

import math
import statistics
from collections import deque
from types import SimpleNamespace

from datamodel import Order, OrderDepth, TradingState
from strategies._base import best_mid, bs_call, bs_vega, find_iv


UNDERLYING_SYMBOL = "VELVETFRUIT_EXTRACT"
MODE_LOW_VOL = "bs"
MODE_HIGH_VOL = "zs"
REGIME_WINDOW = 200

VOUCHER_ZSCORE_WINDOW = 200
VOUCHER_ZSCORE_THR_OPEN = 0.5
VOUCHER_ZSCORE_THR_CLOSE = 0.0
VOUCHER_ZSCORE_VEGA_FLOOR = 0.5
VOUCHER_POSITION_MAX = 300
VOUCHER_MIN_TICKS_FOR_ZSCORE = 30
VOUCHER_ZSCORE_LONG_IV_CAP = 0.20
VOUCHER_ZSCORE_TAKE_EDGE = 0.5
VOUCHER_ZSCORE_MIN_STRIKE = 5100
VOUCHER_ZSCORE_FAIR_IV_CAP = {
    5100: 0.2272,
    5200: 0.2334,
    5300: 0.2300,
    5400: 0.2150,
    5500: 0.2400,
}
VOUCHER_ZSCORE_TRADE_IV_CAP = {
    5100: 0.25,
    5200: 0.25,
    5300: 0.2400,
    5400: 0.2250,
    5500: 0.25,
}
BS_VOUCHER_POSITION_MAX = 50


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
    days_remaining = max(4.0 - (timestamp / 1_000_000.0), 3.0)
    return days_remaining / 252.0


class _VoucherZScoreTrader:
    """Exact W2H voucher z-score path with per-strike rolling IV buffers."""

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
        self.fair_iv_cap = dict(VOUCHER_ZSCORE_FAIR_IV_CAP)
        self.trade_iv_cap = dict(VOUCHER_ZSCORE_TRADE_IV_CAP)
        self.iv_buffers: dict[str, deque[float]] = {}
        self.last_z_scores: dict[str, float | None] = {}

    def load_state(self, raw_state: dict[str, list[float]] | None) -> None:
        self.iv_buffers = {}
        for symbol, values in (raw_state or {}).items():
            if not isinstance(values, list):
                continue
            buf = deque(maxlen=self.window)
            for value in values[-self.window:]:
                if isinstance(value, (int, float)) and math.isfinite(value):
                    buf.append(float(value))
            self.iv_buffers[symbol] = buf

    def dump_state(self) -> dict[str, list[float]]:
        return {symbol: list(buf) for symbol, buf in self.iv_buffers.items() if buf}

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
        results: list[tuple[str, list[Order]]] = []
        self.last_z_scores = {}

        underlying_depth = state.order_depths.get(UNDERLYING_SYMBOL)
        if underlying_depth is None:
            return results

        spot = best_mid(underlying_depth)
        if spot is None or spot <= 0:
            return results

        tte_years = _tte_years(state.timestamp)
        voucher_symbols = sorted(
            (symbol for symbol in state.order_depths if symbol.startswith("VEV_")),
            key=lambda symbol: _strike_from_symbol(symbol) or 0,
        )
        if not voucher_symbols:
            return results

        per_symbol: dict[str, tuple[int, float, float, OrderDepth]] = {}
        max_vega = 0.0

        for symbol in voucher_symbols:
            strike = _strike_from_symbol(symbol)
            order_depth = state.order_depths.get(symbol)
            if strike is None or order_depth is None or strike < self.min_strike:
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
            max_vega = max(max_vega, vega)

        if max_vega <= 0:
            return results

        for symbol in voucher_symbols:
            info = per_symbol.get(symbol)
            self.last_z_scores[symbol] = None
            if info is None:
                continue

            strike, iv, vega, order_depth = info
            buffer = self.iv_buffers.get(symbol)
            if buffer is None or buffer.maxlen != self.window:
                existing = list(buffer) if buffer is not None else []
                buffer = deque(existing[-self.window:], maxlen=self.window)
                self.iv_buffers[symbol] = buffer

            buffer.append(iv)
            if len(buffer) < self.min_ticks_for_zscore:
                continue

            std = statistics.stdev(buffer)
            if std <= 0:
                continue

            mean = statistics.fmean(buffer)
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

            order = self._target_order(symbol, order_depth, current_position, target, fair_price)
            if order is not None and abs(order.quantity) >= 1:
                results.append((symbol, [order]))

        return results


class _BSVoucherTrader:
    """Exact W3A BS-IV anchor path."""

    def __init__(self, threshold: int = 3, anchor_strike: str = "VEV_5300") -> None:
        self.threshold = int(threshold)
        self.anchor_strike = anchor_strike
        self.position_cap = BS_VOUCHER_POSITION_MAX

    def _anchor_state(self, state: TradingState) -> tuple[float, float, float] | None:
        anchor_depth = state.order_depths.get(self.anchor_strike)
        underlying_depth = state.order_depths.get(UNDERLYING_SYMBOL)
        anchor_k = _strike_from_symbol(self.anchor_strike)
        if anchor_depth is None or underlying_depth is None or anchor_k is None:
            return None
        spot = best_mid(underlying_depth)
        anchor_mid = best_mid(anchor_depth)
        tte = _tte_years(state.timestamp)
        if spot is None or anchor_mid is None or spot <= 0 or anchor_mid <= 0 or tte <= 0:
            return None
        iv = find_iv(spot, float(anchor_k), tte, 0.0, anchor_mid)
        if iv is None or not math.isfinite(iv) or iv <= 0:
            return None
        return spot, tte, iv

    def _buy_order(self, symbol: str, order_depth: OrderDepth, position: int) -> Order | None:
        if not order_depth.sell_orders:
            return None
        best_ask = min(order_depth.sell_orders)
        ask_volume = max(0, -order_depth.sell_orders[best_ask])
        buy_capacity = self.position_cap - position
        quantity = min(ask_volume, buy_capacity)
        if quantity <= 0:
            return None
        return Order(symbol, int(best_ask), int(quantity))

    def _sell_order(self, symbol: str, order_depth: OrderDepth, position: int) -> Order | None:
        if not order_depth.buy_orders:
            return None
        best_bid = max(order_depth.buy_orders)
        bid_volume = max(0, order_depth.buy_orders[best_bid])
        sell_capacity = self.position_cap + position
        quantity = min(bid_volume, sell_capacity)
        if quantity <= 0:
            return None
        return Order(symbol, int(best_bid), -int(quantity))

    def run(self, state: TradingState) -> dict[str, list[Order]]:
        anchor_state = self._anchor_state(state)
        if anchor_state is None:
            return {}
        spot, tte, flat_iv = anchor_state
        results: dict[str, list[Order]] = {}
        voucher_symbols = sorted(
            (symbol for symbol in state.order_depths if symbol.startswith("VEV_")),
            key=lambda symbol: (_strike_from_symbol(symbol) or 0, symbol),
        )
        for symbol in voucher_symbols:
            if symbol == self.anchor_strike:
                continue
            strike = _strike_from_symbol(symbol)
            order_depth = state.order_depths.get(symbol)
            if strike is None or order_depth is None:
                continue
            mid = best_mid(order_depth)
            if mid is None:
                continue
            theoretical, _ = bs_call(spot, float(strike), tte, 0.0, flat_iv)
            position = state.position.get(symbol, 0)
            order = None
            if mid > theoretical + self.threshold:
                order = self._sell_order(symbol, order_depth, position)
            elif mid < theoretical - self.threshold:
                order = self._buy_order(symbol, order_depth, position)
            if order is not None:
                results[symbol] = [order]
        return results


class VoucherRegimeRouter:
    """Routes voucher trading by live VELVETFRUIT coefficient of variation."""

    def __init__(
        self,
        regime_threshold: float,
        regime_window: int = REGIME_WINDOW,
        bs_threshold: int = 3,
        bs_anchor_strike: str = "VEV_5300",
    ) -> None:
        self.regime_threshold = float(regime_threshold)
        self.regime_window = int(regime_window)
        self.mid_buffer: deque[float] = deque(maxlen=self.regime_window)
        self.last_regime_indicator = 0.0
        self.current_mode = MODE_LOW_VOL
        self.last_order_mode: dict[str, str] = {}
        self.owned_positions: dict[str, dict[str, int]] = {
            MODE_LOW_VOL: {},
            MODE_HIGH_VOL: {},
        }
        self.bs = _BSVoucherTrader(
            threshold=bs_threshold,
            anchor_strike=bs_anchor_strike,
        )
        self.zscore = _VoucherZScoreTrader()

    def load_state(self, raw_state: dict[str, object] | None) -> None:
        state = raw_state or {}
        self.mid_buffer = deque(maxlen=self.regime_window)
        for value in state.get("mw", []):
            if isinstance(value, (int, float)) and math.isfinite(value):
                self.mid_buffer.append(float(value))
        indicator = state.get("ri", 0.0)
        self.last_regime_indicator = float(indicator) if isinstance(indicator, (int, float)) else 0.0
        mode = state.get("cm", MODE_LOW_VOL)
        self.current_mode = mode if mode in {MODE_LOW_VOL, MODE_HIGH_VOL} else MODE_LOW_VOL

        self.last_order_mode = {}
        for symbol, mode_name in state.get("lm", {}).items():
            if isinstance(symbol, str) and mode_name in {MODE_LOW_VOL, MODE_HIGH_VOL}:
                self.last_order_mode[symbol] = mode_name

        self.owned_positions = {MODE_LOW_VOL: {}, MODE_HIGH_VOL: {}}
        raw_owned = state.get("op", {})
        if isinstance(raw_owned, dict):
            for mode_name in (MODE_LOW_VOL, MODE_HIGH_VOL):
                positions = raw_owned.get(mode_name, {})
                if not isinstance(positions, dict):
                    continue
                for symbol, qty in positions.items():
                    if isinstance(symbol, str) and isinstance(qty, int) and qty != 0:
                        self.owned_positions[mode_name][symbol] = int(qty)

        self.zscore.load_state(state.get("z", {}))

    def dump_state(self) -> dict[str, object]:
        return {
            "mw": list(self.mid_buffer),
            "ri": self.last_regime_indicator,
            "cm": self.current_mode,
            "lm": self.last_order_mode,
            "op": {
                mode: positions
                for mode, positions in self.owned_positions.items()
                if positions
            },
            "z": self.zscore.dump_state(),
        }

    def _voucher_symbols(self, state: TradingState) -> list[str]:
        symbols = {
            symbol
            for symbol in state.order_depths
            if symbol.startswith("VEV_")
        }
        symbols.update(
            symbol
            for symbol in state.position
            if symbol.startswith("VEV_")
        )
        symbols.update(self.last_order_mode)
        for positions in self.owned_positions.values():
            symbols.update(positions)
        return sorted(symbols, key=lambda symbol: (_strike_from_symbol(symbol) or 0, symbol))

    def _set_owned_position(self, mode: str, symbol: str, quantity: int) -> None:
        if quantity == 0:
            self.owned_positions[mode].pop(symbol, None)
        else:
            self.owned_positions[mode][symbol] = int(quantity)

    def _reconcile_positions(self, state: TradingState, voucher_symbols: list[str]) -> None:
        for symbol in voucher_symbols:
            actual = int(state.position.get(symbol, 0))
            current_total = (
                self.owned_positions[MODE_LOW_VOL].get(symbol, 0)
                + self.owned_positions[MODE_HIGH_VOL].get(symbol, 0)
            )
            delta = actual - current_total
            if delta == 0:
                continue
            mode = self.last_order_mode.get(symbol, self.current_mode)
            if mode not in {MODE_LOW_VOL, MODE_HIGH_VOL}:
                mode = MODE_LOW_VOL
            owned = self.owned_positions[mode].get(symbol, 0)
            self._set_owned_position(mode, symbol, owned + delta)

    def _build_mode_state(self, state: TradingState, mode: str, voucher_symbols: list[str]):
        positions = dict(state.position)
        owned = self.owned_positions[mode]
        for symbol in voucher_symbols:
            positions[symbol] = int(owned.get(symbol, 0))
        return SimpleNamespace(
            traderData=state.traderData,
            timestamp=state.timestamp,
            listings=state.listings,
            order_depths=state.order_depths,
            own_trades=state.own_trades,
            market_trades=state.market_trades,
            position=positions,
            observations=state.observations,
        )

    def _update_regime_indicator(self, state: TradingState) -> float:
        underlying_depth = state.order_depths.get(UNDERLYING_SYMBOL)
        spot = best_mid(underlying_depth) if underlying_depth is not None else None
        if spot is not None and math.isfinite(spot) and spot > 0:
            self.mid_buffer.append(float(spot))

        if not self.mid_buffer:
            self.last_regime_indicator = 0.0
            return self.last_regime_indicator

        mean = statistics.fmean(self.mid_buffer)
        if len(self.mid_buffer) < 2 or mean <= 0:
            self.last_regime_indicator = 0.0
            return self.last_regime_indicator

        std = statistics.stdev(self.mid_buffer)
        self.last_regime_indicator = std / mean if mean > 0 else 0.0
        return self.last_regime_indicator

    def _run_zscore(self, state: TradingState) -> dict[str, list[Order]]:
        results: dict[str, list[Order]] = {}
        for symbol, orders in self.zscore.run(state):
            results.setdefault(symbol, []).extend(orders)
        return results

    def _record_order_modes(self, orders_by_symbol: dict[str, list[Order]], mode: str) -> None:
        for symbol, orders in orders_by_symbol.items():
            if any(order.quantity != 0 for order in orders):
                self.last_order_mode[symbol] = mode

    def run(self, state: TradingState) -> dict[str, list[Order]]:
        voucher_symbols = self._voucher_symbols(state)
        self._reconcile_positions(state, voucher_symbols)
        indicator = self._update_regime_indicator(state)
        selected_mode = MODE_HIGH_VOL if indicator > self.regime_threshold else MODE_LOW_VOL

        zscore_state = self._build_mode_state(state, MODE_HIGH_VOL, voucher_symbols)
        zscore_results = self._run_zscore(zscore_state)

        if selected_mode == MODE_HIGH_VOL:
            results = zscore_results
        else:
            bs_state = self._build_mode_state(state, MODE_LOW_VOL, voucher_symbols)
            results = self.bs.run(bs_state)

        self._record_order_modes(results, selected_mode)
        self.current_mode = selected_mode
        return results
