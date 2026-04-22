from __future__ import annotations

import io
import math
import statistics
from contextlib import redirect_stderr
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from datamodel import Order, OrderDepth, TradingState
from strategies.bs_options import bs_greeks, implied_vol

TRADING_DAYS_PER_YEAR = 252.0
TICKS_PER_DAY = 10_000


def _best_mid(order_depth: OrderDepth) -> float | None:
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0


def _voucher_mid(order_depth: OrderDepth) -> float | None:
    bids = order_depth.buy_orders
    asks = order_depth.sell_orders
    if bids and asks:
        best_bid = max(bids)
        best_ask = min(asks)
        bid_volume = max(0, bids[best_bid])
        ask_volume = max(0, -asks[best_ask])
        if bid_volume > 0 and ask_volume > 0:
            return (
                (best_bid * bid_volume) + (best_ask * ask_volume)
            ) / float(bid_volume + ask_volume)
        return (best_bid + best_ask) / 2.0
    return None


def _safe_iv(price: float, spot: float, strike: int, tte_years: float) -> float | None:
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        iv = implied_vol(price, spot, float(strike), tte_years, 0.0, "call")
    if not math.isfinite(iv) or iv <= 0:
        return None
    return iv


class GammaScalper:
    def __init__(
        self,
        underlying: str,
        option_products: dict[int, str],
        tte_func,
        option_limit: int = 30,
        underlying_limit: int = 200,
        trade_qty: int = 10,
        rv_window: int = 200,
        gap_history_window: int = 200,
        min_gap_history: int = 50,
        gap_z_open: float = 0.5,
        hedge_every_ticks: int = 20,
        hedge_delta_threshold: float = 10.0,
        eligible_strikes: tuple[int, ...] = (5000, 5100, 5200, 5300, 5400, 5500),
    ) -> None:
        self.underlying = underlying
        self.option_products = dict(option_products)
        self.tte_func = tte_func
        self.option_limit = option_limit
        self.underlying_limit = underlying_limit
        self.trade_qty = trade_qty
        self.rv_window = rv_window
        self.gap_history_window = gap_history_window
        self.min_gap_history = min_gap_history
        self.gap_z_open = gap_z_open
        self.hedge_every_ticks = hedge_every_ticks
        self.hedge_delta_threshold = hedge_delta_threshold
        self.eligible_strikes = tuple(eligible_strikes)

        self.tick_index = 0
        self.last_timestamp: int | None = None
        self.last_hedge_tick = -10**9
        self.spot_logs: list[float] = []
        self.gap_history: dict[str, list[float]] = {}
        self.active_symbol: str | None = None
        self.option_position = 0
        self.underlying_position = 0

    def load_state(self, raw_state: dict[str, object] | None) -> None:
        state = raw_state or {}
        if not isinstance(state, dict):
            return
        self.tick_index = int(state.get("tick_index", 0))
        self.last_timestamp = state.get("last_timestamp")
        if not isinstance(self.last_timestamp, int):
            self.last_timestamp = None
        self.last_hedge_tick = int(state.get("last_hedge_tick", -10**9))
        self.spot_logs = [
            float(value)
            for value in state.get("spot_logs", [])
            if isinstance(value, (int, float)) and math.isfinite(value)
        ][-self.rv_window :]
        self.active_symbol = state.get("active_symbol")
        if not isinstance(self.active_symbol, str):
            self.active_symbol = None
        self.option_position = int(state.get("option_position", 0))
        self.underlying_position = int(state.get("underlying_position", 0))

        self.gap_history = {}
        raw_gap_history = state.get("gap_history", {})
        if isinstance(raw_gap_history, dict):
            for symbol, values in raw_gap_history.items():
                if not isinstance(symbol, str) or not isinstance(values, list):
                    continue
                self.gap_history[symbol] = [
                    float(value)
                    for value in values[-self.gap_history_window :]
                    if isinstance(value, (int, float)) and math.isfinite(value)
                ]

    def dump_state(self) -> dict[str, object]:
        return {
            "tick_index": self.tick_index,
            "last_timestamp": self.last_timestamp,
            "last_hedge_tick": self.last_hedge_tick,
            "spot_logs": self.spot_logs[-self.rv_window :],
            "gap_history": {
                symbol: values[-self.gap_history_window :]
                for symbol, values in self.gap_history.items()
                if values
            },
            "active_symbol": self.active_symbol,
            "option_position": self.option_position,
            "underlying_position": self.underlying_position,
        }

    def _append_gap(self, symbol: str, gap: float) -> None:
        history = self.gap_history.setdefault(symbol, [])
        history.append(gap)
        if len(history) > self.gap_history_window:
            del history[:-self.gap_history_window]

    def _rolling_rv(self) -> float | None:
        if len(self.spot_logs) < self.rv_window:
            return None
        logs = self.spot_logs[-self.rv_window :]
        path_vars: list[float] = []
        for offset in range(5):
            sampled = logs[offset::5]
            if len(sampled) < 2:
                continue
            realized_var = 0.0
            for idx in range(1, len(sampled)):
                ret = sampled[idx] - sampled[idx - 1]
                realized_var += ret * ret
            path_vars.append(realized_var)
        if not path_vars:
            return None
        annual_var = statistics.fmean(path_vars) * TRADING_DAYS_PER_YEAR * (TICKS_PER_DAY / self.rv_window)
        return math.sqrt(annual_var)

    def _candidate_trigger(
        self,
        symbol: str,
        strike: int,
        option_depth: OrderDepth,
        spot: float,
        tte_years: float,
    ) -> dict[str, float] | None:
        mid = _voucher_mid(option_depth)
        if mid is None or mid <= 0:
            return None
        iv = _safe_iv(mid, spot, strike, tte_years)
        if iv is None:
            return None
        rv = self._rolling_rv()
        if rv is None:
            self._append_gap(symbol, 0.0)
            return None
        gap = rv - iv
        history = self.gap_history.get(symbol, [])
        z_score = None
        if len(history) >= self.min_gap_history:
            sigma = statistics.stdev(history)
            if sigma > 0:
                z_score = (gap - statistics.fmean(history)) / sigma
        self._append_gap(symbol, gap)
        if z_score is None or gap <= 0 or z_score < self.gap_z_open:
            return None
        return {"iv": iv, "gap": gap, "z": z_score}

    def _sweep_option(
        self,
        state: TradingState,
        symbol: str,
        quantity: int,
        results: dict[str, list[Order]],
    ) -> None:
        if quantity == 0:
            return
        order_depth = state.order_depths.get(symbol)
        if order_depth is None:
            return

        if quantity > 0:
            if not order_depth.sell_orders:
                return
            best_ask = min(order_depth.sell_orders)
            available = max(0, -order_depth.sell_orders[best_ask])
            aggregate_position = state.position.get(symbol, 0)
            cap_room = self.option_limit - max(aggregate_position, 0)
            trade_size = min(quantity, available, cap_room)
            if trade_size <= 0:
                return
            results.setdefault(symbol, []).append(Order(symbol, int(best_ask), int(trade_size)))
            self.option_position += int(trade_size)
            return

        if not order_depth.buy_orders:
            return
        best_bid = max(order_depth.buy_orders)
        available = max(0, order_depth.buy_orders[best_bid])
        trade_size = min(-quantity, available, self.option_position)
        if trade_size <= 0:
            return
        results.setdefault(symbol, []).append(Order(symbol, int(best_bid), -int(trade_size)))
        self.option_position -= int(trade_size)
        if self.option_position == 0:
            self.active_symbol = None

    def _sweep_underlying(
        self,
        state: TradingState,
        quantity: int,
        results: dict[str, list[Order]],
    ) -> None:
        if quantity == 0:
            return
        order_depth = state.order_depths.get(self.underlying)
        if order_depth is None:
            return

        aggregate_position = state.position.get(self.underlying, 0)

        if quantity > 0:
            remaining = quantity
            cap_room = self.underlying_limit - aggregate_position
            remaining = min(remaining, cap_room)
            if remaining <= 0:
                return
            for price in sorted(order_depth.sell_orders):
                available = max(0, -order_depth.sell_orders[price])
                trade_size = min(remaining, available)
                if trade_size <= 0:
                    continue
                results.setdefault(self.underlying, []).append(
                    Order(self.underlying, int(price), int(trade_size))
                )
                self.underlying_position += int(trade_size)
                remaining -= trade_size
                if remaining == 0:
                    break
            return

        remaining = -quantity
        cap_room = self.underlying_limit + aggregate_position
        remaining = min(remaining, cap_room)
        if remaining <= 0:
            return
        for price in sorted(order_depth.buy_orders, reverse=True):
            available = max(0, order_depth.buy_orders[price])
            trade_size = min(remaining, available)
            if trade_size <= 0:
                continue
            results.setdefault(self.underlying, []).append(
                Order(self.underlying, int(price), -int(trade_size))
            )
            self.underlying_position -= int(trade_size)
            remaining -= trade_size
            if remaining == 0:
                break

    def run(
        self,
        state: TradingState,
        existing_orders: dict[str, list[Order]] | None = None,
    ) -> dict[str, list[Order]]:
        del existing_orders

        underlying_depth = state.order_depths.get(self.underlying)
        if underlying_depth is None:
            return {}
        spot = _best_mid(underlying_depth)
        if spot is None or spot <= 0:
            return {}

        if self.last_timestamp is not None and state.timestamp < self.last_timestamp:
            self.last_timestamp = None
        self.last_timestamp = state.timestamp
        self.tick_index += 1
        self.spot_logs.append(math.log(spot))
        if len(self.spot_logs) > self.rv_window:
            del self.spot_logs[:-self.rv_window]

        tte_years = self.tte_func(state.timestamp)
        results: dict[str, list[Order]] = {}

        triggers: list[tuple[str, int, dict[str, float]]] = []
        for strike in self.eligible_strikes:
            symbol = self.option_products.get(strike)
            if symbol is None:
                continue
            if state.position.get(symbol, 0) < 0:
                continue
            option_depth = state.order_depths.get(symbol)
            if option_depth is None:
                continue
            info = self._candidate_trigger(symbol, strike, option_depth, spot, tte_years)
            if info is not None:
                triggers.append((symbol, strike, info))

        target_symbol: str | None = None
        if self.active_symbol is not None:
            for symbol, _, _ in triggers:
                if symbol == self.active_symbol:
                    target_symbol = symbol
                    break
        if target_symbol is None and self.option_position == 0 and triggers:
            triggers.sort(key=lambda item: (abs(item[1] - spot), -item[2]["z"], -item[2]["gap"]))
            target_symbol = triggers[0][0]

        if self.active_symbol is not None and self.active_symbol != target_symbol and self.option_position > 0:
            self._sweep_option(state, self.active_symbol, -self.option_position, results)
        elif target_symbol is not None:
            self.active_symbol = target_symbol
            if self.option_position < self.trade_qty:
                self._sweep_option(state, target_symbol, self.trade_qty - self.option_position, results)

        if self.active_symbol is None and self.option_position > 0:
            self.option_position = 0

        current_iv = None
        current_strike = None
        if self.active_symbol is not None and self.option_position > 0:
            current_strike = int(self.active_symbol.split("_", 1)[1])
            current_depth = state.order_depths.get(self.active_symbol)
            current_mid = _voucher_mid(current_depth) if current_depth is not None else None
            if current_mid is not None and current_mid > 0:
                current_iv = _safe_iv(current_mid, spot, current_strike, tte_years)

        option_delta = 0.0
        if current_iv is not None and current_strike is not None and self.option_position > 0:
            greeks = bs_greeks(spot, float(current_strike), tte_years, 0.0, current_iv, "call")
            option_delta = self.option_position * greeks["delta"]

        aggregate_delta = option_delta + self.underlying_position
        target_underlying = 0
        if option_delta != 0.0:
            target_underlying = -int(round(option_delta))
            target_underlying = max(-self.underlying_limit, min(self.underlying_limit, target_underlying))

        if (
            abs(aggregate_delta) >= self.hedge_delta_threshold
            and (self.tick_index - self.last_hedge_tick) >= self.hedge_every_ticks
        ):
            self._sweep_underlying(state, target_underlying - self.underlying_position, results)
            self.last_hedge_tick = self.tick_index
        elif self.option_position == 0 and self.underlying_position != 0:
            self._sweep_underlying(state, -self.underlying_position, results)

        return results
