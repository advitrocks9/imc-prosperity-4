"""Simple rolling-z pair arbitrage module for one configured pair."""
from __future__ import annotations

import math
from collections import deque

if __package__ in {None, ""}:
    _sys = __import__("sys")
    _pathlib = __import__("pathlib")
    _sys.path.append(str(_pathlib.Path(__file__).resolve().parents[1]))

from datamodel import Order, OrderDepth, TradingState
from strategies._base import best_mid


class PairArbTrader:
    def __init__(
        self,
        product_a: str,
        product_b: str,
        beta: float,
        *,
        window: int = 200,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
        leg_cap: int = 50,
        half_life: float = float("inf"),
        lag1_diff_autocorr: float = 0.0,
    ) -> None:
        self.product_a = product_a
        self.product_b = product_b
        self.beta = float(beta)
        self.window = int(window)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.leg_cap = int(leg_cap)
        self.half_life = float(half_life)
        self.lag1_diff_autocorr = float(lag1_diff_autocorr)
        self.spreads: deque[float] = deque(maxlen=self.window)
        self.mode = 0
        self.units_a = 0
        self.units_b = 0

    def gate_passes(self) -> bool:
        return self.half_life < 500.0 and self.lag1_diff_autocorr < -0.05

    def load_state(self, raw_state: dict | None) -> None:
        state = raw_state or {}
        spreads = state.get("spreads", [])
        self.spreads = deque(maxlen=self.window)
        if isinstance(spreads, list):
            for value in spreads[-self.window:]:
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self.spreads.append(float(value))

        mode = state.get("mode", 0)
        units_a = state.get("ua", 0)
        units_b = state.get("ub", 0)
        self.mode = int(mode) if isinstance(mode, int) else 0
        self.units_a = int(units_a) if isinstance(units_a, int) else 0
        self.units_b = int(units_b) if isinstance(units_b, int) else 0

    def dump_state(self) -> dict:
        return {
            "spreads": list(self.spreads),
            "mode": self.mode,
            "ua": self.units_a,
            "ub": self.units_b,
        }

    def _planned_position(
        self,
        state: TradingState,
        existing_orders: dict[str, list[Order]],
        product: str,
    ) -> int:
        return state.position.get(product, 0) + sum(order.quantity for order in existing_orders.get(product, []))

    def _available_caps(
        self,
        state: TradingState,
        existing_orders: dict[str, list[Order]],
        product: str,
    ) -> tuple[int, int]:
        planned = self._planned_position(state, existing_orders, product)
        return max(0, self.leg_cap - planned), max(0, self.leg_cap + planned)

    def _buy_orders(self, product: str, od: OrderDepth, target_qty: int) -> list[Order]:
        remaining = max(0, int(target_qty))
        orders: list[Order] = []
        for ask_price in sorted(od.sell_orders):
            if remaining <= 0:
                break
            ask_volume = -int(od.sell_orders[ask_price])
            if ask_volume <= 0:
                continue
            qty = min(remaining, ask_volume)
            orders.append(Order(product, int(ask_price), int(qty)))
            remaining -= qty
        return orders

    def _sell_orders(self, product: str, od: OrderDepth, target_qty: int) -> list[Order]:
        remaining = max(0, int(target_qty))
        orders: list[Order] = []
        for bid_price in sorted(od.buy_orders, reverse=True):
            if remaining <= 0:
                break
            bid_volume = int(od.buy_orders[bid_price])
            if bid_volume <= 0:
                continue
            qty = min(remaining, bid_volume)
            orders.append(Order(product, int(bid_price), -int(qty)))
            remaining -= qty
        return orders

    def _stats(self) -> tuple[float | None, float | None]:
        n = len(self.spreads)
        if n < self.window:
            return None, None
        total = sum(self.spreads)
        total_sq = sum(value * value for value in self.spreads)
        avg = total / n
        variance = (total_sq - (total * total) / n) / max(n - 1, 1)
        if variance <= 1e-12:
            return avg, None
        return avg, math.sqrt(variance)

    def run(
        self,
        state: TradingState,
        existing_orders: dict[str, list[Order]] | None = None,
    ) -> dict[str, list[Order]]:
        if not self.gate_passes():
            return {}
        if self.product_a not in state.order_depths or self.product_b not in state.order_depths:
            return {}

        existing = existing_orders or {}
        od_a = state.order_depths[self.product_a]
        od_b = state.order_depths[self.product_b]
        mid_a = best_mid(od_a)
        mid_b = best_mid(od_b)
        if mid_a is None or mid_b is None:
            return {}

        spread = mid_a - self.beta * mid_b
        self.spreads.append(spread)
        spread_mean, spread_std = self._stats()
        if spread_mean is None or spread_std is None or spread_std <= 1e-12:
            return {}

        z_score = (spread - spread_mean) / spread_std
        results: dict[str, list[Order]] = {}

        if self.mode == 0:
            if z_score > self.entry_z:
                buy_cap_b, sell_cap_b = self._available_caps(state, existing, self.product_b)
                buy_cap_a, sell_cap_a = self._available_caps(state, existing, self.product_a)
                max_units_a = min(sell_cap_a, int(buy_cap_b / max(self.beta, 1e-9)))
                units_a = max(0, min(self.leg_cap, max_units_a))
                units_b = max(0, min(self.leg_cap, int(round(units_a * self.beta))))
                if units_a > 0 and units_b > 0:
                    orders_a = self._sell_orders(self.product_a, od_a, units_a)
                    orders_b = self._buy_orders(self.product_b, od_b, units_b)
                    sent_a = -sum(order.quantity for order in orders_a)
                    sent_b = sum(order.quantity for order in orders_b)
                    if sent_a > 0 and sent_b > 0:
                        results[self.product_a] = orders_a
                        results[self.product_b] = orders_b
                        self.mode = -1
                        self.units_a = sent_a
                        self.units_b = sent_b
            elif z_score < -self.entry_z:
                buy_cap_a, sell_cap_a = self._available_caps(state, existing, self.product_a)
                buy_cap_b, sell_cap_b = self._available_caps(state, existing, self.product_b)
                max_units_a = min(buy_cap_a, int(sell_cap_b / max(self.beta, 1e-9)))
                units_a = max(0, min(self.leg_cap, max_units_a))
                units_b = max(0, min(self.leg_cap, int(round(units_a * self.beta))))
                if units_a > 0 and units_b > 0:
                    orders_a = self._buy_orders(self.product_a, od_a, units_a)
                    orders_b = self._sell_orders(self.product_b, od_b, units_b)
                    sent_a = sum(order.quantity for order in orders_a)
                    sent_b = -sum(order.quantity for order in orders_b)
                    if sent_a > 0 and sent_b > 0:
                        results[self.product_a] = orders_a
                        results[self.product_b] = orders_b
                        self.mode = 1
                        self.units_a = sent_a
                        self.units_b = sent_b
        elif self.mode < 0 and z_score <= self.exit_z:
            buy_cap_a, sell_cap_a = self._available_caps(state, existing, self.product_a)
            buy_cap_b, sell_cap_b = self._available_caps(state, existing, self.product_b)
            close_a = min(self.units_a, buy_cap_a)
            close_b = min(self.units_b, sell_cap_b)
            if close_a > 0 and close_b > 0:
                results[self.product_a] = self._buy_orders(self.product_a, od_a, close_a)
                results[self.product_b] = self._sell_orders(self.product_b, od_b, close_b)
                self.mode = 0
                self.units_a = 0
                self.units_b = 0
        elif self.mode > 0 and z_score >= self.exit_z:
            buy_cap_b, sell_cap_b = self._available_caps(state, existing, self.product_b)
            buy_cap_a, sell_cap_a = self._available_caps(state, existing, self.product_a)
            close_a = min(self.units_a, sell_cap_a)
            close_b = min(self.units_b, buy_cap_b)
            if close_a > 0 and close_b > 0:
                results[self.product_a] = self._sell_orders(self.product_a, od_a, close_a)
                results[self.product_b] = self._buy_orders(self.product_b, od_b, close_b)
                self.mode = 0
                self.units_a = 0
                self.units_b = 0

        return results
