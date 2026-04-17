from __future__ import annotations

import math
from typing import Iterable

from datamodel import OrderDepth, Trade

AGGRESSOR_MARKS = ("Mark 38", "Mark 67", "Mark 22")
DEFAULT_ENABLED_PAIRS = {
    ("Mark 38", "HYDROGEL_PACK"),
    ("Mark 38", "VEV_4000"),
    ("Mark 22", "VEV_5200"),
    ("Mark 22", "VEV_5300"),
    ("Mark 22", "VEV_5400"),
    ("Mark 22", "VEV_5500"),
}


class FadeBiasTracker:
    """Track decaying post-aggressor mean-reversion pressure by product."""

    def __init__(
        self,
        enabled_pairs: set[tuple[str, str]] | None = None,
        decay_ticks: float = 50.0,
        size_unit: float = 5.0,
    ) -> None:
        self.enabled_pairs = enabled_pairs or set(DEFAULT_ENABLED_PAIRS)
        self.decay_ticks = float(decay_ticks)
        self.size_unit = max(1.0, float(size_unit))
        self._state: dict[str, dict[str, float | int]] = {}

    def update(
        self,
        product: str,
        market_trades: Iterable[Trade],
        timestamp: int,
        order_depth: OrderDepth | None = None,
    ) -> None:
        slot = self._state.setdefault(product, {"pressure": 0.0, "timestamp": int(timestamp)})
        pressure = float(slot.get("pressure", 0.0))
        last_timestamp = int(slot.get("timestamp", timestamp))
        dt = max(0, int(timestamp) - last_timestamp)
        if dt > 0 and self.decay_ticks > 0:
            pressure *= math.exp(-dt / self.decay_ticks)

        signed_size = 0.0
        for signal in self._extract_signals(product, market_trades, order_depth):
            signed_size += float(signal["signed_qty"])

        if signed_size:
            pressure += signed_size / self.size_unit

        slot["pressure"] = pressure
        slot["timestamp"] = int(timestamp)

    def get_pressure(self, product: str) -> float:
        slot = self._state.get(product)
        if slot is None:
            return 0.0
        return float(slot.get("pressure", 0.0))

    def to_dict(self) -> dict[str, dict[str, float | int]]:
        encoded: dict[str, dict[str, float | int]] = {}
        for product, slot in self._state.items():
            encoded[product] = {
                "pressure": round(float(slot.get("pressure", 0.0)), 6),
                "timestamp": int(slot.get("timestamp", 0)),
            }
        return encoded

    @classmethod
    def from_dict(
        cls,
        data: dict[str, dict[str, float | int]] | None,
        enabled_pairs: set[tuple[str, str]] | None = None,
        decay_ticks: float = 50.0,
        size_unit: float = 5.0,
    ) -> "FadeBiasTracker":
        tracker = cls(
            enabled_pairs=enabled_pairs,
            decay_ticks=decay_ticks,
            size_unit=size_unit,
        )
        if data:
            for product, slot in data.items():
                tracker._state[product] = {
                    "pressure": float(slot.get("pressure", 0.0)),
                    "timestamp": int(slot.get("timestamp", 0)),
                }
        return tracker

    def _extract_signals(
        self,
        product: str,
        trades: Iterable[Trade],
        order_depth: OrderDepth | None,
    ) -> list[dict[str, float | str]]:
        best_ask = min(order_depth.sell_orders) if order_depth and order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders) if order_depth and order_depth.buy_orders else None
        by_key: dict[tuple[str, int], dict[str, float | str]] = {}

        for trade in trades:
            mark = None
            side = 0

            if (
                best_ask is not None
                and trade.buyer in AGGRESSOR_MARKS
                and trade.price >= best_ask
            ):
                mark = trade.buyer
                side = -1
            elif (
                best_bid is not None
                and trade.seller in AGGRESSOR_MARKS
                and trade.price <= best_bid
            ):
                mark = trade.seller
                side = 1
            elif best_ask is None or best_bid is None:
                if trade.seller in AGGRESSOR_MARKS:
                    mark = trade.seller
                    side = 1
                elif trade.buyer in AGGRESSOR_MARKS:
                    mark = trade.buyer
                    side = -1

            if mark is None or (mark, product) not in self.enabled_pairs:
                continue

            key = (mark, side)
            slot = by_key.get(key)
            if slot is None:
                by_key[key] = {
                    "mark": mark,
                    "signed_qty": float(side * int(trade.quantity)),
                }
            else:
                slot["signed_qty"] = float(slot["signed_qty"]) + float(side * int(trade.quantity))

        return list(by_key.values())
