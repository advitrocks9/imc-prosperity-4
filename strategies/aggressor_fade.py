from __future__ import annotations

import math
from typing import Iterable

from datamodel import Order, OrderDepth, Trade, TradingState

from strategies._base import best_mid

AGGRESSOR_MARKS = ("Mark 38", "Mark 67", "Mark 22")
DEFAULT_ENABLED_PAIRS = {
    ("Mark 38", "HYDROGEL_PACK"),
    ("Mark 38", "VEV_4000"),
    ("Mark 22", "VEV_5200"),
    ("Mark 22", "VEV_5300"),
    ("Mark 22", "VEV_5400"),
    ("Mark 22", "VEV_5500"),
}


class AggressorFadeStrategy:
    """Fade selected aggressor prints with optional 2-of-3 confirmation."""

    def __init__(
        self,
        enabled_pairs: set[tuple[str, str]] | None = None,
        position_max: int = 50,
        clip_size: int = 10,
        pnl_window_ticks: int = 1000,
        require_confirmation: bool = True,
        imbalance_threshold: float = 0.05,
    ) -> None:
        self.enabled_pairs = enabled_pairs or set(DEFAULT_ENABLED_PAIRS)
        self.position_max = position_max
        self.clip_size = clip_size
        self.pnl_window_ticks = pnl_window_ticks
        self.require_confirmation = require_confirmation
        self.imbalance_threshold = imbalance_threshold
        self.products = {product for _, product in self.enabled_pairs}

    def generate_orders(
        self,
        state: TradingState,
        trader_state: dict | None,
        existing_orders: dict[str, list[Order]] | None = None,
    ) -> tuple[dict[str, list[Order]], dict]:
        next_state = dict(trader_state or {})
        existing_orders = existing_orders or {}
        orders_by_product: dict[str, list[Order]] = {}

        for product in self.products:
            order_depth = state.order_depths.get(product)
            if order_depth is None:
                continue

            product_state = dict(next_state.get(product, {}))
            mid = best_mid(order_depth)
            if mid is None:
                mid = float(product_state.get("last_mid", 0.0))
            if mid <= 0:
                continue

            mids = list(product_state.get("m", []))
            mids.append(round(float(mid), 3))
            if len(mids) > 16:
                mids = mids[-16:]

            imbalance = self._imbalance(order_depth)
            mark_events = {
                mark: self._prune_events(list(product_state.get("ev", {}).get(mark, [])), state.timestamp)
                for mark in AGGRESSOR_MARKS
            }

            candidates = self._extract_candidates(
                state.market_trades.get(product, []),
                order_depth,
                product,
            )
            selected = self._pick_signal(candidates, mids, imbalance, mark_events, mid)

            product_orders: list[Order] = []
            effective_pos = state.position.get(product, 0) + sum(
                order.quantity for order in existing_orders.get(product, [])
            )
            if selected is not None:
                fade_dir = -selected["side"]
                room = (
                    max(0, self.position_max - effective_pos)
                    if fade_dir > 0
                    else max(0, self.position_max + effective_pos)
                )
                qty = min(self.clip_size, room)
                price = self._quote_price(mid, fade_dir)
                if qty > 0:
                    product_orders.append(Order(product, price, qty if fade_dir > 0 else -qty))
                    orders_by_product[product] = product_orders

            for candidate in candidates:
                mark_events[candidate["mark"]].append(
                    [
                        int(state.timestamp),
                        int(candidate["side"]),
                        round(float(candidate["price"]), 3),
                        int(candidate["qty"]),
                    ]
                )

            compact_events = {mark: items for mark, items in mark_events.items() if items}
            if mids or compact_events:
                next_state[product] = {
                    "m": mids,
                    "ev": compact_events,
                    "imb": round(imbalance, 4),
                    "last_mid": round(float(mid), 3),
                }
            else:
                next_state.pop(product, None)

        return orders_by_product, next_state

    def _extract_candidates(
        self,
        trades: Iterable[Trade],
        order_depth: OrderDepth,
        product: str,
    ) -> list[dict[str, int | float | str]]:
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        by_key: dict[tuple[str, int], dict[str, int | float | str]] = {}

        for trade in trades:
            mark = None
            side = 0
            if best_ask is not None and trade.buyer in AGGRESSOR_MARKS and trade.price >= best_ask:
                mark = trade.buyer
                side = 1
            elif best_bid is not None and trade.seller in AGGRESSOR_MARKS and trade.price <= best_bid:
                mark = trade.seller
                side = -1
            if mark is None or (mark, product) not in self.enabled_pairs:
                continue

            key = (mark, side)
            slot = by_key.get(key)
            if slot is None:
                by_key[key] = {
                    "mark": mark,
                    "side": side,
                    "price": float(trade.price),
                    "qty": int(trade.quantity),
                }
            else:
                slot["qty"] = int(slot["qty"]) + int(trade.quantity)
                slot["price"] = float(trade.price)

        return list(by_key.values())

    def _pick_signal(
        self,
        candidates: list[dict[str, int | float | str]],
        mids: list[float],
        imbalance: float,
        mark_events: dict[str, list[list[int | float]]],
        mid: float,
    ) -> dict[str, int | float | str] | None:
        best_candidate = None
        best_score = -1
        for candidate in candidates:
            gates = self._gate_score(
                mark=str(candidate["mark"]),
                side=int(candidate["side"]),
                mids=mids,
                imbalance=imbalance,
                mark_events=mark_events,
                mid=mid,
            )
            if self.require_confirmation and gates < 2:
                continue
            if gates > best_score or (gates == best_score and int(candidate["qty"]) > int(best_candidate["qty"]) if best_candidate else True):
                best_candidate = candidate
                best_score = gates
        return best_candidate

    def _gate_score(
        self,
        mark: str,
        side: int,
        mids: list[float],
        imbalance: float,
        mark_events: dict[str, list[list[int | float]]],
        mid: float,
    ) -> int:
        fade_dir = -side
        score = 0

        if len(mids) >= 11:
            drift10 = mids[-1] - mids[-11]
            if drift10 * fade_dir > 0:
                score += 1

        if imbalance * fade_dir > self.imbalance_threshold:
            score += 1

        pnl = 0.0
        for ts, mark_side, price, qty in mark_events.get(mark, []):
            pnl += float(mark_side) * (mid - float(price)) * int(qty)
        if pnl < 0:
            score += 1

        return score

    def _quote_price(self, mid: float, fade_dir: int) -> int:
        return int(math.floor(mid)) if fade_dir > 0 else int(math.ceil(mid))

    def _prune_events(
        self,
        events: list[list[int | float]],
        timestamp: int,
    ) -> list[list[int | float]]:
        cutoff = timestamp - self.pnl_window_ticks
        return [event for event in events if int(event[0]) >= cutoff]

    def _imbalance(self, order_depth: OrderDepth) -> float:
        bid_vol = sum(max(0, volume) for volume in order_depth.buy_orders.values())
        ask_vol = sum(max(0, -volume) for volume in order_depth.sell_orders.values())
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0
        return (bid_vol - ask_vol) / total
