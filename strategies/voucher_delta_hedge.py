from __future__ import annotations

import math

from datamodel import Order, OrderDepth, Trade, TradingState
from strategies._base import best_mid, bs_call, bs_delta, find_iv

UNDERLYING_SYMBOL = "VELVETFRUIT_EXTRACT"
ALL_10_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
STRIKE_SET_PRESETS = {
    "all_10": ALL_10_STRIKES,
    "subset_4000_5000": [4000, 4500, 5000],
    "subset_4500_5500": [4500, 5000, 5100, 5200, 5300, 5400, 5500],
}


def _strike_symbol(strike: int) -> str:
    return f"VEV_{strike}"


def _strike_from_symbol(symbol: str) -> int | None:
    if not symbol.startswith("VEV_"):
        return None
    try:
        return int(symbol.split("_", 1)[1])
    except ValueError:
        return None


def _tte_years(timestamp: int) -> float:
    days_remaining = max(4.0 - (timestamp / 1_000_000.0), 3.0)
    return days_remaining / 252.0


class VoucherDeltaHedgeTrader:
    def __init__(
        self,
        variant: str = "take",
        threshold: int = 3,
        anchor_strike: str = "VEV_5300",
        enabled_strikes: str | list[int] | None = None,
        max_pos_per_strike: int = 30,
        underlying_limit: int = 200,
        stop_loss_per_strike: float = 2000.0,
        half_spread: int = 1,
        delta_hedge_tolerance: int = 1,
    ) -> None:
        self.variant = variant
        self.threshold = int(threshold)
        self.anchor_strike = anchor_strike
        self.enabled_strikes = self._resolve_enabled_strikes(enabled_strikes)
        self.max_pos_per_strike = int(max_pos_per_strike)
        self.underlying_limit = int(underlying_limit)
        self.stop_loss_per_strike = float(stop_loss_per_strike)
        self.half_spread = int(half_spread)
        self.delta_hedge_tolerance = int(delta_hedge_tolerance)

    def _resolve_enabled_strikes(self, enabled_strikes: str | list[int] | None) -> list[int]:
        if enabled_strikes is None:
            return list(ALL_10_STRIKES)
        if isinstance(enabled_strikes, str):
            return list(STRIKE_SET_PRESETS.get(enabled_strikes, ALL_10_STRIKES))
        return sorted(int(strike) for strike in enabled_strikes)

    def _fresh_td(self, timestamp: int) -> dict:
        return {
            "pt": int(timestamp),
            "bl": [],
            "eap": {},
            "upnl": {},
            "pos": {},
            "ptr": {},
            "hpos": 0,
            "liv": None,
            "mxdelta": 0.0,
            "mxhedge": 0,
            "hedged_qty": 0,
        }

    def _load_td(self, td: dict | None, timestamp: int) -> dict:
        if not isinstance(td, dict):
            return self._fresh_td(timestamp)
        if int(timestamp) < int(td.get("pt", timestamp)):
            return self._fresh_td(timestamp)
        return {
            "pt": int(td.get("pt", timestamp)),
            "bl": [int(x) for x in td.get("bl", [])],
            "eap": {str(k): float(v) for k, v in td.get("eap", {}).items()},
            "upnl": {str(k): float(v) for k, v in td.get("upnl", {}).items()},
            "pos": {str(k): int(v) for k, v in td.get("pos", {}).items()},
            "ptr": {
                str(k): [int(v[0]), int(v[1])]
                for k, v in td.get("ptr", {}).items()
                if isinstance(v, list) and len(v) == 2
            },
            "hpos": int(td.get("hpos", 0)),
            "liv": td.get("liv"),
            "mxdelta": float(td.get("mxdelta", 0.0)),
            "mxhedge": int(td.get("mxhedge", 0)),
            "hedged_qty": int(td.get("hedged_qty", 0)),
        }

    def _trade_cursor(self, trades: list[Trade], cursor: list[int] | None) -> tuple[list[Trade], list[int]]:
        last_ts = int(cursor[0]) if isinstance(cursor, list) and len(cursor) == 2 else -1
        last_count = int(cursor[1]) if isinstance(cursor, list) and len(cursor) == 2 else 0
        if not trades:
            return [], [last_ts, last_count]
        ordered = sorted(
            trades,
            key=lambda trade: (
                int(trade.timestamp),
                int(trade.price),
                int(trade.quantity),
                trade.buyer or "",
                trade.seller or "",
            ),
        )
        counts_by_ts: dict[int, int] = {}
        new_trades: list[Trade] = []
        for trade in ordered:
            trade_ts = int(trade.timestamp)
            counts_by_ts[trade_ts] = counts_by_ts.get(trade_ts, 0) + 1
            seen_count = counts_by_ts[trade_ts]
            if trade_ts < last_ts:
                continue
            if trade_ts == last_ts and seen_count <= last_count:
                continue
            new_trades.append(trade)
        final_ts = int(ordered[-1].timestamp)
        return new_trades, [final_ts, counts_by_ts[final_ts]]

    def _signed_qty(self, trade: Trade) -> int:
        qty = int(trade.quantity)
        if trade.buyer == "SUBMISSION":
            return qty
        if trade.seller == "SUBMISSION":
            return -qty
        return qty

    def _apply_fill(self, position: int, avg_price: float, signed_qty: int, price: int) -> tuple[int, float]:
        if signed_qty == 0:
            return position, avg_price
        new_position = position + signed_qty
        fill_price = float(price)
        if position == 0:
            return new_position, fill_price if new_position != 0 else 0.0
        if position * signed_qty > 0:
            weighted_notional = avg_price * position + fill_price * signed_qty
            return new_position, weighted_notional / new_position if new_position != 0 else 0.0
        if position * new_position > 0:
            return new_position, avg_price
        if new_position == 0:
            return 0, 0.0
        return new_position, fill_price

    def _flatten_order(self, symbol: str, order_depth: OrderDepth, position: int) -> Order | None:
        if position > 0 and order_depth.buy_orders:
            return Order(symbol, int(max(order_depth.buy_orders)), -int(position))
        if position < 0 and order_depth.sell_orders:
            return Order(symbol, int(min(order_depth.sell_orders)), -int(position))
        return None

    def _anchor_state(self, state: TradingState, td_state: dict) -> tuple[float, float, float] | None:
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
            iv = td_state.get("liv")
        if iv is None or not math.isfinite(iv) or iv <= 0:
            return None
        td_state["liv"] = float(iv)
        return spot, tte, float(iv)

    def _take_buy(self, symbol: str, order_depth: OrderDepth, position: int) -> Order | None:
        if not order_depth.sell_orders:
            return None
        best_ask = min(order_depth.sell_orders)
        ask_volume = max(0, -order_depth.sell_orders[best_ask])
        quantity = min(ask_volume, self.max_pos_per_strike - position)
        if quantity <= 0:
            return None
        return Order(symbol, int(best_ask), int(quantity))

    def _take_sell(self, symbol: str, order_depth: OrderDepth, position: int) -> Order | None:
        if not order_depth.buy_orders:
            return None
        best_bid = max(order_depth.buy_orders)
        bid_volume = max(0, order_depth.buy_orders[best_bid])
        quantity = min(bid_volume, self.max_pos_per_strike + position)
        if quantity <= 0:
            return None
        return Order(symbol, int(best_bid), -int(quantity))

    def _make_orders(
        self,
        symbol: str,
        order_depth: OrderDepth,
        theoretical: float,
        position: int,
    ) -> list[Order]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        bid_price = min(int(round(theoretical - self.half_spread)), int(best_ask) - 1)
        ask_price = max(int(round(theoretical + self.half_spread)), int(best_bid) + 1)
        bid_price = max(1, bid_price)
        rem_buy = self.max_pos_per_strike - position
        rem_sell = self.max_pos_per_strike + position
        orders: list[Order] = []
        if rem_buy > 0:
            orders.append(Order(symbol, bid_price, int(rem_buy)))
        if rem_sell > 0:
            orders.append(Order(symbol, ask_price, -int(rem_sell)))
        return orders

    def _trim_existing_underlying(self, existing_orders: dict[str, list[Order]] | None, state_pos: int, signed_qty: int) -> int:
        if existing_orders is None or signed_qty == 0:
            return abs(signed_qty)
        orders = existing_orders.get(UNDERLYING_SYMBOL, [])
        side_buy = signed_qty > 0
        cap = self.underlying_limit - state_pos if side_buy else self.underlying_limit + state_pos
        same_side = sum(max(0, o.quantity) if side_buy else max(0, -o.quantity) for o in orders)
        needed = abs(signed_qty)
        excess = same_side + needed - cap
        if excess > 0:
            for order in orders:
                if side_buy and order.quantity > 0:
                    cut = min(order.quantity, excess)
                    order.quantity -= cut
                    excess -= cut
                elif (not side_buy) and order.quantity < 0:
                    cut = min(-order.quantity, excess)
                    order.quantity += cut
                    excess -= cut
                if excess <= 0:
                    break
            existing_orders[UNDERLYING_SYMBOL] = [order for order in orders if order.quantity != 0]
        remaining_cap = cap - sum(max(0, o.quantity) if side_buy else max(0, -o.quantity) for o in existing_orders.get(UNDERLYING_SYMBOL, []))
        return max(0, min(needed, remaining_cap))

    def _take_underlying(self, order_depth: OrderDepth, signed_qty: int) -> list[Order]:
        if signed_qty == 0:
            return []
        remaining = abs(int(signed_qty))
        orders: list[Order] = []
        if signed_qty > 0:
            for price in sorted(order_depth.sell_orders):
                available = max(0, -order_depth.sell_orders[price])
                if available <= 0:
                    continue
                take_qty = min(remaining, available)
                orders.append(Order(UNDERLYING_SYMBOL, int(price), int(take_qty)))
                remaining -= take_qty
                if remaining == 0:
                    break
            return orders
        for price in sorted(order_depth.buy_orders, reverse=True):
            available = max(0, order_depth.buy_orders[price])
            if available <= 0:
                continue
            take_qty = min(remaining, available)
            orders.append(Order(UNDERLYING_SYMBOL, int(price), -int(take_qty)))
            remaining -= take_qty
            if remaining == 0:
                break
        return orders

    def run(
        self,
        state: TradingState,
        td: dict | None = None,
        existing_orders: dict[str, list[Order]] | None = None,
    ) -> tuple[dict[str, list[Order]], dict]:
        td_state = self._load_td(td, state.timestamp)
        td_state["pt"] = int(state.timestamp)
        blacklist = set(int(x) for x in td_state["bl"])
        entry_avg = td_state["eap"]
        unrealized = td_state["upnl"]
        tracked_pos = td_state["pos"]
        trade_ptr = td_state["ptr"]
        hedge_pos = int(td_state.get("hpos", 0))

        anchor_state = self._anchor_state(state, td_state)
        results: dict[str, list[Order]] = {}
        expected_positions: dict[int, int] = {}

        for strike in self.enabled_strikes:
            symbol = _strike_symbol(strike)
            order_depth = state.order_depths.get(symbol)
            if order_depth is None:
                continue
            strike_key = str(strike)
            fills, trade_ptr[strike_key] = self._trade_cursor(
                state.own_trades.get(symbol, []),
                trade_ptr.get(strike_key),
            )
            position = int(tracked_pos.get(strike_key, 0))
            avg_price = float(entry_avg.get(strike_key, 0.0))
            for trade in fills:
                position, avg_price = self._apply_fill(position, avg_price, self._signed_qty(trade), int(trade.price))
            actual_position = int(state.position.get(symbol, 0))
            if position != actual_position:
                position = actual_position
                if position == 0:
                    avg_price = 0.0
                elif avg_price == 0.0:
                    mid = best_mid(order_depth)
                    avg_price = float(mid) if mid is not None else 0.0
            if position == 0:
                tracked_pos.pop(strike_key, None)
                entry_avg.pop(strike_key, None)
                unrealized[strike_key] = 0.0
            else:
                tracked_pos[strike_key] = position
                if avg_price != 0.0:
                    entry_avg[strike_key] = avg_price

            mid = best_mid(order_depth)
            immediate_qty = 0
            if strike in blacklist:
                flatten_order = self._flatten_order(symbol, order_depth, position)
                if flatten_order is not None:
                    results[symbol] = [flatten_order]
                    immediate_qty += flatten_order.quantity
                expected_positions[strike] = position + immediate_qty
                unrealized[strike_key] = 0.0
                continue

            if position != 0 and mid is not None and strike_key in entry_avg:
                upnl = entry_avg[strike_key] * position - float(mid) * position
                unrealized[strike_key] = round(upnl, 4)
                if upnl < -self.stop_loss_per_strike:
                    blacklist.add(strike)
                    flatten_order = self._flatten_order(symbol, order_depth, position)
                    if flatten_order is not None:
                        results[symbol] = [flatten_order]
                        immediate_qty += flatten_order.quantity
                    entry_avg.pop(strike_key, None)
                    unrealized[strike_key] = 0.0
                    expected_positions[strike] = position + immediate_qty
                    continue
            else:
                unrealized[strike_key] = 0.0

            if anchor_state is None or mid is None:
                expected_positions[strike] = position
                continue

            spot, tte, flat_iv = anchor_state
            theoretical, _ = bs_call(spot, float(strike), tte, 0.0, flat_iv)
            deviation = float(mid) - float(theoretical)
            orders: list[Order] = []
            if self.variant in {"take", "both"} and deviation > self.threshold:
                order = self._take_sell(symbol, order_depth, position)
                if order is not None:
                    orders.append(order)
                    immediate_qty += order.quantity
            elif self.variant in {"take", "both"} and deviation < -self.threshold:
                order = self._take_buy(symbol, order_depth, position)
                if order is not None:
                    orders.append(order)
                    immediate_qty += order.quantity
            elif self.variant in {"make", "both"}:
                orders.extend(self._make_orders(symbol, order_depth, theoretical, position))
            if orders:
                results[symbol] = orders
            expected_positions[strike] = position + immediate_qty

        if anchor_state is not None:
            spot, tte, flat_iv = anchor_state
            target_delta = 0.0
            for strike, position in expected_positions.items():
                if position == 0:
                    continue
                target_delta += position * bs_delta(spot, float(strike), tte, 0.0, flat_iv)
            td_state["mxdelta"] = max(float(td_state["mxdelta"]), abs(target_delta))
            target_hedge_pos = max(-self.underlying_limit, min(self.underlying_limit, -round(target_delta)))
            hedge_qty = int(target_hedge_pos - hedge_pos)
            if abs(hedge_qty) > self.delta_hedge_tolerance:
                underlying_depth = state.order_depths.get(UNDERLYING_SYMBOL)
                if underlying_depth is not None:
                    allowed_qty = self._trim_existing_underlying(
                        existing_orders,
                        int(state.position.get(UNDERLYING_SYMBOL, 0)),
                        hedge_qty,
                    )
                    signed_qty = allowed_qty if hedge_qty > 0 else -allowed_qty
                    hedge_orders = self._take_underlying(underlying_depth, signed_qty)
                    if hedge_orders:
                        results.setdefault(UNDERLYING_SYMBOL, []).extend(hedge_orders)
                        filled_qty = sum(order.quantity for order in hedge_orders)
                        hedge_pos += filled_qty
                        td_state["hedged_qty"] = int(td_state["hedged_qty"]) + sum(abs(order.quantity) for order in hedge_orders)
                        td_state["mxhedge"] = max(int(td_state["mxhedge"]), abs(hedge_pos))

        td_state["bl"] = sorted(blacklist)
        td_state["eap"] = {k: round(v, 4) for k, v in entry_avg.items()}
        td_state["upnl"] = {k: round(v, 4) for k, v in unrealized.items() if abs(v) > 1e-9}
        td_state["pos"] = {k: int(v) for k, v in tracked_pos.items() if int(v) != 0}
        td_state["ptr"] = trade_ptr
        td_state["hpos"] = int(hedge_pos)
        return results, td_state


class VoucherDeltaHedgeV1(VoucherDeltaHedgeTrader):
    def __init__(self, **kwargs) -> None:
        super().__init__(variant="take", **kwargs)


class VoucherDeltaHedgeV2(VoucherDeltaHedgeTrader):
    def __init__(self, **kwargs) -> None:
        super().__init__(variant="make", **kwargs)


class VoucherDeltaHedgeV3(VoucherDeltaHedgeTrader):
    def __init__(self, **kwargs) -> None:
        super().__init__(variant="both", **kwargs)
