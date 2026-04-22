from __future__ import annotations

import math
from typing import TYPE_CHECKING

if __package__ in {None, ""}:
    _sys = __import__("sys")
    _pathlib = __import__("pathlib")
    _sys.path.append(str(_pathlib.Path(__file__).resolve().parents[1]))

if TYPE_CHECKING:
    from datamodel import Order, OrderDepth, Trade, TradingState

UNDERLYING_SYMBOL = "VELVETFRUIT_EXTRACT"
POSITION_CAP_PER_STRIKE = 50


def _strike_from_symbol(symbol: str) -> int | None:
    if not symbol.startswith("VEV_"):
        return None
    try:
        return int(symbol.split("_", 1)[1])
    except ValueError:
        return None


def _best_mid(order_depth: OrderDepth) -> float | None:
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0


def _tte_years(timestamp: int) -> float:
    days_remaining = max(4.0 - (timestamp / 1_000_000.0), 3.0)
    return days_remaining / 252.0


class BSVoucherSmileTrader:
    def __init__(
        self,
        window: int = 100,
        k: float = 1.5,
        min_history: int | None = None,
        stop_loss_per_strike: float | None = None,
    ) -> None:
        self.window = max(2, int(window))
        self.k = float(k)
        self.min_history = max(10, min_history or min(30, self.window))
        self.stop_loss_per_strike = stop_loss_per_strike
        self.position_cap = POSITION_CAP_PER_STRIKE

    def _fresh_td(self, timestamp: int) -> dict:
        return {
            "pt": timestamp,
            "bl": [],
            "eap": {},
            "upnl": {},
            "pos": {},
            "ptr": {},
            "ivh": {},
        }

    def _load_td(self, td: dict | None, timestamp: int) -> dict:
        if not isinstance(td, dict):
            return self._fresh_td(timestamp)
        prev_ts = int(td.get("pt", timestamp))
        if timestamp < prev_ts:
            return self._fresh_td(timestamp)
        iv_history: dict[str, list[float]] = {}
        for key, values in td.get("ivh", {}).items():
            if not isinstance(values, list):
                continue
            iv_history[str(key)] = [
                float(value)
                for value in values[-self.window :]
                if isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            ]
        return {
            "pt": prev_ts,
            "bl": [int(x) for x in td.get("bl", [])],
            "eap": {str(k): float(v) for k, v in td.get("eap", {}).items()},
            "upnl": {str(k): float(v) for k, v in td.get("upnl", {}).items()},
            "pos": {str(k): int(v) for k, v in td.get("pos", {}).items()},
            "ptr": {
                str(k): [int(v[0]), int(v[1])]
                for k, v in td.get("ptr", {}).items()
                if isinstance(v, list) and len(v) == 2
            },
            "ivh": iv_history,
        }

    def _trade_cursor(
        self,
        trades: list[Trade],
        cursor: list[int] | None,
    ) -> tuple[list[Trade], list[int]]:
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
        final_count = counts_by_ts[final_ts]
        return new_trades, [final_ts, final_count]

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

    def _stats(self, history: list[float]) -> tuple[float, float] | None:
        if len(history) < self.min_history:
            return None
        n = len(history)
        mean = sum(history) / n
        if n < 2:
            return None
        variance = sum((value - mean) * (value - mean) for value in history) / (n - 1)
        if variance <= 0:
            return None
        return mean, math.sqrt(variance)

    def _buy_order(self, symbol: str, order_depth: OrderDepth, position: int, target: int | None = None) -> Order | None:
        from datamodel import Order

        if not order_depth.sell_orders:
            return None
        best_ask = min(order_depth.sell_orders)
        ask_volume = max(0, -order_depth.sell_orders[best_ask])
        buy_capacity = self.position_cap - position
        if target is not None:
            buy_capacity = min(buy_capacity, max(0, target - position))
        quantity = min(ask_volume, buy_capacity)
        if quantity <= 0:
            return None
        return Order(symbol, int(best_ask), int(quantity))

    def _sell_order(self, symbol: str, order_depth: OrderDepth, position: int, target: int | None = None) -> Order | None:
        from datamodel import Order

        if not order_depth.buy_orders:
            return None
        best_bid = max(order_depth.buy_orders)
        bid_volume = max(0, order_depth.buy_orders[best_bid])
        sell_capacity = self.position_cap + position
        if target is not None:
            sell_capacity = min(sell_capacity, max(0, position - target))
        quantity = min(bid_volume, sell_capacity)
        if quantity <= 0:
            return None
        return Order(symbol, int(best_bid), -int(quantity))

    def _close_order(self, symbol: str, order_depth: OrderDepth, position: int) -> Order | None:
        if position > 0:
            return self._sell_order(symbol, order_depth, position, target=0)
        if position < 0:
            return self._buy_order(symbol, order_depth, position, target=0)
        return None

    def run(self, state: TradingState, td: dict | None = None) -> tuple[dict[str, list[Order]], dict]:
        from strategies._base import bs_call, find_iv

        td_state = self._load_td(td, state.timestamp)
        td_state["pt"] = int(state.timestamp)
        blacklist = set(int(x) for x in td_state.get("bl", []))
        entry_avg = td_state.get("eap", {})
        unrealized = td_state.get("upnl", {})
        tracked_pos = td_state.get("pos", {})
        trade_ptr = td_state.get("ptr", {})
        iv_history = td_state.get("ivh", {})

        underlying_depth = state.order_depths.get(UNDERLYING_SYMBOL)
        spot = _best_mid(underlying_depth) if underlying_depth is not None else None
        tte = _tte_years(state.timestamp)
        results: dict[str, list[Order]] = {}

        voucher_symbols = sorted(
            (symbol for symbol in state.order_depths if symbol.startswith("VEV_")),
            key=lambda symbol: (_strike_from_symbol(symbol) or 0, symbol),
        )

        for symbol in voucher_symbols:
            strike = _strike_from_symbol(symbol)
            order_depth = state.order_depths.get(symbol)
            if strike is None or order_depth is None:
                continue

            strike_key = str(strike)
            mid = _best_mid(order_depth)
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
                elif avg_price == 0.0 and mid is not None:
                    avg_price = float(mid)

            if position == 0:
                tracked_pos.pop(strike_key, None)
                entry_avg.pop(strike_key, None)
                unrealized[strike_key] = 0.0
            else:
                tracked_pos[strike_key] = position
                if avg_price != 0.0:
                    entry_avg[strike_key] = avg_price

            if strike in blacklist:
                if position != 0:
                    close_order = self._close_order(symbol, order_depth, position)
                    if close_order is not None:
                        results[symbol] = [close_order]
                unrealized[strike_key] = 0.0
                continue

            if position != 0 and mid is not None and strike_key in entry_avg:
                upnl = entry_avg[strike_key] * position - float(mid) * position
                unrealized[strike_key] = round(upnl, 4)
                if self.stop_loss_per_strike is not None and upnl < -self.stop_loss_per_strike:
                    blacklist.add(strike)
                    close_order = self._close_order(symbol, order_depth, position)
                    if close_order is not None:
                        results[symbol] = [close_order]
                    entry_avg.pop(strike_key, None)
                    unrealized[strike_key] = 0.0
                    continue
            else:
                unrealized[strike_key] = 0.0

            if spot is None or spot <= 0 or mid is None or mid <= 0 or tte <= 0:
                continue

            iv = find_iv(spot, float(strike), tte, 0.0, mid)
            if iv is None or not math.isfinite(iv) or iv <= 0:
                continue

            history = iv_history.setdefault(strike_key, [])
            stats = self._stats(history)
            history.append(float(iv))
            if len(history) > self.window:
                del history[:-self.window]
            if stats is None:
                continue

            mean_iv, std_iv = stats
            z_score = (iv - mean_iv) / std_iv
            fair_price, _ = bs_call(spot, float(strike), tte, 0.0, mean_iv)

            close_order = None
            if actual_position > 0 and z_score >= 0.0:
                close_order = self._close_order(symbol, order_depth, actual_position)
            elif actual_position < 0 and z_score <= 0.0:
                close_order = self._close_order(symbol, order_depth, actual_position)
            if close_order is not None:
                results[symbol] = [close_order]
                continue

            order = None
            if z_score < -self.k and order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders)
                if fair_price - best_ask > 1.0:
                    order = self._buy_order(symbol, order_depth, actual_position)
            elif z_score > self.k and order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders)
                if best_bid - fair_price > 1.0:
                    order = self._sell_order(symbol, order_depth, actual_position)
            if order is not None:
                results[symbol] = [order]

        td_state["bl"] = sorted(blacklist)
        td_state["eap"] = {k: round(v, 4) for k, v in entry_avg.items()}
        td_state["upnl"] = {k: round(v, 4) for k, v in unrealized.items() if abs(v) > 1e-9}
        td_state["pos"] = {k: int(v) for k, v in tracked_pos.items() if int(v) != 0}
        td_state["ptr"] = trade_ptr
        td_state["ivh"] = {k: [round(v, 6) for v in values[-self.window :]] for k, values in iv_history.items() if values}
        return results, td_state
