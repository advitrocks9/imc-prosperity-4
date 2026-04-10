"""IMC Prosperity 4 - Round 3 Trader (baseline v1)

Products:
- HYDROGEL_PACK (limit 200)        - STABLE archetype, FV ≈ 10000
- VELVETFRUIT_EXTRACT (limit 200)  - options underlying; reserved entirely for delta hedging
- VEV_4000 ... VEV_6500 (limit 300 each) - European call options on VELVETFRUIT_EXTRACT, 10 strikes

Round 3 final-sim TTE = 5d at start, decays linearly to 4d over 10K ticks.

Strategy:
- HYDROGEL: StableTrader (FV=10000, MAKE inside the 16-wide market spread)
- VELVETFRUIT: no independent MM; full 200-unit capacity reserved for OptionsTrader hedging
- Vouchers: OptionsTrader (smile-fit IV → trade vs theoretical, delta-hedge on underlying)

Skip flags:
- VEV_6000, VEV_6500: pinned 0.5 mid (book is 0×16/1×16). No tradeable edge.
"""
from datamodel import Order, OrderDepth, TradingState
from strategies._base import StateManager
from strategies.hyd_mm import HydrogelMMTrader
from strategies.stable import StableTrader
from strategies.drifting import DriftingTrader
from strategies.stable_anchored import AnchoredStableTrader
from strategies.stable_skew import StableSkewTrader
from strategies.take_only import TakeOnlyTrader
from strategies.counterparty_copy import CounterpartyCopyTrader
from strategies.voucher_zscore import VoucherZScoreTrader

HYDROGEL_LIMIT = 200
VELVETFRUIT_LIMIT = 200
VOUCHER_LIMIT = 300
VOUCHER_POSITION_MAX = 300

# ─── Variant knobs (string-substituted by scripts/build_r3.py) ───
# Defaults reflect best-known backtest variant (v16_h10_v2 = 111,260).
ENABLE_VOUCHER_ZSCORE = True
ENABLE_VOUCHERS = False
ENABLE_VELVETFRUIT_MM = True
ENABLE_COUNTERPARTY_COPY = True
HYDROGEL_FV = 10_000
HYDROGEL_MAKE_SPREAD = 10
HYDROGEL_MAKE_BID_SPREAD = -1   # -1 = use make_spread (symmetric)
HYDROGEL_MAKE_ASK_SPREAD = -1
VELVETFRUIT_FV = 5250
VELVETFRUIT_MAKE_SPREAD = 2
VELVETFRUIT_MAKE_BID_SPREAD = -1
VELVETFRUIT_MAKE_ASK_SPREAD = -1
VELVETFRUIT_TAKE_EDGE = 1.0
VOUCHER_OPEN_THR = 0.5
VOUCHER_TRADE_QTY = 10
VOUCHER_REFIT_EVERY = 200
ENABLE_VEV4000_MM = False
VEV4000_MAKE_SPREAD = 5
VEV4000_TAKE_EDGE = 2.0
VEV4000_STABLE_MODE = False   # True = use StableTrader with VEV4000_FV
VEV4000_FV = 1267   # static FV for VEV_4000 when VEV4000_STABLE_MODE=True
ENABLE_VEV4500_MM = False
VEV4500_MAKE_SPREAD = 7
VEV4500_TAKE_EDGE = 2.0
VELVETFRUIT_MODE = "stable"   # "stable" | "drifting" | "anchored" | "take_only"
VELVETFRUIT_ANCHOR_WEIGHT = 0.7
VELVETFRUIT_EMA_HALFLIFE = 2000
HYDROGEL_MODE = "stable"   # "stable" | "anchored" | "skew" | "follow" | "take_only"
HYDROGEL_SKEW_SENS = 2.0
HYDROGEL_CLEAR_THRESH = 0.5
HYDROGEL_TAKE_EDGE = 1.0
HYDROGEL_ANCHOR_WEIGHT = 0.7
HYDROGEL_EMA_HALFLIFE = 2000
ENABLE_VOUCHER_TAKE = False
VOUCHER_TAKE_EDGE = 2.0
VOUCHER_TAKE_QTY = 10
ENABLE_VOUCHER_Z_IV = False
ENABLE_VOUCHER_MAKE_MM = False
VOUCHER_ZSCORE_WINDOW = 100
VOUCHER_ZSCORE_THR_OPEN = 0.7
VOUCHER_ZSCORE_THR_CLOSE = 0.0
VOUCHER_ZSCORE_VEGA_FLOOR = 0.5
VOUCHER_MIN_TICKS_FOR_ZSCORE = 30
VMM_STRIKES = [5300, 5400, 5500]
VMM_HALF_SPREAD = 1
VMM_MAX_POS_PER_STRIKE = 50
VMM_SKEW_SENS = 0.0
ZIV_OPEN_THR = 1.5
ZIV_CLOSE_THR = 0.5
ZIV_EMA_ALPHA = 0.05
ZIV_STD_WINDOW = 100
ZIV_VEGA_TARGET = 20.0
ZIV_MAX_PER_STRIKE = 25
ZIV_MIN_R2 = 0.85
ZIV_HEDGE_EVERY = 50
ZIV_HEDGE_THRESHOLD = 30
ZIV_HEDGE_FRACTION = 0.5
COPY_BUYERS = {"Mark 67"}
COPY_SELLERS = {"Mark 49"}
COPY_PRODUCTS = {"VELVETFRUIT_EXTRACT"}
COPY_SIZE = 50
COPY_POSITION_MAX = 200
COPY_HOLD_TICKS = 1000
COPY_TAKE_ONLY = True

# Per-strike calibrated IV (single-line so build_r3.py can patch this knob).
STRIKE_IV = {5100: 0.2272, 5200: 0.2334, 5300: 0.2358, 5400: 0.2212, 5500: 0.2400}

# Active vouchers (skip pinned 6000/6500 by default - too far OTM, no tradeable edge)
VOUCHERS = {
    4000: "VEV_4000", 4500: "VEV_4500", 5000: "VEV_5000",
    5100: "VEV_5100", 5200: "VEV_5200", 5300: "VEV_5300",
    5400: "VEV_5400", 5500: "VEV_5500",
}

# TTE schedule: round 3 final-sim starts at 5 days, ends at 4 days.
# tte_years(ts) = (5 - ts/10000) / 365   (linear theta decay across the round)
TTE_START_DAYS = 5.0


def tte_years(timestamp: int) -> float:
    """Linear theta decay across the round."""
    days_remaining = max(TTE_START_DAYS - timestamp / 10_000.0, 1 / 24.0)
    return days_remaining / 365.0


class Trader:

    def __init__(self) -> None:
        # HYDROGEL: stable MM. Mean spread ~16. FV centered at 10000 (round number).
        if HYDROGEL_MODE == "take_only":
            self.hydrogel = TakeOnlyTrader(
                product="HYDROGEL_PACK",
                fv=HYDROGEL_FV,
                limit=HYDROGEL_LIMIT,
                take_edge=0.0,
            )
            self.hydrogel_drifting = False
            self.hydrogel_anchored = False
        elif HYDROGEL_MODE == "follow":
            self.hydrogel = DriftingTrader(
                product="HYDROGEL_PACK",
                limit=HYDROGEL_LIMIT,
                make_spread=HYDROGEL_MAKE_SPREAD,
                take_edge=1.0,
                fv_mode="follow",
                skew_sens=2.0,
                soft_limit_ratio=0.5,
            )
            self.hydrogel_drifting = True
            self.hydrogel_anchored = False
            self.hydrogel_drifting = False
        elif HYDROGEL_MODE == "skew":
            self.hydrogel = StableSkewTrader(
                product="HYDROGEL_PACK",
                fv=HYDROGEL_FV,
                limit=HYDROGEL_LIMIT,
                make_spread=HYDROGEL_MAKE_SPREAD,
                take_edge=HYDROGEL_TAKE_EDGE,
                skew_sens=HYDROGEL_SKEW_SENS,
                clear_thresh=HYDROGEL_CLEAR_THRESH,
            )
            self.hydrogel_anchored = False
            self.hydrogel_drifting = False
        elif HYDROGEL_MODE == "anchored":
            self.hydrogel = AnchoredStableTrader(
                product="HYDROGEL_PACK",
                default_fv=HYDROGEL_FV,
                limit=HYDROGEL_LIMIT,
                make_spread=HYDROGEL_MAKE_SPREAD,
                ema_halflife=HYDROGEL_EMA_HALFLIFE,
                anchor_weight=HYDROGEL_ANCHOR_WEIGHT,
            )
            self.hydrogel_anchored = True
            self.hydrogel_drifting = False
        else:
            self.hydrogel = HydrogelMMTrader(product="HYDROGEL_PACK")
            self.hydrogel_anchored = False
            self.hydrogel_drifting = False

        # VELVETFRUIT_EXTRACT: optional MM (disabled when vouchers enabled - hedge bucket conflict)
        if ENABLE_VELVETFRUIT_MM:
            if VELVETFRUIT_MODE == "take_only":
                self.velvet = TakeOnlyTrader(
                    product="VELVETFRUIT_EXTRACT",
                    fv=VELVETFRUIT_FV,
                    limit=VELVETFRUIT_LIMIT,
                    take_edge=0.0,
                )
                self.velvet_anchored = False
                self.velvet_drifting = False
                self.velvet_take_only = True
            elif VELVETFRUIT_MODE == "anchored":
                self.velvet = AnchoredStableTrader(
                    product="VELVETFRUIT_EXTRACT",
                    default_fv=VELVETFRUIT_FV,
                    limit=VELVETFRUIT_LIMIT,
                    make_spread=VELVETFRUIT_MAKE_SPREAD,
                    ema_halflife=VELVETFRUIT_EMA_HALFLIFE,
                    anchor_weight=VELVETFRUIT_ANCHOR_WEIGHT,
                )
                self.velvet_anchored = True
                self.velvet_drifting = False
                self.velvet_take_only = False
            elif VELVETFRUIT_MODE == "ladder":
                self.velvet = StableLadderTrader(
                    product="VELVETFRUIT_EXTRACT",
                    fv=VELVETFRUIT_FV,
                    limit=VELVETFRUIT_LIMIT,
                    bid_levels=VELVETFRUIT_LADDER_BIDS,
                    ask_levels=VELVETFRUIT_LADDER_ASKS,
                )
                self.velvet_anchored = False
                self.velvet_drifting = False
                self.velvet_take_only = False
            elif VELVETFRUIT_MODE == "regime_adaptive":
                self.velvet = RegimeAdaptiveStable(
                    product="VELVETFRUIT_EXTRACT",
                    peak_fv=VELVETFRUIT_PEAK_FV,
                    peak_mean=VELVETFRUIT_PEAK_MEAN,
                    limit=VELVETFRUIT_LIMIT,
                    make_spread=VELVETFRUIT_MAKE_SPREAD,
                    ema_halflife=VELVETFRUIT_RA_HALFLIFE,
                    adapt_strength=VELVETFRUIT_ADAPT_STRENGTH,
                    max_shift=VELVETFRUIT_MAX_SHIFT,
                    warmup_ticks=VELVETFRUIT_WARMUP_TICKS,
                )
                self.velvet_anchored = True   # td-passing dispatch
                self.velvet_drifting = False
                self.velvet_take_only = False
            elif VELVETFRUIT_MODE == "drifting":
                self.velvet = DriftingTrader(
                    product="VELVETFRUIT_EXTRACT",
                    limit=VELVETFRUIT_LIMIT,
                    make_spread=VELVETFRUIT_MAKE_SPREAD,
                    take_edge=VELVETFRUIT_TAKE_EDGE,
                    fv_mode="fv",
                    skew_sens=2.0,
                    adverse_vol=20,
                    clear_thresh=0.6,
                )
                self.velvet_drifting = True
                self.velvet_anchored = False
                self.velvet_take_only = False
            else:
                self.velvet = StableTrader(
                    product="VELVETFRUIT_EXTRACT",
                    fv=VELVETFRUIT_FV,
                    limit=VELVETFRUIT_LIMIT,
                    make_spread=VELVETFRUIT_MAKE_SPREAD,
                    make_bid_spread=None if VELVETFRUIT_MAKE_BID_SPREAD < 0 else VELVETFRUIT_MAKE_BID_SPREAD,
                    make_ask_spread=None if VELVETFRUIT_MAKE_ASK_SPREAD < 0 else VELVETFRUIT_MAKE_ASK_SPREAD,
                )
                self.velvet_drifting = False
                self.velvet_anchored = False
                self.velvet_take_only = False
        else:
            self.velvet = None
            self.velvet_drifting = False
            self.velvet_anchored = False
            self.velvet_take_only = False

        # VEV_4000 - deep ITM call, delta ~= 1, tracks underlying 1:1.
        # Its own wall_mid tracks underlying - 4000 implicitly.
        if ENABLE_VEV4000_MM:
            if VEV4000_STABLE_MODE:
                self.vev4000 = StableTrader(
                    product="VEV_4000",
                    fv=VEV4000_FV,
                    limit=VOUCHER_LIMIT,
                    make_spread=VEV4000_MAKE_SPREAD,
                )
                self._vev4000_stable = True
            else:
                self.vev4000 = DriftingTrader(
                    product="VEV_4000",
                    limit=VOUCHER_LIMIT,
                    make_spread=VEV4000_MAKE_SPREAD,
                    take_edge=VEV4000_TAKE_EDGE,
                    fv_mode="fv",
                    skew_sens=2.0,
                    adverse_vol=20,
                    clear_thresh=0.6,
                )
                self._vev4000_stable = False
        else:
            self.vev4000 = None
            self._vev4000_stable = False

        # VEV_4500 - ITM call, delta ~0.99, deep ITM behaviour for spot ~5255
        if ENABLE_VEV4500_MM:
            self.vev4500 = DriftingTrader(
                product="VEV_4500",
                limit=VOUCHER_LIMIT,
                make_spread=VEV4500_MAKE_SPREAD,
                take_edge=VEV4500_TAKE_EDGE,
                fv_mode="fv",
                skew_sens=2.0,
                adverse_vol=20,
                clear_thresh=0.6,
            )
        else:
            self.vev4500 = None

        # VoucherTakeTrader: fixed-IV TAKE-only on active strikes
        if ENABLE_VOUCHER_TAKE:
            self.voucher_take = VoucherTakeTrader(
                underlying="VELVETFRUIT_EXTRACT",
                strike_iv=STRIKE_IV,
                option_products={
                    k: f"VEV_{k}" for k in STRIKE_IV.keys()
                },
                option_limit=VOUCHER_LIMIT,
                take_edge=VOUCHER_TAKE_EDGE,
                trade_qty=VOUCHER_TAKE_QTY,
                tte_func=tte_years,
            )
        else:
            self.voucher_take = None

        # VoucherMakeMM: passive MAKE quotes on liquid voucher strikes (Q1e)
        if ENABLE_VOUCHER_MAKE_MM:
            self.voucher_mm = VoucherMakeMM(
                strikes=VMM_STRIKES,
                option_products={s: f"VEV_{s}" for s in VMM_STRIKES},
                option_limit=VOUCHER_LIMIT,
                half_spread=VMM_HALF_SPREAD,
                max_pos_per_strike=VMM_MAX_POS_PER_STRIKE,
                skew_sens=VMM_SKEW_SENS,
            )
        else:
            self.voucher_mm = None

        # VoucherZIVTrader: z-score IV mean reversion (E_002 Signal A)
        if ENABLE_VOUCHER_Z_IV:
            self.voucher_z = VoucherZIVTrader(
                underlying="VELVETFRUIT_EXTRACT",
                option_products=VOUCHERS,
                option_limit=VOUCHER_LIMIT,
                underlying_limit=VELVETFRUIT_LIMIT,
                open_thr=ZIV_OPEN_THR,
                close_thr=ZIV_CLOSE_THR,
                ema_alpha=ZIV_EMA_ALPHA,
                std_window=ZIV_STD_WINDOW,
                vega_target=ZIV_VEGA_TARGET,
                max_pos_per_strike=ZIV_MAX_PER_STRIKE,
                min_r2=ZIV_MIN_R2,
                hedge_every=ZIV_HEDGE_EVERY,
                hedge_threshold=ZIV_HEDGE_THRESHOLD,
                hedge_fraction=ZIV_HEDGE_FRACTION,
                tte_func=tte_years,
            )
        else:
            self.voucher_z = None

        if ENABLE_VOUCHER_ZSCORE:
            self.voucher_zscore_trader = VoucherZScoreTrader()
            self.voucher_zscore_trader.window = VOUCHER_ZSCORE_WINDOW
            self.voucher_zscore_trader.open_thr = VOUCHER_ZSCORE_THR_OPEN
            self.voucher_zscore_trader.close_thr = VOUCHER_ZSCORE_THR_CLOSE
            self.voucher_zscore_trader.vega_floor = VOUCHER_ZSCORE_VEGA_FLOOR
            self.voucher_zscore_trader.position_max = VOUCHER_POSITION_MAX
            self.voucher_zscore_trader.min_ticks_for_zscore = VOUCHER_MIN_TICKS_FOR_ZSCORE
        else:
            self.voucher_zscore_trader = None

        # OptionsTrader: handles all vouchers + uses VELVETFRUIT for delta hedge.
        if ENABLE_VOUCHERS:
            self.options = OptionsTrader(
                underlying="VELVETFRUIT_EXTRACT",
                option_products=VOUCHERS,
                underlying_limit=VELVETFRUIT_LIMIT,
                option_limit=VOUCHER_LIMIT,
                tte=tte_years(0),
                open_thr=VOUCHER_OPEN_THR,
                close_thr=0.0,
                refit_every=VOUCHER_REFIT_EVERY,
                min_vega=0.5,
                trade_qty=VOUCHER_TRADE_QTY,
            )
        else:
            self.options = None

        if ENABLE_COUNTERPARTY_COPY:
            self.counterparty_copy = CounterpartyCopyTrader(
                buyers=COPY_BUYERS,
                sellers=COPY_SELLERS,
                products=COPY_PRODUCTS,
                size=COPY_SIZE,
                position_max=COPY_POSITION_MAX,
                hold_ticks=COPY_HOLD_TICKS,
                take_only=COPY_TAKE_ONLY,
            )
        else:
            self.counterparty_copy = None

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        sm = StateManager(state.traderData)

        # HYDROGEL_PACK - stable MM
        if "HYDROGEL_PACK" in state.order_depths:
            try:
                pos = state.position.get("HYDROGEL_PACK", 0)
                if self.hydrogel_drifting:
                    td_h = sm.get("hyd", {})
                    trades_h = state.market_trades.get("HYDROGEL_PACK", [])
                    orders_h, td_h = self.hydrogel.run(
                        state.order_depths["HYDROGEL_PACK"], pos, td_h,
                        trades_h, state.timestamp,
                    )
                    result["HYDROGEL_PACK"] = orders_h
                    sm.set("hyd", td_h)
                elif self.hydrogel_anchored:
                    td_h = sm.get("hyd", {})
                    orders_h, td_h = self.hydrogel.run(
                        state.order_depths["HYDROGEL_PACK"], pos, td_h
                    )
                    result["HYDROGEL_PACK"] = orders_h
                    sm.set("hyd", td_h)
                else:
                    result["HYDROGEL_PACK"] = self.hydrogel.run(
                        state.order_depths["HYDROGEL_PACK"], pos
                    )
            except Exception:
                result["HYDROGEL_PACK"] = []

        # VELVETFRUIT_EXTRACT - optional standalone MM (skips when vouchers enabled)
        if self.velvet is not None and "VELVETFRUIT_EXTRACT" in state.order_depths:
            try:
                pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
                if self.velvet_take_only:
                    result["VELVETFRUIT_EXTRACT"] = self.velvet.run(
                        state.order_depths["VELVETFRUIT_EXTRACT"], pos
                    )
                elif self.velvet_anchored:
                    td_va = sm.get("vela", {})
                    orders_va, td_va = self.velvet.run(
                        state.order_depths["VELVETFRUIT_EXTRACT"], pos, td_va
                    )
                    result["VELVETFRUIT_EXTRACT"] = orders_va
                    sm.set("vela", td_va)
                elif self.velvet_drifting:
                    td_v = sm.get("vel", {})
                    trades_v = state.market_trades.get("VELVETFRUIT_EXTRACT", [])
                    orders_v, td_v = self.velvet.run(
                        state.order_depths["VELVETFRUIT_EXTRACT"], pos, td_v,
                        trades_v, state.timestamp,
                    )
                    result["VELVETFRUIT_EXTRACT"] = orders_v
                    sm.set("vel", td_v)
                else:
                    result["VELVETFRUIT_EXTRACT"] = self.velvet.run(
                        state.order_depths["VELVETFRUIT_EXTRACT"], pos
                    )
            except Exception:
                result["VELVETFRUIT_EXTRACT"] = []

        if self.voucher_zscore_trader is not None:
            try:
                self.voucher_zscore_trader.load_state(sm.get("vzs", {}))
                for product, orders in self.voucher_zscore_trader.run(state):
                    if product in result:
                        result[product].extend(orders)
                    else:
                        result[product] = orders
                sm.set("vzs", self.voucher_zscore_trader.dump_state())
            except Exception:
                pass

        # VEV_4000 - delta-1 MM (StableTrader for static-FV mode, DriftingTrader otherwise)
        if not ENABLE_VOUCHER_ZSCORE and self.vev4000 is not None and "VEV_4000" in state.order_depths:
            try:
                pos = state.position.get("VEV_4000", 0)
                if self._vev4000_stable:
                    result["VEV_4000"] = self.vev4000.run(
                        state.order_depths["VEV_4000"], pos
                    )
                else:
                    td_v4 = sm.get("v4", {})
                    trades_v4 = state.market_trades.get("VEV_4000", [])
                    orders_v4, td_v4 = self.vev4000.run(
                        state.order_depths["VEV_4000"], pos, td_v4,
                        trades_v4, state.timestamp,
                    )
                    result["VEV_4000"] = orders_v4
                    sm.set("v4", td_v4)
            except Exception:
                result["VEV_4000"] = []

        # VEV_4500 - ITM MM via DriftingTrader
        if not ENABLE_VOUCHER_ZSCORE and self.vev4500 is not None and "VEV_4500" in state.order_depths:
            try:
                pos = state.position.get("VEV_4500", 0)
                td_v45 = sm.get("v45", {})
                trades_v45 = state.market_trades.get("VEV_4500", [])
                orders_v45, td_v45 = self.vev4500.run(
                    state.order_depths["VEV_4500"], pos, td_v45,
                    trades_v45, state.timestamp,
                )
                result["VEV_4500"] = orders_v45
                sm.set("v45", td_v45)
            except Exception:
                result["VEV_4500"] = []

        # Voucher fixed-IV TAKE-only
        if not ENABLE_VOUCHER_ZSCORE and self.voucher_take is not None:
            try:
                vt_orders = self.voucher_take.run(
                    state.order_depths, state.position, state.timestamp
                )
                for product, orders in vt_orders.items():
                    if product in result:
                        result[product].extend(orders)
                    else:
                        result[product] = orders
            except Exception:
                pass

        # VoucherMakeMM run
        if not ENABLE_VOUCHER_ZSCORE and self.voucher_mm is not None:
            try:
                vmm_orders = self.voucher_mm.run(state.order_depths, state.position)
                for product, orders in vmm_orders.items():
                    if product in result:
                        result[product].extend(orders)
                    else:
                        result[product] = orders
            except Exception:
                pass

        # VoucherZIVTrader run
        if not ENABLE_VOUCHER_ZSCORE and self.voucher_z is not None:
            try:
                z_td = sm.get("ziv", {})
                z_orders, z_td = self.voucher_z.run(
                    state.order_depths, state.position, z_td, state.timestamp
                )
                sm.set("ziv", z_td)
                for product, orders in z_orders.items():
                    if product in result:
                        result[product].extend(orders)
                    else:
                        result[product] = orders
            except Exception:
                pass

        # Vouchers + underlying hedge - OptionsTrader
        if not ENABLE_VOUCHER_ZSCORE and ENABLE_VOUCHERS and self.options is not None:
            try:
                opt_td = sm.get("opt", {})
                opt_td["tte"] = tte_years(state.timestamp)
                opt_orders, opt_td = self.options.run(
                    state.order_depths, state.position, opt_td, state.timestamp
                )
                sm.set("opt", opt_td)
                for product, orders in opt_orders.items():
                    if product in result:
                        result[product].extend(orders)
                    else:
                        result[product] = orders
            except Exception:
                pass

        if (
            self.counterparty_copy is not None
            and "VELVETFRUIT_EXTRACT" in state.order_depths
        ):
            try:
                copy_td = sm.get("cpy", {})
                copy_orders, copy_td = self.counterparty_copy.generate_orders(
                    state, copy_td, result
                )
                sm.set("cpy", copy_td)
                for product, orders in copy_orders.items():
                    if product in result:
                        result[product].extend(orders)
                    else:
                        result[product] = orders
            except Exception:
                pass

        return result, 0, sm.dump()
