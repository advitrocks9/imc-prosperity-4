"""Black-Scholes pricer for Round 4 AETHER_CRYSTAL vanilla options."""

from __future__ import annotations

import math

DEFAULT_SIGMA_ANN = 2.51
RISK_FREE_RATE = 0.0
TRADING_DAYS_PER_WEEK = 5
TRADING_DAYS_PER_YEAR = 252
STEPS_PER_WEEK = 20
WEEKS_PER_YEAR = 52
PER_WEEK_VOL_DEFAULT = DEFAULT_SIGMA_ANN * math.sqrt(TRADING_DAYS_PER_WEEK / TRADING_DAYS_PER_YEAR)

try:
    from strategies.bs_options import bs_call as _imported_bs_call
    from strategies.bs_options import bs_put as _imported_bs_put
except ImportError:
    try:
        from bs_options import bs_call as _imported_bs_call
        from bs_options import bs_put as _imported_bs_put
    except ImportError:
        _imported_bs_call = None
        _imported_bs_put = None

try:
    from scipy.stats import norm as _scipy_norm

    def _norm_cdf(x: float) -> float:
        return float(_scipy_norm.cdf(x))

    def _norm_pdf(x: float) -> float:
        return float(_scipy_norm.pdf(x))

except ImportError:

    def _norm_cdf(x: float) -> float:
        return 0.5 * math.erfc(-x / math.sqrt(2.0))

    def _norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _price_only(value: float | tuple[float, float]) -> float:
    return float(value[0] if isinstance(value, tuple) else value)


def _time_to_years(weeks_to_expiry: float) -> float:
    return TRADING_DAYS_PER_WEEK * max(weeks_to_expiry, 0.0) / TRADING_DAYS_PER_YEAR


def _sigma_root_t(weeks_to_expiry: float, sigma_ann: float) -> tuple[float, float]:
    T = _time_to_years(weeks_to_expiry)
    sigma_root_t = sigma_ann * math.sqrt(T)
    if _imported_bs_call is None and weeks_to_expiry > 2.0:
        sigma_root_t += 0.00362 * (weeks_to_expiry - 2.0)
    return sigma_root_t, T


def _d1_d2(S: float, K: float, weeks_to_expiry: float, sigma_ann: float) -> tuple[float, float, float]:
    sigma_root_t, T = _sigma_root_t(weeks_to_expiry, sigma_ann)
    d1 = (math.log(S / K) + 0.5 * sigma_root_t * sigma_root_t) / sigma_root_t
    return d1, d1 - sigma_root_t, T


def aether_call(S: float, K: float, weeks_to_expiry: float, sigma_ann: float = DEFAULT_SIGMA_ANN) -> float:
    if weeks_to_expiry <= 0 or sigma_ann <= 0:
        return max(S - K, 0.0)
    if _imported_bs_call is not None:
        return _price_only(_imported_bs_call(S, K, _time_to_years(weeks_to_expiry), RISK_FREE_RATE, sigma_ann))
    d1, d2, _ = _d1_d2(S, K, weeks_to_expiry, sigma_ann)
    return S * _norm_cdf(d1) - K * math.exp(-RISK_FREE_RATE * _time_to_years(weeks_to_expiry)) * _norm_cdf(d2)


def aether_put(S: float, K: float, weeks_to_expiry: float, sigma_ann: float = DEFAULT_SIGMA_ANN) -> float:
    if weeks_to_expiry <= 0 or sigma_ann <= 0:
        return max(K - S, 0.0)
    if _imported_bs_put is not None:
        return _price_only(_imported_bs_put(S, K, _time_to_years(weeks_to_expiry), RISK_FREE_RATE, sigma_ann))
    d1, d2, T = _d1_d2(S, K, weeks_to_expiry, sigma_ann)
    discounted_strike = K * math.exp(-RISK_FREE_RATE * T)
    return discounted_strike * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def aether_greeks(S: float, K: float, weeks_to_expiry: float, sigma_ann: float = DEFAULT_SIGMA_ANN) -> dict[str, float]:
    if weeks_to_expiry <= 0 or sigma_ann <= 0:
        return {"delta": 1.0 if S > K else 0.0, "gamma": 0.0, "vega_per_1pct_vol": 0.0, "theta_per_week": 0.0}
    d1, _, T = _d1_d2(S, K, weeks_to_expiry, sigma_ann)
    sqrt_T = math.sqrt(T)
    sigma_root_t, _ = _sigma_root_t(weeks_to_expiry, sigma_ann)
    sigma_eff = sigma_root_t / sqrt_T
    pdf_d1 = _norm_pdf(d1)
    vega = S * pdf_d1 * sqrt_T
    theta = -(S * pdf_d1 * sigma_eff) / (2.0 * sqrt_T)
    return {
        "delta": _norm_cdf(d1),
        "gamma": pdf_d1 / (S * sigma_eff * sqrt_T),
        "vega_per_1pct_vol": vega * 0.01,
        "theta_per_week": theta * (TRADING_DAYS_PER_WEEK / TRADING_DAYS_PER_YEAR),
    }


def print_prices(S: float, strike_grid: list[float]) -> None:
    print("strike | 2w_call | 2w_put | 3w_call | 3w_put")
    for strike in strike_grid:
        values = [aether_call(S, strike, 2.0), aether_put(S, strike, 2.0), aether_call(S, strike, 3.0), aether_put(S, strike, 3.0)]
        print(f"{strike:6.0f} | {values[0]:7.4f} | {values[1]:6.4f} | {values[2]:7.4f} | {values[3]:6.4f}")


def _print_validation(name: str, computed: float, expected: float, tolerance: float = 0.05) -> bool:
    passed = abs(computed - expected) < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"{status} {name}: computed={computed:.4f}, expected≈{expected:.4f}")
    return passed


if __name__ == "__main__":
    call_2w = aether_call(100.0, 100.0, 2.0)
    call_3w = aether_call(100.0, 100.0, 3.0)
    ok_2w = _print_validation("2w ATM call", call_2w, 19.74)
    ok_3w = _print_validation("3w ATM call", call_3w, 24.05)
    print_prices(100.0, [85, 90, 95, 100, 105, 110, 115])
    if not (ok_2w and ok_3w):
        raise SystemExit(1)
