import math
import sys
from typing import Literal

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


def cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    root_t = math.sqrt(T)
    vol_t = sigma * root_t
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_t
    return d1, d1 - vol_t


def _zero_vol_call(S: float, K: float, T: float, r: float) -> float:
    return max(S - K * math.exp(-r * T), 0.0)


def _norm_type(option_type: str) -> Literal["call", "put"]:
    opt = option_type.lower()
    if opt not in {"call", "put"}:
        raise ValueError(f"unsupported option_type={option_type!r}")
    return opt


def _warn_nan(message: str) -> float:
    print(f"implied_vol warning: {message}", file=sys.stderr)
    return float("nan")


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return _zero_vol_call(S, K, T, r)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * cdf(d1) - K * math.exp(-r * T) * cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    return bs_call(S, K, T, r, sigma) - S + K * math.exp(-r * T)


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call",
) -> dict[str, float]:
    opt = _norm_type(option_type)
    if T <= 0 or sigma <= 0:
        delta = 1.0 if (opt == "call" and S > K) else 0.0
        delta -= 1.0 if opt == "put" and S < K else 0.0
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc, root_t, nd1 = math.exp(-r * T), math.sqrt(T), pdf(d1)
    delta = cdf(d1) if opt == "call" else cdf(d1) - 1.0
    gamma = nd1 / (S * sigma * root_t)
    vega = S * nd1 * root_t
    carry = -(S * nd1 * sigma) / (2.0 * root_t)
    theta = carry - r * K * disc * cdf(d2) if opt == "call" else carry + r * K * disc * cdf(-d2)
    rho = K * T * disc * cdf(d2) if opt == "call" else -K * T * disc * cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(
    price: float, S: float, K: float, T: float, r: float, option_type: str = "call",
    tol: float = 1e-6, max_iter: int = 100,
) -> float:
    opt = _norm_type(option_type)
    if T <= 0 or S <= 0 or K <= 0:
        return _warn_nan("invalid inputs or expired option")
    disc, pricer = math.exp(-r * T), (bs_call if opt == "call" else bs_put)
    lower = max(0.0, S - K * disc) if opt == "call" else max(0.0, K * disc - S)
    upper = S if opt == "call" else K * disc
    if not lower - tol <= price <= upper + tol:
        return _warn_nan(f"price {price} outside no-arbitrage bounds [{lower}, {upper}]")
    sigma = max(1e-4, min(math.sqrt(2.0 * math.pi / T) * price / S, 5.0))
    for _ in range(max_iter):
        model = pricer(S, K, T, r, sigma)
        vega = bs_greeks(S, K, T, r, sigma, opt)["vega"]
        if abs(model - price) < tol:
            return sigma
        if vega <= 1e-10:
            break
        sigma = max(1e-8, min(sigma - (model - price) / vega, 5.0))
    lo, hi = 1e-8, 5.0
    if (pricer(S, K, T, r, lo) - price) * (pricer(S, K, T, r, hi) - price) > 0:
        return _warn_nan("bisection bracket not found")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        diff = pricer(S, K, T, r, mid) - price
        if abs(diff) < tol or hi - lo < tol:
            return mid
        lo, hi = (lo, mid) if diff > 0 else (mid, hi)
    return _warn_nan("solver did not converge")


def _approx(value: float, target: float, tol: float = 1e-4) -> bool:
    return abs(value - target) <= tol


if __name__ == "__main__":
    call = bs_call(100, 100, 1, 0.05, 0.2)
    put = bs_put(100, 100, 1, 0.05, 0.2)
    parity = call - put - 100 + 100 * math.exp(-0.05)
    deep_itm = bs_call(150, 100, 1, 0.05, 0.2)
    iv = implied_vol(call, 100, 100, 1, 0.05)
    delta = bs_greeks(100, 100, 1, 0.05, 0.2)["delta"]
    impossible = implied_vol(150.0, 100, 100, 1, 0.05)
    checks = [
        ("bs_call", _approx(call, 10.4506, 1e-4)),
        ("bs_put", _approx(put, 5.5735, 1e-4)),
        ("put_call_parity", abs(parity) < 1e-8),
        ("T0_boundary", bs_call(90, 100, 0, 0.05, 0.2) == 0.0),
        ("deep_itm_call", _approx(deep_itm, 54.88, 0.2)),
        ("iv_round_trip", _approx(iv, 0.2, 1e-6)),
        ("delta", _approx(delta, 0.6368, 1e-4)),
        ("impossible_price", math.isnan(impossible)),
    ]
    for name, ok in checks:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
