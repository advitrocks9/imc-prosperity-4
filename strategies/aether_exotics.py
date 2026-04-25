"""Closed-form + Monte Carlo pricers for AETHER_CRYSTAL exotics."""

from __future__ import annotations

import math
from typing import Final

import numpy as np


SQRT_2: Final[float] = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / SQRT_2)


def bs_d1_d2(spot: float, strike: float, time: float, sigma: float) -> tuple[float, float]:
    vol = sigma * math.sqrt(time)
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * time) / vol
    return d1, d1 - vol


def bs_call(spot: float, strike: float, time: float, sigma: float) -> float:
    if time <= 0.0:
        return max(spot - strike, 0.0)
    d1, d2 = bs_d1_d2(spot, strike, time, sigma)
    return spot * norm_cdf(d1) - strike * norm_cdf(d2)


def bs_put(spot: float, strike: float, time: float, sigma: float) -> float:
    if time <= 0.0:
        return max(strike - spot, 0.0)
    d1, d2 = bs_d1_d2(spot, strike, time, sigma)
    return strike * norm_cdf(-d2) - spot * norm_cdf(-d1)


def chooser_price(spot: float, strike: float, t_full: float, t_choice: float, sigma: float) -> float:
    return bs_call(spot, strike, t_full, sigma) + bs_put(spot, strike, t_choice, sigma)


def binary_put_price(spot: float, strike: float, time: float, sigma: float, payout: float) -> float:
    if time <= 0.0:
        return payout if spot < strike else 0.0
    _, d2 = bs_d1_d2(spot, strike, time, sigma)
    return payout * norm_cdf(-d2)


def _mc_stats(payoff: np.ndarray) -> tuple[float, float]:
    mean = float(payoff.mean())
    stderr = float(payoff.std(ddof=1) / math.sqrt(payoff.size))
    return mean, stderr


def _weekly_steps(time: float, steps_per_week: int = 20) -> int:
    return max(1, int(round(time * 52.0 * steps_per_week)))


def _simulate_terminal_spots(
    spot: float, time: float, sigma: float, n_paths: int, n_steps: int
) -> np.ndarray:
    if time <= 0.0:
        return np.full(n_paths, spot, dtype=float)
    dt = time / n_steps
    shocks = np.random.normal(size=(n_paths, n_steps))
    log_spot = math.log(spot) + np.cumsum((-0.5 * sigma * sigma * dt) + sigma * math.sqrt(dt) * shocks, axis=1)
    return np.exp(log_spot[:, -1])


def chooser_mc(
    spot: float, strike: float, t_full: float, t_choice: float, sigma: float, n_paths: int
) -> tuple[float, float]:
    rem_time = t_full - t_choice
    spots_t1 = _simulate_terminal_spots(spot, t_choice, sigma, n_paths, _weekly_steps(t_choice))
    call_vals = np.array([bs_call(s_t1, strike, rem_time, sigma) for s_t1 in spots_t1], dtype=float)
    put_vals = np.array([bs_put(s_t1, strike, rem_time, sigma) for s_t1 in spots_t1], dtype=float)
    return _mc_stats(np.maximum(call_vals, put_vals))


def binary_put_mc(
    spot: float, strike: float, time: float, sigma: float, payout: float, n_paths: int
) -> tuple[float, float]:
    terminal = _simulate_terminal_spots(spot, time, sigma, n_paths, _weekly_steps(time))
    return _mc_stats(np.where(terminal < strike, payout, 0.0))


def _barrier_terms(
    spot: float, strike: float, barrier: float, time: float, sigma: float
) -> tuple[float, float, float, float]:
    vol = sigma * math.sqrt(time)
    mu = -0.5
    mu_sigma = (1.0 + mu) * vol
    hs = barrier / spot
    hs_mu = hs ** (2.0 * mu)
    hs_mu2 = hs_mu * hs * hs
    x1 = math.log(spot / strike) / vol + mu_sigma
    x2 = math.log(spot / barrier) / vol + mu_sigma
    y1 = math.log(barrier * barrier / (spot * strike)) / vol + mu_sigma
    y2 = math.log(barrier / spot) / vol + mu_sigma
    a = strike * norm_cdf(-(x1 - vol)) - spot * norm_cdf(-x1)
    b = strike * norm_cdf(-(x2 - vol)) - spot * norm_cdf(-x2)
    c = hs_mu * strike * norm_cdf(y1 - vol) - hs_mu2 * spot * norm_cdf(y1)
    d = hs_mu * strike * norm_cdf(y2 - vol) - hs_mu2 * spot * norm_cdf(y2)
    return a, b, c, d


def down_and_out_put_price(
    spot: float, strike: float, barrier: float, time: float, sigma: float
) -> float:
    if spot <= barrier:
        return 0.0
    a, b, c, d = _barrier_terms(spot, strike, barrier, time, sigma)
    # Reiner & Rubinstein (1991); Haug (2007), Table 4-18.
    return max(0.0, min(bs_put(spot, strike, time, sigma), a - b + c - d))


def down_and_out_put_mc(
    spot: float, strike: float, barrier: float, time: float, sigma: float, n_paths: int
) -> tuple[float, float]:
    n_steps = max(1, int(round(252.0 * time)))
    dt = time / n_steps
    log_barrier = math.log(barrier)
    spots = np.full(n_paths, spot, dtype=float)
    knocked = np.zeros(n_paths, dtype=bool)
    for _ in range(n_steps):
        prev = spots
        nxt = prev * np.exp((-0.5 * sigma * sigma * dt) + sigma * math.sqrt(dt) * np.random.normal(size=n_paths))
        live = (~knocked) & (prev > barrier) & (nxt > barrier)
        if np.any(live):
            x0 = np.log(prev[live]) - log_barrier
            x1 = np.log(nxt[live]) - log_barrier
            hit_prob = np.exp(-2.0 * x0 * x1 / (sigma * sigma * dt))
            knocked[live] = np.random.random(live.sum()) < np.minimum(hit_prob, 1.0)
        knocked |= (prev <= barrier) | (nxt <= barrier)
        spots = nxt
    payoff = np.where(knocked, 0.0, np.maximum(strike - spots, 0.0))
    return _mc_stats(payoff)


def _print_mc_check(name: str, closed_form: float, mc_mean: float, stderr: float) -> bool:
    within = abs(closed_form - mc_mean) <= 3.0 * stderr
    band = 3.0 * stderr
    status = "PASS" if within else "FAIL"
    print(f"{name} closed-form: {closed_form:.6f}")
    print(f"{name} MC: {mc_mean:.6f} +/- {band:.6f}")
    print(f"{name} within 3 stderr: {status}")
    return within


def _print_dimensional_checks(
    chooser: float, chooser_floor: float, ko_put: float, vanilla_put: float
) -> list[bool]:
    checks = [
        ("chooser >= max(call(T_full), put(T_full))", chooser >= chooser_floor),
        ("ko_put < vanilla_put", ko_put < vanilla_put),
        ("ko_put >= 0", ko_put >= 0.0),
    ]
    for label, passed in checks:
        print(f"{label}: {'PASS' if passed else 'FAIL'}")
    return [passed for _, passed in checks]


def _price_table() -> list[tuple[float, float, float, float]]:
    s0 = 100.0
    sigma = 2.51
    payout = 100.0
    return [
        (
            strike,
            chooser_price(s0, strike, 3.0 / 52.0, 2.0 / 52.0, sigma),
            binary_put_price(s0, strike, 1.0 / 52.0, sigma, payout),
            down_and_out_put_price(s0, strike, 70.0, 3.0 / 52.0, sigma),
        )
        for strike in (80.0, 90.0, 100.0, 110.0, 120.0)
    ]


if __name__ == "__main__":
    np.random.seed(42)
    n_paths = 200_000
    s0 = 100.0
    k = 100.0
    sigma = 2.51
    chooser_cf = chooser_price(s0, k, 3.0 / 52.0, 2.0 / 52.0, sigma)
    chooser_mc_mean, chooser_mc_se = chooser_mc(s0, k, 3.0 / 52.0, 2.0 / 52.0, sigma, n_paths)
    binary_cf = binary_put_price(s0, k, 1.0 / 52.0, sigma, 100.0)
    binary_mc_mean, binary_mc_se = binary_put_mc(s0, k, 1.0 / 52.0, sigma, 100.0, n_paths)
    ko_cf = down_and_out_put_price(s0, k, 70.0, 3.0 / 52.0, sigma)
    ko_mc_mean, ko_mc_se = down_and_out_put_mc(s0, k, 70.0, 3.0 / 52.0, sigma, n_paths)
    all_pass = [
        _print_mc_check("Chooser", chooser_cf, chooser_mc_mean, chooser_mc_se),
        _print_mc_check("Binary Put", binary_cf, binary_mc_mean, binary_mc_se),
        _print_mc_check("KO Put", ko_cf, ko_mc_mean, ko_mc_se),
    ]
    all_pass.extend(
        _print_dimensional_checks(
            chooser_cf,
            max(bs_call(s0, k, 3.0 / 52.0, sigma), bs_put(s0, k, 3.0 / 52.0, sigma)),
            ko_cf,
            bs_put(s0, k, 3.0 / 52.0, sigma),
        )
    )
    print("\nPrice table (S0=100):")
    print("K | chooser | binary_put | ko_put")
    for strike, chooser_px, binary_px, ko_px in _price_table():
        print(f"{strike:>3.0f} | {chooser_px:>8.4f} | {binary_px:>10.4f} | {ko_px:>7.4f}")
    if not all(all_pass):
        raise SystemExit(1)
