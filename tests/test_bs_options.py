import math

import pytest

from strategies.bs_options import bs_call, bs_greeks, bs_put, implied_vol


def test_bs_call_reference_price() -> None:
    assert bs_call(100, 100, 1, 0.05, 0.2) == pytest.approx(10.4506, abs=1e-4)


def test_bs_put_reference_price() -> None:
    assert bs_put(100, 100, 1, 0.05, 0.2) == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity() -> None:
    call = bs_call(100, 100, 1, 0.05, 0.2)
    put = bs_put(100, 100, 1, 0.05, 0.2)
    assert call - put == pytest.approx(100 - 100 * math.exp(-0.05), abs=1e-10)


def test_t_zero_call_boundary() -> None:
    assert bs_call(90, 100, 0, 0.05, 0.2) == 0.0


def test_t_zero_put_boundary() -> None:
    assert bs_put(90, 100, 0, 0.05, 0.2) == 10.0


def test_deep_itm_call_price() -> None:
    assert bs_call(150, 100, 1, 0.05, 0.2) == pytest.approx(54.97, abs=0.02)


def test_call_greeks_ranges() -> None:
    greeks = bs_greeks(100, 100, 1, 0.05, 0.2, "call")
    assert greeks["delta"] == pytest.approx(0.6368, abs=1e-4)
    assert 0.0 < greeks["delta"] < 1.0
    assert greeks["gamma"] > 0.0
    assert greeks["vega"] > 0.0


def test_put_delta_range() -> None:
    greeks = bs_greeks(100, 100, 1, 0.05, 0.2, "put")
    assert -1.0 < greeks["delta"] < 0.0


def test_call_implied_vol_round_trip() -> None:
    price = bs_call(100, 100, 1, 0.05, 0.2)
    assert implied_vol(price, 100, 100, 1, 0.05, "call") == pytest.approx(0.2, abs=1e-6)


def test_put_implied_vol_round_trip() -> None:
    price = bs_put(100, 100, 1, 0.05, 0.3)
    assert implied_vol(price, 100, 100, 1, 0.05, "put") == pytest.approx(0.3, abs=1e-6)


def test_impossible_price_returns_nan() -> None:
    assert math.isnan(implied_vol(150.0, 100, 100, 1, 0.05, "call"))
