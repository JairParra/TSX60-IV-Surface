"""Tests for forward BSM pricing and implied volatility inversion."""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from tsx60_iv.bsm import bsm_greeks, bsm_price, implied_volatility


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_put_call_parity(option_type: str) -> None:
    forward = np.linspace(80.0, 120.0, 17)
    strike = np.full_like(forward, 100.0)
    time = np.full_like(forward, 0.5)
    discount = np.exp(-0.03 * time)
    volatility = np.full_like(forward, 0.23)
    call = bsm_price(forward, strike, time, discount, volatility)
    put = bsm_price(forward, strike, time, discount, volatility, "put")
    assert np.max(np.abs(call - put - discount * (forward - strike))) < 1e-10


@given(
    moneyness=st.floats(min_value=0.75, max_value=1.25),
    volatility=st.floats(min_value=0.03, max_value=1.5),
    time=st.floats(min_value=0.02, max_value=2.0),
    discount=st.floats(min_value=0.85, max_value=0.999),
)
def test_implied_volatility_round_trip(
    moneyness: float,
    volatility: float,
    time: float,
    discount: float,
) -> None:
    forward = 100.0
    strike = forward * moneyness
    price = bsm_price(forward, strike, time, discount, volatility)
    recovered = implied_volatility(price, forward, strike, time, discount)
    assert recovered == pytest.approx(volatility, abs=1e-6)


def test_vega_matches_central_finite_difference() -> None:
    forward = np.array([85.0, 100.0, 115.0])
    strike = np.full(3, 100.0)
    time = np.full(3, 0.4)
    discount = np.full(3, 0.985)
    volatility = np.full(3, 0.25)
    step = 1e-6
    finite_difference = (
        bsm_price(forward, strike, time, discount, volatility + step)
        - bsm_price(forward, strike, time, discount, volatility - step)
    ) / (2 * step)
    assert np.max(
        np.abs(bsm_greeks(forward, strike, time, discount, volatility)["vega"]
        - finite_difference)
    ) < 1e-5


def test_implied_volatility_rejects_arbitrage_price() -> None:
    with pytest.raises(ValueError, match="arbitrage"):
        implied_volatility(100.0, 100.0, 100.0, 1.0, 0.99)
