"""Tests for the American-exercise binomial pricer and bias check."""

import numpy as np
import pytest

from tsx60_iv.american import (
    Dividends,
    american_implied_volatility,
    american_implied_vols,
    american_price,
    early_exercise_bias,
    european_tree_price,
    implied_dividend_yield,
)
from tsx60_iv.bsm import bsm_price


def _bsm(spot, strike, time, rate, q, vol, kind):
    forward = spot * np.exp((rate - q) * time)
    return float(bsm_price(forward, strike, time, np.exp(-rate * time), vol, kind))


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [32.0, 37.7, 44.0])
def test_european_tree_converges_to_bsm(kind: str, strike: float) -> None:
    """The tree's European value matches closed-form forward BSM."""
    args = (37.7, strike, 0.5, 0.03, 0.029, 0.14)
    tree = european_tree_price(*args, kind, steps=400)
    assert tree == pytest.approx(_bsm(*args, kind), abs=2e-4)


def test_american_call_without_dividends_equals_european() -> None:
    """With q = 0 early exercise of a call is never optimal."""
    args = (100.0, 95.0, 1.0, 0.05, 0.0, 0.25)
    assert american_price(*args, "call") == pytest.approx(
        european_tree_price(*args, "call"), abs=1e-9
    )


def test_american_put_matches_published_value() -> None:
    """S=K=100, r=5%, sigma=20%, T=1: European 5.5735, American about 6.09."""
    value = american_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, "put", steps=1000)
    assert value == pytest.approx(6.09, abs=0.01)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_american_dominates_european_and_intrinsic(kind: str) -> None:
    """American >= European >= 0 and American >= intrinsic, for ITM and OTM."""
    for strike in (30.0, 37.7, 45.0):
        args = (37.7, strike, 0.75, 0.04, 0.03, 0.15)
        amer = american_price(*args, kind)
        intrinsic = max(37.7 - strike, 0) if kind == "call" else max(strike - 37.7, 0)
        assert amer >= european_tree_price(*args, kind) - 1e-12
        assert amer >= intrinsic - 1e-12


def test_mcdonald_schroder_put_call_symmetry() -> None:
    """P_A(S, K, r, q) equals C_A(K, S, q, r): exact in theory, close on a tree."""
    put = american_price(40.0, 42.0, 0.5, 0.04, 0.02, 0.2, "put", steps=300)
    call = american_price(42.0, 40.0, 0.5, 0.02, 0.04, 0.2, "call", steps=300)
    assert put == pytest.approx(call, abs=2e-3)


def test_price_is_increasing_in_volatility() -> None:
    """Vega is positive, which the implied-vol inversion relies on."""
    vols = np.linspace(0.05, 0.6, 12)
    prices = [american_price(37.7, 36.0, 0.5, 0.03, 0.03, v, "put") for v in vols]
    assert np.all(np.diff(prices) > 0)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_volatility_round_trip(kind: str) -> None:
    """Inverting an American price recovers the volatility that made it."""
    price = american_price(37.7, 35.0, 0.4, 0.032, 0.029, 0.17, kind)
    recovered = american_implied_volatility(price, 37.7, 35.0, 0.4, 0.032, 0.029, kind)
    assert recovered == pytest.approx(0.17, abs=1e-6)


def test_implied_volatility_rejects_arbitrage_prices() -> None:
    """Price at intrinsic, or above the strike for a put, is rejected."""
    with pytest.raises(ValueError, match="arbitrage"):
        american_implied_volatility(5.0, 35.0, 40.0, 0.5, 0.03, 0.02, "put")
    with pytest.raises(ValueError, match="arbitrage"):
        american_implied_volatility(41.0, 35.0, 40.0, 0.5, 0.03, 0.02, "put")


def test_bias_is_largest_for_deep_itm_put_and_matches_inversion() -> None:
    """Early-exercise bias grows ITM, and first order matches true re-inversion."""
    spot, rate, tau = 37.7, 0.05, 1.0
    forward = spot * np.exp(0.02 * tau)
    q = implied_dividend_yield(spot, forward, tau, rate)
    biases = {}
    for strike in (30.0, 37.7, 46.0):
        _, bias = early_exercise_bias(0.15, spot, forward, strike, tau, rate, "put")
        biases[strike] = bias
    assert biases[46.0] > biases[37.7] > biases[30.0] >= 0
    # True bias: European-invert the American price made with sigma = 0.15.
    from tsx60_iv.bsm import implied_volatility

    amer = american_price(spot, 46.0, tau, rate, q, 0.15, "put")
    european_iv = float(
        implied_volatility(amer, forward, 46.0, tau, np.exp(-rate * tau), "put")
    )
    assert biases[46.0] == pytest.approx(european_iv - 0.15, rel=0.35)


def test_rejects_bad_inputs() -> None:
    """Non-finite or non-positive inputs raise."""
    with pytest.raises(ValueError):
        american_price(37.7, 36.0, 0.5, 0.03, 0.02, 0.0, "put")
    with pytest.raises(ValueError):
        american_price(37.7, 36.0, float("nan"), 0.03, 0.02, 0.2, "put")
    with pytest.raises(ValueError, match="option_type"):
        american_price(37.7, 36.0, 0.5, 0.03, 0.02, 0.2, "straddle")  # type: ignore[arg-type]


def test_vectorized_pricing_matches_scalar_calls() -> None:
    """Pricing a strike/vol array equals pricing each option on its own."""
    strikes = np.array([33.0, 37.7, 42.0])
    vols = np.array([0.18, 0.14, 0.12])
    batch = american_price(37.7, strikes, 0.5, 0.03, 0.025, vols, "put")
    single = [
        american_price(37.7, k, 0.5, 0.03, 0.025, v, "put")
        for k, v in zip(strikes, vols, strict=True)
    ]
    assert batch == pytest.approx(single, abs=1e-12)


def test_batch_inversion_flags_unsolvable_prices_with_nan() -> None:
    """Good prices invert; prices at/below intrinsic or above the strike give NaN."""
    strikes = np.array([35.0, 37.7, 40.0, 41.0])
    good = american_price(
        37.7, strikes[:2], 0.4, 0.03, 0.025, np.array([0.15, 0.2]), "put"
    )
    prices = np.array([good[0], good[1], 2.0, 50.0])  # intrinsic is 2.3; 50 > strike
    vols = american_implied_vols(prices, 37.7, strikes, 0.4, 0.03, 0.025, "put")
    assert vols[:2] == pytest.approx([0.15, 0.2], abs=1e-7)
    assert np.isnan(vols[2:]).all()


def test_bias_accepts_arrays_for_one_expiry() -> None:
    """The array form matches the scalar form."""
    strikes = np.array([40.0, 44.0])
    ivs = np.array([0.15, 0.16])
    forward = 37.7 * np.exp(0.01)
    premium, bias = early_exercise_bias(ivs, 37.7, forward, strikes, 0.5, 0.03, "put")
    for i in range(2):
        p_i, b_i = early_exercise_bias(
            ivs[i], 37.7, forward, strikes[i], 0.5, 0.03, "put"
        )
        assert premium[i] == pytest.approx(p_i) and bias[i] == pytest.approx(b_i)


def test_dividends_validate_and_discount() -> None:
    """PV counts payments strictly after `at` and up to `until`."""
    divs = Dividends((0.1, 0.4, 0.9), (0.25, 0.25, 0.25))
    assert divs.pv(0.0, 0.0, 0.5) == pytest.approx(0.5)
    assert divs.pv(0.1, 0.0, 1.0) == pytest.approx(0.5)  # 0.1 itself is paid
    assert divs.pv(0.0, 0.05, 0.5) == pytest.approx(
        0.25 * (np.exp(-0.005) + np.exp(-0.02))
    )
    with pytest.raises(ValueError):
        Dividends((0.1,), (0.2, 0.3))
    with pytest.raises(ValueError):
        Dividends((-0.1,), (0.2,))


def _consistent_q(spot: float, divs: Dividends, time: float, rate: float) -> float:
    """Yield that reproduces the forward (S - PV(dividends)) * exp(rT)."""
    forward = (spot - divs.pv(0.0, rate, time)) * np.exp(rate * time)
    return implied_dividend_yield(spot, forward, time, rate)


def test_discrete_and_continuous_dividends_share_the_european_price() -> None:
    """Both anchor to the same forward, so only the exercise value differs."""
    divs = Dividends((0.2,), (0.4,))
    q = _consistent_q(37.7, divs, 0.5, 0.03)
    args = (37.7, 37.7, 0.5, 0.03, q, 0.15, "call")
    exact = _bsm(*args)
    # Different trees (S vs S*) discretise differently, but both converge to BSM.
    assert european_tree_price(*args, dividends=divs) == pytest.approx(exact, abs=2e-3)
    assert european_tree_price(*args) == pytest.approx(exact, abs=2e-3)


def test_call_before_an_ex_date_is_worth_more_with_discrete_dividends() -> None:
    """A continuous yield spreads the dividend and understates the call premium."""
    divs = Dividends((0.3,), (1.0,))  # large payment shortly before expiry
    q = _consistent_q(37.7, divs, 0.31, 0.03)
    args = (37.7, 36.0, 0.31, 0.03, q, 0.15, "call")
    discrete = american_price(*args, dividends=divs)
    continuous = american_price(*args)
    european = european_tree_price(*args)
    assert discrete > continuous >= european - 1e-12
    assert discrete - european > 0.05  # exercising just before the ex-date pays


def test_schedule_after_expiry_or_inconsistent_falls_back_to_continuous() -> None:
    """No dividend before expiry, or a schedule 3x off the forward, is ignored."""
    args = (37.7, 36.0, 0.5, 0.03, 0.02, 0.15, "call")
    assert american_price(*args, dividends=Dividends((0.9,), (0.3,))) == american_price(
        *args
    )
    inconsistent = Dividends((0.2,), (0.01,))  # PV 0.01 versus implied S(1-e^-qT)
    assert american_price(*args, dividends=inconsistent) == american_price(*args)


def test_inversion_with_dividends_round_trips() -> None:
    """Prices from the discrete tree invert back to the volatility that made them."""
    divs = Dividends((0.2, 0.45), (0.3, 0.3))
    q = _consistent_q(37.7, divs, 0.5, 0.03)
    strikes = np.array([35.0, 37.7, 40.0])
    prices = american_price(37.7, strikes, 0.5, 0.03, q, 0.16, "call", dividends=divs)
    vols = american_implied_vols(
        prices, 37.7, strikes, 0.5, 0.03, q, "call", dividends=divs
    )
    assert vols == pytest.approx([0.16, 0.16, 0.16], abs=1e-7)
