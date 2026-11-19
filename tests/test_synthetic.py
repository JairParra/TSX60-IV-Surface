"""Tests for the realistic synthetic chain and its SSVI truth surface."""

from datetime import date

import numpy as np
import pytest

from tsx60_iv.svi import durrleman_g, raw_svi_variance
from tsx60_iv.synthetic import (
    DEFAULT_SURFACE,
    SsviSurface,
    expiry_dates,
    generate_realistic_chain,
    strike_grid,
    third_friday,
)


def test_default_surface_satisfies_ssvi_no_arbitrage_conditions() -> None:
    """The truth surface meets the Gatheral-Jacquier conditions."""
    DEFAULT_SURFACE.check_no_arbitrage()


@pytest.mark.parametrize(
    "bad",
    [
        SsviSurface(0.115, 0.135, 0.6, -0.65, 12.0, 0.4),  # eta too large
        SsviSurface(0.6, 0.05, 0.3, -0.65, 0.9, 0.4),  # theta falls with maturity
    ],
)
def test_check_rejects_arbitrageable_surfaces(bad: SsviSurface) -> None:
    """Violating the SSVI conditions raises."""
    with pytest.raises(ValueError):
        bad.check_no_arbitrage()


@pytest.mark.parametrize("time", [0.05, 0.25, 1.0, 2.0])
def test_raw_svi_form_matches_ssvi_and_is_butterfly_free(time: float) -> None:
    """Each SSVI slice is an exact raw SVI with g(k) >= 0."""
    k = np.linspace(-0.6, 0.6, 241)
    params = DEFAULT_SURFACE.raw_svi(time)
    assert raw_svi_variance(k, *params) == pytest.approx(
        DEFAULT_SURFACE.total_variance(k, time), rel=1e-10
    )
    assert np.all(durrleman_g(k, *params) >= 0)


def test_smile_is_a_realistic_equity_index_skew() -> None:
    """Downside skew, ATM vol near 12%, and a 3-month 90% put a few points up."""
    atm = float(DEFAULT_SURFACE.vol(0.0, 0.25))
    put90 = float(DEFAULT_SURFACE.vol(np.log(0.9), 0.25))
    call110 = float(DEFAULT_SURFACE.vol(np.log(1.1), 0.25))
    assert 0.11 < atm < 0.14
    assert 0.02 < put90 - atm < 0.06
    assert call110 < atm


def test_third_friday_and_good_friday_rule() -> None:
    """Expiry is the third Friday, moved to Thursday when it is Good Friday."""
    assert third_friday(2025, 1) == date(2025, 1, 17)
    assert third_friday(2025, 4) == date(2025, 4, 17)  # Good Friday is 18 April 2025
    assert all(d.weekday() in (3, 4) for d in expiry_dates(date(2025, 1, 2), 24))
    assert expiry_dates(date(2025, 1, 2))[0] == date(2025, 1, 17)


def test_chain_is_deterministic_and_market_like() -> None:
    """Same seed, same chain; ticks, strikes and spreads look like listed options."""
    first, second = generate_realistic_chain(), generate_realistic_chain()
    assert first.equals(second)
    assert first is not second
    assert not first.equals(generate_realistic_chain(seed=1))
    for column in ("bid", "ask"):
        assert np.allclose(first[column] / 0.01, np.round(first[column] / 0.01))
    assert (first["ask"] > first["bid"]).all() and (first["bid"] >= 0).all()
    quoted = first[first["bid"] > 0]  # zero bids are the floor, not a spread
    assert (quoted["ask"] - quoted["bid"]).min() >= 0.02 - 1e-9
    assert first["expiration"].nunique() == 13
    assert set(first["strike"]) == set(strike_grid(float(first["spot"].iloc[0])))


def test_deep_itm_puts_quote_at_or_above_intrinsic() -> None:
    """Deep ITM puts never quote below intrinsic value, as American options cannot."""
    chain = generate_realistic_chain()
    spot = float(chain["spot"].iloc[0])
    puts = chain[(chain["option_type"] == "put") & (chain["strike"] > 1.1 * spot)]
    assert len(puts) > 5
    assert (puts["ask"] >= puts["strike"] - spot - 1e-9).all()
