"""Tests for option quote filtering and forward estimation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError

from tsx60_iv import data
from tsx60_iv.data import _filter_quoted_chain, estimate_forward_and_rates

FIXTURE = Path(__file__).parent / "fixtures" / "xiu_sample_chain.parquet"


def test_fixture_contains_four_expirations() -> None:
    """Fixture covers the intended deterministic maturity grid."""
    chain = pd.read_parquet(FIXTURE)
    days = (
        pd.to_datetime(chain["expiration"]) - pd.Timestamp(chain["as_of"].iloc[0])
    ).dt.days
    assert set(days) == {
        14,
        30,
        60,
        90,
    }


def test_zero_bid_and_crossed_markets_are_rejected() -> None:
    """Only positive, uncrossed, sufficiently liquid quotes survive."""
    quotes = pd.DataFrame(
        {
            "strike": [38.0, 38.0, 38.0, 38.0],
            "bid": [0.0, 1.2, 1.0, 1.0],
            "ask": [0.1, 1.1, 2.0, 1.3],
            "expiration": ["2025-02-01"] * 4,
            "option_type": ["call"] * 4,
        }
    )
    result = _filter_quoted_chain(quotes)
    assert result.index.tolist() == [0]
    assert result.iloc[0]["mid"] == 1.15


def test_forward_estimation_preserves_put_call_parity() -> None:
    """Estimated forwards explain paired quote parity within bid-ask spread."""
    chain = pd.read_parquet(FIXTURE)
    enriched = estimate_forward_and_rates(chain, spot=38.50)
    assert np.all(np.isfinite(enriched["log_moneyness"]))
    pairs = enriched.merge(
        enriched,
        on=["expiration", "strike"],
        suffixes=("_call", "_put"),
    )
    pairs = pairs.loc[
        (pairs["option_type_call"] == "call") & (pairs["option_type_put"] == "put")
    ]
    residual = (
        pairs["mid_call"]
        - pairs["mid_put"]
        - pairs["discount_call"] * (pairs["forward_call"] - pairs["strike"])
    )
    assert abs(residual).max() < 0.04


def test_forward_comes_from_parity_not_spot() -> None:
    """A wrong spot must not move the parity-estimated forward."""
    chain = pd.read_parquet(FIXTURE)
    enriched = estimate_forward_and_rates(chain, spot=36.0)
    expected = 38.50 * np.exp(0.0375 * enriched["time_to_expiry"])
    assert np.max(np.abs(enriched["forward"] - expected)) < 1e-9


def test_expiry_on_as_of_is_dropped() -> None:
    """An expiry on its last day is excluded instead of failing the chain."""
    chain = pd.read_parquet(FIXTURE)
    chain["expiration"] = pd.to_datetime(chain["expiration"])
    first = chain["expiration"].min()
    chain.loc[chain["expiration"] == first, "expiration"] = pd.Timestamp(
        chain["as_of"].iloc[0]
    ).normalize()
    result = estimate_forward_and_rates(chain, spot=38.50)
    assert result["expiration"].nunique() == 3
    assert (result["time_to_expiry"] > 0).all()


def test_calls_only_chain_raises() -> None:
    """Without call/put pairs no forward is guessed."""
    chain = pd.read_parquet(FIXTURE)
    calls = chain.loc[chain["option_type"] == "call"]
    with pytest.raises(ValueError, match="no call/put pairs"):
        estimate_forward_and_rates(calls, spot=38.50)


class _FakeTicker:
    """Offline stand-in for ``yf.Ticker`` that counts requests."""

    requests = 0
    fail_after: int | None = None
    options = ("2025-02-21", "2025-03-21", "2025-04-17", "2025-05-16")

    def __init__(self, _symbol: str) -> None:
        pass

    def option_chain(self, date: str | None = None) -> SimpleNamespace:
        cls = type(self)
        cls.requests += 1
        if cls.fail_after is not None and cls.requests > cls.fail_after:
            raise YFRateLimitError()
        quotes = pd.DataFrame({"strike": [38.0], "bid": [1.0], "ask": [1.1]})
        return SimpleNamespace(
            calls=quotes, puts=quotes, underlying={"regularMarketPrice": 38.5}
        )


@pytest.fixture
def fake_yahoo(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTicker]:
    """Patch yfinance and the pause so fetch tests are offline and instant."""
    _FakeTicker.requests, _FakeTicker.fail_after = 0, None
    monkeypatch.setattr(data.yf, "Ticker", _FakeTicker)
    monkeypatch.setattr(data.time, "sleep", lambda _s: None)
    return _FakeTicker


def test_fetch_makes_one_request_per_expiry_and_stores_spot(
    fake_yahoo: type[_FakeTicker], tmp_path: Path
) -> None:
    """Requests equal the capped expiry count; spot and as_of are cached."""
    result = data.fetch_xiu_chain("X", tmp_path / "c.parquet", max_expiries=3)
    assert fake_yahoo.requests == 3
    assert result["expiration"].nunique() == 3
    assert (result["spot"] == 38.5).all() and "as_of" in result
    assert (tmp_path / "c.parquet").exists()


def test_fetch_stops_on_rate_limit_and_writes_nothing(
    fake_yahoo: type[_FakeTicker], tmp_path: Path
) -> None:
    """A rate-limit error is not retried and leaves no cache file."""
    fake_yahoo.fail_after = 2
    with pytest.raises(YFRateLimitError):
        data.fetch_xiu_chain("X", tmp_path / "c.parquet")
    assert fake_yahoo.requests == 3
    assert not (tmp_path / "c.parquet").exists()


def test_fetch_empty_chain_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ticker Yahoo has no options for raises instead of caching nothing."""

    class Empty:
        options: tuple[str, ...] = ()

        def __init__(self, _symbol: str) -> None:
            pass

        def option_chain(self, date: str | None = None) -> SimpleNamespace:
            return SimpleNamespace(calls=None, puts=None, underlying=None)

    monkeypatch.setattr(data.yf, "Ticker", Empty)
    with pytest.raises(ValueError, match="no options"):
        data.fetch_xiu_chain("XIU.TO", tmp_path / "c.parquet")


def test_rate_curve_gives_each_expiry_its_own_rate() -> None:
    """A downward-sloping curve changes discounts per expiry; a float still works."""
    from tsx60_iv.rates import RateCurve

    chain = pd.read_parquet(FIXTURE)
    curve = RateCurve("2025-01-02", (0.01, 2.0), (0.04, 0.02), "test")
    enriched = estimate_forward_and_rates(chain, spot=38.50, r_cad=curve)
    per_expiry = enriched.groupby("time_to_expiry")["rate"].first()
    assert per_expiry.is_monotonic_decreasing and per_expiry.nunique() == 4
    assert np.allclose(
        enriched["discount"], np.exp(-enriched["rate"] * enriched["time_to_expiry"])
    )
    flat = estimate_forward_and_rates(chain, spot=38.50, r_cad=0.03)
    assert (flat["rate"] == 0.03).all()


def test_mid_adjustment_shifts_the_parity_forward_only() -> None:
    """Lowering call mids by d lowers each parity forward by d / D, mids untouched."""
    chain = pd.read_parquet(FIXTURE)
    base = estimate_forward_and_rates(chain, spot=38.50)
    calls = base["option_type"] == "call"
    adjusted = estimate_forward_and_rates(
        chain, spot=38.50, mid_adjustment=pd.Series(np.where(calls, 0.10, 0.0))
    )
    assert np.allclose(adjusted["mid"], base["mid"])
    shift = base["forward"] - adjusted["forward"]
    assert np.allclose(shift, 0.10 / base["discount"], atol=1e-9)
