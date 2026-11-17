"""Tests for option quote filtering and forward estimation."""

from pathlib import Path

import numpy as np
import pandas as pd

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
