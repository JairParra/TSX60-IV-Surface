"""Offline tests for the Yahoo-derived market parameters."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tsx60_iv.market_params import derive_params


def _history(days: int = 300) -> pd.DataFrame:
    index = pd.bdate_range(end="2025-01-03", periods=days, tz="America/Toronto")
    rng = np.random.default_rng(0)
    close = 30.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.007, days)))
    dividends = np.zeros(days)
    for position, amount in zip((60, 150, 240), (0.25, 0.26, 0.27), strict=True):
        if position < days:
            dividends[position] = amount
    return pd.DataFrame({"Close": close, "Dividends": dividends}, index=index)


def test_derive_params_matches_direct_computation() -> None:
    """Spot is the last close up to as_of; vols are annualized log-return stds."""
    history = _history()
    params = derive_params(history, date(2025, 1, 2), "XIU.TO")
    kept = history.loc[: pd.Timestamp("2025-01-02", tz="America/Toronto")]
    returns = np.log(kept["Close"]).diff().dropna()
    assert params["spot"] == round(float(kept["Close"].iloc[-1]), 2)
    assert params["realized_vol"]["63d"] == pytest.approx(
        float(returns.tail(63).std() * np.sqrt(252)), abs=1e-4
    )
    assert "price series" in params["provenance"]


def test_only_dividends_in_the_trailing_year_are_kept() -> None:
    """Dividends older than 365 days before as_of are excluded."""
    params = derive_params(_history(), date(2025, 1, 2), "XIU.TO")
    amounts = [d["amount"] for d in params["dividends"]]
    assert amounts == [0.25, 0.26, 0.27][-len(amounts) :]
    assert all(
        pd.Timestamp(d["ex_date"]) > pd.Timestamp("2024-01-02")
        for d in params["dividends"]
    )


def test_too_little_history_raises() -> None:
    """Realized volatility needs a full year of returns."""
    with pytest.raises(ValueError, match="not enough history"):
        derive_params(_history(100), date(2025, 1, 2), "XIU.TO")
