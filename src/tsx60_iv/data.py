"""Option-chain acquisition and forward estimation."""

from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import yfinance as yf

_DEFAULT_TICKER: Final = "XIU.TO"


def fetch_xiu_chain(
    ticker: str = _DEFAULT_TICKER,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch, normalize, filter, and optionally cache an ETF option chain.

    Parameters
    ----------
    ticker
        Symbol understood by Yahoo Finance.
    cache_path
        Optional Parquet destination. Its parents are created automatically.

    Returns
    -------
    pandas.DataFrame
        Valid call and put quotes with calculated ``mid`` prices.
    """
    instrument = yf.Ticker(ticker)
    frames: list[pd.DataFrame] = []
    for expiration in instrument.options:
        chain = instrument.option_chain(expiration)
        for option_type, quotes in (("call", chain.calls), ("put", chain.puts)):
            frame = quotes.copy()
            frame["option_type"] = option_type
            frame["expiration"] = expiration
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result["mid"] = (result["bid"] + result["ask"]) / 2.0
    relative_spread = (result["ask"] - result["bid"]) / result["mid"]
    result = result.loc[
        (result["bid"] > 0)
        & (result["ask"] >= result["bid"])
        & (relative_spread <= 0.35)
        & np.isfinite(result["mid"])
    ].reset_index(drop=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)
    return result


def _filter_quoted_chain(chain_df: pd.DataFrame) -> pd.DataFrame:
    """Return valid quote rows for testable, provider-independent filtering."""
    required = {"strike", "bid", "ask", "expiration", "option_type"}
    missing = required.difference(chain_df.columns)
    if missing:
        raise ValueError(f"chain_df is missing required columns: {sorted(missing)}")
    result = chain_df.copy()
    result["mid"] = (result["bid"] + result["ask"]) / 2.0
    relative_spread = (result["ask"] - result["bid"]) / result["mid"]
    return result.loc[
        (result["bid"] > 0)
        & (result["ask"] >= result["bid"])
        & (relative_spread <= 0.35)
        & np.isfinite(result["mid"])
    ].reset_index(drop=True)


def estimate_forward_and_rates(
    chain_df: pd.DataFrame,
    spot: float,
    r_cad: float = 0.0375,
) -> pd.DataFrame:
    """Estimate forwards by put-call parity and calculate log-moneyness.

    Parameters
    ----------
    chain_df
        Quotes containing strike, bid, ask, expiration, and option_type.
    spot
        Current ETF spot price.
    r_cad
        Continuously compounded CAD risk-free rate.

    Returns
    -------
    pandas.DataFrame
        Clean quotes augmented with time, forward, discount, and
        ``log_moneyness`` columns.
    """
    if spot <= 0 or not np.isfinite(spot):
        raise ValueError("spot must be positive and finite")
    clean = _filter_quoted_chain(chain_df)
    as_of = pd.Timestamp(
        chain_df.get("as_of", pd.Series([datetime.utcnow().date()])).iloc[0]
    ).normalize()
    clean["expiration"] = pd.to_datetime(clean["expiration"])
    clean["time_to_expiry"] = (clean["expiration"] - as_of).dt.days / 365.0
    if np.any(clean["time_to_expiry"] <= 0):
        raise ValueError("expirations must be after as_of")
    clean["discount"] = np.exp(-r_cad * clean["time_to_expiry"])
    calls = clean.loc[clean["option_type"].str.lower() == "call"]
    puts = clean.loc[clean["option_type"].str.lower() == "put"]
    pairs = calls.merge(
        puts[["expiration", "strike", "mid"]],
        on=["expiration", "strike"],
        suffixes=("_call", "_put"),
    )
    pairs["forward_estimate"] = pairs["strike"] + (
        pairs["mid_call"] - pairs["mid_put"]
    ) / pairs["discount"]
    pairs["distance"] = abs(pairs["strike"] - spot)
    if pairs.empty:
        clean["forward"] = spot * np.exp(r_cad * clean["time_to_expiry"])
    else:
        ranked = pairs.sort_values(["expiration", "distance"])
        forwards = ranked.groupby("expiration")["forward_estimate"].median()
        clean["forward"] = clean["expiration"].map(forwards)
        clean["forward"] = clean["forward"].fillna(
            spot * np.exp(r_cad * clean["time_to_expiry"])
        )
    clean["log_moneyness"] = np.log(clean["strike"] / clean["forward"])
    return clean


if __name__ == "__main__":
    output = Path("data/xiu_chain.parquet")
    print(fetch_xiu_chain(cache_path=output).shape)
