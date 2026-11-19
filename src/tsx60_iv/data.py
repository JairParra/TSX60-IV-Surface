"""Option-chain acquisition and forward estimation."""

import time
from datetime import date
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import yfinance as yf

from tsx60_iv.rates import RateCurve

_DEFAULT_TICKER: Final = "XIU.TO"
_PAUSE_SECONDS: Final = 1.5


def fetch_xiu_chain(
    ticker: str = _DEFAULT_TICKER,
    cache_path: Path | None = None,
    max_expiries: int = 6,
) -> pd.DataFrame:
    """Fetch, normalize, and optionally cache an ETF option chain.

    Makes one Yahoo request per expiry (at most ``max_expiries``), pausing
    between them. There are no retries: a rate-limit error propagates and
    nothing is written.

    Parameters
    ----------
    ticker
        Symbol understood by Yahoo Finance.
    cache_path
        Optional Parquet destination. Its parents are created automatically.
    max_expiries
        Number of nearest expiries to download.

    Returns
    -------
    pandas.DataFrame
        Raw call and put quotes (unfiltered, no ``mid``) with ``as_of`` and
        ``spot`` columns.
    """
    instrument = yf.Ticker(ticker)
    # One request returns the nearest chain, every expiry date and the spot quote.
    first = instrument.option_chain()
    expirations = instrument.options[:max_expiries]
    if not expirations or first.calls is None:
        raise ValueError(f"Yahoo returned no options for {ticker}")
    frames: list[pd.DataFrame] = []
    for index, expiration in enumerate(expirations):
        if index:
            time.sleep(_PAUSE_SECONDS)
        chain = instrument.option_chain(expiration) if index else first
        for option_type, quotes in (("call", chain.calls), ("put", chain.puts)):
            frame = quotes.copy()
            frame["option_type"] = option_type
            frame["expiration"] = expiration
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    today = pd.Timestamp.now(tz="America/Toronto").normalize()
    result["as_of"] = today.tz_localize(None)
    result["spot"] = float(first.underlying["regularMarketPrice"])
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
    r_cad: float | RateCurve = 0.0375,
    mid_adjustment: pd.Series | None = None,
) -> pd.DataFrame:
    """Estimate forwards by put-call parity and calculate log-moneyness.

    Parameters
    ----------
    chain_df
        Quotes containing strike, bid, ask, expiration, and option_type.
    spot
        Current ETF spot price.
    r_cad
        Continuously compounded CAD risk-free rate: a constant, or a
        ``RateCurve`` evaluated at each expiry's time to expiry.
    mid_adjustment
        Optional amount subtracted from each filtered quote's mid in the parity
        step only (indexed like this function's own output), for example the
        early-exercise premium of American quotes.

    Returns
    -------
    pandas.DataFrame
        Clean quotes augmented with time, rate, discount, forward, and
        ``log_moneyness`` columns.
    """
    if spot <= 0 or not np.isfinite(spot):
        raise ValueError("spot must be positive and finite")
    clean = _filter_quoted_chain(chain_df)
    if "as_of" in chain_df:
        as_of = pd.Timestamp(chain_df["as_of"].iloc[0]).normalize()
    else:
        as_of = pd.Timestamp(date.today())
    clean["expiration"] = pd.to_datetime(clean["expiration"])
    clean["time_to_expiry"] = (clean["expiration"] - as_of).dt.days / 365.0
    clean = clean.loc[clean["time_to_expiry"] > 0].reset_index(drop=True)
    if clean.empty:
        raise ValueError("no expirations after as_of")
    clean["rate"] = r_cad(clean["time_to_expiry"]) if callable(r_cad) else float(r_cad)
    clean["discount"] = np.exp(-clean["rate"] * clean["time_to_expiry"])
    adjustment = 0.0 if mid_adjustment is None else mid_adjustment
    clean["parity_mid"] = clean["mid"] - pd.Series(
        adjustment, index=clean.index
    ).fillna(0.0)
    calls = clean.loc[clean["option_type"].str.lower() == "call"]
    puts = clean.loc[clean["option_type"].str.lower() == "put"]
    pairs = calls.merge(
        puts[["expiration", "strike", "parity_mid"]],
        on=["expiration", "strike"],
        suffixes=("_call", "_put"),
    )
    pairs["forward_estimate"] = (
        pairs["strike"]
        + (pairs["parity_mid_call"] - pairs["parity_mid_put"]) / pairs["discount"]
    )
    pairs["distance"] = abs(pairs["strike"] - spot)
    if pairs.empty:
        raise ValueError("no call/put pairs to estimate forwards")
    ranked = pairs.sort_values(["expiration", "distance"])
    near_atm = ranked.groupby("expiration", group_keys=False).head(3)
    forwards = near_atm.groupby("expiration")["forward_estimate"].median()
    clean["forward"] = clean["expiration"].map(forwards)
    clean = clean.dropna(subset=["forward"]).reset_index(drop=True)
    clean["log_moneyness"] = np.log(clean["strike"] / clean["forward"])
    return clean


if __name__ == "__main__":
    output = Path("data/xiu_chain.parquet")
    print(fetch_xiu_chain(cache_path=output).shape)
