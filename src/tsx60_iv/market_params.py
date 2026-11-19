"""Free Yahoo-derived parameters that anchor the synthetic chain to reality.

One explicit history request yields the closing spot, realized volatilities
and trailing dividends of an ETF. Only these derived statistics are stored;
no price series is kept or redistributed.
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from tsx60_iv.american import Dividends

REALIZED_WINDOWS = (21, 63, 252)  # trading days
TRADING_DAYS = 252


def derive_params(history: pd.DataFrame, as_of: date, ticker: str) -> dict[str, Any]:
    """Compute spot, realized vols and trailing-year dividends as of ``as_of``."""
    frame = history.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame = frame.loc[: pd.Timestamp(as_of)]
    if len(frame) < max(REALIZED_WINDOWS) + 1:
        raise ValueError("not enough history to compute realized volatility")
    returns = np.log(frame["Close"]).diff().dropna()
    dividends = frame["Dividends"]
    year_ago = pd.Timestamp(as_of) - pd.Timedelta(days=365)
    paid = dividends.loc[(dividends > 0) & (dividends.index > year_ago)]
    return {
        "ticker": ticker,
        "as_of": as_of.isoformat(),
        "spot": round(float(frame["Close"].iloc[-1]), 2),
        "realized_vol": {
            f"{n}d": round(float(returns.tail(n).std() * np.sqrt(TRADING_DAYS)), 4)
            for n in REALIZED_WINDOWS
        },
        "dividends": [
            {"ex_date": stamp.date().isoformat(), "amount": round(float(amount), 4)}
            for stamp, amount in paid.items()
        ],
        "provenance": (
            "Derived statistics from Yahoo Finance daily history via yfinance "
            "(free, personal use). No price series is stored or redistributed."
        ),
    }


def expected_dividends(params: dict[str, Any]) -> Dividends:
    """Return the expected dividends: the trailing ones repeated 364 days later.

    # shortcut: no growth and no announced schedule; replace with the issuer's
    # declared distributions if they are available.
    """
    as_of = date.fromisoformat(params["as_of"])
    pairs = [
        (
            (date.fromisoformat(d["ex_date"]) + timedelta(days=364) - as_of).days
            / 365.0,
            float(d["amount"]),
        )
        for d in params["dividends"]
    ]
    pairs = [(t, a) for t, a in sorted(pairs) if t > 0]
    return Dividends(tuple(t for t, _ in pairs), tuple(a for _, a in pairs))


def load_params(path: Path) -> dict[str, Any]:
    """Read a parameter file written by ``main``."""
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_market_params(ticker: str, as_of: date) -> dict[str, Any]:
    """Download about 14 months of daily history (one request) and derive."""
    start = as_of - timedelta(days=430)
    history = yf.Ticker(ticker).history(
        start=start.isoformat(),
        end=(as_of + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        actions=True,
    )
    return derive_params(history, as_of, ticker)


def main() -> None:
    """Fetch and write the market parameters (explicit network call)."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--ticker", default="XIU.TO")
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    params = fetch_market_params(args.ticker, args.as_of)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: spot {params['spot']}, vols {params['realized_vol']}")


if __name__ == "__main__":
    main()
