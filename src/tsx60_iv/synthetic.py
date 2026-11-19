"""Synthetic XIU-style option chain for the offline fixture and the demo surface."""

import numpy as np
import pandas as pd

from tsx60_iv.bsm import bsm_price

SPOT = 38.50
RATE = 0.0375
AS_OF = "2025-01-02"
DEMO_DAYS = (45, 75, 105, 135, 180, 270, 365)


def generate_xiu_sample_chain(
    days: tuple[int, ...] = (14, 30, 60, 90),
) -> pd.DataFrame:
    """Build synthetic XIU quotes with a realistic downward volatility skew."""
    as_of = pd.Timestamp(AS_OF)
    spot = SPOT
    rate = RATE
    strikes = np.arange(31.0, 46.5, 1.0)
    rows: list[dict[str, float | str]] = []
    for day_count in days:
        expiry = as_of + pd.Timedelta(days=day_count)
        time = day_count / 365.0
        discount = np.exp(-rate * time)
        forward = spot / discount
        for strike in strikes:
            log_moneyness = np.log(strike / forward)
            volatility = 0.205 - 0.11 * log_moneyness + 0.02 * np.sqrt(time)
            for option_type in ("call", "put"):
                fair_price = float(
                    bsm_price(
                        forward,
                        strike,
                        time,
                        discount,
                        volatility,
                        option_type,
                    )
                )
                spread = max(0.01, fair_price * 0.03)
                rows.append(
                    {
                        "contractSymbol": (
                            f"XIU{expiry:%y%m%d}{option_type[0].upper()}"
                            f"{strike * 1000:08.0f}"
                        ),
                        "ticker": "XIU.TO",
                        "as_of": as_of.isoformat(),
                        "expiration": expiry.isoformat(),
                        "option_type": option_type,
                        "strike": float(strike),
                        "bid": max(0.005, fair_price - spread / 2),
                        "ask": fair_price + spread / 2,
                        "volume": 300.0,
                        "openInterest": 1200.0,
                    }
                )
    return pd.DataFrame(rows)

