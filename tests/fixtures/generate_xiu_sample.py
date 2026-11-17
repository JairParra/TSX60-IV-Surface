"""Generate the deterministic XIU option-chain regression fixture."""

from pathlib import Path

import numpy as np
import pandas as pd

from tsx60_iv.bsm import bsm_price

FIXTURE_PATH = Path(__file__).with_name("xiu_sample_chain.parquet")


def generate_xiu_sample_chain() -> pd.DataFrame:
    """Build synthetic XIU quotes with a realistic downward volatility skew."""
    as_of = pd.Timestamp("2025-01-02")
    spot = 38.50
    rate = 0.0375
    strikes = np.arange(31.0, 46.5, 1.0)
    rows: list[dict[str, float | str]] = []
    for days in (14, 30, 60, 90):
        expiry = as_of + pd.Timedelta(days=days)
        time = days / 365.0
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


if __name__ == "__main__":
    generate_xiu_sample_chain().to_parquet(FIXTURE_PATH, index=False)
