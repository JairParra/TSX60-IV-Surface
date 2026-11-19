"""Synthetic XIU-style option chain for the offline fixture and the demo surface."""

import json
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dateutil.easter import easter

from tsx60_iv.american import american_price
from tsx60_iv.bsm import bsm_price
from tsx60_iv.rates import RateCurve

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


# ---------------------------------------------------------------------------
# Realistic demo chain: arbitrage-free SSVI truth, American prices, real rates
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data_files"
MARKET_PARAMS_PATH = DATA_DIR / "xiu_market_params.json"
BOC_CURVE_PATH = DATA_DIR / "boc_curve_2025-01-02.json"
TICK = 0.01
SEED = 20250102
TREE_STEPS = 100


@dataclass(frozen=True)
class SsviSurface:
    """Gatheral-Jacquier SSVI total variance with a power-law ``phi``.

    ``w(k, T) = theta/2 * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2))``
    where ``theta(T) = atm_vol(T)^2 * T`` and ``phi(theta) = eta * theta^-gamma``.
    """

    atm_short: float
    atm_long: float
    decay: float
    rho: float
    eta: float
    gamma: float

    def atm_vol(self, time: Any) -> Any:
        """ATM volatility term structure, rising from short to long."""
        return self.atm_short + (self.atm_long - self.atm_short) * (
            1.0 - np.exp(-np.asarray(time) / self.decay)
        )

    def theta(self, time: Any) -> Any:
        """ATM total variance."""
        return self.atm_vol(time) ** 2 * np.asarray(time)

    def phi(self, theta: Any) -> Any:
        """Skew-curvature function."""
        return self.eta * np.asarray(theta) ** -self.gamma

    def total_variance(self, k: Any, time: float) -> Any:
        """Total implied variance at log-moneyness ``k``."""
        theta = self.theta(time)
        phi = self.phi(theta)
        x = phi * np.asarray(k)
        root = np.sqrt((x + self.rho) ** 2 + 1.0 - self.rho**2)
        return 0.5 * theta * (1.0 + self.rho * x + root)

    def vol(self, k: Any, time: float) -> Any:
        """Implied volatility at log-moneyness ``k``."""
        return np.sqrt(self.total_variance(k, time) / time)

    def raw_svi(self, time: float) -> tuple[float, float, float, float, float]:
        """Exact raw-SVI ``(a, b, rho, m, sigma)`` of the slice at ``time``."""
        theta = float(self.theta(time))
        phi = float(self.phi(theta))
        rho = self.rho
        return (
            0.5 * theta * (1.0 - rho**2),
            0.5 * theta * phi,
            rho,
            -rho / phi,
            float(np.sqrt(1.0 - rho**2) / phi),
        )

    def check_no_arbitrage(self, max_time: float = 3.0) -> None:
        """Raise unless the Gatheral-Jacquier conditions hold on (0, max_time]."""
        times = np.linspace(1e-3, max_time, 600)
        theta = self.theta(times)
        phi = self.phi(theta)
        rho = abs(self.rho)
        if np.any(np.diff(theta) <= 0):
            raise ValueError("theta must increase with maturity (calendar)")
        if np.any(theta * phi * (1 + rho) >= 4):
            raise ValueError("SSVI butterfly condition theta*phi*(1+|rho|) < 4 fails")
        if np.any(theta * phi**2 * (1 + rho) > 4):
            raise ValueError(
                "SSVI butterfly condition theta*phi^2*(1+|rho|) <= 4 fails"
            )
        slope = (1.0 - self.gamma) * theta**-self.gamma * self.eta
        limit = (1.0 + np.sqrt(1.0 - self.rho**2)) / self.rho**2 * phi
        if np.any(slope < 0) or np.any(slope > limit):
            raise ValueError("SSVI calendar condition on theta*phi fails")


# Calm-regime index smile: realized vol was about 10% in Jan 2025, so ATM implied
# vol starts near 11.5% and drifts up with maturity; equity-like negative skew.
DEFAULT_SURFACE = SsviSurface(
    atm_short=0.115, atm_long=0.135, decay=0.6, rho=-0.65, eta=0.9, gamma=0.4
)


def third_friday(year: int, month: int) -> date:
    """Monthly expiry: third Friday, or the Thursday before if that is Good Friday."""
    first = date(year, month, 1)
    day = first + timedelta(days=(4 - first.weekday()) % 7 + 14)
    return day - timedelta(days=1) if day == easter(year) - timedelta(days=2) else day


def expiry_dates(as_of: date, months: int = 12, min_days: int = 7) -> list[date]:
    """Monthly third-Friday expiries over the next ``months`` months."""
    found = []
    for offset in range(months + 1):
        month_index = as_of.month - 1 + offset
        expiry = third_friday(as_of.year + month_index // 12, month_index % 12 + 1)
        if (expiry - as_of).days >= min_days:
            found.append(expiry)
    return found


def strike_grid(spot: float) -> np.ndarray:
    """Half-dollar strikes within 12% of spot, whole dollars out to 25%."""
    candidates = np.arange(np.floor(0.75 * spot), np.ceil(1.25 * spot) + 0.5, 0.5)
    near = np.abs(candidates / spot - 1.0) <= 0.12
    return candidates[near | (candidates == np.round(candidates))]


def _forward_inputs(
    spot: float,
    time: float,
    curve: RateCurve,
    dividends: list[tuple[float, float]],
) -> tuple[float, float, float]:
    """Return (rate, forward, implied_yield) using escrowed discrete dividends."""
    rate = float(curve(time))
    pv = sum(
        amount * np.exp(-float(curve(t)) * t) for t, amount in dividends if t <= time
    )
    forward = (spot - pv) * np.exp(rate * time)
    return rate, float(forward), rate - float(np.log(forward / spot)) / time


@lru_cache(maxsize=4)
def _realistic_chain(seed: int, steps: int, surface: SsviSurface) -> pd.DataFrame:
    surface.check_no_arbitrage()
    params = json.loads(MARKET_PARAMS_PATH.read_text(encoding="utf-8"))
    curve = RateCurve.from_json(BOC_CURVE_PATH)
    as_of, spot = date.fromisoformat(params["as_of"]), float(params["spot"])
    # shortcut: next year's dividends repeat the trailing four, one year later.
    dividends = [
        (
            (date.fromisoformat(d["ex_date"]) + timedelta(days=364) - as_of).days
            / 365.0,
            float(d["amount"]),
        )
        for d in params["dividends"]
    ]
    rng = np.random.default_rng(seed)
    strikes = strike_grid(spot)
    rows: list[dict[str, Any]] = []
    for expiry in expiry_dates(as_of):
        time = (expiry - as_of).days / 365.0
        rate, forward, div_yield = _forward_inputs(spot, time, curve, dividends)
        vols = surface.vol(np.log(strikes / forward), time)
        for kind in ("call", "put"):
            mid = american_price(
                spot, strikes, time, rate, div_yield, vols, kind, steps
            )
            spread = np.maximum(0.02, 0.05 * mid)
            shifted = mid + rng.uniform(-1.0, 1.0, mid.size) * 0.25 * spread
            bid = np.floor((shifted - spread / 2) / TICK + 1e-9) * TICK
            ask = np.ceil((shifted + spread / 2) / TICK - 1e-9) * TICK
            liquidity = np.exp(-((np.log(strikes / forward) / 0.08) ** 2))
            volume = np.round(rng.lognormal(3.0, 1.0, mid.size) * liquidity)
            for i, strike in enumerate(strikes):
                symbol = f"XIU{expiry:%y%m%d}{kind[0].upper()}{strike * 1000:08.0f}"
                rows.append(
                    {
                        "contractSymbol": symbol,
                        "ticker": "XIU.TO",
                        "as_of": as_of.isoformat(),
                        "expiration": expiry.isoformat(),
                        "option_type": kind,
                        "strike": float(strike),
                        "bid": round(max(float(bid[i]), 0.0), 2),
                        "ask": round(float(ask[i]), 2),
                        "volume": float(volume[i]),
                        "openInterest": float(round(volume[i] * rng.uniform(5, 20))),
                        "spot": spot,
                    }
                )
    return pd.DataFrame(rows)


def generate_realistic_chain(
    seed: int = SEED,
    steps: int = TREE_STEPS,
    surface: SsviSurface = DEFAULT_SURFACE,
) -> pd.DataFrame:
    """Build a deterministic, market-like XIU chain (synthetic, not quotes).

    - Truth: an arbitrage-free SSVI surface anchored to Jan 2025 realized vol.
    - Spot and dividends: derived statistics of XIU.TO (``market_params``).
    - Rates: the Bank of Canada curve of the same date (per-expiry zero rate).
    - Prices: American-style CRR tree, so European inversion shows a real bias.
    - Quotes: third-Friday expiries, 0.01 ticks, spreads of at least 0.02 or 5%.
    """
    return _realistic_chain(seed, steps, surface).copy()
