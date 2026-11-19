"""American-exercise pricing (CRR binomial) and the early-exercise bias check.

XIU options are American-style while ``bsm.py`` prices European options. This
module quantifies how much that matters and can invert American prices.

Every expiry has a parity forward ``F``, so the tree is anchored to it through
the implied continuous yield ``q = r - ln(F / S) / T``: the tree's forward
equals ``F`` exactly. Two ways to realise the dividends:

- continuous yield ``q`` (default, needs no schedule), or
- escrowed discrete cash dividends (``dividends=Dividends(...)``): the tree
  runs on the dividend-free price ``S* = S e^(-qT)`` and, at each node, early
  exercise is valued on ``S* + PV(remaining dividends)``. The schedule only
  times the dividends; its present value is scaled to the forward-implied
  total so the forward is unchanged. Calls are then worth most just before an
  ex-date, which a continuous yield spreads out.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from tsx60_iv.bsm import bsm_greeks

OptionType = Literal["call", "put"]
DEFAULT_STEPS = 200
MAX_SCALE = 2.0  # schedule and forward may disagree by at most this factor


@dataclass(frozen=True)
class Dividends:
    """Discrete cash dividends as ``(time in years, amount)`` pairs."""

    times: tuple[float, ...]
    amounts: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the schedule."""
        if len(self.times) != len(self.amounts):
            raise ValueError("times and amounts must have the same length")
        if any(t <= 0 for t in self.times) or any(a <= 0 for a in self.amounts):
            raise ValueError("dividend times and amounts must be positive")

    def pv(self, at: float, rate: float, until: float) -> float:
        """Value at ``at`` of dividends paid strictly after ``at`` up to ``until``."""
        return float(
            sum(
                amount * np.exp(-rate * (t - at))
                for t, amount in zip(self.times, self.amounts, strict=True)
                if at < t <= until
            )
        )


@lru_cache(maxsize=512)
def _pv_levels(
    dividends: Dividends, time: float, rate: float, steps: int
) -> NDArray[np.float64]:
    """PV at every tree level of the dividends still to be paid before expiry."""
    levels = np.arange(steps + 1) * time / steps
    times = np.array(dividends.times)
    amounts = np.array(dividends.amounts)
    owed = (levels[:, None] < times[None, :]) & (times[None, :] <= time)
    values = amounts[None, :] * np.exp(-rate * (times[None, :] - levels[:, None]))
    result = np.where(owed, values, 0.0).sum(axis=1)
    result.setflags(write=False)
    return result


def implied_dividend_yield(
    spot: float, forward: float, time_to_expiry: float, rate: float
) -> float:
    """Continuous yield ``q`` that makes ``S * exp((r - q) T)`` equal ``forward``."""
    if spot <= 0 or forward <= 0 or time_to_expiry <= 0:
        raise ValueError("spot, forward and time_to_expiry must be positive")
    return rate - float(np.log(forward / spot)) / time_to_expiry


def _validate(
    spot: Any, strike: Any, time: float, rate: float, q: float, vol: Any
) -> None:
    values = (*np.atleast_1d(strike), *np.atleast_1d(vol), spot, time, rate, q)
    if not np.all(np.isfinite(values)):
        raise ValueError("inputs must be finite")
    if np.any(np.atleast_1d(strike) <= 0) or spot <= 0 or time <= 0:
        raise ValueError("spot, strike and time must be positive")
    if np.any(np.atleast_1d(vol) <= 0):
        raise ValueError("vol must be positive")


def _escrow(
    dividends: Dividends | None,
    spot: float,
    time: float,
    rate: float,
    q: float,
    steps: int,
) -> tuple[float, float, NDArray[np.float64] | None]:
    """Return ``(tree_spot, tree_yield, pv_levels)`` for the chosen dividend model."""
    if dividends is None:
        return spot, q, None
    pv0 = dividends.pv(0.0, rate, time)
    star = spot * float(np.exp(-q * time))
    scale = (spot - star) / pv0 if pv0 > 0 else 0.0
    if not 1.0 / MAX_SCALE <= scale <= MAX_SCALE:
        return spot, q, None  # no dividend before expiry, or forward disagrees
    return star, 0.0, scale * _pv_levels(dividends, time, rate, steps)


def _tree(
    spot: float,
    strike: NDArray[np.float64],
    time: float,
    rate: float,
    q: float,
    vol: NDArray[np.float64],
    is_call: bool,
    steps: int,
    american: bool,
    pv_levels: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """CRR tree values at the root, vectorized over options sharing an expiry."""
    dt = time / steps
    log_u = (vol * np.sqrt(dt))[:, None]
    up, down = np.exp(log_u), np.exp(-log_u)
    p_up = (np.exp((rate - q) * dt) - down) / (up - down)
    if not np.all((p_up > 0.0) & (p_up < 1.0)):
        raise ValueError("tree probability outside (0, 1); increase steps")
    discount = np.exp(-rate * dt)
    sign = 1.0 if is_call else -1.0
    strike = strike[:, None]
    # ladder[:, n + k] = u**k, so the node prices of every level are strided slices.
    ladder = np.exp(np.arange(-steps, steps + 1)[None, :] * log_u)
    value = np.maximum(sign * (spot * ladder[:, 0::2] - strike), 0.0)
    for level in range(steps - 1, -1, -1):
        value = discount * (
            p_up * value[:, 1 : level + 2] + (1.0 - p_up) * value[:, : level + 1]
        )
        if american:
            nodes = ladder[:, steps - level : steps + level + 1 : 2]
            owed = 0.0 if pv_levels is None else pv_levels[level]
            value = np.maximum(value, sign * (spot * nodes + owed - strike))
    return value[:, 0]


def _price(
    spot: float,
    strike: Any,
    time: float,
    rate: float,
    q: float,
    vol: Any,
    option_type: OptionType,
    steps: int,
    american: bool,
    dividends: Dividends | None = None,
) -> Any:
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    _validate(spot, strike, time, rate, q, vol)
    scalar = np.ndim(strike) == 0 and np.ndim(vol) == 0
    strike_arr, vol_arr = np.broadcast_arrays(
        np.atleast_1d(np.asarray(strike, float)), np.atleast_1d(np.asarray(vol, float))
    )
    values = []
    # Averaging n and n+1 steps damps the odd-even oscillation of CRR trees.
    for n in (steps, steps + 1):
        tree_spot, tree_q, levels = _escrow(dividends, spot, time, rate, q, n)
        args = (tree_spot, strike_arr, time, rate, tree_q, vol_arr)
        values.append(_tree(*args, option_type == "call", n, american, levels))
    value = 0.5 * (values[0] + values[1])
    return float(value[0]) if scalar else value


def american_price(
    spot: float,
    strike: Any,
    time: float,
    rate: float,
    div_yield: float,
    vol: Any,
    option_type: OptionType = "call",
    steps: int = DEFAULT_STEPS,
    dividends: Dividends | None = None,
) -> Any:
    """American value from a CRR tree; ``strike`` and ``vol`` may be arrays."""
    return _price(
        spot, strike, time, rate, div_yield, vol, option_type, steps, True, dividends
    )


def european_tree_price(
    spot: float,
    strike: Any,
    time: float,
    rate: float,
    div_yield: float,
    vol: Any,
    option_type: OptionType = "call",
    steps: int = DEFAULT_STEPS,
    dividends: Dividends | None = None,
) -> Any:
    """European value from the same tree (to isolate the exercise premium)."""
    return _price(
        spot, strike, time, rate, div_yield, vol, option_type, steps, False, dividends
    )


def american_implied_vols(
    prices: Any,
    spot: float,
    strikes: Any,
    time: float,
    rate: float,
    div_yield: float,
    option_type: OptionType,
    steps: int = DEFAULT_STEPS,
    dividends: Dividends | None = None,
) -> NDArray[np.float64]:
    """Invert American prices of one expiry by safeguarded Newton iteration.

    Returns NaN where a price is at or beyond its arbitrage bounds (intrinsic
    value, or the spot for a call and the strike for a put) or where the
    volatility is outside the bracket [floor, 5]. The floor keeps the tree
    probabilities valid: ``vol * sqrt(dt) > |r - q| * dt``.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    price = np.atleast_1d(np.asarray(prices, dtype=float))
    strike = np.atleast_1d(np.asarray(strikes, dtype=float))
    _validate(spot, strike, time, rate, div_yield, 1.0)
    sign = 1.0 if option_type == "call" else -1.0
    intrinsic = np.maximum(sign * (spot - strike), 0.0)
    upper = spot if option_type == "call" else strike
    valid = np.isfinite(price) & (price > intrinsic) & (price < upper)
    roots = np.full(price.shape, np.nan)
    if not valid.any():
        return roots
    k, target = strike[valid], price[valid]
    drift = max(abs(rate - div_yield), abs(rate))  # either tree convention
    floor = 1e-4 + 2.0 * drift * float(np.sqrt(time / steps))
    low, high = np.full(k.shape, floor), np.full(k.shape, 5.0)

    def gap(vol: NDArray[np.float64]) -> NDArray[np.float64]:
        value = american_price(
            spot, k, time, rate, div_yield, vol, option_type, steps, dividends
        )
        return value - target

    bracketed = (gap(low) < 0.0) & (gap(high) > 0.0)
    forward = spot * np.exp((rate - div_yield) * time)
    discount = float(np.exp(-rate * time))
    vol = np.clip(np.full(k.shape, 0.2), low, high)
    # Safeguarded Newton: the European vega is a good enough slope, and any
    # step outside the bracket falls back to bisection. Price rises with vol.
    for _ in range(40):
        err = gap(vol)
        low, high = np.where(err < 0.0, vol, low), np.where(err > 0.0, vol, high)
        vega = bsm_greeks(forward, k, time, discount, vol, option_type)["vega"]
        step = vol - err / vega
        new = np.where((step > low) & (step < high), step, 0.5 * (low + high))
        done = np.max(np.abs(new - vol)) < 1e-9
        vol = new
        if done:
            break
    roots[valid] = np.where(bracketed, vol, np.nan)
    return roots


def american_implied_volatility(
    price: float,
    spot: float,
    strike: float,
    time: float,
    rate: float,
    div_yield: float,
    option_type: OptionType = "call",
    steps: int = DEFAULT_STEPS,
    dividends: Dividends | None = None,
) -> float:
    """Invert one American price; raises ``ValueError`` when there is no solution."""
    intrinsic = (
        max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    )
    upper = spot if option_type == "call" else strike
    if not (np.isfinite(price) and intrinsic < price < upper):
        raise ValueError("price violates intrinsic-value arbitrage bounds")
    vol = float(
        american_implied_vols(
            price, spot, strike, time, rate, div_yield, option_type, steps, dividends
        )[0]
    )
    if not np.isfinite(vol):
        raise ValueError("American implied volatility has no solution in [floor, 5.0]")
    return vol


def early_exercise_bias(
    european_iv: Any,
    spot: float,
    forward: float,
    strike: Any,
    time: float,
    rate: float,
    option_type: OptionType,
    steps: int = DEFAULT_STEPS,
    dividends: Dividends | None = None,
) -> tuple[Any, Any]:
    """Return ``(premium, vol_bias)`` of treating an American quote as European.

    ``premium`` is the American minus European tree value at ``european_iv``.
    ``vol_bias`` is the first-order volatility overstatement ``premium / vega``,
    in volatility units (0.01 is one vol point). ``strike`` and ``european_iv``
    may be arrays for one expiry.
    """
    q = implied_dividend_yield(spot, forward, time, rate)
    kwargs = (spot, strike, time, rate, q, european_iv, option_type, steps)
    premium = _price(*kwargs, True, dividends) - _price(*kwargs, False, dividends)
    discount = float(np.exp(-rate * time))
    vega = bsm_greeks(forward, strike, time, discount, european_iv, option_type)["vega"]
    return premium, premium / vega
