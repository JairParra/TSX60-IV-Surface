"""American-exercise pricing (CRR binomial) and the early-exercise bias check.

XIU options are American-style while ``bsm.py`` prices European options. This
module quantifies how much that matters and can invert American prices.

Dividends enter as a continuous yield ``q``. For an expiry with parity forward
``F`` the matching yield is ``q = r - ln(F / S) / T``, so the tree's forward
equals ``F`` exactly.
# shortcut: continuous yield stands in for discrete quarterly dividends, so
# the premium of calls just before an ex-date is understated; upgrade to a
# discrete-dividend tree if that matters.
"""

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from tsx60_iv.bsm import bsm_greeks

OptionType = Literal["call", "put"]
DEFAULT_STEPS = 200


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
            value = np.maximum(value, sign * (spot * nodes - strike))
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
) -> Any:
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    _validate(spot, strike, time, rate, q, vol)
    scalar = np.ndim(strike) == 0 and np.ndim(vol) == 0
    strike_arr, vol_arr = np.broadcast_arrays(
        np.atleast_1d(np.asarray(strike, float)), np.atleast_1d(np.asarray(vol, float))
    )
    args = (spot, strike_arr, time, rate, q, vol_arr, option_type == "call")
    # Averaging n and n+1 steps damps the odd-even oscillation of CRR trees.
    value = 0.5 * (_tree(*args, steps, american) + _tree(*args, steps + 1, american))
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
) -> Any:
    """American value from a CRR tree; ``strike`` and ``vol`` may be arrays."""
    return _price(spot, strike, time, rate, div_yield, vol, option_type, steps, True)


def european_tree_price(
    spot: float,
    strike: Any,
    time: float,
    rate: float,
    div_yield: float,
    vol: Any,
    option_type: OptionType = "call",
    steps: int = DEFAULT_STEPS,
) -> Any:
    """European value from the same tree (to isolate the exercise premium)."""
    return _price(spot, strike, time, rate, div_yield, vol, option_type, steps, False)


def american_implied_vols(
    prices: Any,
    spot: float,
    strikes: Any,
    time: float,
    rate: float,
    div_yield: float,
    option_type: OptionType,
    steps: int = DEFAULT_STEPS,
) -> NDArray[np.float64]:
    """Invert American prices of one expiry by vectorized bisection.

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
    floor = 1e-4 + 2.0 * abs(rate - div_yield) * float(np.sqrt(time / steps))
    low, high = np.full(k.shape, floor), np.full(k.shape, 5.0)

    def gap(vol: NDArray[np.float64]) -> NDArray[np.float64]:
        value = american_price(spot, k, time, rate, div_yield, vol, option_type, steps)
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
            price, spot, strike, time, rate, div_yield, option_type, steps
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
) -> tuple[Any, Any]:
    """Return ``(premium, vol_bias)`` of treating an American quote as European.

    ``premium`` is the American minus European tree value at ``european_iv``.
    ``vol_bias`` is the first-order volatility overstatement ``premium / vega``,
    in volatility units (0.01 is one vol point). ``strike`` and ``european_iv``
    may be arrays for one expiry.
    """
    q = implied_dividend_yield(spot, forward, time, rate)
    kwargs = (spot, strike, time, rate, q, european_iv, option_type, steps)
    premium = _price(*kwargs, True) - _price(*kwargs, False)
    discount = float(np.exp(-rate * time))
    vega = bsm_greeks(forward, strike, time, discount, european_iv, option_type)["vega"]
    return premium, premium / vega
