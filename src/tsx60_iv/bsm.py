"""Forward Black-Scholes-Merton pricing, Greeks, and volatility inversion."""

from collections.abc import Mapping
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import ndtr

OptionType = Literal["call", "put"]
FloatArray = NDArray[np.float64]


def _inputs(
    forward: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike,
    volatility: ArrayLike,
) -> tuple[FloatArray, ...]:
    """Broadcast and validate BSM inputs."""
    values = np.broadcast_arrays(
        *(np.asarray(value, dtype=float) for value in (
            forward,
            strike,
            time_to_expiry,
            discount,
            volatility,
        ))
    )
    forward_array, strike_array, time_array, discount_array, vol_array = values
    if np.any(forward_array <= 0) or np.any(strike_array <= 0):
        raise ValueError("forward and strike must be positive")
    if np.any(time_array <= 0) or np.any(discount_array <= 0):
        raise ValueError("time_to_expiry and discount must be positive")
    if np.any(vol_array < 0):
        raise ValueError("volatility must be nonnegative")
    return values


def bsm_price(
    forward: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType = "call",
) -> FloatArray | np.float64:
    """Price European options under forward BSM.

    Parameters
    ----------
    forward
        Forward price for the expiry, ``F(T)``.
    strike
        Option strike.
    time_to_expiry
        Time to expiry in years.
    discount
        Risk-free discount factor ``D(T)``.
    volatility
        Annualized implied volatility.
    option_type
        ``"call"`` or ``"put"``.

    Returns
    -------
    numpy.ndarray or numpy.float64
        Option prices, broadcast to the input shape.
    """
    f, k, time, disc, vol = _inputs(
        forward, strike, time_to_expiry, discount, volatility
    )
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_variance = vol * np.sqrt(time)
        d1 = np.log(f / k) / sqrt_variance + 0.5 * sqrt_variance
        d2 = d1 - sqrt_variance
    if option_type == "call":
        price = disc * (f * ndtr(d1) - k * ndtr(d2))
        intrinsic = disc * np.maximum(f - k, 0.0)
    else:
        price = disc * (k * ndtr(-d2) - f * ndtr(-d1))
        intrinsic = disc * np.maximum(k - f, 0.0)
    price = np.where(vol == 0, intrinsic, price)
    return price.item() if price.ndim == 0 else price


def bsm_greeks(
    forward: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType = "call",
) -> dict[str, FloatArray | np.float64]:
    """Return analytical delta, gamma, vega, vanna, and volga.

    Delta, gamma, vanna, and volga are forward-space derivatives. Vega and
    volga are derivatives per unit volatility, rather than per volatility
    percentage point.
    """
    f, k, time, disc, vol = _inputs(
        forward, strike, time_to_expiry, discount, volatility
    )
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if np.any(vol == 0):
        raise ValueError("Greeks are undefined at zero volatility")
    root_t = np.sqrt(time)
    sqrt_variance = vol * root_t
    d1 = np.log(f / k) / sqrt_variance + 0.5 * sqrt_variance
    d2 = d1 - sqrt_variance
    density = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    sign = 1.0 if option_type == "call" else -1.0
    delta = disc * (ndtr(d1) if sign > 0 else ndtr(d1) - 1.0)
    gamma = disc * density / (f * sqrt_variance)
    vega = disc * f * density * root_t
    vanna = -disc * density * d2 / vol
    volga = vega * d1 * d2 / vol
    result = {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "vanna": vanna,
        "volga": volga,
    }
    return {
        name: value.item() if np.ndim(value) == 0 else value
        for name, value in result.items()
    }


def implied_volatility(
    price: ArrayLike,
    forward: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike,
    option_type: OptionType = "call",
) -> FloatArray | np.float64:
    """Invert BSM prices using Brent's method with arbitrage validation.

    Arrays are broadcast together. Prices at exact intrinsic boundaries are
    rejected because they have no finite, positive-volatility solution.
    """
    market, f, k, time, disc = np.broadcast_arrays(
        *(np.asarray(value, dtype=float) for value in (
            price,
            forward,
            strike,
            time_to_expiry,
            discount,
        ))
    )
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if (
        np.any(f <= 0)
        or np.any(k <= 0)
        or np.any(time <= 0)
        or np.any(disc <= 0)
        or np.any(disc > 1)
    ):
        raise ValueError("forward, strike, time and discount must be valid")
    intrinsic = disc * (
        np.maximum(f - k, 0.0)
        if option_type == "call"
        else np.maximum(k - f, 0.0)
    )
    upper = disc * (f if option_type == "call" else k)
    valid = (market > intrinsic) & (market < upper) & np.isfinite(market)
    if not np.all(valid):
        raise ValueError("price violates intrinsic-value arbitrage bounds")

    flat_market, flat_f, flat_k, flat_t, flat_d = (
        value.ravel() for value in (market, f, k, time, disc)
    )
    roots = np.empty(flat_market.size)
    for index, (market_price, fwd, strike_price, expiry, df) in enumerate(
        zip(flat_market, flat_f, flat_k, flat_t, flat_d, strict=True)
    ):
        roots[index] = brentq(
            lambda sigma: float(
                bsm_price(fwd, strike_price, expiry, df, sigma, option_type)
            )
            - market_price,
            1e-5,
            5.0,
        )
    roots = roots.reshape(market.shape)
    return roots.item() if roots.ndim == 0 else roots
