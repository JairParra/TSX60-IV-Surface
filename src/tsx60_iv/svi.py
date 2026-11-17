"""Interfaces for Raw SVI calibration and static arbitrage checks."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def raw_svi_variance(
    k: ArrayLike,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Evaluate Gatheral's Raw SVI total implied variance."""
    if b < 0 or abs(rho) >= 1 or sigma <= 0:
        raise ValueError("SVI requires b >= 0, |rho| < 1, and sigma > 0")
    if a + b * sigma * np.sqrt(1.0 - rho**2) < 0:
        raise ValueError("SVI parameterization has negative minimum variance")
    x = np.asarray(k, dtype=float)
    return a + b * (
        rho * (x - m) + np.sqrt((x - m) ** 2 + sigma**2)
    )


def check_butterfly_arbitrage(
    k: ArrayLike,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> NDArray[np.bool_]:
    """Check Durrleman's butterfly-arbitrage condition on a log-strike grid."""
    x = np.asarray(k, dtype=float)
    y = x - m
    root = np.sqrt(y**2 + sigma**2)
    variance = raw_svi_variance(x, a, b, rho, m, sigma)
    first = b * (rho + y / root)
    second = b * sigma**2 / root**3
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (
            (1 - x * first / (2 * variance)) ** 2
            - first**2 / 4 * (1 / variance + 0.25)
            + second / 2
        )
    return np.asarray(g >= 0)


def calibrate_svi_slice(
    k: ArrayLike,
    total_variance: ArrayLike,
    **kwargs: Any,
) -> NDArray[np.float64]:
    """Calibrate Raw SVI parameters (reserved for the next implementation)."""
    raise NotImplementedError("SVI slice calibration is not implemented yet")
