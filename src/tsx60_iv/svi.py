"""Interfaces for Raw SVI calibration and static arbitrage checks."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize


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
    return a + b * (rho * (x - m) + np.sqrt((x - m) ** 2 + sigma**2))


def durrleman_g(
    k: ArrayLike,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Durrleman's ``g(k)``; the smile is butterfly-free where ``g >= 0``."""
    x = np.asarray(k, dtype=float)
    y = x - m
    root = np.sqrt(y**2 + sigma**2)
    variance = raw_svi_variance(x, a, b, rho, m, sigma)
    first = b * (rho + y / root)
    second = b * sigma**2 / root**3
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            (1 - x * first / (2 * variance)) ** 2
            - first**2 / 4 * (1 / variance + 0.25)
            + second / 2
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
    return np.asarray(durrleman_g(k, a, b, rho, m, sigma) >= 0)


def calibrate_svi_slice(
    k: ArrayLike,
    total_variance: ArrayLike,
) -> NDArray[np.float64]:
    """Fit Raw SVI ``[a, b, rho, m, sigma]`` to one expiry's total variance.

    Raises ``ValueError`` if the best fit has butterfly arbitrage; the fit is
    never clamped or penalized into validity.
    """
    x = np.asarray(k, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    if x.shape != w.shape or x.ndim != 1 or x.size < 5:
        raise ValueError("k and total_variance must be 1-D with at least 5 points")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(w)) and np.all(w > 0)):
        raise ValueError("k must be finite and total_variance finite and positive")

    scale = w.mean() ** 2  # keeps the objective O(1) so SLSQP's ftol is meaningful

    def objective(p: NDArray[np.float64]) -> float:
        a, b, rho, m, sigma = p
        fit = a + b * (rho * (x - m) + np.sqrt((x - m) ** 2 + sigma**2))
        return float(np.sum((fit - w) ** 2) / scale)

    def min_variance(p: NDArray[np.float64]) -> float:
        a, b, rho, _, sigma = p
        return float(a + b * sigma * np.sqrt(1.0 - rho**2))

    eps = 1e-6
    bounds = [(None, None), (0.0, None), (-1 + eps, 1 - eps), (None, None), (eps, None)]
    constraints = [{"type": "ineq", "fun": min_variance}]
    m0 = x[np.argmin(w)]
    b0 = max(np.ptp(w) / np.ptp(x), 1e-3)
    # shortcut: multi-start local SLSQP, not a global optimizer; upgrade if real
    # chains fail to converge.
    best = None
    for rho0 in (-0.5, 0.0, 0.5):
        guess = [0.5 * w.min(), b0, rho0, m0, 0.1]
        fit = minimize(
            objective,
            guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )
        if fit.success and (best is None or fit.fun < best.fun):
            best = fit
    if best is None:
        raise ValueError("SVI calibration did not converge")
    params = best.x
    grid = np.concatenate([x, np.linspace(x.min() - 0.5, x.max() + 0.5, 201)])
    if not check_butterfly_arbitrage(grid, *params).all():  # also checks min variance
        raise ValueError("fitted SVI slice has butterfly arbitrage")
    return params


def check_calendar_arbitrage(
    k: ArrayLike,
    slices: Sequence[tuple[float, Sequence[float]]],
) -> NDArray[np.bool_]:
    """Check ``w(k, T)`` is nondecreasing in ``T`` across ``(T, params)`` slices.

    Returns a boolean array of shape ``(len(slices) - 1, len(k))``.
    """
    ordered = sorted(slices, key=lambda item: item[0])
    variances = np.array([raw_svi_variance(k, *params) for _, params in ordered])
    return np.asarray(np.diff(variances, axis=0) >= 0)
