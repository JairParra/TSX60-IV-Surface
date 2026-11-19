"""Tests for Raw SVI evaluation and butterfly-arbitrage checks."""

import numpy as np
import pytest

from tsx60_iv.svi import (
    calibrate_svi_slice,
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    raw_svi_variance,
)

K = np.linspace(-1, 1, 201)


def test_arbitrage_free_parameters_pass() -> None:
    """A benign smile satisfies Durrleman's condition everywhere."""
    assert check_butterfly_arbitrage(K, 0.04, 0.1, -0.3, 0.0, 0.2).all()


def test_butterfly_arbitrage_is_detected() -> None:
    """A steep, narrow smile violates the condition."""
    assert not check_butterfly_arbitrage(K, 0.001, 1.5, 0.0, 0.0, 0.01).all()


@pytest.mark.parametrize(
    "params",
    [
        (0.04, -0.1, 0.0, 0.0, 0.2),
        (0.04, 0.1, 1.0, 0.0, 0.2),
        (0.04, 0.1, -1.0, 0.0, 0.2),
        (0.04, 0.1, 0.0, 0.0, 0.0),
        (-1.0, 0.1, 0.0, 0.0, 0.2),
    ],
)
def test_invalid_parameters_raise(params: tuple[float, ...]) -> None:
    """Invalid Raw SVI parameters are rejected."""
    with pytest.raises(ValueError):
        raw_svi_variance(K, *params)


def test_calibration_recovers_variance_curve() -> None:
    """Fitting exact SVI data reproduces the curve."""
    k = np.linspace(-1, 1, 41)
    truth = (0.04, 0.1, -0.3, 0.0, 0.2)
    w = raw_svi_variance(k, *truth)
    fitted = calibrate_svi_slice(k, w)
    assert np.max(np.abs(raw_svi_variance(k, *fitted) - w)) < 1e-6


def test_calibration_rejects_arbitrage() -> None:
    """Data from an arbitrageable smile is rejected, not clamped."""
    k = np.linspace(-1, 1, 41)
    w = raw_svi_variance(k, 0.001, 1.5, 0.0, 0.0, 0.01)
    with pytest.raises(ValueError, match="arbitrage"):
        calibrate_svi_slice(k, w)


@pytest.mark.parametrize(
    "w",
    [np.full(10, np.nan), np.full(10, -0.1), np.full(4, 0.04)],
)
def test_calibration_rejects_bad_input(w: np.ndarray) -> None:
    """NaN, nonpositive variance and too few points raise."""
    with pytest.raises(ValueError):
        calibrate_svi_slice(np.linspace(-1, 1, w.size), w)


def test_calendar_arbitrage_check() -> None:
    """Variance must not decrease with maturity."""
    early = (0.5, (0.02, 0.1, -0.3, 0.0, 0.2))
    late = (1.0, (0.04, 0.1, -0.3, 0.0, 0.2))
    assert check_calendar_arbitrage(K, [late, early]).all()
    assert not check_calendar_arbitrage(K, [(0.5, late[1]), (1.0, early[1])]).any()


def test_calibration_handles_small_short_expiry_variance() -> None:
    """Total variance near 1e-3 (a 14-day smile) must still converge."""
    k = np.linspace(-0.2, 0.2, 15)
    w = raw_svi_variance(k, 0.0008, 0.004, -0.4, 0.0, 0.1)
    fitted = calibrate_svi_slice(k, w)
    assert np.max(np.abs(raw_svi_variance(k, *fitted) - w)) < 1e-6
