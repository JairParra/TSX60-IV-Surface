"""Offline tests for the chain -> surface pipeline on the realistic demo chain."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from tsx60_iv.data import estimate_forward_and_rates
from tsx60_iv.market_params import expected_dividends, load_params
from tsx60_iv.rates import RateCurve
from tsx60_iv.surface import SUBSTEPS, build_surface, write_surface
from tsx60_iv.synthetic import (
    BOC_CURVE_PATH,
    DEFAULT_SURFACE,
    MARKET_PARAMS_PATH,
    _forward_inputs,
    generate_realistic_chain,
    generate_xiu_sample_chain,
)

FIXTURE = Path(__file__).parent / "fixtures" / "xiu_sample_chain.parquet"
CURVE = RateCurve.from_json(BOC_CURVE_PATH)
PARAMS = load_params(MARKET_PARAMS_PATH)
SCHEDULE = expected_dividends(PARAMS)


def _build(
    chain: pd.DataFrame, exercise: str = "american", *, schedule: bool = True
) -> dict[str, Any]:
    return build_surface(
        chain,
        float(chain["spot"].iloc[0]),
        CURVE,
        ticker="XIU (synthetic)",
        source="synthetic_demo",
        synthetic=True,
        exercise=exercise,
        dividends=SCHEDULE if schedule else None,
    )


def _atm_errors(surface: dict[str, Any]) -> np.ndarray:
    """Fitted minus true ATM vol in vol points, per expiry."""
    return np.array(
        [
            100
            * (
                np.sqrt(sl["atm_w"] / sl["T"])
                - float(DEFAULT_SURFACE.vol(0.0, sl["T"]))
            )
            for sl in surface["expiries"]
        ]
    )


@pytest.fixture(scope="module")
def short_chain() -> pd.DataFrame:
    """Return the first six expiries only (keeps comparison tests quick)."""
    chain = generate_realistic_chain()
    first_six = sorted(chain["expiration"].unique())[:6]
    return chain[chain["expiration"].isin(first_six)]


@pytest.fixture(scope="module")
def american() -> dict[str, Any]:
    """Return the full demo surface inverted with the American tree."""
    return _build(generate_realistic_chain())


@pytest.fixture(scope="module")
def short_american(short_chain: pd.DataFrame) -> dict[str, Any]:
    """Return the short-chain surface with American inversion and dividends."""
    return _build(short_chain)


def test_fixture_matches_generator() -> None:
    """The checked-in fixture is exactly what the legacy generator produces."""
    pdt.assert_frame_equal(generate_xiu_sample_chain(), pd.read_parquet(FIXTURE))


def test_demo_surface_schema_and_validity(
    american: dict[str, Any], tmp_path: Path
) -> None:
    """Expiries fit and pass both checks, rates follow the curve, JSON is strict."""
    meta = american["meta"]
    assert meta["synthetic"] is True and meta["exercise"] == "american"
    assert meta["rate_curve"]["source"] == "Bank of Canada Valet API"
    assert len(american["expiries"]) >= 11
    rejected = {r["expiration"]: r["reason"] for r in american["rejected"]}
    assert "2025-01-17" in rejected  # too few quotes
    # The ex-date expiry (8 quotes) is fragile: SLSQP may reject it on some
    # platforms, and the pipeline then reports it instead of repairing it.
    assert set(rejected) <= {"2025-01-17", "2025-02-21"}, rejected
    assert american["calendar"]["ok"] is True
    for sl in american["expiries"]:
        assert sl["min_g"] >= 0
        assert sl["rmse_vol_pts"] < 0.5
        assert sl["rate"] == pytest.approx(float(CURVE(sl["T"])))
        assert 0.0 < sl["dividend_yield"] < 0.06
        assert len(sl["k_grid"]) == len(sl["iv"]) == len(sl["g"]) == 81
    grid = american["grid"]
    rows = (len(american["expiries"]) - 1) * SUBSTEPS + 1
    assert len(grid["iv"]) == len(grid["days"]) == len(grid["T"]) == rows
    assert sum(grid["fit"]) == len(american["expiries"])
    out = tmp_path / "surface.json"
    write_surface(american, out)  # allow_nan=False: raises on any NaN
    assert json.loads(out.read_text())["meta"]["spot"] == meta["spot"]


def test_fit_recovers_the_true_surface(american: dict[str, Any]) -> None:
    """ATM level and every inverted quote stay within a fraction of a vol point."""
    for sl in american["expiries"]:
        truth_atm = float(DEFAULT_SURFACE.vol(0.0, sl["T"]))
        assert 100 * abs(np.sqrt(sl["atm_w"] / sl["T"]) - truth_atm) < 0.15
        truth = DEFAULT_SURFACE.vol(np.array(sl["k_obs"]), sl["T"])
        assert 100 * np.max(np.abs(np.array(sl["iv_obs"]) - truth)) < 0.5


def test_american_inversion_removes_the_european_bias(
    short_chain: pd.DataFrame, short_american: dict[str, Any]
) -> None:
    """Treating American quotes as European overstates ATM vol; the tree does not."""
    european = _build(short_chain, "european")
    assert np.mean(_atm_errors(european)) > np.mean(_atm_errors(short_american)) + 0.02


def test_exercise_diagnostics_are_reported(american: dict[str, Any]) -> None:
    """Each slice carries the early-exercise bias; OTM quotes are barely affected."""
    for sl in american["expiries"]:
        diag = sl["american"]
        assert 0.0 <= diag["median_vol_bias_pts"] <= diag["max_vol_bias_pts"]
        assert 0 <= diag["n_material"] <= sl["n_points"]
    # Only expiries at an ex-date (21 Feb, and 21 Nov one day after 20 Nov) have
    # a large call premium; every other expiry stays within half a vol point.
    ex_date = {"2025-02-21", "2025-11-21"}
    bias = {
        s["expiration"]: s["american"]["max_vol_bias_pts"] for s in american["expiries"]
    }
    assert bias["2025-11-21"] > 0.9
    assert all(v < 0.6 for k, v in bias.items() if k not in ex_date)


def test_bad_quote_is_skipped_not_fatal(
    short_chain: pd.DataFrame, short_american: dict[str, Any]
) -> None:
    """One arbitrage-violating quote is counted and the rest still fit."""
    chain = short_chain.copy()
    mask = (
        (chain["option_type"] == "put")
        & (chain["strike"] == 35.0)
        & (chain["expiration"] == "2025-03-21")
    )
    assert mask.sum() == 1
    chain.loc[mask, ["bid", "ask"]] = [40.0, 40.1]
    broken = _build(chain)
    assert broken["meta"]["dropped"]["arbitrage_bounds"] == 1
    assert (
        broken["meta"]["iv_points_used"] == short_american["meta"]["iv_points_used"] - 1
    )
    assert len(broken["expiries"]) == len(short_american["expiries"])


def test_short_expiry_with_few_quotes_is_reported(american: dict[str, Any]) -> None:
    """The 15-day expiry has too few usable quotes and is listed, not repaired."""
    (item,) = (r for r in american["rejected"] if r["expiration"] == "2025-01-17")
    assert "usable quotes" in item["reason"]


def test_mesh_total_variance_never_falls_with_maturity(
    american: dict[str, Any],
) -> None:
    """Interpolated mesh rows keep w = iv^2 * T nondecreasing in T."""
    grid = american["grid"]
    iv = np.array(grid["iv"], dtype=float)  # None becomes NaN
    w = iv**2 * np.array(grid["T"])[:, None]
    assert np.nanmin(np.diff(w, axis=0)) >= -1e-12


def test_unknown_exercise_is_rejected() -> None:
    """Only european and american are valid."""
    with pytest.raises(ValueError, match="exercise"):
        _build(generate_realistic_chain(), "bermudan")


def test_discrete_dividends_remove_the_continuous_yield_bias(
    short_chain: pd.DataFrame, short_american: dict[str, Any]
) -> None:
    """With a discrete-dividend truth, a continuous yield overstates ATM vol."""
    discrete = short_american
    continuous = _build(short_chain, schedule=False)
    assert discrete["meta"]["dividends"] == "schedule"
    assert continuous["meta"]["dividends"] == "continuous"
    assert np.mean(np.abs(_atm_errors(discrete))) < 0.5 * np.mean(
        np.abs(_atm_errors(continuous))
    )


def test_parity_forward_is_corrected_for_early_exercise(
    short_chain: pd.DataFrame, short_american: dict[str, Any]
) -> None:
    """Raw parity on American mids misses the forward; the refined one does not."""
    surface = short_american
    spot = float(short_chain["spot"].iloc[0])
    raw = estimate_forward_and_rates(short_chain, spot, CURVE)
    raw_forward = raw.groupby("time_to_expiry")["forward"].first()
    refined_error, raw_error = [], []
    for sl in surface["expiries"]:
        _, true_forward, _ = _forward_inputs(spot, sl["T"], CURVE, SCHEDULE)
        refined_error.append(abs(sl["forward"] - true_forward) / true_forward)
        raw_error.append(abs(raw_forward[sl["T"]] - true_forward) / true_forward)
    assert max(refined_error) < 3e-4  # 0.03% of the forward
    assert max(raw_error) > 2 * max(refined_error)
