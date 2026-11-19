"""Offline tests for the chain -> surface pipeline on the realistic demo chain."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from tsx60_iv.rates import RateCurve
from tsx60_iv.surface import SUBSTEPS, build_surface, write_surface
from tsx60_iv.synthetic import (
    BOC_CURVE_PATH,
    DEFAULT_SURFACE,
    generate_realistic_chain,
    generate_xiu_sample_chain,
)

FIXTURE = Path(__file__).parent / "fixtures" / "xiu_sample_chain.parquet"
CURVE = RateCurve.from_json(BOC_CURVE_PATH)


def _build(chain: pd.DataFrame, exercise: str = "american") -> dict[str, Any]:
    return build_surface(
        chain,
        float(chain["spot"].iloc[0]),
        CURVE,
        ticker="XIU (synthetic)",
        source="synthetic_demo",
        synthetic=True,
        exercise=exercise,
    )


@pytest.fixture(scope="module")
def american() -> dict[str, Any]:
    """Demo surface inverted with the American tree (built once)."""
    return _build(generate_realistic_chain())


@pytest.fixture(scope="module")
def european() -> dict[str, Any]:
    """Demo surface inverted with forward BSM (built once)."""
    return _build(generate_realistic_chain(), "european")


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
    assert [r["expiration"] for r in american["rejected"]] == ["2025-01-17"]
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
    american: dict[str, Any], european: dict[str, Any]
) -> None:
    """Treating American quotes as European overstates ATM vol; the tree does not."""

    def mean_bias(surface: dict[str, Any]) -> float:
        errors = [
            np.sqrt(sl["atm_w"] / sl["T"]) - float(DEFAULT_SURFACE.vol(0.0, sl["T"]))
            for sl in surface["expiries"]
        ]
        return 100 * float(np.mean(errors))

    assert mean_bias(european) > mean_bias(american) + 0.02


def test_exercise_diagnostics_are_reported(american: dict[str, Any]) -> None:
    """Each slice carries the early-exercise bias; OTM quotes are barely affected."""
    for sl in american["expiries"]:
        diag = sl["american"]
        assert 0.0 <= diag["median_vol_bias_pts"] <= diag["max_vol_bias_pts"] < 1.0
        assert 0 <= diag["n_material"] <= sl["n_points"]


def test_bad_quote_is_skipped_not_fatal(american: dict[str, Any]) -> None:
    """One arbitrage-violating quote is counted and the rest still fit."""
    chain = generate_realistic_chain()
    mask = (
        (chain["option_type"] == "put")
        & (chain["strike"] == 35.0)
        & (chain["expiration"] == "2025-03-21")
    )
    assert mask.sum() == 1
    chain.loc[mask, ["bid", "ask"]] = [40.0, 40.1]
    broken = _build(chain)
    assert broken["meta"]["dropped"]["arbitrage_bounds"] == 1
    assert broken["meta"]["iv_points_used"] == american["meta"]["iv_points_used"] - 1
    assert len(broken["expiries"]) == len(american["expiries"])


def test_short_expiry_with_few_quotes_is_reported(american: dict[str, Any]) -> None:
    """The 15-day expiry has too few usable quotes and is listed, not repaired."""
    (item,) = american["rejected"]
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
