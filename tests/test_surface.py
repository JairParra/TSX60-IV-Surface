"""Offline tests for the chain -> surface pipeline."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas.testing as pdt

from tsx60_iv.surface import SUBSTEPS, build_surface, write_surface
from tsx60_iv.synthetic import DEMO_DAYS, RATE, SPOT, generate_xiu_sample_chain

FIXTURE = Path(__file__).parent / "fixtures" / "xiu_sample_chain.parquet"


def _build(chain: pd.DataFrame) -> dict[str, Any]:
    return build_surface(
        chain,
        SPOT,
        RATE,
        ticker="XIU (synthetic)",
        source="synthetic_demo",
        synthetic=True,
    )


def test_fixture_matches_generator() -> None:
    """The checked-in fixture is exactly what the generator produces."""
    pdt.assert_frame_equal(generate_xiu_sample_chain(), pd.read_parquet(FIXTURE))


def test_demo_surface_schema_and_validity(tmp_path: Path) -> None:
    """Every demo expiry fits, passes the checks, and the JSON is strict."""
    surface = _build(generate_xiu_sample_chain(DEMO_DAYS))
    assert surface["meta"]["synthetic"] is True
    assert len(surface["expiries"]) == len(DEMO_DAYS)
    assert surface["rejected"] == []
    assert surface["calendar"]["ok"] is True
    for sl in surface["expiries"]:
        assert sl["min_g"] >= 0
        assert sl["rmse_vol_pts"] < 0.5
        assert len(sl["k_grid"]) == len(sl["iv"]) == len(sl["g"]) == 81
    grid = surface["grid"]
    rows = (len(DEMO_DAYS) - 1) * SUBSTEPS + 1
    assert len(grid["iv"]) == len(grid["days"]) == len(grid["T"]) == rows
    assert sum(grid["fit"]) == len(DEMO_DAYS)
    assert grid["days"] == sorted(grid["days"])
    assert all(len(row) == len(grid["k"]) for row in grid["iv"])
    out = tmp_path / "surface.json"
    write_surface(surface, out)  # allow_nan=False: raises on any NaN
    assert json.loads(out.read_text())["meta"]["spot"] == SPOT


def test_bad_quote_is_skipped_not_fatal() -> None:
    """One arbitrage-violating quote is counted and the rest still fit."""
    chain = generate_xiu_sample_chain(DEMO_DAYS)
    clean = _build(chain)
    # An OTM put priced above its discounted strike breaks the upper bound.
    mask = (
        (chain["option_type"] == "put")
        & (chain["strike"] == 36.0)
        & (chain["expiration"] == chain["expiration"].iloc[0])
    )
    chain.loc[mask, ["bid", "ask"]] = [40.0, 40.1]
    broken = _build(chain)
    assert broken["meta"]["dropped"]["arbitrage_bounds"] == 1
    assert broken["meta"]["iv_points_used"] == clean["meta"]["iv_points_used"] - 1
    assert len(broken["expiries"]) == len(DEMO_DAYS)


def test_expiry_with_too_few_points_is_reported() -> None:
    """Short fixture expiries lack usable quotes and are listed, not repaired."""
    surface = _build(pd.read_parquet(FIXTURE))
    reasons = {item["expiration"]: item["reason"] for item in surface["rejected"]}
    assert "usable quotes" in reasons["2025-01-16"]
    assert all(np.isfinite(sl["rmse_vol_pts"]) for sl in surface["expiries"])


def test_mesh_total_variance_never_falls_with_maturity() -> None:
    """Interpolated mesh rows keep w = iv^2 * T nondecreasing in T."""
    grid = _build(generate_xiu_sample_chain(DEMO_DAYS))["grid"]
    iv = np.array(grid["iv"], dtype=float)  # None becomes NaN
    w = iv**2 * np.array(grid["T"])[:, None]
    steps = np.diff(w, axis=0)
    assert np.nanmin(steps) >= -1e-12
