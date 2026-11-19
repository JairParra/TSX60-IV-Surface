"""Offline tests for the Bank of Canada curve (no network)."""

import io
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from tsx60_iv import rates
from tsx60_iv.rates import RateCurve, curve_from_observations, to_zero_rate
from tsx60_iv.synthetic import BOC_CURVE_PATH

ROWS = [
    {
        "d": "2025-01-01",
        "V80691342": {"v": "3.19"},
        "V80691344": {"v": "3.16"},
        "V80691345": {"v": "3.09"},
        "V80691346": {"v": "3.03"},
    },
    {"d": "2025-01-02", "AVG.INTWO": {"v": "3.30"}, "BD.CDN.2YR.DQ.YLD": {"v": "2.93"}},
    {"d": "2025-01-03", "AVG.INTWO": {"v": "3.29"}},  # after as_of: must be ignored
]


def test_money_market_yield_converts_to_continuous_rate() -> None:
    """A 3-month simple 4% yield has DF = 1/(1 + 0.04 * 0.25)."""
    r = to_zero_rate(4.0, 0.25, "money")
    assert np.exp(-r * 0.25) == pytest.approx(1 / (1 + 0.04 * 0.25), rel=1e-12)


def test_par_yield_converts_semiannual_to_continuous() -> None:
    """A 3% semi-annual yield is 2 * ln(1.015) continuously compounded."""
    assert to_zero_rate(3.0, 2.0, "par") == pytest.approx(2 * np.log(1.015))
    with pytest.raises(ValueError, match="convention"):
        to_zero_rate(3.0, 1.0, "act")


def test_curve_interpolates_and_extrapolates_flat() -> None:
    """Linear between nodes, constant outside them; arrays work."""
    curve = RateCurve("2025-01-02", (0.25, 1.0), (0.03, 0.02), "test")
    assert float(curve(0.625)) == pytest.approx(0.025)
    assert float(curve(0.01)) == 0.03 and float(curve(5.0)) == 0.02
    assert curve(np.array([0.25, 1.0])).tolist() == [0.03, 0.02]
    assert float(RateCurve.flat(0.0375)(0.7)) == 0.0375


@pytest.mark.parametrize(
    ("tenors", "zero_rates"),
    [
        ((), ()),
        ((0.5, 0.25), (0.03, 0.03)),
        ((0.25,), (0.03, 0.02)),
        ((0.25,), (np.nan,)),
    ],
)
def test_curve_validates_nodes(tenors: tuple, zero_rates: tuple) -> None:
    """Empty, unsorted, mismatched or non-finite nodes are rejected."""
    with pytest.raises(ValueError):
        RateCurve("x", tenors, zero_rates, "test")


def test_observations_use_last_print_on_or_before_as_of() -> None:
    """Weekly bills come from 1 Jan, daily series from 2 Jan; later rows are ignored."""
    curve = curve_from_observations(date(2025, 1, 2), ROWS)
    assert curve.observation_dates == (
        "2025-01-02",
        "2025-01-01",
        "2025-01-01",
        "2025-01-01",
        "2025-01-01",
        "2025-01-02",
    )
    assert float(curve(1 / 365)) == pytest.approx(to_zero_rate(3.30, 1 / 365, "money"))


def test_missing_or_stale_observations_raise() -> None:
    """No silent carry-forward of old prints."""
    with pytest.raises(ValueError, match="on or before"):
        curve_from_observations(date(2024, 12, 1), ROWS)
    with pytest.raises(ValueError, match="too old"):
        curve_from_observations(date(2025, 2, 1), ROWS)


def test_snapshot_round_trip(tmp_path: Path) -> None:
    """to_json then from_json restores the curve."""
    curve = curve_from_observations(date(2025, 1, 2), ROWS)
    curve.to_json(tmp_path / "c.json")
    assert RateCurve.from_json(tmp_path / "c.json") == curve


def test_bundled_snapshot_is_a_plausible_cad_curve() -> None:
    """The checked-in 2025-01-02 curve: six nodes, rates 2-4%, no future prints."""
    curve = RateCurve.from_json(BOC_CURVE_PATH)
    assert len(curve.tenors) == 6 and curve.as_of == "2025-01-02"
    assert all(0.02 < r < 0.04 for r in curve.zero_rates)
    assert all(d <= curve.as_of for d in curve.observation_dates)


def test_fetch_makes_one_request_for_all_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A snapshot costs one Valet request; the URL names every series and the date."""
    urls: list[str] = []

    def fake_urlopen(request: object, timeout: float) -> io.StringIO:
        urls.append(request.full_url)  # type: ignore[attr-defined]
        return io.StringIO(json.dumps({"observations": ROWS}))

    monkeypatch.setattr(rates.urllib.request, "urlopen", fake_urlopen)
    curve = rates.fetch_boc_curve(date(2025, 1, 2))
    assert len(urls) == 1
    assert all(series in urls[0] for series, *_ in rates.NODES)
    assert "end_date=2025-01-02" in urls[0]
    assert len(curve.tenors) == 6
