"""Bank of Canada risk-free curve: per-expiry continuously compounded rates.

The curve comes from the Bank of Canada Valet API (public open data, one
request per snapshot, never called by tests). Snapshots are stored as JSON so
every run is reproducible offline.
"""

import argparse
import json
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

VALET_URL = "https://www.bankofcanada.ca/valet/observations/{series}/json"
TERMS_URL = "https://www.bankofcanada.ca/terms/"
MAX_STALENESS_DAYS = 10  # weekly T-bill series: the last print can be a week old
# (series id, label, tenor in years, quote convention)
NODES: tuple[tuple[str, str, float, str], ...] = (
    ("AVG.INTWO", "CORRA overnight", 1 / 365, "money"),
    ("V80691342", "1 month T-bill", 1 / 12, "money"),
    ("V80691344", "3 month T-bill", 3 / 12, "money"),
    ("V80691345", "6 month T-bill", 6 / 12, "money"),
    ("V80691346", "1 year T-bill", 1.0, "money"),
    ("BD.CDN.2YR.DQ.YLD", "2 year benchmark bond", 2.0, "par"),
)


@dataclass(frozen=True)
class RateCurve:
    """Continuously compounded zero rates by tenor; flat beyond the end nodes."""

    as_of: str
    tenors: tuple[float, ...]
    zero_rates: tuple[float, ...]
    source: str
    observation_dates: tuple[str, ...] = ()
    quoted_yields_pct: tuple[float, ...] = ()  # raw prints, for audit

    def __post_init__(self) -> None:
        """Validate the nodes."""
        tenors, rates = np.array(self.tenors), np.array(self.zero_rates)
        if tenors.size == 0 or tenors.shape != rates.shape:
            raise ValueError("tenors and zero_rates must be equal-length and nonempty")
        if not (np.all(np.isfinite(tenors)) and np.all(np.isfinite(rates))):
            raise ValueError("tenors and zero_rates must be finite")
        if np.any(tenors <= 0) or np.any(np.diff(tenors) <= 0):
            raise ValueError("tenors must be positive and strictly increasing")

    def __call__(self, time: Any) -> Any:
        """Zero rate at ``time`` years (scalar or array); flat extrapolation."""
        return np.interp(time, self.tenors, self.zero_rates)

    @classmethod
    def flat(cls, rate: float, as_of: str = "") -> "RateCurve":
        """Return a constant-rate curve (the previous scalar behaviour)."""
        return cls(as_of, (1.0,), (float(rate),), "flat")

    def to_json(self, path: Path) -> None:
        """Write the snapshot."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self) | {"terms": TERMS_URL}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "RateCurve":
        """Load a snapshot written by ``to_json``."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            raw["as_of"],
            tuple(raw["tenors"]),
            tuple(raw["zero_rates"]),
            raw["source"],
            tuple(raw.get("observation_dates", ())),
            tuple(raw.get("quoted_yields_pct", ())),
        )


def to_zero_rate(percent_yield: float, tenor: float, convention: str) -> float:
    """Convert a quoted money-market yield in percent to a continuous rate.

    ``money``: simple annualized yield, ``DF = 1 / (1 + y * tenor)``. Par yields
    need the shorter nodes and go through ``bootstrap_par_zero``.
    """
    if convention != "money":
        raise ValueError(f"unknown convention {convention!r}")
    return float(np.log1p(percent_yield / 100.0 * tenor) / tenor)


def bootstrap_par_zero(
    percent_par: float,
    maturity: float,
    known_tenors: tuple[float, ...],
    known_rates: tuple[float, ...],
) -> float:
    """Zero rate at ``maturity`` that prices a semi-annual par bond at par.

    Coupon dates every half year; their zero rates come from the known nodes
    (linear, flat before the first), and rates between the last known tenor
    and ``maturity`` are linear in tenor toward the unknown rate.
    """
    coupon = percent_par / 100.0 / 2.0
    dates = np.arange(0.5, maturity + 1e-9, 0.5)
    last = known_tenors[-1]

    def zero_at(t: float, rate: float) -> float:
        if t <= last:
            return float(np.interp(t, known_tenors, known_rates))
        return known_rates[-1] + (rate - known_rates[-1]) * (t - last) / (
            maturity - last
        )

    def pricing_gap(rate: float) -> float:
        value = sum(coupon * np.exp(-zero_at(t, rate) * t) for t in dates[:-1])
        value += (1.0 + coupon) * np.exp(-rate * maturity)
        return float(value - 1.0)

    return float(brentq(pricing_gap, -0.05, 0.25))


def curve_from_observations(
    as_of: date, observations: list[dict[str, Any]]
) -> RateCurve:
    """Build a curve from Valet rows, using the last print on or before ``as_of``."""
    rates: list[float] = []
    stamps: list[str] = []
    quoted: list[float] = []
    for series, label, tenor, convention in NODES:
        rows = [
            (row["d"], row[series]["v"])
            for row in observations
            if series in row and date.fromisoformat(row["d"]) <= as_of
        ]
        if not rows:
            raise ValueError(f"no {label} observation on or before {as_of}")
        stamp, value = max(rows)
        if (as_of - date.fromisoformat(stamp)).days > MAX_STALENESS_DAYS:
            raise ValueError(f"{label} observation {stamp} is too old for {as_of}")
        if convention == "par":
            known = tuple(node[2] for node in NODES[: len(rates)])
            rates.append(bootstrap_par_zero(float(value), tenor, known, tuple(rates)))
        else:
            rates.append(to_zero_rate(float(value), tenor, convention))
        stamps.append(stamp)
        quoted.append(float(value))
    return RateCurve(
        as_of.isoformat(),
        tuple(node[2] for node in NODES),
        tuple(rates),
        "Bank of Canada Valet API",
        tuple(stamps),
        tuple(quoted),
    )


def fetch_boc_curve(as_of: date, timeout: float = 20.0) -> RateCurve:
    """Download the curve with a single Valet request (explicit network call)."""
    query = urllib.parse.urlencode(
        {
            "start_date": (as_of - timedelta(days=MAX_STALENESS_DAYS + 4)).isoformat(),
            "end_date": as_of.isoformat(),
        }
    )
    url = VALET_URL.format(series=",".join(node[0] for node in NODES)) + "?" + query
    request = urllib.request.Request(url, headers={"User-Agent": "tsx60-iv-surface"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.load(response)
    return curve_from_observations(as_of, payload["observations"])


def main() -> None:
    """Fetch one Bank of Canada snapshot and write it as JSON."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    curve = fetch_boc_curve(args.as_of)
    curve.to_json(args.out)
    pairs = zip(curve.tenors, curve.zero_rates, strict=True)
    print(f"wrote {args.out}")
    for tenor, rate in pairs:
        print(f"  {tenor:7.4f}y  {100 * rate:.4f}%")


if __name__ == "__main__":
    main()
