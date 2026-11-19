"""Chain -> implied vols -> per-expiry SVI -> validated surface JSON."""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tsx60_iv.american import american_implied_vols, early_exercise_bias
from tsx60_iv.bsm import implied_volatility
from tsx60_iv.data import estimate_forward_and_rates, fetch_xiu_chain
from tsx60_iv.rates import RateCurve
from tsx60_iv.svi import (
    calibrate_svi_slice,
    check_calendar_arbitrage,
    durrleman_g,
    raw_svi_variance,
)
from tsx60_iv.synthetic import BOC_CURVE_PATH, generate_realistic_chain

SCHEMA_VERSION = 2
DEFAULT_RATE = 0.0375  # flat CAD rate when no curve snapshot is given
EXERCISES = ("european", "american")
# shortcut: fixed 0.05 floor on mid; quotes on the tick floor give junk vols.
# Make it relative to tick size and vega when real data arrives.
MIN_MID = 0.05
MIN_POINTS = 5
FETCH_COOLDOWN_SECONDS = 30 * 60
DEFAULT_CHAIN = Path("data/xiu_chain.parquet")
DEFAULT_OUT = Path("data/surface.json")
PARAM_NAMES = ("a", "b", "rho", "m", "sigma")


def _floats(values: Any) -> list[float | None]:
    """Convert to JSON-safe floats; non-finite values become ``None``."""
    return [float(v) if np.isfinite(v) else None for v in np.asarray(values, float)]


def _european_vols(otm: pd.DataFrame, counts: dict[str, int]) -> np.ndarray:
    """Invert mids with forward BSM row by row, counting rows that fail."""
    vols: list[float] = []
    for row in otm.itertuples():
        try:
            vols.append(
                float(
                    implied_volatility(
                        row.mid,
                        row.forward,
                        row.strike,
                        row.time_to_expiry,
                        row.discount,
                        row.kind,
                    )
                )
            )
        except ValueError as exc:
            counts[
                "arbitrage_bounds" if "arbitrage" in str(exc) else "solver_bracket"
            ] += 1
            vols.append(np.nan)
    return np.array(vols)


def _american_vols(
    otm: pd.DataFrame, spot: float, counts: dict[str, int]
) -> np.ndarray:
    """Invert mids with a binomial tree per expiry and side, counting failures."""
    vols = np.full(len(otm), np.nan)
    for _, g in otm.assign(pos=np.arange(len(otm))).groupby(["expiration", "kind"]):
        first = g.iloc[0]
        q = first["rate"] - np.log(first["forward"] / spot) / first["time_to_expiry"]
        result = american_implied_vols(
            g["mid"].to_numpy(),
            spot,
            g["strike"].to_numpy(),
            first["time_to_expiry"],
            first["rate"],
            q,
            first["kind"],
        )
        vols[g["pos"].to_numpy()] = result
        is_call = first["kind"] == "call"
        intrinsic = np.maximum(
            (spot - g["strike"]) if is_call else (g["strike"] - spot), 0.0
        )
        upper = spot if is_call else g["strike"]
        outside = ((g["mid"] <= intrinsic) | (g["mid"] >= upper)).to_numpy()
        failed = np.isnan(result)
        counts["arbitrage_bounds"] += int((failed & outside).sum())
        counts["solver_bracket"] += int((failed & ~outside).sum())
    return vols


def _otm_implied_vols(
    clean: pd.DataFrame, min_mid: float, spot: float, exercise: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Invert out-of-the-money mids and measure the early-exercise bias."""
    kind = clean["option_type"].str.lower()
    is_otm = ((kind == "put") & (clean["strike"] < clean["forward"])) | (
        (kind == "call") & (clean["strike"] >= clean["forward"])
    )
    otm = clean.loc[is_otm & (clean["mid"] >= min_mid)].assign(kind=kind).copy()
    counts = {
        "below_min_mid": int((is_otm & (clean["mid"] < min_mid)).sum()),
        "arbitrage_bounds": 0,
        "solver_bracket": 0,
    }
    if exercise == "american":
        otm["iv"] = _american_vols(otm, spot, counts)
    else:
        otm["iv"] = _european_vols(otm, counts)
    otm = otm.dropna(subset=["iv"]).copy()
    # Early-exercise premium at the fitted vol, and its first-order vol bias.
    otm["premium"], otm["vol_bias"] = np.nan, np.nan
    for _, g in otm.groupby(["expiration", "kind"]):
        first = g.iloc[0]
        premium, bias = early_exercise_bias(
            g["iv"].to_numpy(),
            spot,
            first["forward"],
            g["strike"].to_numpy(),
            first["time_to_expiry"],
            first["rate"],
            first["kind"],
        )
        otm.loc[g.index, "premium"] = premium
        otm.loc[g.index, "vol_bias"] = bias
    return otm, counts


def _fit_slice(
    expiration: pd.Timestamp, g: pd.DataFrame, spot: float
) -> dict[str, Any]:
    """Fit one expiry and precompute everything the page draws."""
    g = g.sort_values("log_moneyness")
    t = float(g["time_to_expiry"].iloc[0])
    k = g["log_moneyness"].to_numpy()
    params = calibrate_svi_slice(k, (g["iv"] ** 2 * t).to_numpy())
    pad = 0.1 * float(np.ptp(k))
    grid = np.linspace(k.min() - pad, k.max() + pad, 81)
    w = raw_svi_variance(grid, *params)
    fit_iv = np.sqrt(raw_svi_variance(k, *params) / t)
    rate, forward = float(g["rate"].iloc[0]), float(g["forward"].iloc[0])
    half_spread = ((g["ask"] - g["bid"]) / 2.0).to_numpy()
    return {
        "expiration": expiration.date().isoformat(),
        "T": t,
        "days": round(t * 365),
        "forward": forward,
        "discount": float(g["discount"].iloc[0]),
        "rate": rate,
        "dividend_yield": float(rate - np.log(forward / spot) / t),
        "american": {
            "max_vol_bias_pts": float(100 * np.max(np.abs(g["vol_bias"]))),
            "median_vol_bias_pts": float(100 * np.median(np.abs(g["vol_bias"]))),
            "n_material": int((g["premium"].to_numpy() > half_spread).sum()),
        },
        "params": dict(zip(PARAM_NAMES, map(float, params), strict=True)),
        "n_points": len(g),
        "rmse_vol_pts": float(
            100 * np.sqrt(np.mean((fit_iv - g["iv"].to_numpy()) ** 2))
        ),
        "atm_w": float(raw_svi_variance(0.0, *params)),
        "k_obs": _floats(k),
        "iv_obs": _floats(g["iv"]),
        "strike_obs": _floats(g["strike"]),
        "bid_obs": _floats(g["bid"]),
        "ask_obs": _floats(g["ask"]),
        "k_grid": _floats(grid),
        "w": _floats(w),
        "iv": _floats(np.sqrt(w / t)),
        "g": _floats(durrleman_g(grid, *params)),
        "min_g": float(np.nanmin(durrleman_g(grid, *params))),
        "_k_range": (float(k.min()), float(k.max())),
    }


SUBSTEPS = 6  # interpolated mesh rows between two fitted expiries


def _mesh(slices: list[dict[str, Any]]) -> dict[str, Any]:
    """Mesh rows for the 3D plot: fitted expiries plus rows between them.

    Between expiries total variance is interpolated linearly in T at fixed k,
    which preserves calendar monotonicity. Cells outside the quoted strikes
    (of both neighbours, for an interpolated row) stay empty: no extrapolation.
    """
    grid: dict[str, Any] = {
        "k": [],
        "days": [],
        "T": [],
        "forward": [],
        "fit": [],
        "iv": [],
    }
    if not slices:
        return grid
    k = np.linspace(
        min(s["_k_range"][0] for s in slices),
        max(s["_k_range"][1] for s in slices),
        81,
    )
    grid["k"] = _floats(k)
    w = [raw_svi_variance(k, *s["params"].values()) for s in slices]
    inside = [(k >= s["_k_range"][0]) & (k <= s["_k_range"][1]) for s in slices]
    for i, sl in enumerate(slices):
        last = i + 1 == len(slices)
        for j in range(1 if last else SUBSTEPS):
            lam = 0.0 if last else j / SUBSTEPS
            nxt = i if last else i + 1
            t = (1 - lam) * sl["T"] + lam * slices[nxt]["T"]
            var = (1 - lam) * w[i] + lam * w[nxt]
            mask = inside[i] if lam == 0 else inside[i] & inside[nxt]
            grid["days"].append(t * 365)
            grid["T"].append(t)
            grid["forward"].append(
                (1 - lam) * sl["forward"] + lam * slices[nxt]["forward"]
            )
            grid["fit"].append(lam == 0)
            grid["iv"].append(_floats(np.where(mask, np.sqrt(var / t), np.nan)))
    return grid


def build_surface(
    chain: pd.DataFrame,
    spot: float,
    rate: float | RateCurve,
    *,
    ticker: str,
    source: str,
    synthetic: bool,
    min_mid: float = MIN_MID,
    exercise: str = "american",
) -> dict[str, Any]:
    """Build the surface document; rejected expiries are reported, never repaired."""
    if exercise not in EXERCISES:
        raise ValueError(f"exercise must be one of {EXERCISES}")
    as_of = pd.Timestamp(chain["as_of"].iloc[0]).date().isoformat()
    curve = rate if isinstance(rate, RateCurve) else RateCurve.flat(rate, as_of)
    clean = estimate_forward_and_rates(chain, spot, curve)
    otm, counts = _otm_implied_vols(clean, min_mid, spot, exercise)
    slices: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for expiration in sorted(clean["expiration"].unique()):
        g = otm.loc[otm["expiration"] == expiration]
        stamp = pd.Timestamp(expiration)
        if len(g) < MIN_POINTS:
            reason = f"only {len(g)} usable quotes (need {MIN_POINTS})"
        else:
            try:
                slices.append(_fit_slice(stamp, g, spot))
                continue
            except ValueError as exc:
                reason = str(exc)
        rejected.append({"expiration": stamp.date().isoformat(), "reason": reason})

    calendar: dict[str, Any] = {"ok": None, "violations": [], "note": ""}
    if len(slices) >= 2:
        low = max(s["_k_range"][0] for s in slices)
        high = min(s["_k_range"][1] for s in slices)
        if low < high:
            ok = check_calendar_arbitrage(
                np.linspace(low, high, 41),
                [(s["T"], list(s["params"].values())) for s in slices],
            )
            calendar["ok"] = bool(ok.all())
            calendar["violations"] = [
                [slices[i]["expiration"], slices[i + 1]["expiration"]]
                for i in np.flatnonzero(~ok.all(axis=1))
            ]
        else:
            calendar["note"] = "slices share no common log-moneyness range"

    grid = _mesh(slices)
    for s in slices:
        del s["_k_range"]

    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "as_of": as_of,
            "ticker": ticker,
            "source": source,
            "synthetic": synthetic,
            "spot": float(spot),
            "exercise": exercise,
            "rate_curve": {
                "source": curve.source,
                "as_of": curve.as_of,
                "tenors": list(curve.tenors),
                "zero_rates": list(curve.zero_rates),
            },
            "min_mid": min_mid,
            "quotes_raw": len(chain),
            "quotes_after_quality_filter": len(clean),
            "iv_points_used": int(sum(s["n_points"] for s in slices)),
            "dropped": counts,
        },
        "calendar": calendar,
        "rejected": rejected,
        "expiries": slices,
        "grid": grid,
    }


def write_surface(surface: dict[str, Any], out: Path) -> None:
    """Write atomically so a running API never reads half a file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(surface, allow_nan=False), encoding="utf-8")
    os.replace(tmp, out)


def main() -> None:
    """Build ``data/surface.json`` from the demo chain or a cached chain."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="synthetic offline chain")
    parser.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    parser.add_argument("--fetch", action="store_true", help="explicit live download")
    parser.add_argument("--force", action="store_true", help="ignore fetch cooldown")
    parser.add_argument("--ticker", default="XIU.TO")
    parser.add_argument("--max-expiries", type=int, default=6)
    parser.add_argument("--spot", type=float)
    parser.add_argument("--rate", type=float, help="flat CAD rate (default 3.75%%)")
    parser.add_argument("--curve", type=Path, help="Bank of Canada curve snapshot")
    parser.add_argument("--exercise", choices=EXERCISES, default="american")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    args = parser.parse_args()

    if args.rate is not None and args.curve is not None:
        raise SystemExit("pass either --rate or --curve, not both")
    if args.demo:
        chain = generate_realistic_chain()
        spot = float(chain["spot"].iloc[0])
        source, synthetic = "synthetic_demo", True
        curve_path = args.curve or (None if args.rate is not None else BOC_CURVE_PATH)
    else:
        if args.fetch:
            fresh = args.chain.exists() and (
                time.time() - args.chain.stat().st_mtime < FETCH_COOLDOWN_SECONDS
            )
            if fresh and not args.force:
                raise SystemExit(f"{args.chain} is under 30 min old; use --force")
            fetch_xiu_chain(args.ticker, args.chain, max_expiries=args.max_expiries)
        chain = pd.read_parquet(args.chain)
        spot = args.spot or (float(chain["spot"].iloc[0]) if "spot" in chain else None)
        if spot is None:
            raise SystemExit("chain has no spot column; pass --spot")
        source, synthetic = ("yahoo_live" if args.fetch else "cached_chain"), False
        curve_path = args.curve
    rate = (
        RateCurve.from_json(curve_path)
        if curve_path
        else (DEFAULT_RATE if args.rate is None else args.rate)
    )

    surface = build_surface(
        chain,
        spot,
        rate,
        ticker=args.ticker if not args.demo else "XIU (synthetic)",
        source=source,
        synthetic=synthetic,
        exercise=args.exercise,
    )
    write_surface(surface, args.out)
    meta = surface["meta"]
    print(
        f"wrote {args.out}: {len(surface['expiries'])} slices, "
        f"{len(surface['rejected'])} rejected, dropped={meta['dropped']}, "
        f"calendar_ok={surface['calendar']['ok']}"
    )
    for item in surface["rejected"]:
        print(f"  rejected {item['expiration']}: {item['reason']}")
    if args.mlflow:
        from tsx60_iv.tracking import log_surface_run

        log_surface_run(
            {
                "ticker": meta["ticker"],
                "as_of": meta["as_of"],
                "spot": spot,
                "rate_source": meta["rate_curve"]["source"],
                "exercise": meta["exercise"],
            },
            {
                "slices": float(len(surface["expiries"])),
                "rejected": float(len(surface["rejected"])),
            },
            [str(args.out)],
        )


if __name__ == "__main__":
    main()
