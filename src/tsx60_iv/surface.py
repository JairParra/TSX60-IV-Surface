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

from tsx60_iv.bsm import implied_volatility
from tsx60_iv.data import estimate_forward_and_rates, fetch_xiu_chain
from tsx60_iv.svi import (
    calibrate_svi_slice,
    check_calendar_arbitrage,
    durrleman_g,
    raw_svi_variance,
)
from tsx60_iv.synthetic import DEMO_DAYS, RATE, SPOT, generate_xiu_sample_chain

SCHEMA_VERSION = 1
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


def _otm_implied_vols(
    clean: pd.DataFrame, min_mid: float
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Invert out-of-the-money mids row by row, counting rows that cannot be."""
    kind = clean["option_type"].str.lower()
    is_otm = ((kind == "put") & (clean["strike"] < clean["forward"])) | (
        (kind == "call") & (clean["strike"] >= clean["forward"])
    )
    otm = clean.loc[is_otm & (clean["mid"] >= min_mid)].copy()
    counts = {
        "below_min_mid": int((is_otm & (clean["mid"] < min_mid)).sum()),
        "arbitrage_bounds": 0,
        "solver_bracket": 0,
    }
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
                        row.option_type.lower(),
                    )
                )
            )
        except ValueError as exc:
            counts[
                "arbitrage_bounds" if "arbitrage" in str(exc) else "solver_bracket"
            ] += 1
            vols.append(np.nan)
    otm["iv"] = vols
    return otm.dropna(subset=["iv"]), counts


def _fit_slice(expiration: pd.Timestamp, g: pd.DataFrame) -> dict[str, Any]:
    """Fit one expiry and precompute everything the page draws."""
    g = g.sort_values("log_moneyness")
    t = float(g["time_to_expiry"].iloc[0])
    k = g["log_moneyness"].to_numpy()
    params = calibrate_svi_slice(k, (g["iv"] ** 2 * t).to_numpy())
    pad = 0.1 * float(np.ptp(k))
    grid = np.linspace(k.min() - pad, k.max() + pad, 81)
    w = raw_svi_variance(grid, *params)
    fit_iv = np.sqrt(raw_svi_variance(k, *params) / t)
    return {
        "expiration": expiration.date().isoformat(),
        "T": t,
        "days": round(t * 365),
        "forward": float(g["forward"].iloc[0]),
        "discount": float(g["discount"].iloc[0]),
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
    rate: float,
    *,
    ticker: str,
    source: str,
    synthetic: bool,
    min_mid: float = MIN_MID,
) -> dict[str, Any]:
    """Build the surface document; rejected expiries are reported, never repaired."""
    clean = estimate_forward_and_rates(chain, spot, rate)
    otm, counts = _otm_implied_vols(clean, min_mid)
    slices: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for expiration in sorted(clean["expiration"].unique()):
        g = otm.loc[otm["expiration"] == expiration]
        stamp = pd.Timestamp(expiration)
        if len(g) < MIN_POINTS:
            reason = f"only {len(g)} usable quotes (need {MIN_POINTS})"
        else:
            try:
                slices.append(_fit_slice(stamp, g))
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
            "as_of": pd.Timestamp(chain["as_of"].iloc[0]).date().isoformat(),
            "ticker": ticker,
            "source": source,
            "synthetic": synthetic,
            "spot": float(spot),
            "rate": float(rate),
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
    parser.add_argument("--rate", type=float, default=RATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    args = parser.parse_args()

    if args.demo:
        chain, spot = generate_xiu_sample_chain(DEMO_DAYS), SPOT
        source, synthetic = "synthetic_demo", True
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

    surface = build_surface(
        chain,
        spot,
        args.rate,
        ticker=args.ticker if not args.demo else "XIU (synthetic)",
        source=source,
        synthetic=synthetic,
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
                "rate": args.rate,
            },
            {
                "slices": float(len(surface["expiries"])),
                "rejected": float(len(surface["rejected"])),
            },
            [str(args.out)],
        )


if __name__ == "__main__":
    main()
