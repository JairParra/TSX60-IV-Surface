# TSX60 IV Surface

## Commands

- Install: `uv sync`
- Test: `uv run pytest`
- Lint: `uv run ruff check .`
- Demo surface: `uv run python -m tsx60_iv.surface --demo`, then `uv run uvicorn tsx60_iv.api:app` and open `http://127.0.0.1:8000/`
- Live fetch (explicit only): `uv run python -m tsx60_iv.surface --fetch` (Yahoo has no XIU.TO options)

## Layout

- `src/tsx60_iv/data.py`: `yfinance` ingestion, quote quality, forwards/rates.
- `src/tsx60_iv/bsm.py`: forward Black-Scholes pricing, Greeks, IV inversion.
- `src/tsx60_iv/svi.py`: SVI parameterization, slice calibration, arbitrage checks.
- `src/tsx60_iv/surface.py`: chain to implied vols to per-expiry SVI to `data/surface.json` (`python -m tsx60_iv.surface --demo`).
- `src/tsx60_iv/synthetic.py`: legacy fixture chain and the realistic demo chain (SSVI truth, American prices).
- `src/tsx60_iv/american.py`: CRR American pricer, tree inversion, early-exercise bias.
- `src/tsx60_iv/rates.py`: Bank of Canada curve (Valet snapshot), per-expiry zero rates.
- `src/tsx60_iv/market_params.py`: free Yahoo-derived anchors (spot, realized vol, dividends).
- `src/tsx60_iv/data_files/`: checked-in snapshots of the two anchors.
- `src/tsx60_iv/static/index.html`: 3D surface page served by the API at `/`.
- `src/tsx60_iv/tracking.py`: MLflow logging interface.
- `src/tsx60_iv/api.py`: FastAPI surface-serving interface.
- `tests/`: deterministic unit and property tests.

## Numerical invariants

All prices use forward $F(T)$ and discount $D(T)=e^{-rT}$. Validate call and put
prices against intrinsic value and their discounted forward bounds. Put-call
parity must be preserved. Use finite, positive expiry and volatility; `w(k,T)`
must be nonnegative and monotone in calendar time. Do not clamp arbitrages into
apparent validity. Keep tests offline: the checked-in Parquet chain is the
regression fixture and live downloads must remain explicitly requested.

## Session Memory & Progress Log
- Read `@docs/PROGRESS.md` at session start to know current project state, recent skill runs, and next steps.
- Update `docs/PROGRESS.md` (via `/log`) before ending a session or running `/clear`.