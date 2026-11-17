# TSX60 IV Surface

## Commands

- Install: `uv sync`
- Test: `uv run pytest`
- Lint: `uv run ruff check .`
- Run pipeline: `uv run python -m tsx60_iv.data`

## Layout

- `src/tsx60_iv/data.py`: `yfinance` ingestion, quote quality, forwards/rates.
- `src/tsx60_iv/bsm.py`: forward Black-Scholes pricing, Greeks, IV inversion.
- `src/tsx60_iv/svi.py`: SVI parameterization and arbitrage stubs.
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
