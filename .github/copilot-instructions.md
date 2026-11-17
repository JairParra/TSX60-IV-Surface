# Copilot instructions

- Python 3.12+, typed `src/tsx60_iv` package, with NumPy-style docstrings.
- Run `uv run pytest` and `uv run ruff check .` before submitting changes.
- Keep quantitative routines vectorized; avoid abstractions without a use case.
- Never silently relax no-arbitrage bounds. Put-call parity, implied-volatility
  round trips, positive total variance, and calendar monotonicity are strict invariants.
- Keep tests offline and deterministic. Live `yfinance` access belongs behind
  `fetch_xiu_chain`, not in numerical tests.
- Treat strike, expiry, forward, volatility, and discount inputs explicitly; do
  not substitute spot for forward in pricing formulas.
