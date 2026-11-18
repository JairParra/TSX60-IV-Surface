# Progress

## Session Log

## 2026-11-17 — SLSQP calibration and whitepaper
- **Model & Skills Used:** `sonnet` + `/ponytail:ponytail`, `session-logger`
- **Completed:**
  - `svi.py`: `calibrate_svi_slice` now uses multi-start `minimize(method="SLSQP")` with box bounds and the `a + b*sigma*sqrt(1-rho^2) >= 0` constraint; `pyproject.toml` excludes `.claude` from ruff.
  - `docs/tsx60_iv_whitepaper.tex` written (it did not exist) and built to `docs/tsx60_iv_whitepaper.pdf` with `tectonic` 0.17.0 (conda env `tectonic`, not on `PATH`); the rendered pages were not viewed.
  - `uv run pytest -v`: 27 passed. `uv run ruff check .`: clean.
- **Key Decisions / Invariants:** SLSQP alone handles bounds and the nonlinear constraint (no L-BFGS-B). `tracking.py` and `api.py` are interfaces not wired to the core modules. The fixture chain is synthetic.
- **Next Step:** Build the pipeline: chain, OTM mids, `implied_volatility`, `w = iv^2 * T`, per-expiry fit, calendar check, `log_surface_run`.

## 2026-11-17 — Audit fixes and SVI calibration
- **Model & Skills Used:** `sonnet` + `/ponytail:ponytail`, `session-logger`
- **Completed:**
  - Audit fixes 1-11 in `data.py` (as_of stamp, raw cache, expiry-day drop, parity-only forwards), `bsm.py` (finite and option_type checks in `_inputs`), `api.py` (503 until loaded), `pyproject.toml` (plotly out, httpx in).
  - `svi.py`: `calibrate_svi_slice` (multi-start `least_squares`, raises on butterfly arbitrage) and `check_calendar_arbitrage`; README and CLAUDE.md updated.
  - New tests: Greeks by finite differences, SVI, API, data forwards. `uv run pytest`: 27 passed. `uv run ruff check src tests`: clean. `ruff check .` also flags the untracked third-party `.claude/skills/latex-document-skill/` (210 errors), not project code.
- **Key Decisions / Invariants:** Forward comes only from put-call parity, with no spot fallback. Arbitrageable fits raise, never clamp. Cache holds raw quotes plus `as_of`. Fixes 12 and 13 skipped; nothing committed yet.
- **Next Step:** Add `[tool.ruff] extend-exclude = [".claude"]` to `pyproject.toml`, then build the pipeline: chain, OTM mids, `implied_volatility`, `w = iv^2 * T`, per-expiry fit, calendar check, `log_surface_run`.
