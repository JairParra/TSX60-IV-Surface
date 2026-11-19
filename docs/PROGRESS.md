# Progress

## Session Log

## 2026-11-20 — Whitepaper rewritten as a self-contained tutorial
- **Model & Skills Used:** `sonnet` + `/ponytail:ponytail`
- **Completed:**
  - `docs/tsx60_iv_whitepaper.tex` rewritten in four parts: foundations (options, no-arbitrage pricing, forwards, rates and dividends, American options and the CRR tree, implied volatility, SVI/SSVI and arbitrage), the algorithm (module map, five pseudo-code algorithms, the exercise and dividend variants, safeguards), a worked example (synthetic market, one expiry step by step with real quote tables, a Python session, all expiries), and results, limits and references.
  - `docs/build_walkthrough.py` computes every table, figure and the Python session output with the library (offline), into `docs/generated/*.tex`, so the PDF cannot drift from the code. TikZ figures are plain TikZ (no pgfplots).
  - `uv run ruff check .` clean (per-file ignores for the generator); PDF builds with no LaTeX warnings.
- **Key Decisions / Invariants:** Added `tikz` (needed for figures). Corrections made while writing: a continuous yield improves ATM error only slightly (+0.24 vs +0.27 vol pts European); the dividend schedule is what removes the bias.
- **Next Step:** Look at the rendered PDF (no renderer was available when writing) and tighten wording or figures.

## 2026-11-20 — Shortcuts hardened: discrete dividends, forward correction, bootstrap
- **Model & Skills Used:** `sonnet` + `/ponytail:ponytail`, built-in browser (DOM checks)
- **Completed:**
  - `american.Dividends` and an escrowed discrete-dividend tree (`--dividends schedule|continuous`); the generator now prices with the discrete schedule, so the pipeline faces the effect it must undo. Earlier numbers were optimistic: generator and inversion shared the same continuous-yield tree.
  - American-aware parity forward (3 passes, premium of near-ATM quotes subtracted from mids): forward error 0.33% -> 0.01% of F on the ex-date expiry. 2y par yield now bootstrapped (2.9072% vs 2.9087%, immaterial); raw yields stored in the snapshot; `--min-mid` flag; `python-dateutil` declared explicitly.
  - Demo vs truth: ATM error mean +0.003 vol pts (max 0.07) with schedule + corrected forward; European inversion +0.27 (max 0.74); continuous yield worse than schedule (tested).
  - `uv run pytest -v`: 100 passed. `uv run ruff check .`: clean. Light mode and phone width checked via DOM geometry and theme colors; whitepaper rebuilt.
- **Key Decisions / Invariants:** Tried an active-set butterfly constraint for SVI speed: no gain, reverted. Real WebGL click still unverified (browser pane hidden, no input delivered).
- **Next Step:** Click the 3D surface once in a real browser; optionally build a bootstrapped curve beyond 2y and a test-suite speedup (about 40-50 s).

## 2026-11-19 — Simulation realism: BoC curve, American exercise, SSVI demo chain
- **Model & Skills Used:** `sonnet` + `/ponytail:ponytail`, built-in browser for UI checks
- **Completed:**
  - Scope decision: simulation only, no paid or real quotes. Free sources used for anchors only: Bank of Canada Valet (curve snapshot, one request) and Yahoo (XIU.TO spot, realized vol, dividends; derived statistics only).
  - New `rates.py` (per-expiry zero rates, snapshot JSON), `american.py` (vectorized CRR tree, Newton inversion, early-exercise bias), `market_params.py`, `data_files/`; `synthetic.generate_realistic_chain` (arbitrage-free SSVI truth, American prices, third-Friday expiries, 0.01 ticks); `estimate_forward_and_rates` accepts a `RateCurve`.
  - `calibrate_svi_slice` now imposes Durrleman `g >= 0` as a constraint (12 of 13 demo expiries fit, was 7 of 13 with rejects); `surface.py` gains `--exercise`, `--curve`, per-slice rate, implied dividend yield and bias diagnostics; page, README, `docs/README.md`, whitepaper and PDF updated.
  - `uv run pytest -v`: 90 passed. `uv run ruff check .`: clean. Page checked in the built-in browser.
- **Key Decisions / Invariants:** Butterfly freedom is a fit constraint and is re-checked, never repaired. Demo recovers the SSVI truth within 0.15 vol pts ATM; European inversion overstates ATM vol by about 0.06 pts versus the tree. Continuous dividend yield stands in for discrete dividends (flagged shortcut).
- **Next Step:** Optional polish: discrete-dividend tree, bootstrapped zero curve, a tracked list of the Valet series codes; otherwise open the PR and review CI.

## 2026-11-19 — Repository documentation and OpenAPI 304 response
- **Completed:**
  - Authored comprehensive architectural and functional documentation in `docs/README.md`.
  - Added explicit OpenAPI / Swagger response documentation for HTTP 200, 304 (Not Modified with ETag), and 503 on `GET /surface` in `src/tsx60_iv/api.py`.
  - Added unit test `test_surface_openapi_schema_documents_304` to `tests/test_api.py`.
  - `uv run pytest`: 39 passed. `uv run ruff check .`: clean.

## 2026-11-18 — Pipeline, API and 3D demo page
- **Model & Skills Used:** `sonnet` + `/ponytail:ponytail`, `session-logger`, built-in browser for checks
- **Completed:**
  - New `surface.py` (chain, row-by-row IV, per-expiry SVI, calendar check, atomic `data/surface.json`), `synthetic.py` (generator moved out of the fixture script; fixture equality test), `static/index.html` (Plotly 3D surface, smile, Durrleman g, term structure, KaTeX formulas with live numbers, 30 s ETag polling), `api.py` serves the file (ETag/304, 503 when missing) and the page.
  - Fixed `calibrate_svi_slice`: `ftol` 1e-14 made SLSQP hit the iteration limit on every real-looking slice; now 1e-10 / 1000 iterations. `fetch_xiu_chain` now makes one request per expiry (capped), stores spot, stops on rate limit, raises on empty.
  - `uv run pytest -v`: 37 passed. `uv run ruff check .`: clean. Page checked in the built-in browser (no local errors, modes and keyboard work, 304 confirmed). Whitepaper and PDF updated.
  - Follow-up fixes: clicking the 3D plot overflowed the stack (re-fired `plotly_click`); now nearest-expiry selection with a re-entrancy guard and 400 ms debounce. Total-variance view works; mesh now has 6 interpolated rows between expiries (linear in w, no extrapolation) and the page has an all-expiries smile chart. A real mouse click on the WebGL canvas could not be simulated in the test browser, so that path is covered by emitting the event.
- **Key Decisions / Invariants:** Yahoo has no XIU.TO options (probed: 0 expirations; SPY 29, EWC 8), so the demo is synthetic and labelled. Quotes with mid < 0.05 are excluded (tick-floor noise). Rejected expiries and a failed calendar check are reported, never repaired. No `POST /refresh`; refresh is a CLI action.
- **Next Step:** Research how to get Montreal Exchange XIU option data (free delayed quotes and terms, TMX Datalinx, broker APIs such as Interactive Brokers, vendors) and recommend one with the adapter needed.

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
