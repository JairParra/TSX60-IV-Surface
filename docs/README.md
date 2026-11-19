# TSX60 Implied Volatility Surface: Architecture, Flow & API Documentation

## 1. Project Mission & High-Level Purpose

The **TSX60-IV-Surface** repository provides a quantitative, production-oriented modeling library and visualization engine for the Canadian S&P/TSX 60 equity index implied volatility surface, proxied via the liquid **XIU** (iShares S&P/TSX 60 Index ETF) option chains.

### Core Philosophy & Strict Invariants
1. **Forward BSM Foundation**: All options are priced in forward space $F(T)$ with continuous CAD discount factor $D(T) = e^{-rT}$.
2. **Put-Call Parity Forwards**: Forward prices are estimated strictly from put-call parity of near-ATM quote pairs ($C - P = D(T)(F(T) - K)$). There is no spot fallback.
3. **No Clamping / No Arbitrage Masking**: If a quote violates intrinsic bounds or if an SVI slice produces butterfly or calendar arbitrage, the code **rejects and reports** it. It never clamps or penalizes parameters into artificial validity.
4. **Deterministic & Offline**: Offline tests use a synthetic option chain fixture; live network ingestion is strictly explicit (`--fetch`). The demo chain is generated: an arbitrage-free SSVI surface anchored to XIU.TO realized volatility, priced American-style with the Bank of Canada curve (section 5).

---

## 2. System Architecture & Component Connections

The repository is modularized into discrete responsibilities:

```mermaid
flowchart TD
    subgraph Data Layer ["1. Ingestion & Preprocessing"]
        Y[Yahoo Finance / Live Fetch] -->|fetch_xiu_chain| RAW[Raw Option Chain DataFrame]
        SYN[synthetic.py: generate_realistic_chain] -->|--demo| RAW
        BOC[rates.py: Bank of Canada curve snapshot] -->|r of T per expiry| FWD
        RAW --> FILT["_filter_quoted_chain\n(bid>0, ask>=bid, spread<=35%)"]
        FILT --> FWD["estimate_forward_and_rates\n(Put-Call Parity -> F(T), discount, log-moneyness k)"]
    end

    subgraph Analytics Layer ["2. Forward BSM & Volatility Inversion"]
        FWD --> OTM["_otm_implied_vols\n(Filter OTM puts/calls, mid >= 0.05)"]
        OTM --> BRENT["bsm.py: implied_volatility (European, Brent)\nor american.py: american_implied_vols (CRR tree, default)"]
        BRENT --> BSM_PRICE["bsm.py: bsm_price\n(Forward Black-Scholes-Merton)"]
        BRENT --> TOTAL_VAR["w = iv^2 * T (Total Variance)"]
    end

    subgraph Calibration Layer ["3. SVI Slice Calibration & Arbitrage Checks"]
        TOTAL_VAR --> SVI_FIT["svi.py: calibrate_svi_slice\n(Multi-start SLSQP on Raw SVI,\nmin-variance and butterfly constraints)"]
        SVI_FIT --> BUTTERFLY["svi.py: check_butterfly_arbitrage\n(Durrleman g(k) >= 0)"]
        BUTTERFLY --> CALENDAR["svi.py: check_calendar_arbitrage\n(d w / dT >= 0 across slices)"]
    end

    subgraph Assembly & Pipeline ["4. Surface Pipeline"]
        SVI_FIT --> SLICE["_fit_slice\n(Params, RMSE, grid, w, iv, Durrleman g)"]
        CALENDAR --> MESH["_mesh\n(81x(N_expiries*6) 3D grid, no extrapolation)"]
        SLICE --> SURF_DOC["build_surface\n(JSON surface document)"]
        MESH --> SURF_DOC
        SURF_DOC --> ATOMIC["write_surface\n(Atomic tmp -> data/surface.json)"]
    end

    subgraph Presentation & Serving ["5. Serving & Tracking"]
        ATOMIC --> API["api.py: FastAPI Server"]
        API -->|GET /surface (ETag / 304)| UI["static/index.html (Plotly 3D + KaTeX)"]
        API -->|GET /| UI
        ATOMIC -.->|--mlflow| TRACK["tracking.py: MLflow Run Logging"]
    end
```

---

## 3. Entry Points

### 1. Offline Interactive Demo (Recommended Entry Point)
Generates a realistic synthetic chain, calibrates SVI slices, creates `data/surface.json`, and starts the FastAPI server:
```bash
# Generate surface JSON
uv run python -m tsx60_iv.surface --demo

# Serve UI & API
uv run uvicorn tsx60_iv.api:app --host 127.0.0.1 --port 8000
```
Then navigate to `http://127.0.0.1:8000/`.

### 2. Live Data Ingestion & Pipeline Run
Fetches live chains from Yahoo Finance (if available for the ticker, e.g. EWC or SPY), enforces a 30-minute cooldown, fits the surface, and optionally logs to MLflow:
```bash
uv run python -m tsx60_iv.surface --fetch --ticker EWC --rate 0.043 --max-expiries 6 --mlflow
```

### 3. Direct Module Execution
- `python -m tsx60_iv.data`: Runs `fetch_xiu_chain` and saves `data/xiu_chain.parquet`.
- `python -m tsx60_iv.surface`: Runs the end-to-end calibration CLI.

---

## 4. API Design (`src/tsx60_iv/api.py`)

The API layer is built on **FastAPI** and is designed for high-concurrency, low-latency, and zero-downtime hot reloading.

### Endpoints

| HTTP Method | Route | Description | Status Codes & Headers |
|---|---|---|---|
| `GET` | `/` | Serves the standalone 3D visualizer SPA (`static/index.html`). | `200 OK` (`text/html`) |
| `GET` | `/health` | Health-check endpoint for load balancers and orchestrators. | `200 OK` (`{"status": "ok"}`) |
| `GET` | `/surface` | Returns the latest calibrated surface JSON document. | `200 OK` (with `ETag`), `304 Not Modified`, `503 Service Unavailable` |

### Architectural Highlights of the API:
- **Hot-Reload Cache with ETag Validation**:
  `_current()` inspects the filesystem modification timestamp (`st_mtime_ns`) and file size (`st_size`). The ETag is computed as `"{st_mtime_ns}-{st_size}"`. If the browser sends `If-None-Match` matching this ETag, the API immediately returns `304 Not Modified` with zero JSON serialization overhead.
- **Fail-Safe Startup (503 Handling)**:
  If `data/surface.json` does not exist yet (e.g. server starts before calibration runs), `/surface` returns `503 Service Unavailable` with `{"detail": "surface not loaded"}`. The frontend gracefully informs the user how to generate it.
- **In-Memory Injection Hook**:
  `set_surface(surface: dict)` allows programmatic injection of a surface dictionary (used in unit tests) that bypasses file reading.
- **Atomic File Reads**:
  The CLI writes to a temporary file (`surface.json.tmp`) and uses `os.replace` to atomically swap it into place, preventing the API from ever reading a partially written JSON file.

---

## 5. End-to-End Data Flow

The lifecycle of data through the repository progresses through 8 stages:

1. **Option Chain Ingestion**:
   - `fetch_xiu_chain()` or `generate_realistic_chain()` provides raw quotes: `strike`, `bid`, `ask`, `expiration`, `option_type`, `volume`, `openInterest`, `spot`, `as_of`.
2. **Quote Quality Filtering**:
   - `_filter_quoted_chain()` retains only quotes where $bid > 0$, $ask \ge bid$, mid price $m = (bid + ask) / 2$ is finite, and relative bid-ask spread $(ask - bid) / m \le 0.35$.
3. **Discounting & Forward Estimation**:
   - `estimate_forward_and_rates()` calculates $T = \text{days} / 365$ and $D(T) = e^{-r(T)\,T}$, where $r(T)$ is the continuously compounded zero rate read from the Bank of Canada curve (`rates.RateCurve`, linear between nodes, flat beyond them) or a flat float. The 2-year par yield is bootstrapped to a zero rate (`rates.bootstrap_par_zero`).
   - With `--exercise american`, the parity forward is then corrected for the early-exercise premium of its own near-the-money quotes (3 fixed-point passes, `surface._parity_premium`), because parity holds for European prices.
   - Put-call parity $C - P = D(T)(F(T) - K)$ is computed for matched strikes.
   - For each expiry, the 3 quote pairs closest to ATM ($|K - \text{spot}|$) are selected, and $F(T) = \text{median}(\hat{F}_i)$ is assigned. Log-moneyness $k = \ln(K / F(T))$ is derived.
4. **OTM Slice Filtering**:
   - `_otm_implied_vols()` filters to Out-of-the-Money options ($K < F(T)$ for puts, $K \ge F(T)$ for calls) to eliminate liquidity distortions from in-the-money options.
   - Any quote with $mid < 0.05$ is dropped to avoid tick-floor noise.
5. **Implied Volatility Inversion**:
   - `--exercise european`: for each quote, `implied_volatility()` verifies static no-arbitrage bounds ($D\max(F-K,0) < C < DF$) and Brent's method solves for $\sigma_{\text{imp}}$ in $[10^{-5}, 5.0]$.
   - `--exercise american` (default): XIU options are American-style, so `american_implied_vols()` inverts a Cox-Ross-Rubinstein tree per expiry and side with a safeguarded Newton iteration. The tree is anchored to the parity forward through $q = r - \ln(F/S)/T$. With a dividend schedule (`american.Dividends`) it runs on $S^* = Se^{-qT}$ and values early exercise on $S^* + \text{PV(remaining dividends)}$; without one it uses $q$ as a continuous yield.
   - Either way, every slice records the early-exercise premium at its fitted vol and the first-order vol bias $\text{premium}/\mathcal{V}$.
   - Total variance is formed: $w(k, T) = \sigma_{\text{imp}}^2 \cdot T$.
6. **Raw SVI Slice Calibration**:
   - `calibrate_svi_slice()` fits Gatheral's 5-parameter Raw SVI formula:
     $$w(k) = a + b\left[\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right]$$
   - Uses constrained `scipy.optimize.minimize(method="SLSQP")` with multi-start guesses ($\rho_0 \in \{-0.5, 0.0, 0.5\}$) the minimum variance constraint $a + b\sigma\sqrt{1-\rho^2} \ge 0$, and the Durrleman constraint $g(k) \ge 0$ on a grid extending 0.5 beyond the quoted strikes (butterfly freedom is imposed by the fit, never repaired afterwards).
7. **No-Arbitrage Validation**:
   - **Butterfly Arbitrage**: Durrleman's condition $g(k) \ge 0$ is both a constraint of the fit and re-checked on the same dense grid (quoted strikes plus 201 points). If the fit still violates it, the slice is rejected.
   - **Calendar Arbitrage**: `check_calendar_arbitrage()` evaluates $\partial_T w(k, T) \ge 0$ across all valid slices.
8. **Mesh Generation & Atomic Delivery**:
   - `_mesh()` computes an $81 \times (N_{\text{slices}} \times 6)$ grid linearly interpolated in $w$ across calendar time without extrapolating outside quoted strikes.
   - `write_surface()` serializes the result to `data/surface.json`.

---

## 6. Detailed Module & Function Reference

### `tsx60_iv.data`
- `fetch_xiu_chain(ticker, cache_path, max_expiries)`: Calls Yahoo Finance `yf.Ticker`, downloads chain per expiration, normalizes timezone to `America/Toronto`, appends `as_of` and `spot`, and optionally writes Parquet.
- `_filter_quoted_chain(chain_df)`: Filters out crossed quotes, negative bids, and spreads $> 35\%$.
- `estimate_forward_and_rates(chain_df, spot, r_cad)`: `r_cad` is a float or a `RateCurve`. Applies put-call parity to estimate $F(T)$, assigns per-expiry rates and discount factors, and computes $k = \ln(K / F)$.

### `tsx60_iv.bsm`
- `_inputs(...)`: Validates broadcast shapes, strictly checks positive finite forward, strike, time, and discount.
- `bsm_price(forward, strike, time_to_expiry, discount, volatility, option_type)`: Closed-form forward BSM formula using `scipy.special.ndtr`.
- `bsm_greeks(forward, strike, time_to_expiry, discount, volatility, option_type)`: Computes forward-space analytical $\Delta, \Gamma, \mathcal{V}, \text{vanna}, \text{volga}$.
- `implied_volatility(price, forward, strike, time_to_expiry, discount, option_type)`: Inverts market prices to Black vol using Brent's method with static arbitrage checks.

### `tsx60_iv.svi`
- `raw_svi_variance(k, a, b, rho, m, sigma)`: Evaluates Raw SVI total variance $w(k)$.
- `durrleman_g(k, a, b, rho, m, sigma)`: Computes Durrleman's $g(k)$ curvature formula.
- `check_butterfly_arbitrage(k, a, b, rho, m, sigma)`: Validates $g(k) \ge 0$.
- `calibrate_svi_slice(k, total_variance)`: Multi-start SLSQP calibration of Raw SVI parameters with min-variance and butterfly constraints; raises if the result still shows butterfly arbitrage.
- `check_calendar_arbitrage(k, slices)`: Ensures $w(k, T_2) \ge w(k, T_1)$ for $T_2 > T_1$.

### `tsx60_iv.surface`
- `_otm_implied_vols(clean, min_mid, spot, exercise)`: Selects OTM contracts, inverts European or American, tracks failure categories (`below_min_mid`, `arbitrage_bounds`, `solver_bracket`) and the early-exercise bias per quote.
- `_fit_slice(expiration, g)`: Calibrates SVI, computes RMSE in vol points, precomputes smile grids and Durrleman $g$.
- `_mesh(slices)`: Generates 6 intermediate interpolated rows between each pair of expiries for smooth 3D rendering.
- `build_surface(...)`: Orchestrates the entire pipeline into the final schema.
- `write_surface(surface, out)`: Writes to temporary file and atomically replaces destination.
- `main()`: CLI argument parsing and driver.

### `tsx60_iv.synthetic`
- `generate_xiu_sample_chain(days)`: The small legacy chain behind the checked-in test fixture (downward skew $\sigma(k) = 0.205 - 0.11k + 0.02\sqrt{T}$).
- `generate_realistic_chain(seed, steps, surface)`: The demo chain. Truth is an `SsviSurface` (Gatheral-Jacquier SSVI, conditions checked), anchored to XIU.TO realized volatility; expiries are third Fridays (Thursday before Good Friday); spot and dividends come from `market_params`; rates from the Bank of Canada snapshot; prices from the American tree; quotes have 0.01 ticks and spreads of at least 0.02 or 5%. Synthetic, not market quotes.

### `tsx60_iv.american`
- `american_price`, `european_tree_price`: CRR tree vectorized over strikes of one expiry (average of $n$ and $n+1$ steps to damp oscillation).
- `american_implied_vols`, `american_implied_volatility`: tree inversion; NaN or `ValueError` outside the arbitrage bounds.
- `early_exercise_bias`: premium of American over European at a given vol and its first-order vol overstatement.
- `Dividends(times, amounts)`: discrete cash dividends; `american_price(..., dividends=...)` switches the tree to escrowed discrete dividends (scaled so the forward is unchanged).

### `tsx60_iv.rates`
- `RateCurve`: zero rates by tenor, snapshot to and from JSON.
- `fetch_boc_curve(as_of)`: one Bank of Canada Valet request for CORRA, 1m/3m/6m/1y T-bills and the 2y benchmark bond; money-market yields convert as $r = \ln(1+y\tau)/\tau$ and the 2y par yield is bootstrapped. The snapshot keeps the raw quoted yields.
- `python -m tsx60_iv.rates --as-of DATE --out PATH` writes a snapshot (explicit network call; tests never call it).

### `tsx60_iv.market_params`
- `derive_params`, `fetch_market_params`: spot, 21/63/252-day realized vol and trailing dividends of XIU.TO from one free Yahoo history request. Only these statistics are stored.

### `tsx60_iv.tracking`
- `log_surface_run(parameters, metrics, artifacts, experiment_name)`: Connects to MLflow, starts a run, logs parameters, metrics (slices, rejected count), and saves artifacts.

### `tsx60_iv.api`
- `set_surface(surface)`: In-memory override for testing.
- `_current()`: ETag-aware file reader with hot-reload detection.
- `page()`: Serves `static/index.html`.
- `health()`: Returns health dictionary.
- `surface(request)`: Evaluates `If-None-Match`, returns 304 or full JSON surface.

### `tsx60_iv.static.index.html`
- Interactive WebGL/Plotly 3D visualizer.
- Supports switching Z-axis (Implied Vol % vs Total Variance $w$), X-axis (Log-moneyness $k$ vs Strike $K$).
- Interactive slice slider, 2D smile fit plot, Durrleman $g(k)$ plot, and term structure.
- KaTeX mathematical display dynamically bound to the selected slice's calibrated parameters.
- Polls `/surface` every 30 seconds with ETag header for live updates.
