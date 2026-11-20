r"""Generate the tables, figures and Python session that the whitepaper inputs.

Run from the repository root:

    uv run python docs/build_walkthrough.py

Everything is computed by the library itself (offline, deterministic), so the
numbers in the PDF cannot drift from the code. Output goes to
``docs/generated/*.tex``; the whitepaper includes those files with ``\\input``.
"""

import contextlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tsx60_iv.american import american_price, european_tree_price
from tsx60_iv.market_params import expected_dividends, load_params
from tsx60_iv.rates import RateCurve
from tsx60_iv.surface import build_surface
from tsx60_iv.synthetic import (
    BOC_CURVE_PATH,
    DEFAULT_SURFACE,
    MARKET_PARAMS_PATH,
    generate_realistic_chain,
)

OUT = Path(__file__).parent / "generated"
EXPIRY = "2025-03-21"
WINDOW = (35.0, 41.0)  # strikes shown in the raw-quote table


def tex_num(value: float, digits: int = 2) -> str:
    """Format a number for LaTeX text mode, with a true minus sign."""
    return f"{value:.{digits}f}".replace("-", "$-$")


def tabular(header: list[str], rows: list[list[str]], spec: str) -> str:
    """Return a ruled tabular environment (no extra packages)."""
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\hline", " & ".join(header) + r" \\ \hline"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def write(name: str, text: str) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(text, encoding="utf-8")


# --------------------------------------------------------------------- TikZ
def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round tick positions between ``low`` and ``high``."""
    raw = (high - low) / count
    step = 10 ** np.floor(np.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        if raw <= factor * step:
            step *= factor
            break
    first = np.ceil(low / step) * step
    return [float(v) for v in np.arange(first, high + 1e-9, step)]


def tikz_plot(
    series: list[dict[str, Any]],
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    width: float = 11.5,
    height: float = 6.0,
    legend: bool = True,
) -> str:
    """Draw axes, polylines and marks with plain TikZ (no pgfplots)."""

    def point(x: float, y: float) -> tuple[float, float]:
        return (
            (x - xlim[0]) / (xlim[1] - xlim[0]) * width,
            (y - ylim[0]) / (ylim[1] - ylim[0]) * height,
        )

    out = [r"\begin{tikzpicture}[font=\small]"]
    for tick in nice_ticks(*xlim):
        px, _ = point(tick, ylim[0])
        out.append(rf"\draw[gray!30] ({px:.3f},0) -- ({px:.3f},{height});")
        out.append(rf"\node[below] at ({px:.3f},0) {{{tick:g}}};")
    for tick in nice_ticks(*ylim):
        _, py = point(xlim[0], tick)
        out.append(rf"\draw[gray!30] (0,{py:.3f}) -- ({width},{py:.3f});")
        out.append(rf"\node[left] at (0,{py:.3f}) {{{tick:g}}};")
    out.append(rf"\draw (0,0) rectangle ({width},{height});")
    out.append(rf"\node[below=14pt] at ({width / 2},0) {{{xlabel}}};")
    out.append(rf"\node[rotate=90,above=22pt] at (0,{height / 2}) {{{ylabel}}};")
    for item in series:
        color = item.get("color", "black")
        coords = [point(x, y) for x, y in zip(item["x"], item["y"], strict=True)]
        coords = [
            (px, py)
            for (px, py) in coords
            if -0.01 <= px <= width + 0.01 and -0.01 <= py <= height + 0.01
        ]
        if item["kind"] == "line":
            style = item.get("style", "thick")
            path = " ".join(f"({px:.3f},{py:.3f})" for px, py in coords)
            out.append(rf"\draw[{color},{style}] plot coordinates {{{path}}};")
        else:
            for px, py in coords:
                out.append(rf"\fill[{color}] ({px:.3f},{py:.3f}) circle (1.6pt);")
    if legend:
        for i, item in enumerate(s for s in series if s.get("label")):
            y = height - 0.45 - 0.45 * i
            out.append(rf"\fill[white,opacity=0.85] ({width - 5.2},{y - 0.18}) rectangle ({width - 0.1},{y + 0.18});")
            if item["kind"] == "line":
                out.append(rf"\draw[{item.get('color', 'black')},{item.get('style', 'thick')}] ({width - 5.0},{y}) -- ({width - 4.4},{y});")
            else:
                out.append(rf"\fill[{item.get('color', 'black')}] ({width - 4.7},{y}) circle (1.6pt);")
            out.append(rf"\node[right] at ({width - 4.3},{y}) {{{item['label']}}};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------ Python session
SESSION = [
    (
        "The chain",
        """from tsx60_iv.synthetic import generate_realistic_chain
chain = generate_realistic_chain()
print(chain.shape)
cols = ["expiration", "option_type", "strike", "bid", "ask", "volume"]
print(chain[cols].head(4).to_string(index=False))""",
    ),
    (
        "One expiry, calls and puts side by side",
        """e = chain[chain.expiration == "2025-03-21"]
window = e[e.strike.between(36, 39)]
print(window.pivot(index="strike", columns="option_type",
                   values=["bid", "ask"]).to_string())""",
    ),
    (
        "Quality filter, time to expiry, rate and discount",
        """from tsx60_iv.rates import RateCurve
from tsx60_iv.synthetic import BOC_CURVE_PATH
from tsx60_iv.data import estimate_forward_and_rates

curve = RateCurve.from_json(BOC_CURVE_PATH)
spot = float(chain.spot.iloc[0])
clean = estimate_forward_and_rates(chain, spot, curve)
c = clean[clean.expiration == "2025-03-21"]
print(len(e), "quotes,", len(c), "pass the quality filter")
cols = ["strike", "option_type", "mid", "time_to_expiry", "rate", "discount"]
print(c[cols].head(3).to_string(index=False))""",
    ),
    (
        "Put-call parity by hand: the forward from the three at-the-money pairs",
        """D = c.discount.iloc[0]
pairs = c.pivot(index="strike", columns="option_type", values="mid").dropna()
K = pairs.index.to_numpy()
pairs["F_hat"] = K + (pairs.call - pairs.put) / D
near = pairs.iloc[np.argsort(np.abs(K - spot))[:3]]
print(near.round(4).to_string())
print("median F =", round(near.F_hat.median(), 4))""",
    ),
    (
        "Correcting the forward for early exercise (three passes)",
        """from tsx60_iv.bsm import implied_volatility
from tsx60_iv.american import early_exercise_bias
from tsx60_iv.market_params import expected_dividends, load_params
from tsx60_iv.synthetic import MARKET_PARAMS_PATH

dividends = expected_dividends(load_params(MARKET_PARAMS_PATH))
T, r = c.time_to_expiry.iloc[0], c.rate.iloc[0]
Kn = near.index.to_numpy()
F = near.F_hat.median()
for step in range(3):
    adjusted = {}
    for side in ("call", "put"):
        iv = [implied_volatility(m, F, k, T, D, side)
              for k, m in zip(Kn, near[side])]
        premium, _ = early_exercise_bias(np.array(iv), spot, F, Kn, T, r, side,
                                         dividends=dividends)
        adjusted[side] = near[side].to_numpy() - premium
    F = np.median(Kn + (adjusted["call"] - adjusted["put"]) / D)
    print("pass", step + 1, "F =", round(float(F), 4))""",
    ),
    (
        "Out-of-the-money quotes inverted two ways",
        """from tsx60_iv.american import american_implied_vols
otm = c[((c.option_type == "put") & (c.strike < F))
        | ((c.option_type == "call") & (c.strike >= F))]
q = r - np.log(F / spot) / T          # yield implied by the forward
parts = []
for side, g in otm.groupby("option_type"):
    iv_a = american_implied_vols(g.mid.to_numpy(), spot, g.strike.to_numpy(),
                                 T, r, q, side, dividends=dividends)
    iv_e = [implied_volatility(m, F, k, T, D, side)
            for m, k in zip(g.mid, g.strike)]
    parts.append(g.assign(iv_european=100 * np.array(iv_e), iv_american=100 * iv_a))
table = pd.concat(parts).sort_values("strike")
cols = ["strike", "option_type", "mid", "iv_european", "iv_american"]
print(table[cols].round(3).to_string(index=False))""",
    ),
    (
        "Fitting one SVI slice and checking the butterfly condition",
        """from tsx60_iv.svi import calibrate_svi_slice, durrleman_g
k = np.log(table.strike.to_numpy() / F)
w = (table.iv_american.to_numpy() / 100) ** 2 * T      # total variance
params = calibrate_svi_slice(k, w)
names = ["a", "b", "rho", "m", "sigma"]
print({n: round(float(v), 5) for n, v in zip(names, params)})
pad = 0.1 * np.ptp(k)
grid = np.linspace(k.min() - pad, k.max() + pad, 81)
print("min g on the grid:", round(float(durrleman_g(grid, *params).min()), 3))""",
    ),
    (
        "The whole pipeline, then the European variant for comparison",
        """from tsx60_iv.surface import build_surface
kw = dict(ticker="XIU (synthetic)", source="synthetic_demo", synthetic=True)
amer = build_surface(chain, spot, curve, exercise="american",
                     dividends=dividends, **kw)
euro = build_surface(chain, spot, curve, exercise="european", **kw)
print("fitted:", len(amer["expiries"]), " rejected:",
      [r["expiration"] for r in amer["rejected"]])
print("calendar arbitrage-free:", amer["calendar"]["ok"])
for a, b in list(zip(amer["expiries"], euro["expiries"]))[2:5]:
    atm = lambda s: 100 * (s["atm_w"] / s["T"]) ** 0.5
    print(a["expiration"], "ATM vol  american %.2f  european %.2f"
          % (atm(a), atm(b)))""",
    ),
]


def run_session() -> dict[str, Any]:
    """Execute the session blocks, write them with their real output."""
    pd.set_option("display.width", 92)
    namespace: dict[str, Any] = {"np": np, "pd": pd}
    parts = []
    for title, code in SESSION:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(code, namespace)  # noqa: S102 - our own snippets
        output = buffer.getvalue().rstrip()
        parts.append(
            rf"\paragraph{{{title}.}}" "\n"
            r"{\footnotesize" "\n"
            r"\begin{verbatim}" "\n" + code + "\n" r"\end{verbatim}" "\n"
            r"\vspace{-0.6em}" "\n"
            r"\begin{verbatim}" "\n" + output + "\n" r"\end{verbatim}" "\n"
            "}\n"
        )
    write("python_session.tex", "\n".join(parts))
    return namespace


# ------------------------------------------------------------------- tables
def quote_tables(ns: dict[str, Any]) -> None:
    chain: pd.DataFrame = ns["chain"]
    e = chain[chain.expiration == EXPIRY]
    rows = []
    for strike in sorted(e.strike.unique()):
        if not WINDOW[0] <= strike <= WINDOW[1]:
            continue
        cells = [f"{strike:.1f}"]
        for side in ("call", "put"):
            q = e[(e.strike == strike) & (e.option_type == side)].iloc[0]
            mid = (q.bid + q.ask) / 2
            spread = (q.ask - q.bid) / mid if mid > 0 else float("inf")
            bad = q.bid <= 0 or spread > 0.35
            flag = r"\textsuperscript{*}" if bad else ""
            cells += [f"{q.bid:.2f}", f"{q.ask:.2f}", f"{mid:.3f}{flag}", f"{100 * spread:.0f}\\%"]
        rows.append(cells)
    header = ["K", "bid", "ask", "mid", "spr.", "bid", "ask", "mid", "spr."]
    write("quotes_table.tex", r"\begin{center}\small" "\n" + tabular(header, rows, "|r|rrrr|rrrr|") + r"\end{center}" "\n")

    clean = ns["clean"]
    c = clean[clean.expiration == EXPIRY]
    write("filter_counts.tex", rf"\newcommand{{\nRaw}}{{{len(e)}}}\newcommand{{\nPass}}{{{len(c)}}}" "\n")


def curve_table() -> None:
    curve = RateCurve.from_json(BOC_CURVE_PATH)
    labels = ["CORRA overnight", "1 month T-bill", "3 month T-bill", "6 month T-bill", "1 year T-bill", "2 year bond (par)"]
    rows = [
        [lab, f"{t:.4f}", f"{y:.2f}", f"{100 * z:.4f}", d]
        for lab, t, y, z, d in zip(
            labels, curve.tenors, curve.quoted_yields_pct, curve.zero_rates, curve.observation_dates, strict=True
        )
    ]
    header = ["Node", "tenor (y)", "quoted \\%", "zero rate \\%", "print date"]
    write("curve_table.tex", r"\begin{center}\small" "\n" + tabular(header, rows, "|l|r|r|r|l|") + r"\end{center}" "\n")


def walkthrough_tables(ns: dict[str, Any]) -> None:
    near = ns["near"]
    rows = [
        [f"{k:.1f}", f"{r.call:.4f}", f"{r.put:.4f}", f"{r.F_hat:.4f}"]
        for k, r in near.sort_index().iterrows()
    ]
    write("parity_table.tex", r"\begin{center}\small" "\n" + tabular(["K", "call mid", "put mid", r"$\hat F=K+(C-P)/D$"], rows, "|r|r|r|r|") + r"\end{center}" "\n")

    table = ns["table"]
    surf = ns["amer"]
    sl = next(s for s in surf["expiries"] if s["expiration"] == EXPIRY)
    T = sl["T"]
    truth = DEFAULT_SURFACE.vol(np.log(table.strike.to_numpy() / sl["forward"]), T) * 100
    body = []
    for (_, row), t in zip(table.iterrows(), truth, strict=True):
        body.append(
            [f"{row.strike:.1f}", row.option_type, f"{row.mid:.3f}", f"{row.iv_european:.2f}", f"{row.iv_american:.2f}", f"{t:.2f}"]
        )
    header = ["K", "side", "mid", r"$\sigma_E$ \%", r"$\sigma_A$ \%", "truth \\%"]
    write("otm_table.tex", r"\begin{center}\small" "\n" + tabular(header, body, "|r|l|r|r|r|r|") + r"\end{center}" "\n")

    p = sl["params"]
    prm = [
        ["$a$", f"{p['a']:.5f}"],
        ["$b$", f"{p['b']:.5f}"],
        [r"$\rho$", f"{p['rho']:.4f}"],
        ["$m$", f"{p['m']:.4f}"],
        ["$s$", f"{p['sigma']:.4f}"],
        ["quotes used", str(sl["n_points"])],
        ["RMSE (vol points)", f"{sl['rmse_vol_pts']:.3f}"],
        [r"$\min g(k)$", f"{sl['min_g']:.3f}"],
        [r"$a+bs\sqrt{1-\rho^2}$", f"{p['a'] + p['b'] * p['sigma'] * np.sqrt(1 - p['rho'] ** 2):.5f}"],
    ]
    write("svi_table.tex", r"\begin{center}\small" "\n" + tabular(["quantity", "value"], prm, "|l|r|") + r"\end{center}" "\n")

    rows = []
    for s in surf["expiries"]:
        truth_atm = float(DEFAULT_SURFACE.vol(0.0, s["T"])) * 100
        fit_atm = 100 * np.sqrt(s["atm_w"] / s["T"])
        rows.append(
            [
                s["expiration"],
                str(s["days"]),
                f"{s['forward']:.3f}",
                f"{100 * s['rate']:.3f}",
                f"{100 * s['dividend_yield']:.2f}",
                str(s["n_points"]),
                f"{fit_atm:.2f}",
                f"{truth_atm:.2f}",
                tex_num(fit_atm - truth_atm, 2),
                f"{s['american']['max_vol_bias_pts']:.2f}",
            ]
        )
    header = ["expiry", "days", "$F$", "$r$ \\%", "$q$ \\%", "n", "ATM fit", "ATM true", "err.", "max bias"]
    write("expiry_table.tex", r"\begin{center}\footnotesize" "\n" + tabular(header, rows, "|l|r|r|r|r|r|r|r|r|r|") + r"\end{center}" "\n")


# ------------------------------------------------------------------ figures
def figures(ns: dict[str, Any]) -> None:
    surf = ns["amer"]
    sl = next(s for s in surf["expiries"] if s["expiration"] == EXPIRY)
    k_obs = np.array(sl["k_obs"])
    grid = np.array(sl["k_grid"])
    truth = 100 * DEFAULT_SURFACE.vol(grid, sl["T"])
    series = [
        {"kind": "line", "x": grid, "y": truth, "color": "gray", "style": "thick,dashed", "label": "true surface"},
        {"kind": "line", "x": grid, "y": 100 * np.array(sl["iv"]), "color": "blue", "label": "SVI fit"},
        {"kind": "marks", "x": k_obs, "y": 100 * np.array(sl["iv_obs"]), "color": "red", "label": "inverted quotes"},
    ]
    lo, hi = float(truth.min()) - 0.5, float(truth.max()) + 0.5
    write("fig_smile.tex", tikz_plot(series, "log-moneyness $k$", r"implied vol (\%)", (float(grid.min()), float(grid.max())), (np.floor(lo), np.ceil(hi))))

    series = []
    colors = ["blue", "teal", "green!60!black", "orange", "red", "violet"]
    for i, s in enumerate(surf["expiries"]):
        series.append({"kind": "line", "x": s["k_grid"], "y": 100 * np.array(s["iv"]), "color": colors[i % len(colors)], "style": "thin", "label": ""})
    ymax = max(100 * max(s["iv"]) for s in surf["expiries"])
    ymin = min(100 * min(s["iv"]) for s in surf["expiries"])
    write("fig_smiles.tex", tikz_plot(series, "log-moneyness $k$", r"implied vol (\%)", (-0.3, 0.25), (np.floor(ymin), np.ceil(ymax)), legend=False))

    days = np.array([s["days"] for s in surf["expiries"]])
    fit = np.array([100 * np.sqrt(s["atm_w"] / s["T"]) for s in surf["expiries"]])
    tt = np.linspace(0.04, 1.05, 60)
    series = [
        {"kind": "line", "x": tt * 365, "y": 100 * DEFAULT_SURFACE.atm_vol(tt), "color": "gray", "style": "thick,dashed", "label": "true ATM vol"},
        {"kind": "marks", "x": days, "y": fit, "color": "red", "label": "fitted ATM vol"},
    ]
    write("fig_term.tex", tikz_plot(series, "days to expiry", r"ATM implied vol (\%)", (0, 400), (11, 14), height=4.5))

    # early-exercise premium by strike: calls with a dividend before expiry, and puts
    from tsx60_iv.american import Dividends

    spot, T, r, vol = 37.7, 0.31, 0.03, 0.14
    divs = Dividends((0.30,), (0.6,))
    q = -np.log(1 - divs.pv(0.0, r, T) / spot) / T
    strikes = np.linspace(30, 45, 61)
    put = american_price(spot, strikes, T, r, 0.0, vol, "put") - european_tree_price(spot, strikes, T, r, 0.0, vol, "put")
    call_c = american_price(spot, strikes, T, r, q, vol, "call") - european_tree_price(spot, strikes, T, r, q, vol, "call")
    call_d = american_price(spot, strikes, T, r, q, vol, "call", dividends=divs) - european_tree_price(spot, strikes, T, r, q, vol, "call", dividends=divs)
    series = [
        {"kind": "line", "x": strikes, "y": put, "color": "blue", "label": "put, no dividends"},
        {"kind": "line", "x": strikes, "y": call_c, "color": "gray", "style": "thick,dashed", "label": "call, continuous yield"},
        {"kind": "line", "x": strikes, "y": call_d, "color": "red", "label": "call, discrete dividend"},
    ]
    top = float(max(put.max(), call_d.max(), call_c.max()))
    write("fig_premium.tex", tikz_plot(series, "strike $K$", "early-exercise premium (CAD)", (30, 45), (0, np.ceil(top * 10) / 10)))


# ------------------------------------------------------------------ results
def results(ns: dict[str, Any]) -> None:
    chain, curve = ns["chain"], ns["curve"]
    spot = float(chain.spot.iloc[0])
    kw = {"ticker": "x", "source": "s", "synthetic": True}
    runs = {
        "American, schedule (default)": ns["amer"],
        "American, continuous yield": build_surface(chain, spot, curve, exercise="american", dividends=None, **kw),
        "European": ns["euro"],
    }
    rows, numbers = [], []
    for label, surface in runs.items():
        errors, worst = [], []
        for s in surface["expiries"]:
            truth = DEFAULT_SURFACE.vol(np.array(s["k_obs"]), s["T"])
            errors.append(100 * (np.sqrt(s["atm_w"] / s["T"]) - float(DEFAULT_SURFACE.vol(0.0, s["T"]))))
            worst.append(100 * float(np.max(np.abs(np.array(s["iv_obs"]) - truth))))
        errors = np.array(errors)
        rows.append([label, str(len(surface["expiries"])), f"{errors.mean():+.3f}", f"{np.abs(errors).max():.2f}", f"{max(worst):.2f}"])
        numbers.append((label, errors.mean(), np.abs(errors).max(), max(worst)))
    header = ["variant", "expiries fitted", "mean ATM error", "worst ATM error", "worst quote error"]
    write("results_table.tex", r"\begin{center}\footnotesize" "\n" + tabular(header, rows, "|l|r|r|r|r|") + r"\end{center}" "\n")
    am, co, eu = numbers
    macros = [
        ("atmErrAmerican", f"{am[1]:+.3f}"), ("atmWorstAmerican", f"{am[2]:.2f}"),
        ("atmErrContinuous", f"{co[1]:+.2f}"), ("atmErrEuropean", f"{eu[1]:+.2f}"),
        ("atmWorstEuropean", f"{eu[2]:.2f}"), ("quoteWorstEuropean", f"{eu[3]:.2f}"),
    ]
    write("numbers.tex", "".join(rf"\newcommand{{\{n}}}{{{v}}}" "\n" for n, v in macros))


def main() -> None:
    ns = run_session()
    ns["curve"] = RateCurve.from_json(BOC_CURVE_PATH)
    ns["dividends"] = expected_dividends(load_params(MARKET_PARAMS_PATH))
    ns["chain"] = generate_realistic_chain()
    quote_tables(ns)
    curve_table()
    walkthrough_tables(ns)
    figures(ns)
    results(ns)
    print(f"wrote {len(list(OUT.glob('*.tex')))} files to {OUT}")


if __name__ == "__main__":
    main()
