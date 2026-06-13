"""
================================================================================
 KALMAN 3-MONTH SCREENER — iPhone-friendly Streamlit app (LIVE DATA)
================================================================================
Run on your computer:
    pip install streamlit yfinance pandas numpy scipy statsmodels
    streamlit run kalman_screener_app.py

Open on your iPhone (same Wi-Fi):
    Safari -> http://<your-computer-IP>:8501
    (Streamlit prints the "Network URL" when it starts.)

Requires kalman_trading_framework.py in the same folder.
Demo mode (no internet needed): toggle "Demo data" in Settings.
================================================================================
"""

import os
import time

import numpy as np
import pandas as pd
import streamlit as st

import kalman_trading_framework as ktf

try:
    import yfinance as yf
except ImportError:
    yf = None

# ------------------------------------------------------------------ page setup
st.set_page_config(page_title="Kalman Screener", page_icon="📈",
                   layout="centered", initial_sidebar_state="collapsed")

C_KF, C_BUY, C_SELL, C_HOLD = "#2B5DA8", "#18A065", "#E05252", "#C99A2E"
# Translucent zone/badge backgrounds render correctly on light AND dark themes
BUY_BG, SELL_BG, HOLD_BG = ("rgba(24,160,101,.16)", "rgba(224,82,82,.16)",
                            "rgba(201,154,46,.16)")
C_BUY_BG, C_SELL_BG, C_HOLD_BG = BUY_BG, SELL_BG, HOLD_BG   # strip zones & badges

st.markdown("""
<style>
  /* mobile-first tightening — theme-adaptive (works in light & dark) */
  .block-container { padding: 1.0rem 0.9rem 3rem; max-width: 560px; }
  #MainMenu, footer, header { visibility: hidden; }
  div[data-testid="stMetric"] {
      background: var(--secondary-background-color, rgba(128,128,128,.08));
      border: 1px solid rgba(128,128,128,.28);
      border-radius: 12px; padding: 8px 10px; }
  div[data-testid="stMetricValue"] { font-size: 1.05rem;
      color: var(--text-color);
      font-family: ui-monospace, Menlo, monospace; }
  div[data-testid="stMetricLabel"] { font-size: 0.70rem; }
  .badge { font-family: ui-monospace, Menlo, monospace; font-weight: 700;
      font-size: 0.75rem; letter-spacing: .08em; border-radius: 6px;
      padding: 3px 9px; float: right; }
  .tkr { font-family: ui-monospace, Menlo, monospace; font-weight: 700;
      font-size: 1.0rem; color: var(--text-color); }
  .sub { color: #8A919C; font-size: 0.74rem; }
  .strip-wrap { margin: 8px 0 2px; }
  .strip { position: relative; height: 8px; border-radius: 4px;
      background: rgba(128,128,128,.25); overflow: visible; }
  .zone { position: absolute; top: 0; bottom: 0; }
  .marker { position: absolute; top: -2px; width: 10px; height: 12px;
      border-radius: 3px; background: var(--text-color, #16191D);
      border: 2px solid var(--background-color, #fff);
      box-shadow: 0 1px 2px rgba(0,0,0,.25); }
  .striplbl { display: flex; justify-content: space-between; color: #8A919C;
      font-family: ui-monospace, Menlo, monospace; font-size: 0.62rem;
      margin-top: 3px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def badge(label: str) -> str:
    fg, bg = {"BUY": (C_BUY, C_BUY_BG), "SELL": (C_SELL, C_SELL_BG)}.get(
        label, (C_HOLD, C_HOLD_BG))
    return f'<span class="badge" style="color:{fg};background:{bg}">{label}</span>'


def decision_strip(value: float, entry: float, rng: float,
                   left: str, mid: str, right: str,
                   left_bg: str, right_bg: str) -> str:
    v = max(-rng, min(rng, 0.0 if value is None or np.isnan(value) else value))
    pos = (v + rng) / (2 * rng) * 100
    e_l = (rng - entry) / (2 * rng) * 100
    e_r = (rng + entry) / (2 * rng) * 100
    return f"""
    <div class="strip-wrap">
      <div class="strip">
        <div class="zone" style="left:0;width:{e_l:.1f}%;background:{left_bg}"></div>
        <div class="zone" style="left:{e_r:.1f}%;right:0;background:{right_bg}"></div>
        <div class="marker" style="left:calc({pos:.1f}% - 5px)"></div>
      </div>
      <div class="striplbl"><span>{left}</span><span>{mid}</span><span>{right}</span></div>
    </div>"""


def make_demo(tickers: list[str], interval: str = "1d"
              ) -> dict[str, pd.DataFrame]:
    """Synthetic OHLCV so the app works with no internet (any interval)."""
    ppy = ktf.periods_per_year(interval)
    if interval == "1d":
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=520)
        bar_vol, hl = 0.013, 0.008
    else:
        n_req = 60 * (ppy // 252)                   # ~60 sessions of bars
        freq = {"1h": "1h", "15m": "15min"}.get(interval, "15min")
        idx = pd.date_range(end=pd.Timestamp.now().floor("min"),
                            periods=n_req, freq=freq)
        bar_vol = 0.013 * np.sqrt(252 / ppy)        # scale noise to bar size
        hl = bar_vol * 0.6
    n = len(idx)                                    # source of truth for length
    out = {}
    for i, t in enumerate(tickers):
        rs = np.random.RandomState(100 + i)
        d2 = rs.choice([0.30, 0.10, -0.20]) / ppy   # annualised regime drift
        cut = int(n * 0.6)
        drift = np.concatenate([np.full(cut, 0.08 / ppy), np.full(n - cut, d2)])
        px = 100 * np.exp(np.cumsum(drift + rs.normal(0, bar_vol, n)))
        out[t] = pd.DataFrame(
            {"Open": px, "High": px * (1 + hl), "Low": px * (1 - hl),
             "Close": px, "Volume": rs.lognormal(15, 0.3, n)}, index=idx)
    return out


@st.cache_data(show_spinner=False)
def load_data(tickers: tuple[str, ...], years: float, interval: str,
              demo: bool, freshness_key: int) -> dict[str, pd.DataFrame]:
    # freshness_key buckets time so intraday data refetches every 5 min
    # while daily data is reused for an hour.
    if demo:
        return make_demo(list(tickers), interval)
    return ktf.DataHandler(lookback_years=years,
                           interval=interval).fetch(list(tickers))


# ------------------------------------------------------------------ header
st.markdown('<div style="font-size:1.35rem;font-weight:700;'
            'letter-spacing:-0.02em">Kalman Screener</div>'
            '<div class="sub">3-month horizon · live Yahoo Finance data</div>',
            unsafe_allow_html=True)
st.write("")

# ------------------------------------------------------------------ glossary
GLOSSARY_VALUATION = [
    ("WACC / Discount Rate", C_KF, [
        "The annual rate used to discount future cash flows back to today — "
        "the return you demand for the risk taken.",
        "<b>Lower WACC → higher fair value</b> (future cash worth more today); "
        "<b>higher WACC → lower fair value</b>.",
        "It also sets the exit multiple: high-quality megacaps arguably "
        "deserve a <b>lower</b> WACC (~7%) than the 9% default.",
    ]),
    ("FCF Growth Override", C_KF, [
        "Bypasses the historical / analyst growth estimate and applies "
        "<b>your own</b> annual free-cash-flow growth assumption.",
        "Off → uses Yahoo's trailing revenue growth as a proxy. "
        "On → uses your slider value for the full 5-year projection.",
        "Use it to <b>stress-test</b>: the caption always shows both the rate "
        "in use and the historical baseline for comparison.",
    ]),
    ("Fade to 2.5% (Fade Mechanism)", C_KF, [
        "The model does <b>not</b> hold the starting growth rate flat. It "
        "decays it <b>linearly</b> toward the 2.5% terminal rate over 5 years.",
        "Example: a +17% start steps down roughly 17 → 14 → 11 → 8 → 2.5%.",
        "This mirrors reality — <b>no company compounds at peak growth "
        "forever</b>; competition and scale pull it toward the economy's rate.",
    ]),
    ("Terminal Growth & Terminal Multiple", C_HOLD, [
        "<b>Terminal growth</b> (2.5%) = the perpetual rate cash flows grow "
        "after year 5, roughly long-run nominal GDP.",
        "The Gordon formula locks in the <b>exit multiple = 1 / (WACC − "
        "terminal growth)</b>. At 9% / 2.5% that's ~15.8× FCF.",
        "This is why the DCF is anchored: change WACC or terminal growth and "
        "the entire valuation re-prices through that multiple.",
    ]),
    ("DCF Fair Value vs. Market Price", C_HOLD, [
        "<b>Fair value</b> = present value of 5 years of projected FCF + the "
        "discounted terminal value, divided by shares.",
        "A strict DCF is <b>structurally conservative</b>: it caps the exit "
        "multiple and won't extrapolate hyper-growth beyond the window.",
        "The market often pays far more (e.g. ~43× FCF for Apple), pricing in "
        "lower risk or longer growth — the gap is a <b>question to "
        "investigate</b>, not proof of mispricing.",
    ]),
    ("Medium-Term (Kalman) Drift", C_BUY, [
        "The Kalman filter strips daily noise from price to estimate the "
        "<b>true underlying trend</b> — a lag-free moving average.",
        "<b>Drift</b> = that trend's annualised slope. Positive = structural "
        "uptrend; negative = downtrend.",
        "It's a <b>momentum overlay</b> on the fundamental view: the matrix "
        "crosses 'is it cheap?' (DCF) with 'is it trending?' (Kalman).",
    ]),
    ("ROE — Megacap Caution", C_SELL, [
        "Return on Equity = net income ÷ shareholder equity. Normally a "
        "quality gauge — but <b>distorted by buybacks</b>.",
        "Heavy repurchases (Apple) <b>shrink book equity toward zero</b>, "
        "inflating the denominator's effect and pushing ROE to 100%+.",
        "<b>ROIC is more reliable</b> for these names — read a sky-high ROE as "
        "a balance-sheet artifact, not 100%+ returns on capital.",
    ]),
    ("Scorecard & Verdict", C_HOLD, [
        "A 0–10 cross-check on the DCF scoring ROE, P/E, PEG and revenue "
        "growth — guards against a single bad input swinging the verdict.",
        "<b>Undervalued</b> = DCF upside &gt; +20%; <b>Overvalued</b> = "
        "downside &gt; 15%; otherwise <b>Fairly Valued</b> (asymmetry = "
        "built-in margin of safety).",
    ]),
    ("Hybrid Decision Matrix", C_BUY, [
        "Combines the two timeframes into one action: fundamental verdict "
        "(1–5y) × Kalman trend (≈3-month).",
        "<b>Undervalued + BUY → Accumulate</b> (strongest); "
        "<b>Overvalued + BUY → Speculative</b> (momentum only, tight stops); "
        "<b>Undervalued + SELL → Wait</b> (value-trap risk).",
        "When the two views <b>disagree</b>, the matrix surfaces it rather "
        "than averaging them away.",
    ]),
]


GLOSSARY_KALMAN = [
    ("Kalman Filter", C_KF, [
        "A recursive estimator that separates the <b>true signal</b> from "
        "random noise. Here it smooths price into a clean trend line.",
        "Unlike a moving average it has <b>no lag</b> — it updates its estimate "
        "the instant a new price arrives, weighting it against its prediction.",
    ]),
    ("Q — Process Noise (log\u2081\u2080)", C_KF, [
        "How much the 'true' trend is allowed to move each bar. Shown as "
        "<b>log\u2081\u2080</b>: a value of \u22125 means Q = 10\u207b\u2075 = "
        "0.00001.",
        "<b>Higher Q (e.g. \u22123)</b> → line reacts fast, hugs price, more "
        "noise. <b>Lower Q (e.g. \u22126)</b> → very smooth, slow to turn.",
    ]),
    ("R — Measurement Noise (log\u2081\u2080)", C_KF, [
        "How noisy the model assumes each raw price is. Also log\u2081\u2080, so "
        "\u22123 = R = 0.001.",
        "<b>Higher R</b> → trust each tick less, lean on the model's own "
        "prediction (smoother). It's the <b>counterweight to Q</b>.",
        "What matters is the <b>ratio Q/R</b>: large Q/R hugs price, small Q/R "
        "produces a calm, slow line.",
    ]),
    ("Drift (Annualised Slope)", C_BUY, [
        "The slope of the Kalman trend, scaled to a yearly rate. <b>+45% "
        "drift</b> = the current trend implies a 45%/yr rise if sustained.",
        "Positive = structural uptrend, negative = downtrend. The "
        "<b>entry-drift</b> slider sets how steep it must be to trigger a BUY.",
        "Scaling adapts to the data: daily uses 252 bars/yr, 15-min uses 6,552, "
        "so the % is comparable across resolutions.",
    ]),
    ("Z-Score (Pairs Spread)", C_HOLD, [
        "For two related assets, the <b>spread</b> is what's left after hedging "
        "one against the other. Z-score = how many standard deviations that "
        "spread sits from its mean.",
        "<b>|z| &gt; entry</b> (e.g. 2.0) → the spread is stretched → bet on it "
        "reverting. <b>|z| &lt; exit</b> (e.g. 0.5) → reverted → close.",
        "Self-normalising, so it works across pairs with different volatility.",
    ]),
    ("Hedge Ratio \u03b2 (beta)", C_HOLD, [
        "How many units of the hedge asset offset one unit of the traded asset. "
        "A second Kalman filter estimates it <b>dynamically</b> over time.",
        "\u03b2 = 1.25 → short 1.25 shares of the hedge for every share held — "
        "the combination is what mean-reverts.",
    ]),
    ("Cointegration (p-value)", C_HOLD, [
        "Tests whether two prices share a <b>long-run equilibrium</b> they keep "
        "returning to — the precondition for pairs trading.",
        "<b>p &lt; 0.05</b> → statistically cointegrated, pair accepted. Higher "
        "→ rejected (the spread could wander forever — a value trap).",
    ]),
    ("ATR · Stop · Target · R:R", C_SELL, [
        "<b>ATR</b> (Average True Range) measures a stock's typical move. Stops "
        "and targets are set as multiples of it, so they adapt to volatility.",
        "Default stop = 2.5\u00d7 ATR, target = 5\u00d7 ATR → <b>Risk:Reward = "
        "2:1</b>. You risk 1 to make 2.",
    ]),
    ("Regime Veto (Vol Percentile)", C_SELL, [
        "Ranks current volatility against the past year. When it's in the top "
        "decile (90th pct+), <b>new entries are blocked</b>.",
        "Keeps you out during market-wide panics where signals are unreliable. "
        "Shows as 'regime OK' or 'VETO' on each card.",
    ]),
    ("Data Resolution", C_KF, [
        "<b>Daily (2y)</b> — the core 3-month strategy; cleanest signal.",
        "<b>1h / 15m (60d)</b> — near-real-time price for timing entries. "
        "Per-bar noise is far higher, so thresholds auto-loosen and these are "
        "best as a <b>timing overlay</b>, not the primary signal.",
    ]),
    ("Decision Strip", C_BUY, [
        "The little bar under each card. The marker shows where current drift "
        "(or z-score) sits between the <b>sell zone</b>, neutral band, and "
        "<b>buy zone</b> — so you see <i>why</i> a card is HOLD, not just that "
        "it is.",
    ]),
]


def render_glossary(entries):
    """Scannable dark/light-adaptive glossary. Body text inherits Streamlit's
    themed text color (no hardcoded fallback, so it's readable on any theme)."""
    cards = []
    for title, accent, points in entries:
        items = "".join(f'<li style="margin:3px 0">{p}</li>' for p in points)
        cards.append(
            f'<div style="border:1px solid rgba(128,128,128,.22);'
            f'border-left:3px solid {accent};border-radius:10px;'
            f'padding:10px 12px;margin-bottom:8px;'
            f'background:rgba(128,128,128,.08)">'
            f'<div style="font-family:ui-monospace,Menlo,monospace;'
            f'font-weight:700;font-size:0.82rem;color:{accent};'
            f'margin-bottom:4px">{title}</div>'
            f'<ul style="margin:0;padding-left:16px;font-size:0.76rem;'
            f'line-height:1.45;opacity:0.92">{items}</ul>'
            f'</div>')
    st.markdown("".join(cards), unsafe_allow_html=True)


tab_screen, tab_value = st.tabs(["Screener", "Long-Term Valuation"])

with tab_screen:
    # ------------------------------------------------------------------ universe
    basket_names = {k: b["name"] for k, b in ktf.BASKETS.items()}
    basket_names["4"] = "Custom tickers"
    sel = st.radio("Universe", list(basket_names),
                   format_func=lambda k: basket_names[k], horizontal=False,
                   label_visibility="collapsed")

    if sel == "4":
        custom = st.text_input("Tickers", placeholder="AAPL, JPM, XOM",
                               label_visibility="collapsed")
        tickers, pairs, label = ktf.resolve_universe("custom", custom or "")
    else:
        tickers, pairs, label = ktf.resolve_universe(sel)
        st.markdown(f'<div class="sub">{", ".join(tickers)}</div>',
                    unsafe_allow_html=True)

    # ------------------------------------------------------------------ settings
    RESOLUTIONS = {
        "Daily · 2y history": "1d",
        "1-hour bars · last 60d (near-real-time)": "1h",
        "15-min bars · last 60d (near-real-time)": "15m",
    }

    with st.expander("Settings"):
        res_label = st.selectbox(
            "Data resolution", list(RESOLUTIONS),
            help="Daily uses 2 years of end-of-day bars (best for the "
                 "3-month strategy). 1h / 15m pull the last 60 days of "
                 "intraday bars so the latest price is near-real-time — "
                 "better for timing entries than for the core signal.")
        interval = RESOLUTIONS[res_label]
        ppy = ktf.periods_per_year(interval)
        bars_per_day = max(1, ppy // 252)

        years = st.slider("History (years, daily mode only)", 1.0, 5.0, 2.0, 0.5,
                          disabled=(interval != "1d"),
                          help="How much daily history to load. More history "
                               "= more backtest trades but older regimes.")

        st.caption("Kalman filter tuning — higher Q tracks live price faster "
                   "(less smoothing); higher R trusts the model over raw ticks.")
        q_exp = st.slider(
            "Kalman sensitivity Q (log\u2081\u2080)", -6.0, -2.0, -5.0, 0.5,
            help="Process noise. Shown as log\u2081\u2080, so \u22125 means "
                 "Q = 10\u207b\u2075 = 0.00001. HIGHER (e.g. \u22123) = the "
                 "Kalman line reacts fast to new prices (less smoothing, more "
                 "noise). LOWER (e.g. \u22126) = very smooth, slow to turn.")
        r_exp = st.slider(
            "Measurement noise R (log\u2081\u2080)", -5.0, -1.0, -3.0, 0.5,
            help="Measurement noise. Also log\u2081\u2080, so \u22123 means "
                 "R = 0.001. HIGHER R = the filter distrusts each raw tick and "
                 "leans on its own prediction (smoother). It's the counterweight "
                 "to Q: the ratio Q/R sets how tightly the line hugs price.")
        kf_q, kf_r = 10.0 ** q_exp, 10.0 ** r_exp

        drift_default = 10 if interval == "1d" else 25
        slope_entry = st.slider(
            "Trend entry drift (ann. %)", 2, 80, drift_default, 1,
            help="Minimum annualised Kalman slope to trigger a BUY. The drift "
                 "is the filter's trend, annualised. 10% = only act on trends "
                 "implying \u2265+10%/yr.") / 100
        persist = st.slider(
            "Slope persistence (bars)", 2, 60,
            5 if interval == "1d" else bars_per_day, 1,
            help="The drift must stay above the threshold this many bars in a "
                 "row before a signal fires — filters out one-day head-fakes.")
        vol_veto = st.slider(
            "Vol veto percentile", 70, 99, 90, 1,
            help="Regime filter. Blocks new entries when recent volatility is "
                 "in its top X% vs the past year — keeps you out during "
                 "market-wide stress. 90 = veto the most volatile 10%.") / 100
        entry_z = st.slider(
            "Pairs entry |z|", 1.0, 3.5, 2.0, 0.1,
            help="Z-score = how many standard deviations the pair's spread is "
                 "from its mean. Enter a mean-reversion trade when |z| exceeds "
                 "this. 2.0 = act when the spread is a 2-sigma stretch.")
        exit_z = st.slider(
            "Pairs exit |z|", 0.0, 1.5, 0.5, 0.1,
            help="Close the pairs trade once the spread reverts back inside "
                 "this z-score band. 0.5 = exit near the mean.")
        run_pairs = st.toggle("Run pairs leg", value=True,
                              help="Also test each name against its hedge ETF "
                                   "for mean-reversion (pairs) opportunities.")
        demo = st.toggle("Demo data (offline / no Yahoo)",
                         value=bool(os.environ.get("KALMAN_DEMO")),
                         help="Use synthetic data so the UI works with no "
                              "internet / when Yahoo rate-limits.")

    run = st.button("Run screen", type="primary", use_container_width=True)

    # ------------------------------------------------------------------ screen
    if run and not tickers:
        st.warning("Enter at least one ticker.")

    if run and tickers:
        hedges = sorted({t for p in pairs for t in p} - set(tickers)) if run_pairs else []
        span = f"{years:.0f}y daily" if interval == "1d" else f"60d @ {interval}"
        with st.spinner(f"Fetching {len(tickers) + len(hedges)} tickers ({span})..."):
            ttl = 3600 if interval == "1d" else 300
            data = load_data(tuple(sorted(set(tickers) | set(hedges))), years,
                             interval, demo, int(time.time() // ttl))

        missing = [t for t in tickers if t not in data]
        if missing:
            st.warning(f"No data for: {', '.join(missing)} — skipped.")
        if not any(t in data for t in tickers):
            st.error("Nothing downloaded. Check tickers or your connection, "
                     "or switch on Demo data in Settings.")
            st.stop()

        risk = ktf.RiskParams(vol_pctile_veto=vol_veto)
        tp = ktf.TrendParams(slope_entry_ann=slope_entry, slope_persist=persist)
        pp = ktf.PairsParams(entry_z=entry_z, exit_z=exit_z)
        rf = ktf.RegimeFilter(risk, ppy)
        # Slope noise scaled by (252/ppy)^2 -> comparable annualised-drift
        # mobility at any bar frequency (reduces to Q/100 in daily mode).
        q_slope = (kf_q / 100.0) * (252 / ppy) ** 2
        trend = ktf.TrendStrategy(
            tp, ktf.KalmanTrendFilter(q_level=kf_q, q_slope=q_slope, r=kf_r,
                                      periods_per_year=ppy), rf)
        pairs_strat = ktf.PairsStrategy(pp, ktf.KalmanPairsFilter(), rf)

        # ---------------- trend cards ----------------
        results = []
        for t in tickers:
            if t not in data:
                continue
            sig_df = trend.generate(data[t])
            bt = ktf.Backtester(risk, mode="trend",
                                periods_per_year=ppy).run(sig_df)
            last = sig_df.iloc[-1]
            lab = {1: "BUY", -1: "SELL", 0: "HOLD"}[int(last["signal"])]
            results.append((t, sig_df, bt, last, lab))
        results.sort(key=lambda r: {"BUY": 0, "SELL": 1, "HOLD": 2}[r[4]])

        n_active = sum(1 for r in results if r[4] != "HOLD")
        st.markdown(f'<div class="sub" style="margin:6px 0 2px">'
                    f'{label} · {n_active} active trend signal'
                    f'{"" if n_active == 1 else "s"}</div>', unsafe_allow_html=True)

        for t, sig_df, bt, last, lab in results:
            with st.container(border=True):
                ts = sig_df.index[-1]
                ts_str = (ts.strftime("%d %b") if interval == "1d"
                          else ts.strftime("%d %b %H:%M"))
                st.markdown(
                    f'<span class="tkr">{t}</span>{badge(lab)}'
                    f'<div class="sub">Trend · KF drift '
                    f'{last["kf_slope_ann"]*100:.1f}% ann. · regime '
                    f'{"OK" if last["regime_ok"] else "VETO"} · '
                    f'last bar {ts_str}</div>',
                    unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Price", f'{last["Close"]:.2f}')
                c2.metric("Kalman", f'{last["kf_price"]:.2f}')
                rr = risk.atr_target_mult / risk.atr_stop_mult
                c3.metric("R : R", f"{rr:.1f} : 1")

                st.markdown(decision_strip(
                    last["kf_slope_ann"], slope_entry, max(0.35, slope_entry * 3),
                    "sell zone", "neutral", "buy zone", C_SELL_BG, C_BUY_BG),
                    unsafe_allow_html=True)

                with st.expander("Chart & backtest"):
                    chart = sig_df[["Close", "kf_price"]].rename(
                        columns={"Close": "Price", "kf_price": "Kalman"})
                    st.line_chart(chart, height=200,
                                  color=["#9CA3AF", C_KF])
                    m = bt["metrics"]
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Return", m["Total Return"].strip())
                    b2.metric("Sharpe", m["Sharpe Ratio"].strip())
                    b3.metric("Max DD", m["Max Drawdown"].strip())
                    b4.metric("Win rate", str(m["Win Rate"]).strip())

        # ---------------- pairs cards ----------------
        if run_pairs and pairs:
            st.markdown('<div class="sub" style="margin:14px 0 2px">'
                        'Pairs / mean reversion (Kalman β)</div>',
                        unsafe_allow_html=True)
            for a, b in pairs:
                if a not in data or b not in data:
                    continue
                sig_df = pairs_strat.generate(data[a], data[b])
                if sig_df is None:
                    st.markdown(f'<div class="sub">· {a}/{b} — rejected '
                                f'(no cointegration on this window)</div>',
                                unsafe_allow_html=True)
                    continue
                last = sig_df.iloc[-1]
                z = last["z"]
                lab = ("BUY" if last["signal"] == 1
                       else "SELL" if last["signal"] == -1 else "HOLD")
                with st.container(border=True):
                    st.markdown(
                        f'<span class="tkr">{a} / {b}</span>{badge(lab)}'
                        f'<div class="sub">β = {last["beta"]:.3f} · spread z = '
                        f'{z:.2f} · coint p = '
                        f'{sig_df.attrs.get("coint_pval", float("nan")):.3f}</div>',
                        unsafe_allow_html=True)
                    st.markdown(decision_strip(
                        z, entry_z, 3.5, "long spread", "flat", "short spread",
                        C_BUY_BG, C_SELL_BG), unsafe_allow_html=True)
                    if lab == "BUY":
                        st.markdown(f'<div class="sub">Long {a}, short '
                                    f'{last["beta"]:.2f}× {b}; exit when '
                                    f'|z| &lt; {exit_z:.1f}.</div>',
                                    unsafe_allow_html=True)
                    elif lab == "SELL":
                        st.markdown(f'<div class="sub">Short {a}, long '
                                    f'{last["beta"]:.2f}× {b}; exit when '
                                    f'|z| &lt; {exit_z:.1f}.</div>',
                                    unsafe_allow_html=True)
                    with st.expander("Spread z-score"):
                        st.line_chart(sig_df[["z"]], height=180, color=[C_KF])

        st.markdown('<div class="sub" style="margin-top:14px">Research tool only — '
                    'Kalman local-linear-trend + dynamic-β models, ATR stops, '
                    'vol-percentile regime veto, 63-day time exit. '
                    'Not investment advice.</div>', unsafe_allow_html=True)

    # ---- Kalman / screener glossary (always visible) ----
    with st.expander("📖 Glossary — Kalman, Q/R, z-score & screener terms"):
        render_glossary(GLOSSARY_KALMAN)


# ==============================================================================
# LONG-TERM VALUATION TAB — fundamentals + DCF + hybrid decision matrix
# ==============================================================================
def _row(df, *names):
    """First matching row (as Series, newest column first) from a statement."""
    if df is None or getattr(df, "empty", True):
        return None
    for nm in names:
        if nm in df.index:
            ser = df.loc[nm].dropna()
            if len(ser):
                return ser
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def get_fundamentals(t: str, demo: bool) -> dict:
    """Core metrics from the lightweight `ticker.info` endpoint only.

    Avoids ticker.financials / .balance_sheet / .cashflow, which hit a
    heavier endpoint that Yahoo rate-limits hard on shared servers.
    """
    if demo:
        rs = np.random.RandomState(abs(hash(t)) % (2 ** 31))
        shares = rs.uniform(0.5, 8) * 1e9
        price = rs.uniform(40, 400)
        g = rs.uniform(-0.05, 0.22)
        return {"name": f"{t} (demo)", "price": price, "shares": shares,
                "trailing_pe": rs.uniform(11, 42),
                "forward_pe": rs.uniform(10, 35),
                "ps": rs.uniform(1.5, 12),
                "peg": rs.uniform(0.6, 3.5),
                "roa": rs.uniform(0.02, 0.22),
                "roe": rs.uniform(0.08, 0.55),
                "rev_growth": g,
                "debt_to_equity": rs.uniform(20, 200),
                "fcf": shares * price * rs.uniform(0.02, 0.07),
                "trailing_growth": g, "currency": "USD"}

    if yf is None:
        raise ImportError("yfinance is required for live fundamentals.")

    try:
        info = yf.Ticker(t).info or {}
    except Exception as e:
        raise RuntimeError(f"info endpoint failed: {e}")

    def g(key, default=np.nan):
        v = info.get(key, default)
        return v if v is not None else default

    d2e = g("debtToEquity")                      # Yahoo reports as a percent
    return {
        "name": info.get("shortName") or info.get("longName") or t,
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "shares": info.get("sharesOutstanding"),
        "trailing_pe": g("trailingPE"),
        "forward_pe": g("forwardPE"),
        "ps": g("priceToSalesTrailing12Months"),
        "peg": g("trailingPegRatio"),
        "roa": g("returnOnAssets"),
        "roe": g("returnOnEquity"),            # capital-efficiency proxy
        "rev_growth": g("revenueGrowth"),
        "debt_to_equity": d2e,
        "fcf": g("freeCashflow"),
        # Revenue growth is the best forward-growth proxy available from info
        "trailing_growth": g("revenueGrowth"),
        "currency": info.get("currency", "USD"),
    }


def dcf_fair_value(fcf0, g0, wacc, shares, g_term=0.025, years=5):
    """5y explicit FCF with growth fading linearly to a terminal rate,
    Gordon terminal value. Equity value per share (net debt omitted —
    the lightweight info endpoint has no reliable balance-sheet cash)."""
    if not fcf0 or fcf0 <= 0 or not shares or wacc <= g_term:
        return None
    g0 = float(np.clip(g0 if np.isfinite(g0) else 0.05, -0.10, 0.25))
    pv, fcf = 0.0, fcf0
    for k in range(1, years + 1):
        g_k = g0 + (g_term - g0) * (k - 1) / (years - 1)     # linear fade
        fcf *= (1 + g_k)
        pv += fcf / (1 + wacc) ** k
    tv = fcf * (1 + g_term) / (wacc - g_term) / (1 + wacc) ** years
    return (pv + tv) / shares


def scorecard(f, upside):
    """0-10 multi-factor score as a cross-check on the DCF (info fields)."""
    pts, notes = 0, []
    roe = f.get("roe")
    if roe is not None and np.isfinite(roe):
        pts += 2 if roe > 0.20 else 1 if roe > 0.10 else 0
        notes.append(f"ROE {roe:.0%}")
    pe = f.get("trailing_pe")
    if pe and np.isfinite(pe) and pe > 0:
        pts += 2 if pe < 18 else 1 if pe < 28 else 0
        notes.append(f"P/E {pe:.1f}")
    peg = f.get("peg")
    if peg and np.isfinite(peg) and peg > 0:
        pts += 2 if peg < 1.0 else 1 if peg < 2.0 else 0
        notes.append(f"PEG {peg:.2f}")
    rg = f.get("rev_growth")
    if rg is not None and np.isfinite(rg):
        pts += 2 if rg > 0.12 else 1 if rg > 0.04 else 0
        notes.append(f"Rev growth {rg:+.0%}")
    if upside is not None:
        pts += 2 if upside > 0.20 else 1 if upside > 0 else 0
    return pts, " · ".join(notes)




HYBRID_MATRIX = {
    ("Undervalued", "BUY"):  ("ACCUMULATE", C_BUY,
        "Fundamentals cheap and trend confirms — strongest setup; size normally."),
    ("Undervalued", "HOLD"): ("WATCHLIST", C_HOLD,
        "Cheap but no trend yet — set an alert for the Kalman drift turning positive."),
    ("Undervalued", "SELL"): ("WAIT", C_HOLD,
        "Value trap risk: cheap with a falling trend. Wait for the knife to land."),
    ("Fairly Valued", "BUY"):  ("MOMENTUM ONLY", C_BUY,
        "Trend trade, not an investment — keep the 3-month exits strict."),
    ("Fairly Valued", "HOLD"): ("NEUTRAL", C_HOLD,
        "Nothing to do here today."),
    ("Fairly Valued", "SELL"): ("REDUCE", C_SELL,
        "No valuation cushion and trend is down — trim or avoid."),
    ("Overvalued", "BUY"):  ("SPECULATIVE", C_HOLD,
        "Expensive momentum — only with tight stops; never a long-term hold."),
    ("Overvalued", "HOLD"): ("AVOID", C_SELL,
        "Expensive with no trend support."),
    ("Overvalued", "SELL"): ("EXIT / AVOID", C_SELL,
        "Both views negative — clearest no."),
}


with tab_value:
    st.markdown('<div class="sub" style="margin:4px 0 8px">Structural 1-5y '
                'view: lightweight Yahoo <code>info</code> fundamentals + '
                'simplified DCF, cross-checked against the medium-term '
                'Kalman trend.</div>', unsafe_allow_html=True)

    default_t = tickers[0] if tickers else "AAPL"
    val_ticker = st.text_input("Ticker to value", value=default_t).strip().upper()
    wacc = st.slider("Discount rate / WACC (%)", 6.0, 14.0, 9.0, 0.5) / 100

    # ---- FCF growth override ----
    override_on = st.toggle("Override FCF growth manually", value=False,
                            help="Off = use the trailing revenue-growth "
                                 "proxy from Yahoo. On = use your slider value "
                                 "for the 5-year DCF projection.")
    manual_growth = st.slider("FCF growth override (% / yr)", -10, 25, 8, 1,
                              disabled=not override_on) / 100

    analyze = st.button("Analyze fundamentals", type="primary",
                        use_container_width=True)

    if analyze and val_ticker:
        try:
            with st.spinner(f"Fetching {val_ticker} (info endpoint)..."):
                f = get_fundamentals(val_ticker, demo)
        except Exception as e:
            st.error(f"Could not load fundamentals for {val_ticker}: {e}")
            st.stop()

        if not f.get("price") or not f.get("shares"):
            st.error(f"{val_ticker}: no price/share data — "
                     "ETFs and some foreign listings aren't covered.")
            st.stop()

        # ---------- which growth rate feeds the DCF ----------
        hist_g = f.get("trailing_growth", np.nan)
        if override_on:
            used_g = manual_growth
            g_source = (f"manual override <b>{manual_growth:+.0%}</b> "
                        f"(historical proxy {hist_g:+.0%})"
                        if np.isfinite(hist_g) else
                        f"manual override <b>{manual_growth:+.0%}</b>")
        else:
            used_g = hist_g if np.isfinite(hist_g) else 0.05
            g_source = (f"historical proxy <b>{hist_g:+.0%}</b> "
                        f"(Yahoo revenue growth)"
                        if np.isfinite(hist_g) else
                        "default <b>+5%</b> (no Yahoo growth field)")

        # ---------- fundamentals row 1: valuation ----------
        st.markdown(f'<span class="tkr">{f["name"]}</span>'
                    f'<div class="sub">Trailing fundamentals · '
                    f'{f["currency"]} · source: info endpoint</div>',
                    unsafe_allow_html=True)
        pe = lambda v: "—" if not v or not np.isfinite(v) else f"{v:.1f}"
        pc = lambda v: "—" if v is None or not np.isfinite(v) else f"{v:+.1%}"
        rt = lambda v: "—" if v is None or not np.isfinite(v) else f"{v:.1%}"

        c1, c2, c3 = st.columns(3)
        c1.metric("P/E (ttm)", pe(f.get("trailing_pe")))
        c2.metric("Forward P/E", pe(f.get("forward_pe")))
        c3.metric("P/S (ttm)", pe(f.get("ps")))

        c4, c5, c6 = st.columns(3)
        c4.metric("PEG", pe(f.get("peg")))
        c5.metric("ROE", rt(f.get("roe")))
        c6.metric("ROA", rt(f.get("roa")))

        c7, c8, c9 = st.columns(3)
        c7.metric("Rev growth", pc(f.get("rev_growth")))
        d2e = f.get("debt_to_equity")
        c8.metric("Debt/Equity", "—" if not d2e or not np.isfinite(d2e)
                  else f"{d2e/100:.2f}" if d2e > 5 else f"{d2e:.2f}")
        fcf = f.get("fcf")
        c9.metric("Free cash flow", "—" if not fcf or not np.isfinite(fcf)
                  else f"{fcf/1e9:.1f}B")

        # ---------- DCF ----------
        fair = dcf_fair_value(f.get("fcf"), used_g, wacc, f["shares"])
        upside = (fair / f["price"] - 1) if fair else None

        if fair:
            verdict = ("Undervalued" if upside > 0.20 else
                       "Overvalued" if upside < -0.15 else "Fairly Valued")
        else:
            pts, _ = scorecard(f, None)
            verdict = ("Undervalued" if pts >= 6 else
                       "Overvalued" if pts <= 2 else "Fairly Valued")
            st.info("Free cash flow unavailable — verdict from the factor "
                    "scorecard only.")

        pts, notes = scorecard(f, upside)
        v_color = {"Undervalued": C_BUY, "Overvalued": C_SELL,
                   "Fairly Valued": C_HOLD}[verdict]
        with st.container(border=True):
            st.markdown(
                f'<span class="tkr">DCF (5y, fade to 2.5%)</span>'
                f'<span class="badge" style="color:{v_color};'
                f'background:rgba(128,128,128,.12)">{verdict.upper()}</span>',
                unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            d1.metric("Price", f'{f["price"]:.2f}')
            d2.metric("DCF fair value", "—" if not fair else f"{fair:.2f}")
            d3.metric("Upside", "—" if upside is None else f"{upside:+.0%}")
            st.markdown(f'<div class="sub">Growth assumption: {g_source} · '
                        f'fading to 2.5% · WACC {wacc:.1%}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="sub">Scorecard {pts}/10 · {notes}</div>',
                        unsafe_allow_html=True)

        # ---------- medium-term Kalman view (always daily, 2y) ----------
        with st.spinner("Computing medium-term Kalman trend (daily, 2y)..."):
            ddata = load_data((val_ticker,), 2.0, "1d", demo,
                              int(time.time() // 3600))
        if val_ticker in ddata:
            kf_d = ktf.KalmanTrendFilter(q_level=kf_q, q_slope=kf_q / 100,
                                         r=kf_r, periods_per_year=252)
            tr = ktf.TrendStrategy(
                ktf.TrendParams(slope_entry_ann=slope_entry,
                                slope_persist=persist),
                kf_d, ktf.RegimeFilter(ktf.RiskParams(), 252))
            sig_df = tr.generate(ddata[val_ticker])
            last = sig_df.iloc[-1]
            tech = {1: "BUY", -1: "SELL", 0: "HOLD"}[int(last["signal"])]
            drift = last["kf_slope_ann"]
        else:
            tech, drift = "HOLD", np.nan
            st.warning("No daily price history — technical view defaulted "
                       "to HOLD.")

        # ---------- hybrid decision matrix ----------
        action, a_color, advice = HYBRID_MATRIX[(verdict, tech)]
        with st.container(border=True):
            st.markdown(
                f'<span class="tkr">Hybrid Decision Matrix</span>'
                f'<span class="badge" style="color:{a_color};'
                f'background:rgba(128,128,128,.12)">{action}</span>',
                unsafe_allow_html=True)
            h1, h2 = st.columns(2)
            h1.metric("Long-term (fundamental)", verdict)
            h2.metric("Medium-term (Kalman)",
                      f"{tech}" + ("" if not np.isfinite(drift)
                                   else f" · {drift:+.0%} drift"))
            st.markdown(f'<div class="sub">{advice}</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="sub" style="margin-top:10px">Simplified '
                    'model on Yahoo summary fields — revenue growth is a '
                    'proxy for FCF growth, net debt is omitted, and earnings '
                    'quality / guidance are not captured. Use the override '
                    'to stress-test. Not investment advice.</div>',
                    unsafe_allow_html=True)

    # ---- Glossary (always visible, independent of analysis) ----
    with st.expander("📖 Glossary — valuation terms & how they move the output"):
        render_glossary(GLOSSARY_VALUATION)
