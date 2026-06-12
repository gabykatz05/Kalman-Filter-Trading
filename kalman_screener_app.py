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

import numpy as np
import pandas as pd
import streamlit as st

import kalman_trading_framework as ktf

# ------------------------------------------------------------------ page setup
st.set_page_config(page_title="Kalman Screener", page_icon="📈",
                   layout="centered", initial_sidebar_state="collapsed")

C_KF, C_BUY, C_SELL, C_HOLD = "#2B5DA8", "#0E7C4F", "#C03434", "#8A6D1F"
C_BUY_BG, C_SELL_BG, C_HOLD_BG = "#E4F2EB", "#F9E7E7", "#FBF3DC"

st.markdown(f"""
<style>
  /* mobile-first tightening */
  .block-container {{ padding: 1.0rem 0.9rem 3rem; max-width: 560px; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  div[data-testid="stMetric"] {{ background: #fff; border: 1px solid #E4E6EA;
      border-radius: 12px; padding: 8px 10px; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.05rem;
      font-family: ui-monospace, Menlo, monospace; }}
  div[data-testid="stMetricLabel"] {{ font-size: 0.70rem; }}
  .badge {{ font-family: ui-monospace, Menlo, monospace; font-weight: 700;
      font-size: 0.75rem; letter-spacing: .08em; border-radius: 6px;
      padding: 3px 9px; float: right; }}
  .tkr {{ font-family: ui-monospace, Menlo, monospace; font-weight: 700;
      font-size: 1.0rem; }}
  .sub {{ color: #6B7280; font-size: 0.74rem; }}
  .strip-wrap {{ margin: 8px 0 2px; }}
  .strip {{ position: relative; height: 8px; border-radius: 4px;
      background: #E4E6EA; overflow: visible; }}
  .zone {{ position: absolute; top: 0; bottom: 0; }}
  .marker {{ position: absolute; top: -2px; width: 10px; height: 12px;
      border-radius: 3px; background: #16191D; border: 2px solid #fff;
      box-shadow: 0 1px 2px rgba(0,0,0,.25); }}
  .striplbl {{ display: flex; justify-content: space-between; color: #6B7280;
      font-family: ui-monospace, Menlo, monospace; font-size: 0.62rem;
      margin-top: 3px; }}
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


def make_demo(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Synthetic OHLCV so the app works with no internet."""
    out, n = {}, 520
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    for i, t in enumerate(tickers):
        rs = np.random.RandomState(100 + i)
        d2 = rs.choice([1.2e-3, 4e-4, -8e-4])
        drift = np.concatenate([np.full(300, 3e-4), np.full(n - 300, d2)])
        px = 100 * np.exp(np.cumsum(drift + rs.normal(0, 0.013, n)))
        out[t] = pd.DataFrame(
            {"Open": px, "High": px * 1.008, "Low": px * 0.992,
             "Close": px, "Volume": rs.lognormal(15, 0.3, n)}, index=idx)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers: tuple[str, ...], years: float,
              demo: bool) -> dict[str, pd.DataFrame]:
    if demo:
        return make_demo(list(tickers))
    return ktf.DataHandler(lookback_years=years).fetch(list(tickers))


# ------------------------------------------------------------------ header
st.markdown('<div style="font-size:1.35rem;font-weight:700;'
            'letter-spacing:-0.02em">Kalman Screener</div>'
            '<div class="sub">3-month horizon · live Yahoo Finance data</div>',
            unsafe_allow_html=True)
st.write("")

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
with st.expander("Settings"):
    years = st.slider("History (years)", 1.0, 5.0, 2.0, 0.5)
    slope_entry = st.slider("Trend entry drift (ann. %)", 2, 30, 10, 1) / 100
    persist = st.slider("Slope persistence (days)", 2, 15, 5, 1)
    vol_veto = st.slider("Vol veto percentile", 70, 99, 90, 1) / 100
    entry_z = st.slider("Pairs entry |z|", 1.0, 3.5, 2.0, 0.1)
    exit_z = st.slider("Pairs exit |z|", 0.0, 1.5, 0.5, 0.1)
    run_pairs = st.toggle("Run pairs leg", value=True)
    demo = st.toggle("Demo data (offline / no Yahoo)",
                     value=bool(os.environ.get("KALMAN_DEMO")))

run = st.button("Run screen", type="primary", use_container_width=True)

# ------------------------------------------------------------------ screen
if run and not tickers:
    st.warning("Enter at least one ticker.")

if run and tickers:
    hedges = sorted({t for p in pairs for t in p} - set(tickers)) if run_pairs else []
    with st.spinner(f"Fetching {len(tickers) + len(hedges)} tickers "
                    f"({years:.0f}y daily)..."):
        data = load_data(tuple(sorted(set(tickers) | set(hedges))), years, demo)

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
    rf = ktf.RegimeFilter(risk)
    trend = ktf.TrendStrategy(tp, ktf.KalmanTrendFilter(), rf)
    pairs_strat = ktf.PairsStrategy(pp, ktf.KalmanPairsFilter(), rf)

    # ---------------- trend cards ----------------
    results = []
    for t in tickers:
        if t not in data:
            continue
        sig_df = trend.generate(data[t])
        bt = ktf.Backtester(risk, mode="trend").run(sig_df)
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
            st.markdown(
                f'<span class="tkr">{t}</span>{badge(lab)}'
                f'<div class="sub">Trend · KF drift '
                f'{last["kf_slope_ann"]*100:.1f}% ann. · regime '
                f'{"OK" if last["regime_ok"] else "VETO"}</div>',
                unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Price", f'{last["Close"]:.2f}')
            c2.metric("Kalman", f'{last["kf_price"]:.2f}')
            rr = risk.atr_target_mult / risk.atr_stop_mult
            c3.metric("R : R", f"{rr:.1f} : 1")

            st.markdown(decision_strip(
                last["kf_slope_ann"], slope_entry, 0.35,
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
