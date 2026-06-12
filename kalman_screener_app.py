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
        n = 520
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
        bar_vol, hl = 0.013, 0.008
    else:
        n = 60 * (ppy // 252)                       # ~60 sessions of bars
        freq = {"1h": "1h", "15m": "15min"}.get(interval, "15min")
        idx = pd.date_range(end=pd.Timestamp.now().floor("min"),
                            periods=n, freq=freq)
        bar_vol = 0.013 * np.sqrt(252 / ppy)        # scale noise to bar size
        hl = bar_vol * 0.6
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
        res_label = st.selectbox("Data resolution", list(RESOLUTIONS))
        interval = RESOLUTIONS[res_label]
        ppy = ktf.periods_per_year(interval)
        bars_per_day = max(1, ppy // 252)

        years = st.slider("History (years, daily mode only)", 1.0, 5.0, 2.0, 0.5,
                          disabled=(interval != "1d"))

        st.caption("Kalman filter tuning — higher Q tracks live price faster "
                   "(less smoothing); higher R trusts the model over raw ticks.")
        q_exp = st.slider("Kalman sensitivity Q (log\u2081\u2080)",
                          -6.0, -2.0, -5.0, 0.5)
        r_exp = st.slider("Measurement noise R (log\u2081\u2080)",
                          -5.0, -1.0, -3.0, 0.5)
        kf_q, kf_r = 10.0 ** q_exp, 10.0 ** r_exp

        drift_default = 10 if interval == "1d" else 25
        slope_entry = st.slider("Trend entry drift (ann. %)", 2, 80,
                                drift_default, 1) / 100
        persist = st.slider("Slope persistence (bars)", 2, 60,
                            5 if interval == "1d" else bars_per_day, 1)
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
    """ROIC, P/E, FCF history/growth, net debt, shares, price."""
    if demo:
        rs = np.random.RandomState(abs(hash(t)) % (2**31))
        shares = rs.uniform(0.5, 8) * 1e9
        price = rs.uniform(40, 400)
        fcf0 = shares * price * rs.uniform(0.02, 0.07)      # FCF yield 2-7%
        g = rs.uniform(-0.05, 0.22)
        fcf_hist = [fcf0 / (1 + g) ** k for k in range(4)]   # newest first
        return {"name": f"{t} (demo)", "price": price, "shares": shares,
                "pe": rs.uniform(11, 42), "roic": rs.uniform(0.04, 0.30),
                "fcf": fcf_hist, "fcf_growth": g,
                "net_debt": shares * price * rs.uniform(-0.05, 0.25),
                "currency": "USD"}

    if yf is None:
        raise ImportError("yfinance is required for live fundamentals.")
    tk = yf.Ticker(t)
    info = tk.info or {}
    inc, bs, cf = tk.income_stmt, tk.balance_sheet, tk.cashflow

    fcf_ser = _row(cf, "Free Cash Flow")
    ebit = _row(inc, "EBIT", "Operating Income")
    tax = _row(inc, "Tax Provision")
    pretax = _row(inc, "Pretax Income")
    ic = _row(bs, "Invested Capital")
    debt = _row(bs, "Total Debt")
    equity = _row(bs, "Stockholders Equity",
                  "Total Equity Gross Minority Interest")
    cash = _row(bs, "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments")

    # --- ROIC = NOPAT / Invested Capital ---
    tax_rate = 0.21
    if tax is not None and pretax is not None and pretax.iloc[0]:
        tax_rate = float(np.clip(tax.iloc[0] / pretax.iloc[0], 0.0, 0.35))
    roic = np.nan
    icap = None
    if ic is not None:
        icap = float(ic.iloc[0])
    elif debt is not None and equity is not None:
        icap = float(debt.iloc[0] + equity.iloc[0]
                     - (cash.iloc[0] if cash is not None else 0.0))
    if ebit is not None and icap:
        roic = float(ebit.iloc[0]) * (1 - tax_rate) / icap

    # --- FCF growth: CAGR across available annual statements ---
    fcf_hist, fcf_g = [], np.nan
    if fcf_ser is not None:
        fcf_hist = [float(v) for v in fcf_ser.values]        # newest first
        if len(fcf_hist) >= 2 and fcf_hist[-1] > 0 and fcf_hist[0] > 0:
            yrs = len(fcf_hist) - 1
            fcf_g = (fcf_hist[0] / fcf_hist[-1]) ** (1 / yrs) - 1

    net_debt = 0.0
    if debt is not None:
        net_debt = float(debt.iloc[0]
                         - (cash.iloc[0] if cash is not None else 0.0))

    return {"name": info.get("shortName", t),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "shares": info.get("sharesOutstanding"),
            "pe": info.get("trailingPE"),
            "roic": roic, "fcf": fcf_hist, "fcf_growth": fcf_g,
            "net_debt": net_debt,
            "currency": info.get("currency", "USD")}


def dcf_fair_value(fcf0, g0, wacc, shares, net_debt,
                   g_term=0.025, years=5):
    """5y explicit FCF with growth fading to terminal, Gordon TV."""
    if not fcf0 or fcf0 <= 0 or not shares or wacc <= g_term:
        return None
    g0 = float(np.clip(g0 if np.isfinite(g0) else 0.05, -0.05, 0.25))
    pv, fcf = 0.0, fcf0
    for k in range(1, years + 1):
        g_k = g0 + (g_term - g0) * (k - 1) / (years - 1)     # linear fade
        fcf *= (1 + g_k)
        pv += fcf / (1 + wacc) ** k
    tv = fcf * (1 + g_term) / (wacc - g_term) / (1 + wacc) ** years
    return (pv + tv - net_debt) / shares


def scorecard(f, upside):
    """0-8 multi-factor score as a cross-check on the DCF."""
    pts, notes = 0, []
    r = f.get("roic")
    if r is not None and np.isfinite(r):
        pts += 2 if r > 0.15 else 1 if r > 0.08 else 0
        notes.append(f"ROIC {r:.0%}")
    pe = f.get("pe")
    if pe and np.isfinite(pe) and pe > 0:
        pts += 2 if pe < 18 else 1 if pe < 28 else 0
        notes.append(f"P/E {pe:.1f}")
    g = f.get("fcf_growth")
    if g is not None and np.isfinite(g):
        pts += 2 if g > 0.12 else 1 if g > 0.04 else 0
        notes.append(f"FCF growth {g:+.0%}")
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
                'view: yfinance fundamentals + simplified DCF, cross-checked '
                'against the medium-term Kalman trend.</div>',
                unsafe_allow_html=True)

    default_t = tickers[0] if tickers else "AAPL"
    val_ticker = st.text_input("Ticker to value", value=default_t).strip().upper()
    wacc = st.slider("Discount rate / WACC (%)", 6.0, 14.0, 9.0, 0.5) / 100
    analyze = st.button("Analyze fundamentals", type="primary",
                        use_container_width=True)

    if analyze and val_ticker:
        try:
            with st.spinner(f"Fetching {val_ticker} financial statements..."):
                f = get_fundamentals(val_ticker, demo)
        except Exception as e:
            st.error(f"Could not load fundamentals for {val_ticker}: {e}")
            st.stop()

        if not f.get("price") or not f.get("shares"):
            st.error(f"{val_ticker}: missing price/share data — "
                     "ETFs and some foreign listings have no statements.")
            st.stop()

        # ---------- fundamentals ----------
        st.markdown(f'<span class="tkr">{f["name"]}</span>'
                    f'<div class="sub">Trailing fundamentals · '
                    f'{f["currency"]}</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("ROIC", "—" if not np.isfinite(f.get("roic", np.nan))
                  else f'{f["roic"]:.1%}')
        c2.metric("P/E (ttm)", "—" if not f.get("pe") else f'{f["pe"]:.1f}')
        c3.metric("FCF growth", "—" if not np.isfinite(f.get("fcf_growth", np.nan))
                  else f'{f["fcf_growth"]:+.1%}')

        # ---------- DCF ----------
        fcf0 = f["fcf"][0] if f.get("fcf") else None
        fair = dcf_fair_value(fcf0, f.get("fcf_growth", 0.05), wacc,
                              f["shares"], f.get("net_debt", 0.0))
        upside = (fair / f["price"] - 1) if fair else None

        if fair:
            verdict = ("Undervalued" if upside > 0.20 else
                       "Overvalued" if upside < -0.15 else "Fairly Valued")
        else:
            pts, _ = scorecard(f, None)
            verdict = ("Undervalued" if pts >= 5 else
                       "Overvalued" if pts <= 1 else "Fairly Valued")
            st.info("FCF unavailable — verdict from the factor scorecard only.")

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
            st.markdown(f'<div class="sub">Scorecard {pts}/8 · {notes} · '
                        f'WACC {wacc:.1%}</div>', unsafe_allow_html=True)

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
                    'model on trailing statements — earnings quality, '
                    'cyclicality and guidance are not captured. '
                    'Not investment advice.</div>', unsafe_allow_html=True)
