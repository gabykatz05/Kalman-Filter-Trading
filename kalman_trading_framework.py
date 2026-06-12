"""
================================================================================
 KALMAN-BASED MEDIUM-TERM (3-MONTH) TRADING ARCHITECTURE
================================================================================
Author : Quantitative Research
Purpose: Identify attractive 90-day-horizon opportunities using Kalman
         filtering for (1) noise-free trend estimation and (2) dynamic
         hedge-ratio estimation for pairs/mean-reversion trading.

--------------------------------------------------------------------------------
 STATE-SPACE MODELS USED
--------------------------------------------------------------------------------
1) TREND MODEL — "Local Linear Trend" (Strategy B: Trend Following)

   State vector:        x_t = [ level_t , slope_t ]'
   Transition matrix:   F   = [[1, 1],
                               [0, 1]]          (level_t = level_{t-1} + slope_{t-1})
   Observation matrix:  H   = [1, 0]            (we only observe the noisy price)
   Process noise:       Q   = diag(q_level, q_slope)   -> how fast the "true"
                              trend is allowed to evolve (small = smooth)
   Measurement noise:   R   = r                  -> daily market noise variance

   The filtered slope_t is a *lag-free* analogue of a moving-average slope.
   A persistent sign change in slope_t = structural regime shift.

2) PAIRS MODEL — Dynamic Hedge Ratio (Strategy A: Mean Reversion)

   Observation eq.:     y_t = beta_t * x_t + alpha_t + eps_t ,  eps ~ N(0, R)
   State vector:        theta_t = [ beta_t , alpha_t ]'
   Transition matrix:   F = I_2  (random-walk coefficients)
   Observation matrix:  H_t = [ x_t , 1 ]   (time-varying!)
   Process noise:       Q = (delta / (1 - delta)) * I_2
                            delta in (0,1): larger -> faster-adapting hedge ratio
   Measurement noise:   R = scalar

   The innovation e_t = y_t - H_t * theta_{t|t-1} is the *spread*.
   Its predictive std  sqrt(S_t) gives a self-normalising z-score:
        z_t = e_t / sqrt(S_t)
   |z| > entry_z  -> spread is statistically stretched -> mean-reversion trade.

--------------------------------------------------------------------------------
 ARCHITECTURE
--------------------------------------------------------------------------------
 DataHandler          -> fetch & clean OHLCV (yfinance)
 KalmanTrendFilter    -> local linear trend smoothing (level + slope)
 KalmanPairsFilter    -> dynamic beta/alpha + innovation z-score
 RegimeFilter         -> ATR%, rolling vol percentile, market stress veto
 TrendStrategy        -> Option B signals (slope regime shift + volume)
 PairsStrategy        -> Option A signals (z-score thresholds, cointegration)
 Backtester           -> event-driven, with stop / target / 90-day time exit
 Metrics              -> Total Return, Sharpe, Max Drawdown, Win Rate
 Screener             -> "Current Active Recommendations" DataFrame

--------------------------------------------------------------------------------
 HOW TO RUN
--------------------------------------------------------------------------------
    pip install yfinance numpy pandas scipy scikit-learn
    python kalman_trading_framework.py

 Edit the CONFIG block at the bottom (tickers, pairs, dates) to taste.
================================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import zscore  # noqa: F401  (handy for ad-hoc research)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# Optional dependencies handled gracefully
# ------------------------------------------------------------------------------
try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

TRADING_DAYS = 252
HOLDING_DAYS = 63          # ~3 calendar months of trading days


# ==============================================================================
# 1. DATA ACQUISITION & PREPROCESSING
# ==============================================================================
class DataHandler:
    """Fetches and cleans daily OHLCV data."""

    def __init__(self, start: str = None, end: str = None, lookback_years: float = 3.0):
        self.end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        self.start = (pd.Timestamp(start) if start
                      else self.end - pd.DateOffset(years=lookback_years))

    def fetch(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """Returns {ticker: DataFrame[Open, High, Low, Close, Volume]}."""
        if not _HAS_YF:
            raise ImportError("yfinance is required: pip install yfinance")

        raw = yf.download(
            tickers, start=self.start, end=self.end,
            auto_adjust=True, progress=False, group_by="ticker", threads=True,
        )
        out: dict[str, pd.DataFrame] = {}
        for t in tickers:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(df) > TRADING_DAYS:          # need >= 1y of data
                    out[t] = df
                else:
                    print(f"[WARN] {t}: insufficient history, skipped.")
            except KeyError:
                print(f"[WARN] {t}: download failed, skipped.")
        return out

    @staticmethod
    def from_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Inject pre-loaded data (testing / offline use)."""
        return {t: df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                for t, df in frames.items()}


# ==============================================================================
# 2a. KALMAN FILTER — LOCAL LINEAR TREND  (price smoothing, Strategy B core)
# ==============================================================================
class KalmanTrendFilter:
    """
    Local Linear Trend Kalman Filter.

    x_t = [level, slope]',  F = [[1,1],[0,1]],  H = [1,0]
    Q   = diag(q_level, q_slope),  R = measurement noise variance.

    `q_slope` controls responsiveness of the trend:
      smaller  -> smoother, slower regime detection
      larger   -> faster, noisier regime detection
    Defaults are tuned for a ~3-month horizon (slow structural trend).
    """

    def __init__(self, q_level: float = 1e-5, q_slope: float = 1e-7,
                 r: float = 1e-3):
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q_scale = np.diag([q_level, q_slope])
        self.r_scale = r

    def filter(self, prices: pd.Series) -> pd.DataFrame:
        """Run the filter on a price series.

        Works in log-price space so noise parameters are scale-free.
        Returns DataFrame[kf_price, kf_slope, kf_slope_ann] indexed like input.
        """
        y = np.log(prices.values.astype(float))
        n = len(y)

        # State init: level = first obs, slope = 0; diffuse-ish covariance
        x = np.array([y[0], 0.0])
        P = np.eye(2) * 1.0

        Q = self.Q_scale
        R = np.array([[self.r_scale]])
        F, H = self.F, self.H

        levels = np.empty(n)
        slopes = np.empty(n)

        for t in range(n):
            # --- Predict ---
            x = F @ x
            P = F @ P @ F.T + Q
            # --- Update ---
            innov = y[t] - (H @ x)[0]
            S = (H @ P @ H.T + R)[0, 0]
            K = (P @ H.T / S).flatten()
            x = x + K * innov
            P = (np.eye(2) - np.outer(K, H)) @ P

            levels[t], slopes[t] = x[0], x[1]

        return pd.DataFrame(
            {
                "kf_price": np.exp(levels),                     # smoothed price
                "kf_slope": slopes,                             # daily log-slope
                "kf_slope_ann": slopes * TRADING_DAYS,          # annualised drift
            },
            index=prices.index,
        )


# ==============================================================================
# 2b. KALMAN FILTER — DYNAMIC HEDGE RATIO  (Strategy A core)
# ==============================================================================
class KalmanPairsFilter:
    """
    Time-varying regression  y_t = beta_t * x_t + alpha_t + eps_t.

    State theta = [beta, alpha]', F = I, H_t = [x_t, 1] (time-varying).
    Q = (delta / (1-delta)) * I  — `delta` is the classic discount factor:
        delta -> 0  : nearly static OLS beta
        delta -> 1  : beta adapts very quickly
    Returns beta, alpha, spread (innovation) and self-normalised z-score
    z_t = innovation / sqrt(innovation predictive variance S_t).
    """

    def __init__(self, delta: float = 1e-4, r: float = 1e-3):
        self.delta = delta
        self.r = r

    def filter(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        idx = y.index.intersection(x.index)
        yv = np.log(y.loc[idx].values.astype(float))
        xv = np.log(x.loc[idx].values.astype(float))
        n = len(idx)

        theta = np.zeros(2)                      # [beta, alpha]
        P = np.eye(2) * 1.0
        Q = np.eye(2) * (self.delta / (1.0 - self.delta))
        R = self.r

        betas = np.empty(n)
        alphas = np.empty(n)
        spread = np.empty(n)
        zs = np.empty(n)

        for t in range(n):
            H = np.array([xv[t], 1.0])
            # --- Predict (F = I) ---
            P = P + Q
            # --- Innovation ---
            y_hat = H @ theta
            e = yv[t] - y_hat
            S = H @ P @ H + R
            # --- Update ---
            K = P @ H / S
            theta = theta + K * e
            P = P - np.outer(K, H) @ P

            betas[t], alphas[t] = theta
            spread[t] = e
            zs[t] = e / np.sqrt(S)

        out = pd.DataFrame(
            {"beta": betas, "alpha": alphas, "spread": spread, "z_innov": zs},
            index=idx,
        )
        # Trading z-score: spread normalised by its own rolling std (63d).
        # More robust than e/sqrt(S) when R is imperfectly calibrated.
        out["z"] = out["spread"] / out["spread"].rolling(63, min_periods=40).std()
        # Burn-in: first ~60 obs are unreliable while the filter converges
        out.iloc[:60, out.columns.get_loc("z")] = np.nan
        return out


# ------------------------------------------------------------------------------
# Lightweight Engle-Granger cointegration check (statsmodels optional)
# ------------------------------------------------------------------------------
def coint_test(y: pd.Series, x: pd.Series) -> float:
    """Returns p-value of Engle-Granger cointegration test (lower = better)."""
    try:
        from statsmodels.tsa.stattools import coint
        idx = y.index.intersection(x.index)
        _, pval, _ = coint(np.log(y.loc[idx]), np.log(x.loc[idx]))
        return float(pval)
    except ImportError:
        # Fallback: ADF-like heuristic on OLS residuals via variance-ratio
        idx = y.index.intersection(x.index)
        ly, lx = np.log(y.loc[idx]), np.log(x.loc[idx])
        b = np.polyfit(lx, ly, 1)[0]
        resid = ly - b * lx
        vr = resid.diff().var() / resid.var()      # high VR ~ stationary resid
        return 0.01 if vr > 0.05 else 0.50          # crude proxy


# ==============================================================================
# 3. RISK MANAGEMENT & REGIME FILTERING
# ==============================================================================
@dataclass
class RiskParams:
    atr_window: int = 14
    vol_window: int = 21
    vol_pctile_veto: float = 0.90      # veto entries when vol in top decile
    atr_stop_mult: float = 2.5         # stop  = entry -+ 2.5 * ATR
    atr_target_mult: float = 5.0       # target= entry +- 5.0 * ATR  (R:R = 2)
    max_holding_days: int = HOLDING_DAYS


class RegimeFilter:
    """ATR + rolling-volatility regime veto."""

    def __init__(self, params: RiskParams):
        self.p = params

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.p
        hl = df["High"] - df["Low"]
        hc = (df["High"] - df["Close"].shift()).abs()
        lc = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.ewm(span=p.atr_window, adjust=False).mean()

        ret = df["Close"].pct_change()
        vol = ret.rolling(p.vol_window).std() * np.sqrt(TRADING_DAYS)
        vol_pct = vol.rolling(TRADING_DAYS, min_periods=63).rank(pct=True)

        return pd.DataFrame(
            {
                "atr": atr,
                "atr_pct": atr / df["Close"],
                "ann_vol": vol,
                "vol_pctile": vol_pct,
                "regime_ok": vol_pct < p.vol_pctile_veto,   # entry allowed?
            },
            index=df.index,
        )


# ==============================================================================
# 4a. STRATEGY B — TREND FOLLOWING (Kalman slope regime shift + volume)
# ==============================================================================
@dataclass
class TrendParams:
    slope_entry_ann: float = 0.10      # require >= +10% annualised Kalman drift
    slope_persist: int = 5             # slope must hold sign N consecutive days
    vol_confirm_mult: float = 1.10     # volume >= 1.1x its 20d average
    vol_avg_window: int = 20


class TrendStrategy:
    name = "TREND"

    def __init__(self, tp: TrendParams, kf: KalmanTrendFilter,
                 rf: RegimeFilter):
        self.tp, self.kf, self.rf = tp, kf, rf

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns df + [kf_price, kf_slope_ann, signal] where signal in
        {+1 BUY, 0 HOLD/FLAT, -1 SELL/EXIT}."""
        tp = self.tp
        k = self.kf.filter(df["Close"])
        r = self.rf.compute(df)

        slope = k["kf_slope_ann"]
        pos_persist = (slope > tp.slope_entry_ann).rolling(tp.slope_persist).sum() == tp.slope_persist
        neg_persist = (slope < -tp.slope_entry_ann).rolling(tp.slope_persist).sum() == tp.slope_persist

        vol_ok = df["Volume"] >= tp.vol_confirm_mult * df["Volume"].rolling(tp.vol_avg_window).mean()
        # Volume confirmation within last 5 days of the regime shift
        vol_recent = vol_ok.rolling(5).max().astype(bool)

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[pos_persist & vol_recent & r["regime_ok"]] = 1
        sig[neg_persist] = -1                      # exit / short signal

        out = df.copy()
        out["kf_price"] = k["kf_price"]
        out["kf_slope_ann"] = slope
        out = out.join(r[["atr", "regime_ok", "vol_pctile"]])
        out["signal"] = sig
        return out


# ==============================================================================
# 4b. STRATEGY A — PAIRS / MEAN REVERSION (dynamic hedge ratio z-score)
# ==============================================================================
@dataclass
class PairsParams:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5                # spread blow-out stop
    coint_pval_max: float = 0.05       # require cointegration at 5%


class PairsStrategy:
    name = "PAIRS"

    def __init__(self, pp: PairsParams, kf: KalmanPairsFilter,
                 rf: RegimeFilter):
        self.pp, self.kf, self.rf = pp, kf, rf

    def generate(self, df_y: pd.DataFrame, df_x: pd.DataFrame
                 ) -> pd.DataFrame | None:
        """Signal on the *spread*: +1 = long y / short beta*x (z < -entry),
        -1 = short y / long beta*x (z > +entry), 0 = flat."""
        pp = self.pp
        pval = coint_test(df_y["Close"], df_x["Close"])
        if pval > pp.coint_pval_max:
            return None                            # pair rejected

        k = self.kf.filter(df_y["Close"], df_x["Close"])
        r = self.rf.compute(df_y).reindex(k.index)

        z = k["z"]
        sig = pd.Series(0, index=k.index, dtype=int)
        sig[(z < -pp.entry_z) & r["regime_ok"]] = 1
        sig[(z > pp.entry_z) & r["regime_ok"]] = -1

        out = pd.DataFrame(index=k.index)
        out["Close"] = df_y["Close"].reindex(k.index)        # traded leg
        out["Close_x"] = df_x["Close"].reindex(k.index)      # hedge leg
        out["High"] = df_y["High"].reindex(k.index)
        out["Low"] = df_y["Low"].reindex(k.index)
        out["beta"], out["z"] = k["beta"], z
        out["atr"] = r["atr"]
        out["regime_ok"] = r["regime_ok"]
        out["signal"] = sig
        out.attrs["coint_pval"] = pval
        return out


# ==============================================================================
# 5. BACKTESTING ENGINE  (event-driven, long-only on the signal leg)
# ==============================================================================
@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_px: float
    stop: float
    target: float
    direction: int                     # +1 long, -1 short
    exit_date: pd.Timestamp = None
    exit_px: float = None
    reason: str = ""
    # Pairs-only fields (spread P&L): long y / short beta*x when direction=+1
    beta: float = None
    entry_px_x: float = None
    exit_px_x: float = None

    @property
    def pnl_pct(self) -> float:
        if self.beta is None:                       # single-leg directional
            return self.direction * (self.exit_px / self.entry_px - 1.0)
        # Spread return on gross notional (1 unit y + |beta| units x)
        leg_y = self.exit_px / self.entry_px - 1.0
        leg_x = self.exit_px_x / self.entry_px_x - 1.0
        return self.direction * (leg_y - self.beta * leg_x) / (1 + abs(self.beta))


class Backtester:
    """
    Event-driven backtest with:
      - entry on signal (next bar handled implicitly: signals use info to t)
      - ATR-based stop & profit target
      - hard time exit at `max_holding_days` (90-day mandate)
      - opposite-signal exit
      - one position at a time, full notional, costs in bps per side
    """

    def __init__(self, risk: RiskParams, cost_bps: float = 5.0,
                 mode: str = "trend"):
        self.risk = risk
        self.cost = cost_bps / 1e4
        self.mode = mode               # 'trend' (price exits) or 'pairs' (z exits)

    # --------------------------------------------------------------
    def run(self, df: pd.DataFrame,
            pairs_params: PairsParams | None = None) -> dict:
        rp = self.risk
        trades: list[Trade] = []
        pos: Trade | None = None
        equity = [1.0]
        eq = 1.0
        dates = df.index

        for i in range(1, len(df)):
            row, prev = df.iloc[i], df.iloc[i - 1]
            px = row["Close"]

            # ---------- manage open position ----------
            if pos is not None:
                days_held = (dates[i] - pos.entry_date).days
                exit_now, reason, exit_px = False, "", px

                if self.mode == "trend":
                    if pos.direction == 1:
                        if row["Low"] <= pos.stop:
                            exit_now, reason, exit_px = True, "STOP", pos.stop
                        elif row["High"] >= pos.target:
                            exit_now, reason, exit_px = True, "TARGET", pos.target
                    if not exit_now and row["signal"] == -pos.direction:
                        exit_now, reason = True, "REVERSE"
                else:  # pairs: exit on z mean-reversion or blow-out
                    z = row["z"]
                    if pos.direction == 1 and z >= -pairs_params.exit_z:
                        exit_now, reason = True, "Z_EXIT"
                    elif pos.direction == -1 and z <= pairs_params.exit_z:
                        exit_now, reason = True, "Z_EXIT"
                    elif abs(z) >= pairs_params.stop_z:
                        exit_now, reason = True, "Z_STOP"

                if not exit_now and days_held >= rp.max_holding_days * 1.6:
                    # 63 trading days ~ 100 calendar days; 1.6x buffer in cal days
                    exit_now, reason = True, "TIME"

                if exit_now:
                    pos.exit_date, pos.exit_px, pos.reason = dates[i], exit_px, reason
                    if self.mode == "pairs":
                        pos.exit_px_x = row["Close_x"]
                    eq *= (1 + pos.pnl_pct - 2 * self.cost)
                    trades.append(pos)
                    pos = None

            # ---------- mark-to-market ----------
            if pos is not None:
                if self.mode == "pairs":
                    leg_y = px / pos.entry_px - 1
                    leg_x = row["Close_x"] / pos.entry_px_x - 1
                    mtm = pos.direction * (leg_y - pos.beta * leg_x) / (1 + abs(pos.beta))
                else:
                    mtm = pos.direction * (px / pos.entry_px - 1)
                eq_mtm = eq * (1 + mtm)
            else:
                eq_mtm = eq
            equity.append(eq_mtm)

            # ---------- new entries (use yesterday's completed signal) ----------
            if pos is None and prev["signal"] != 0 and not np.isnan(prev.get("atr", np.nan)):
                d = int(prev["signal"])
                if self.mode == "trend" and d == -1:
                    continue          # long-only in trend mode by default
                atr = prev["atr"]
                pos = Trade(
                    entry_date=dates[i],
                    entry_px=px,
                    stop=px - d * rp.atr_stop_mult * atr,
                    target=px + d * rp.atr_target_mult * atr,
                    direction=d,
                    beta=prev["beta"] if self.mode == "pairs" else None,
                    entry_px_x=row["Close_x"] if self.mode == "pairs" else None,
                )

        eq_series = pd.Series(equity, index=dates[: len(equity)])
        return {"trades": trades, "equity": eq_series,
                "metrics": Metrics.compute(eq_series, trades)}


# ==============================================================================
# 6. PERFORMANCE METRICS
# ==============================================================================
class Metrics:
    @staticmethod
    def compute(equity: pd.Series, trades: list[Trade]) -> dict:
        ret = equity.pct_change().dropna()
        total = equity.iloc[-1] / equity.iloc[0] - 1
        sharpe = (ret.mean() / ret.std() * np.sqrt(TRADING_DAYS)
                  if ret.std() > 0 else 0.0)
        dd = (equity / equity.cummax() - 1).min()
        wins = [t for t in trades if t.pnl_pct > 0]
        wr = len(wins) / len(trades) if trades else np.nan
        return {
            "Total Return": f"{total: .2%}",
            "Sharpe Ratio": f"{sharpe: .2f}",
            "Max Drawdown": f"{dd: .2%}",
            "Win Rate": f"{wr: .2%}" if trades else "n/a",
            "Trades": len(trades),
        }


# ==============================================================================
# 7. ACTIONABLE SCREENER
# ==============================================================================
class Screener:
    """
    Scans tickers (trend) and pairs (mean reversion); returns the
    'Current Active Recommendations' DataFrame.
    """

    def __init__(self, risk: RiskParams = RiskParams(),
                 trend_params: TrendParams = TrendParams(),
                 pairs_params: PairsParams = PairsParams()):
        self.risk = risk
        self.rf = RegimeFilter(risk)
        self.trend = TrendStrategy(trend_params, KalmanTrendFilter(), self.rf)
        self.pairs = PairsStrategy(pairs_params, KalmanPairsFilter(), self.rf)
        self.pairs_params = pairs_params

    # ------------------------------------------------------------------
    def scan(self, data: dict[str, pd.DataFrame],
             pair_list: list[tuple[str, str]] | None = None,
             backtest: bool = True) -> pd.DataFrame:
        rows = []

        # ---- Trend leg ------------------------------------------------
        for tkr, df in data.items():
            sig_df = self.trend.generate(df)
            last = sig_df.iloc[-1]
            label = {1: "BUY", -1: "SELL", 0: "HOLD"}[int(last["signal"])]
            rr = (self.risk.atr_target_mult / self.risk.atr_stop_mult)
            row = {
                "Ticker": tkr,
                "Strategy": "TREND (3M)",
                "Price": round(last["Close"], 2),
                "Kalman Value": round(last["kf_price"], 2),
                "KF Drift (ann.)": f"{last['kf_slope_ann']: .1%}",
                "Signal": label,
                "Risk/Reward": f"{rr:.1f} : 1",
                "Regime OK": bool(last["regime_ok"]),
            }
            if backtest:
                bt = Backtester(self.risk, mode="trend").run(sig_df)
                row.update({"BT Sharpe": bt["metrics"]["Sharpe Ratio"],
                            "BT Return": bt["metrics"]["Total Return"],
                            "BT MaxDD": bt["metrics"]["Max Drawdown"],
                            "BT WinRate": bt["metrics"]["Win Rate"]})
            rows.append(row)

        # ---- Pairs leg -------------------------------------------------
        for (a, b) in (pair_list or []):
            if a not in data or b not in data:
                continue
            sig_df = self.pairs.generate(data[a], data[b])
            if sig_df is None:
                rows.append({"Ticker": f"{a}/{b}", "Strategy": "PAIRS",
                             "Signal": "REJECTED (no cointegration)"})
                continue
            last = sig_df.iloc[-1]
            z = last["z"]
            label = ("BUY spread (long %s / short %s)" % (a, b) if last["signal"] == 1
                     else "SELL spread (short %s / long %s)" % (a, b) if last["signal"] == -1
                     else "HOLD")
            rr = (abs(z) - self.pairs_params.exit_z) / max(
                self.pairs_params.stop_z - abs(z), 1e-9)
            row = {
                "Ticker": f"{a}/{b}",
                "Strategy": "PAIRS (KF beta)",
                "Price": round(last["Close"], 2),
                "Kalman Value": round(last["beta"], 3),       # dynamic beta
                "KF Drift (ann.)": f"z = {z: .2f}",
                "Signal": label,
                "Risk/Reward": f"{max(rr, 0):.1f} : 1",
                "Regime OK": bool(last["regime_ok"]),
            }
            if backtest:
                bt = Backtester(self.risk, mode="pairs").run(
                    sig_df, pairs_params=self.pairs_params)
                row.update({"BT Sharpe": bt["metrics"]["Sharpe Ratio"],
                            "BT Return": bt["metrics"]["Total Return"],
                            "BT MaxDD": bt["metrics"]["Max Drawdown"],
                            "BT WinRate": bt["metrics"]["Win Rate"]})
            rows.append(row)

        cols = ["Ticker", "Strategy", "Price", "Kalman Value", "KF Drift (ann.)",
                "Signal", "Risk/Reward", "Regime OK",
                "BT Return", "BT Sharpe", "BT MaxDD", "BT WinRate"]
        out = pd.DataFrame(rows)
        return out.reindex(columns=[c for c in cols if c in out.columns])


# ==============================================================================
# 8. BUILT-IN BASKETS, SELECTION & EXECUTION
# ==============================================================================
BASKETS = {
    "1": {
        "name": "Mega-Cap Tech (Magnificent 7)",
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
        # Each name vs. its natural hedge (QQQ) for the pairs leg
        "pairs": [("AAPL", "QQQ"), ("MSFT", "QQQ"), ("GOOGL", "QQQ"),
                  ("AMZN", "QQQ"), ("NVDA", "QQQ"), ("META", "QQQ"),
                  ("TSLA", "QQQ")],
    },
    "2": {
        "name": "High-Yield / Real Estate (REITs)",
        "tickers": ["SPG", "PLD", "AMT", "CCI", "O"],
        # Sector-ETF hedges + the classic cell-tower pair AMT/CCI
        "pairs": [("SPG", "VNQ"), ("PLD", "VNQ"), ("O", "VNQ"),
                  ("AMT", "CCI")],
    },
    "3": {
        "name": "Major Index ETFs (macro trend/regime)",
        "tickers": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"],
        "pairs": [("QQQ", "SPY"), ("IWM", "SPY"), ("DIA", "SPY"),
                  ("XLK", "QQQ"), ("XLF", "SPY")],
    },
}


def resolve_universe(choice: str, custom: str = "") -> tuple[list[str], list[tuple[str, str]], str]:
    """Map a menu choice (or custom ticker string) to (tickers, pairs, label)."""
    if choice in BASKETS:
        b = BASKETS[choice]
        return b["tickers"], b["pairs"], b["name"]
    # Custom: comma/space separated tickers; pairs auto-built vs SPY
    raw = custom or choice
    tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
    pairs = [(t, "SPY") for t in tickers if t != "SPY"]
    return tickers, pairs, f"Custom ({len(tickers)} tickers)"


def run_screen(tickers: list[str], pairs: list[tuple[str, str]],
               label: str, lookback_years: float = 2.0,
               run_pairs: bool = True) -> pd.DataFrame:
    """Fetch live Yahoo data and run the full screen on one universe."""
    hedge_tickers = {t for p in pairs for t in p} if run_pairs else set()
    all_tickers = sorted(set(tickers) | hedge_tickers)

    print(f"\n[{label}] Fetching {len(all_tickers)} tickers · "
          f"{lookback_years:.0f}y daily history from Yahoo Finance...")
    data = DataHandler(lookback_years=lookback_years).fetch(all_tickers)
    if not data:
        print("No data downloaded — check tickers / connection.")
        return pd.DataFrame()

    # Only screen the requested names on trend (hedges are inputs only)
    trend_data = {t: df for t, df in data.items() if t in tickers}
    screener = Screener()
    table = screener.scan(trend_data | {t: data[t] for t in hedge_tickers if t in data},
                          pair_list=pairs if run_pairs else None,
                          backtest=True)
    # Drop hedge-only rows from the trend section for a cleaner report
    hedge_only = hedge_tickers - set(tickers)
    table = table[~table["Ticker"].isin(hedge_only)].reset_index(drop=True)

    active = table[table["Signal"].astype(str).str.contains("BUY|SELL", na=False)]
    print("\n================ CURRENT ACTIVE RECOMMENDATIONS ================")
    print(active.to_string(index=False) if len(active) else
          "No active BUY/SELL signals right now — all HOLD.")
    print("\n======================= FULL SCREEN ============================")
    print(table.to_string(index=False))
    return table


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Kalman 3-month screener")
    ap.add_argument("--basket", choices=list(BASKETS), default=None,
                    help="1=Mag7 tech, 2=REITs, 3=Index ETFs")
    ap.add_argument("--tickers", default=None,
                    help="Custom comma-separated tickers, e.g. AAPL,JPM,XOM")
    ap.add_argument("--years", type=float, default=2.0,
                    help="Lookback window in years (default 2)")
    ap.add_argument("--no-pairs", action="store_true",
                    help="Skip the pairs/mean-reversion leg")
    args = ap.parse_args()

    # ---- Non-interactive mode (flags supplied) ----
    if args.basket or args.tickers:
        choice = args.basket or "custom"
        tickers, pairs, label = resolve_universe(choice, args.tickers or "")
        run_screen(tickers, pairs, label, args.years, not args.no_pairs)
        return

    # ---- Interactive mode ----
    print("=" * 64)
    print(" KALMAN 3-MONTH SCREENER — choose a universe")
    print("=" * 64)
    for k, b in BASKETS.items():
        print(f"  [{k}] {b['name']:40s} {', '.join(b['tickers'])}")
    print("  [4] Custom — type your own tickers")
    choice = input("\nSelect 1-4: ").strip()

    if choice == "4" or choice not in BASKETS:
        custom = choice if choice not in {"4"} and choice not in BASKETS else \
            input("Enter tickers (comma or space separated): ").strip()
        tickers, pairs, label = resolve_universe("custom", custom)
    else:
        tickers, pairs, label = resolve_universe(choice)

    run_screen(tickers, pairs, label, lookback_years=2.0)


if __name__ == "__main__":
    main()
