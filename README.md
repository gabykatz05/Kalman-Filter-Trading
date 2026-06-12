# Kalman 3-Month Screener — iPhone app

A mobile web app that screens stock baskets (Magnificent 7, REITs, index
ETFs, or custom tickers) using Kalman-filter trend and pairs models on
live 2-year Yahoo Finance data.

## Deploy free in ~3 minutes (Streamlit Community Cloud)

1. **GitHub**: create a new repository (e.g. `kalman-screener`) and upload
   ALL files in this folder, keeping the `.streamlit/` subfolder.
2. **Streamlit**: go to https://share.streamlit.io → "Create app" →
   pick the repo → main file: `kalman_screener_app.py` → Deploy.
3. You get a permanent URL like `https://<name>.streamlit.app`.

## Make it feel native on iPhone

Open the URL in Safari → Share button → **Add to Home Screen**.
You get an app icon that opens full-screen, works on any network
(cellular included), and always pulls fresh market data.

## Run locally instead

    pip install -r requirements.txt
    streamlit run kalman_screener_app.py

Then open the printed Network URL from your iPhone on the same Wi-Fi.

## Notes

- First screen of a basket takes ~10–20 s (download + cointegration
  tests + backtests); results are cached for 1 hour.
- "Demo data" toggle in Settings lets you explore the UI offline.
- Research tool only — not investment advice.
