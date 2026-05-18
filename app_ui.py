"""Streamlit UI: candlestick chart, FastAPI prediction, and backtest equity curves."""

from __future__ import annotations

import os

import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Stock direction", layout="wide")
st.title("Tech stock direction")
st.caption("Candlesticks from Yahoo Finance · prediction from your FastAPI `/predict` service")

ticker = st.text_input("Ticker", value="MSFT").strip().upper()
if st.button("Run", type="primary"):
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        hist = yf.Ticker(ticker).history(period="6mo", auto_adjust=False, interval="1d")
        if hist.empty:
            st.error("No price history returned for this symbol.")
        else:
            fig_c = go.Figure(
                data=[
                    go.Candlestick(
                        x=hist.index,
                        open=hist["Open"],
                        high=hist["High"],
                        low=hist["Low"],
                        close=hist["Close"],
                        name=ticker,
                    )
                ]
            )
            fig_c.update_layout(
                title=f"{ticker} — last 6 months",
                xaxis_title="Date",
                yaxis_title="Price (split-adjusted close in chart)",
                height=480,
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig_c, use_container_width=True)

        try:
            resp = requests.post(
                f"{API_URL.rstrip('/')}/predict",
                json={"ticker": ticker},
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            st.error(f"Could not reach API at {API_URL}: {exc}")
        else:
            prob = float(data.get("probability_up", 0.0))
            label = data.get("prediction", "?")
            as_of = data.get("as_of_date", "")

            st.markdown("### Next session (from model)")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    label="Direction",
                    value=label,
                    help="Up = model expects next Adj Close > current Adj Close (training definition).",
                )
            with m2:
                st.metric(label="P(Up)", value=f"{prob*100:.1f}%")
            with m3:
                st.metric(label="Features as of", value=str(as_of))

            bt = data.get("backtest") or {}
            dates = bt.get("dates") or []
            eq_s = bt.get("equity_strategy") or []
            eq_occ = bt.get("equity_open_to_close_benchmark") or []
            eq_bh = bt.get("equity_buy_and_hold") or []
            if dates and eq_s:
                st.markdown("### Backtest (API test window)")
                fig_e = go.Figure()
                fig_e.add_trace(
                    go.Scatter(
                        x=dates,
                        y=eq_s,
                        mode="lines",
                        name="Model strategy",
                        line=dict(color="#2563eb", width=2),
                    )
                )
                if eq_occ:
                    fig_e.add_trace(
                        go.Scatter(
                            x=dates,
                            y=eq_occ,
                            mode="lines",
                            name="Open→close daily (benchmark)",
                            line=dict(color="#94a3b8", width=1, dash="dash"),
                        )
                    )
                if eq_bh and len(eq_bh) == len(dates):
                    fig_e.add_trace(
                        go.Scatter(
                            x=dates,
                            y=eq_bh,
                            mode="lines",
                            name="Buy & hold (adj. close)",
                            line=dict(color="#16a34a", width=1, dash="dot"),
                        )
                    )
                fig_e.update_layout(
                    height=420,
                    yaxis_title="Portfolio (US$)",
                    xaxis_title="Date",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_e, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric(
                        "Strategy return",
                        f"{bt.get('cumulative_return_strategy', 0)*100:.2f}%",
                    )
                with c2:
                    st.metric(
                        "Open→close benchmark",
                        f"{bt.get('cumulative_return_open_to_close_benchmark', 0)*100:.2f}%",
                    )
                with c3:
                    st.metric(
                        "Buy & hold return",
                        f"{bt.get('cumulative_return_buy_and_hold', 0)*100:.2f}%",
                    )
                with c4:
                    st.metric(
                        "Sharpe (strategy)",
                        f"{bt.get('sharpe_annualized_strategy', float('nan')):.2f}",
                    )
                entry_p = bt.get("buy_and_hold_entry_price")
                exit_p = bt.get("buy_and_hold_exit_price")
                if entry_p is not None and exit_p is not None:
                    st.caption(
                        f"Buy & hold: {bt.get('buy_and_hold_entry_date', '?')} @ {entry_p:.2f} → "
                        f"{bt.get('buy_and_hold_exit_date', '?')} @ {exit_p:.2f}"
                    )
                st.metric(
                    "Max drawdown (strategy)",
                    f"{bt.get('max_drawdown_strategy', float('nan'))*100:.2f}%",
                )

            with st.expander("Test-set classification metrics"):
                st.json(data.get("test_classification_metrics", {}))
