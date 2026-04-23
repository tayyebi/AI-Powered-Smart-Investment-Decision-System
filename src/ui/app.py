"""
src/ui/app.py
-------------
Streamlit web UI for the Smart Investment Decision Fusion System.

Run with:
    streamlit run streamlit_app.py
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf

from pipeline import run_analysis
from src.ui.charts import (
    chart_feature_importance,
    chart_pred_vs_actual,
    chart_price_trend,
    chart_sentiment,
    fig_to_bytes,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Investment Decision System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }

    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricLabel"] { color: #aaa; font-size: 0.82rem; }
    [data-testid="stMetricValue"] { font-weight: 700; font-size: 1.4rem; }

    .rec-banner {
        border-radius: 16px; padding: 20px 32px; text-align: center;
        font-size: 2.4rem; font-weight: 800; letter-spacing: 2px;
        margin: 8px 0 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .rec-buy  { background: linear-gradient(120deg,#0db560,#00e676); color:#000; }
    .rec-hold { background: linear-gradient(120deg,#f5a623,#ffd740); color:#000; }
    .rec-sell { background: linear-gradient(120deg,#e53935,#ff1744); color:#fff; }

    .score-wrap { margin: 8px 0 4px; }
    .score-bar-track {
        height: 14px; border-radius: 99px;
        background: rgba(255,255,255,0.10); overflow: hidden;
    }
    .score-bar-fill { height: 100%; border-radius: 99px; transition: width .5s ease; }

    .reason-item {
        background: rgba(255,255,255,0.05);
        border-left: 3px solid #4F8EF7;
        border-radius: 0 10px 10px 0;
        padding: 9px 16px; margin: 6px 0; font-size: 0.93rem;
    }

    h2, h3 { color: #90caf9 !important; }
    hr { border-color: rgba(255,255,255,0.08) !important; }

    .stButton > button {
        background: linear-gradient(120deg, #4F8EF7, #a78bfa);
        color: white; border: none; border-radius: 10px;
        padding: 12px 32px; font-weight: 700; font-size: 1rem;
        transition: opacity .2s; width: 100%;
    }
    .stButton > button:hover { opacity: 0.87; }

    .stTextInput input, .stSelectbox select {
        background: rgba(255,255,255,0.06) !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────

CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}
RISK_COLOUR      = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}


def _fmt(price: float, currency: str = "USD") -> str:
    sym = CURRENCY_SYMBOLS.get(currency, f"{currency} ")
    return f"{sym}{price:,.2f}"


def _score_bar_html(score: float) -> str:
    pct    = score * 100
    colour = "#00e676" if score > 0.70 else "#ffd740" if score >= 0.40 else "#ff1744"
    return f"""
    <div class="score-wrap">
        <div class="score-bar-track">
            <div class="score-bar-fill" style="width:{pct:.1f}%;background:{colour};"></div>
        </div>
    </div>
    """


def _resolve_symbol(raw: str) -> tuple[str, str | None]:
    """Auto-resolve a bare ticker to the correct yfinance symbol.

    Strategy (for tickers without an exchange suffix):
      1. Try <TICKER>.NS  (NSE India)
      2. Try <TICKER>.BO  (BSE India)
      3. Try <TICKER>     (US markets)

    If the user already typed a suffix (TCS.NS, TSLA), it is used as-is.
    """
    raw = raw.strip().upper()
    if not raw:
        return raw, None

    if "." in raw:
        return raw, None

    for candidate in (f"{raw}.NS", f"{raw}.BO", raw):
        try:
            info = yf.Ticker(candidate).info
            if info.get("longName") or info.get("shortName") or info.get("regularMarketPrice"):
                note = None if candidate == raw else f"Auto-resolved **{raw}** → **{candidate}**"
                return candidate, note
        except Exception:
            continue

    return raw, f"⚠️ Could not auto-resolve **{raw}**; trying as-is."


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Investment Analyser")
    st.markdown("---")

    symbol_raw = st.text_input(
        "Stock Symbol",
        value="TSLA",
        placeholder="e.g. TSLA, TCS, RELIANCE, AAPL",
        help=(
            "Just type the ticker — e.g. **TCS**, **INFY**, **RELIANCE** for Indian stocks "
            "or **TSLA**, **AAPL** for US stocks. The exchange suffix (.NS / .BO) is "
            "added automatically. You can also specify it manually: TCS.NS, RELIANCE.BO."
        ),
    ).strip().upper()

    period = st.selectbox(
        "Historical Period",
        options=["1y", "2y", "3y", "4y", "5y"],
        index=1,
        help="More data = better model. 2y–5y recommended.",
    )

    st.markdown("---")
    run_btn = st.button("🚀 Analyse Now", use_container_width=True)

    st.markdown("""
    **Example tickers:**
    | Type | Just type… |
    |------|------------|
    | 🇮🇳 NSE | `TCS` `INFY` `RELIANCE` |
    | 🇮🇳 BSE | `TCS.BO` `WIPRO.BO` |
    | 🇺🇸 US  | `TSLA` `AAPL` `MSFT` |
    """)
    st.markdown("---")

    st.markdown("""
    **How it works:**
    - 📡 Live market data via **Alpha Vantage** / **yfinance**
    - 📰 News from **Google News** (free) + **NewsAPI** (optional key)
    - 🧠 Next-day price via **XGBoost**
    - 💬 Sentiment scored with **VADER**
    - ⚖️ Scores fused: `0.5×Trend + 0.3×Sent + 0.2×Vol`
    """)

    st.markdown("---")
    st.caption("🔐 Optional API keys loaded from `.env`")


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding: 20px 0 10px;">
    <h1 style="font-size:2.6rem; font-weight:800; color:#90caf9; margin:0;">
        📊 Smart Investment Decision System
    </h1>
    <p style="color:#888; font-size:1rem; margin:6px 0 0;">
        Multi-source data fusion · ML price prediction · Explainable AI decisions
    </p>
</div>
<hr>
""", unsafe_allow_html=True)


# ── Main analysis ─────────────────────────────────────────────────────────────

if run_btn:
    if not symbol_raw:
        st.error("Please enter a stock symbol in the sidebar.")
    else:
        with st.spinner(f"Resolving ticker **{symbol_raw}** …"):
            symbol, resolve_note = _resolve_symbol(symbol_raw)

        if resolve_note:
            if resolve_note.startswith("⚠️"):
                st.warning(resolve_note)
            else:
                st.info(f"🔍 {resolve_note}")

        progress_bar = st.progress(0, text="Initialising …")
        status_text  = st.empty()

        steps = [
            "Fetching historical stock data …",
            "Fetching ticker info …",
            "Fetching latest news …",
            "Running sentiment analysis …",
            "Training ML model …",
            "Predicting next price …",
            "Computing volatility …",
            "Fusing signals …",
            "Building explanation …",
        ]

        try:
            from src.analysis import fusion as fus_mod
            from src.analysis import model as mdl_mod
            from src.analysis import sentiment as sent_mod
            from src.analysis import volatility as vol_mod
            from src.data import news as news_mod
            from src.data import stock as stock_mod

            # 1
            progress_bar.progress(5, text=steps[0])
            stock_df = stock_mod.fetch_stock_data(symbol, period=period)

            # 2
            progress_bar.progress(15, text=steps[1])
            info           = stock_mod.get_current_info(symbol)
            current_price  = info["current_price"]  or float(stock_df["Close"].iloc[-1])
            previous_close = info["previous_close"] or float(stock_df["Close"].iloc[-2])
            currency       = info["currency"]
            name           = info["name"]

            # 3
            progress_bar.progress(30, text=steps[2])
            news_query = symbol.split(".")[0]
            articles   = news_mod.fetch_news(f"{news_query} stock")

            # 4
            progress_bar.progress(42, text=steps[3])
            sentiment_result = sent_mod.analyse_news(articles)
            avg_compound     = sentiment_result["average_compound"]
            sentiment_label  = sentiment_result["label"]
            sentiment_norm   = sent_mod.normalise_sentiment(avg_compound)

            # 5 & 6
            progress_bar.progress(55, text=steps[4])
            train_result = mdl_mod.train_model(stock_df)
            progress_bar.progress(70, text=steps[5])
            predicted_price = mdl_mod.predict_next_price(train_result)

            # 7
            progress_bar.progress(78, text=steps[6])
            vol_result       = vol_mod.compute_volatility_signal(stock_df)
            volatility_score = vol_result["score"]
            volatility_label = vol_result["label"]

            # 8
            progress_bar.progress(88, text=steps[7])
            trend_result   = fus_mod.compute_trend_signal(current_price, predicted_price)
            trend_label    = trend_result["label"]
            trend_score    = trend_result["score"]
            fusion_score   = fus_mod.compute_fusion_score(trend_score, sentiment_norm, volatility_score)
            recommendation = fus_mod.apply_decision_rules(fusion_score)
            risk_level     = fus_mod.compute_risk_level(volatility_label, fusion_score)

            # 9
            progress_bar.progress(95, text=steps[8])
            reasons = fus_mod.build_reasons(
                trend_label, sentiment_label, volatility_label,
                current_price, predicted_price, recommendation,
            )

            progress_bar.progress(100, text="✅ Analysis complete!")
            time.sleep(0.4)
            progress_bar.empty()
            status_text.empty()

            # ── Company header ────────────────────────────────
            st.markdown(f"""
            <div style="text-align:center; margin:10px 0 6px;">
                <span style="font-size:1.1rem; color:#888;">Analysing</span><br>
                <span style="font-size:1.9rem; font-weight:800; color:#fff;">{name}</span>
                <span style="font-size:1.1rem; color:#90caf9;"> ({symbol})</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Recommendation banner ─────────────────────────
            rec_cls = {"BUY": "rec-buy", "HOLD": "rec-hold", "SELL": "rec-sell"}[recommendation]
            st.markdown(
                f'<div class="rec-banner {rec_cls}">{recommendation}</div>',
                unsafe_allow_html=True,
            )

            # ── Top metrics ───────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            price_delta = current_price - previous_close
            price_pct   = (price_delta / previous_close * 100) if previous_close else 0

            c1.metric("Current Price",
                      _fmt(current_price, currency),
                      f"{price_delta:+.2f} ({price_pct:+.2f}%)")
            c2.metric("Predicted Price",
                      _fmt(predicted_price, currency),
                      f"{predicted_price - current_price:+.2f}")
            c3.metric("Fusion Score",
                      f"{fusion_score:.4f}",
                      f"{fusion_score * 100:.1f}%")
            c4.metric("Risk Level",
                      f"{RISK_COLOUR[risk_level]} {risk_level}",
                      f"Model R² {train_result['r2']:.4f}")

            st.markdown("---")

            # ── Fusion score bar ──────────────────────────────
            st.markdown("#### ⚖️ Fusion Score Breakdown")
            st.markdown(_score_bar_html(fusion_score), unsafe_allow_html=True)
            col_t, col_s, col_v = st.columns(3)
            col_t.metric("Trend Score",      f"{trend_score:.2f}",      trend_label)
            col_s.metric("Sentiment Score",  f"{sentiment_norm:.4f}",   sentiment_label)
            col_v.metric("Volatility Score", f"{volatility_score:.4f}", volatility_label)

            st.markdown("---")

            # ── Signal cards ──────────────────────────────────
            st.markdown("#### 🔍 Signal Summary")
            sig1, sig2, sig3 = st.columns(3)

            with sig1:
                trend_icon = "🔼" if trend_label == "UP" else "🔽"
                st.info(f"**Trend** {trend_icon}\n\n**{trend_label}**\n\n"
                        f"Predicted: {_fmt(predicted_price, currency)}")
            with sig2:
                sent_icon = "😊" if sentiment_label == "Positive" else "😟" if sentiment_label == "Negative" else "😐"
                st.info(f"**Sentiment** {sent_icon}\n\n**{sentiment_label}**\n\n"
                        f"Compound: {avg_compound:.4f}")
            with sig3:
                vol_icon = "🟢" if volatility_label == "Low" else "🟡" if volatility_label == "Medium" else "🔴"
                st.info(f"**Volatility** {vol_icon}\n\n**{volatility_label}**\n\n"
                        f"Score: {volatility_score:.4f}")

            st.markdown("---")

            # ── ML model performance ──────────────────────────
            st.markdown("#### 🤖 ML Model Performance")
            dir_acc = train_result.get("dir_accuracy")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("R² Score",             f"{train_result['r2']:.4f}",
                      help="Predictive power on next-day closing price.")
            m2.metric("Directional Accuracy",
                      f"{dir_acc * 100:.1f}%" if dir_acc is not None else "N/A",
                      help="% of test days where up/down direction was correct.")
            m3.metric("MAE",                  _fmt(train_result["mae"], currency),
                      help="Mean Absolute Error – average prediction error.")
            m4.metric("RMSE",                 _fmt(train_result.get("rmse", 0), currency),
                      help="Root Mean Squared Error.")
            m5.metric("Features",             f"{len(train_result['features'])} indicators")

            st.markdown("---")

            # ── Reasoning ─────────────────────────────────────
            st.markdown("#### 💡 Reasoning")
            for reason in reasons:
                st.markdown(f'<div class="reason-item">• {reason}</div>',
                            unsafe_allow_html=True)

            st.markdown("---")

            # ── Charts ────────────────────────────────────────
            st.markdown("#### 📈 Visualisations")
            tab1, tab2, tab3, tab4 = st.tabs([
                "📉 Price Trend",
                "🎯 Pred vs Actual",
                "📰 Sentiment",
                "🌳 Feature Importance",
            ])

            with tab1:
                fig = chart_price_trend(stock_df, symbol)
                st.image(fig_to_bytes(fig), use_column_width=True)
                plt.close(fig)

            with tab2:
                fig = chart_pred_vs_actual(train_result["y_test"], train_result["y_pred"], symbol)
                st.image(fig_to_bytes(fig), use_column_width=True)
                plt.close(fig)

            with tab3:
                fig = chart_sentiment(sentiment_result["individual"], symbol)
                if fig:
                    st.image(fig_to_bytes(fig), use_column_width=True)
                    plt.close(fig)
                else:
                    st.info("No news articles were fetched.")

            with tab4:
                fig = chart_feature_importance(train_result["feature_importance"], symbol)
                st.image(fig_to_bytes(fig), use_column_width=True)
                plt.close(fig)

            st.markdown("---")

            # ── News table ────────────────────────────────────
            if sentiment_result["individual"]:
                st.markdown("#### 📰 Analysed News Articles")
                news_df = pd.DataFrame([
                    {"Title": r["title"][:80], "Compound": r["compound"], "Label": r["label"]}
                    for r in sentiment_result["individual"]
                ])
                st.dataframe(news_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.caption(
                "⚠️ This is a demonstration tool for educational purposes only. "
                "Do NOT use this as the sole basis for real investment decisions. "
                "Always consult a qualified financial advisor."
            )

        except ValueError as exc:
            progress_bar.empty()
            st.error(f"**Invalid symbol or insufficient data:** {exc}")
        except RuntimeError as exc:
            progress_bar.empty()
            st.error(f"**Data fetch failed:** {exc}")
        except Exception as exc:
            progress_bar.empty()
            st.error(f"**Unexpected error:** {exc}")

else:
    # Landing placeholder
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#555;">
        <div style="font-size:5rem;">📊</div>
        <h2 style="color:#4F8EF7; margin:16px 0 8px;">Enter a stock symbol to begin</h2>
        <p style="font-size:1.05rem;">
            Use the <strong>sidebar</strong> on the left to enter a ticker symbol<br>
            and click <strong>🚀 Analyse Now</strong>.
        </p>
        <br>
        <table style="margin:auto; color:#888; border-collapse:collapse;">
            <tr><th style="padding:6px 20px;">Market</th><th style="padding:6px 20px;">Just type…</th></tr>
            <tr><td>🇮🇳 NSE (India)</td><td>TCS &nbsp;·&nbsp; INFY &nbsp;·&nbsp; RELIANCE</td></tr>
            <tr><td>🇮🇳 BSE (India)</td><td>TCS.BO &nbsp;·&nbsp; INFY.BO &nbsp;·&nbsp; RELIANCE.BO</td></tr>
            <tr><td>🇺🇸 NASDAQ / NYSE</td><td>TSLA &nbsp;·&nbsp; AAPL &nbsp;·&nbsp; MSFT &nbsp;·&nbsp; NVDA</td></tr>
        </table>
        <p style="font-size:0.9rem; margin-top:18px; color:#666;">
            💡 No suffix needed for Indian stocks — the exchange is detected automatically.
        </p>
    </div>
    """, unsafe_allow_html=True)
