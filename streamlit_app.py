"""
streamlit_app.py
----------------
Streamlit-based Web UI for the Smart Investment Decision Fusion System.

Run with:
    streamlit run streamlit_app.py
"""

import sys
import time
from pathlib import Path
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Local modules ──────────────────────────────────────────────────────────────
import data_fetcher as df_mod
import sentiment    as sent_mod
import model        as mdl_mod
import fusion       as fus_mod
import utils        as util_mod

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Investment Decision System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricLabel"] { color: #aaa; font-size: 0.82rem; }
    [data-testid="stMetricValue"] { font-weight: 700; font-size: 1.4rem; }

    /* ── Recommendation banner ── */
    .rec-banner {
        border-radius: 16px;
        padding: 20px 32px;
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: 2px;
        margin: 8px 0 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .rec-buy  { background: linear-gradient(120deg,#0db560,#00e676); color:#000; }
    .rec-hold { background: linear-gradient(120deg,#f5a623,#ffd740); color:#000; }
    .rec-sell { background: linear-gradient(120deg,#e53935,#ff1744); color:#fff; }

    /* ── Score bar ── */
    .score-wrap { margin: 8px 0 4px; }
    .score-bar-track {
        height: 14px; border-radius: 99px;
        background: rgba(255,255,255,0.10);
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%; border-radius: 99px;
        transition: width .5s ease;
    }

    /* ── Reason bullets ── */
    .reason-item {
        background: rgba(255,255,255,0.05);
        border-left: 3px solid #4F8EF7;
        border-radius: 0 10px 10px 0;
        padding: 9px 16px;
        margin: 6px 0;
        font-size: 0.93rem;
    }

    /* ── Section headers ── */
    h2, h3 { color: #90caf9 !important; }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(120deg, #4F8EF7, #a78bfa);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-weight: 700;
        font-size: 1rem;
        transition: opacity .2s;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.87; }

    /* ── Input ── */
    .stTextInput input, .stSelectbox select {
        background: rgba(255,255,255,0.06) !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}
RISK_COLOUR      = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
TREND_ICON       = {"UP": "🔼", "DOWN": "🔽"}

def resolve_symbol(raw: str) -> tuple[str, str | None]:
    """
    Auto-resolve ANY bare ticker to the correct yfinance symbol.

    Strategy (for tickers with no '.' suffix):
      1. Try <TICKER>.NS  (NSE India)
      2. Try <TICKER>.BO  (BSE India)
      3. Try <TICKER>     (US markets — NASDAQ / NYSE)

    If the user already typed a suffix (TCS.NS, RELIANCE.BO, TSLA),
    it is used as-is with no lookup.

    Returns:
        (resolved_symbol, note_string | None)
    """
    import yfinance as yf

    raw = raw.strip().upper()
    if not raw:
        return raw, None

    # Already has exchange suffix — trust the user
    if "." in raw:
        return raw, None

    # For any bare ticker: probe NSE → BSE → US in order
    candidates = [f"{raw}.NS", f"{raw}.BO", raw]

    for candidate in candidates:
        try:
            ticker = yf.Ticker(candidate)
            info = ticker.info
            if info.get("longName") or info.get("shortName") or info.get("regularMarketPrice"):
                note = None if candidate == raw else f"Auto-resolved **{raw}** → **{candidate}**"
                return candidate, note
        except Exception:
            continue

    # Fall back to raw and let the pipeline raise a proper error
    return raw, f"⚠️ Could not auto-resolve **{raw}**; trying as-is."


def fmt(price: float, currency: str = "USD") -> str:
    sym = CURRENCY_SYMBOLS.get(currency, currency + " ")
    return f"{sym}{price:,.2f}"


def fig_to_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="none", transparent=True)
    buf.seek(0)
    return buf.read()


def score_bar_html(score: float) -> str:
    pct = score * 100
    if score > 0.70:
        colour = "#00e676"
    elif score >= 0.40:
        colour = "#ffd740"
    else:
        colour = "#ff1744"
    return f"""
    <div class="score-wrap">
        <div class="score-bar-track">
            <div class="score-bar-fill" style="width:{pct:.1f}%;background:{colour};"></div>
        </div>
    </div>
    """


# ─────────────────────────────────────────────
# PLOT HELPERS (inline Matplotlib, dark theme)
# ─────────────────────────────────────────────

DARK_BG   = "#0d1117"
DARK_GRID = "#1e2a38"
BLUE      = "#4F8EF7"
ORANGE    = "#F7A24F"
PURPLE    = "#A24FF7"


def dark_fig(w=11, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(DARK_GRID)
    ax.tick_params(colors="#aaa")
    ax.xaxis.label.set_color("#aaa")
    ax.yaxis.label.set_color("#aaa")
    ax.title.set_color("#e0e0e0")
    ax.grid(color=DARK_GRID, linewidth=0.6)
    return fig, ax


def chart_price_trend(df: pd.DataFrame, symbol: str):
    fig, ax = dark_fig(11, 4.5)
    ax.plot(df.index, df["Close"], label="Close", linewidth=1.6, color=BLUE)
    ax.plot(df.index, df["Close"].rolling(20).mean(),
            label="MA-20", linewidth=1.2, linestyle="--", color=ORANGE)
    ax.plot(df.index, df["Close"].rolling(50).mean(),
            label="MA-50", linewidth=1.2, linestyle=":", color=PURPLE)
    ax.set_title(f"{symbol} – Price History & Moving Averages", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Price")
    ax.legend(facecolor=DARK_BG, labelcolor="#ccc")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=25)
    plt.tight_layout()
    return fig


def chart_pred_vs_actual(y_test, y_pred, symbol: str):
    fig, ax = dark_fig(6, 6)
    ax.scatter(y_test, y_pred, alpha=0.6, color=BLUE,
               edgecolors="white", linewidths=0.3, s=45)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_title(f"{symbol} – Predicted vs Actual (Test Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Price"); ax.set_ylabel("Predicted Price")
    ax.legend(facecolor=DARK_BG, labelcolor="#ccc")
    plt.tight_layout()
    return fig


def chart_sentiment(individual: list, symbol: str):
    if not individual:
        return None
    titles    = [r["title"][:35] + "…" if len(r["title"]) > 35 else r["title"]
                 for r in individual]
    compounds = [r["compound"] for r in individual]
    colours   = ["#4CAF50" if c >= 0.05 else "#F44336" if c <= -0.05 else "#9E9E9E"
                 for c in compounds]
    h = max(5, len(titles) * 0.42)
    fig, ax = dark_fig(11, h)
    ax.barh(range(len(titles)), compounds, color=colours, edgecolor=DARK_BG)
    ax.set_yticks(range(len(titles)))
    ax.set_yticklabels(titles, fontsize=8, color="#ccc")
    ax.axvline(0, color="#aaa", linewidth=0.8)
    ax.axvline(0.05,  color="#4CAF50", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(-0.05, color="#F44336", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("VADER Compound Score"); 
    ax.set_title(f"{symbol} – News Sentiment Scores", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_feature_importance(importance: pd.Series, symbol: str):
    top = importance.head(10)
    fig, ax = dark_fig(9, 4)
    colours = [BLUE] * len(top)
    ax.barh(top.index[::-1], top.values[::-1], color=colours[::-1], edgecolor=DARK_BG)
    ax.set_title(f"{symbol} – Top Feature Importances", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

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
        help="More data = better model. 2y–5y recommended for strong R² and directional accuracy.",
    )

    st.markdown("---")
    run_btn = st.button("🚀 Analyse Now", use_container_width=True)

    # Show quick examples
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
    - 📡 Fetches live market data via **Alpha Vantage (fallback to yfinance)**
    - 📰 Retrieves latest news via **NewsAPI**
    - 🧠 Trains **RandomForestRegressor** on 6 months of data
    - 💬 Sentiment scored with **VADER**
    - ⚖️ Scores fused: `0.5×Trend + 0.3×Sent + 0.2×Vol`
    """)

    st.markdown("---")
    st.caption("🔐 API key loaded from `.env` file")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────

if run_btn:
    if not symbol_raw:
        st.error("Please enter a stock symbol in the sidebar.")
    else:
        # ── Auto-resolve symbol ───────────────────────────────
        with st.spinner(f"Resolving ticker **{symbol_raw}** …"):
            symbol, resolve_note = resolve_symbol(symbol_raw)

        if resolve_note:
            if resolve_note.startswith("⚠️"):
                st.warning(resolve_note)
            else:
                st.info(f"🔍 {resolve_note}")

        # ── Progress bar ─────────────────────────────────────
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
            # ── 1: Stock data ──────────────────────────────────
            progress_bar.progress(5,  text=steps[0])
            stock_df = df_mod.fetch_stock_data(symbol, period=period)

            # ── 2: Info ────────────────────────────────────────
            progress_bar.progress(15, text=steps[1])
            info = df_mod.get_current_info(symbol)
            current_price  = info["current_price"]  or float(stock_df["Close"].iloc[-1])
            previous_close = info["previous_close"] or float(stock_df["Close"].iloc[-2])
            currency       = info["currency"]
            name           = info["name"]

            # ── 3: News ────────────────────────────────────────
            progress_bar.progress(30, text=steps[2])
            news_query = symbol.split(".")[0]
            articles   = df_mod.fetch_news(f"{news_query} stock")

            # ── 4: Sentiment ───────────────────────────────────
            progress_bar.progress(42, text=steps[3])
            sentiment_result = sent_mod.analyse_news(articles)
            avg_compound     = sentiment_result["average_compound"]
            sentiment_label  = sentiment_result["label"]
            sentiment_norm   = sent_mod.normalise_sentiment(avg_compound)

            # ── 5 & 6: Model + Prediction ──────────────────────
            progress_bar.progress(55, text=steps[4])
            train_result = mdl_mod.train_model(stock_df)
            progress_bar.progress(70, text=steps[5])
            predicted_price = mdl_mod.predict_next_price(train_result)

            # ── 7: Volatility ──────────────────────────────────
            progress_bar.progress(78, text=steps[6])
            vol_result       = mdl_mod.compute_volatility_signal(stock_df)
            volatility_score = vol_result["score"]
            volatility_label = vol_result["label"]

            # ── 8: Fusion ──────────────────────────────────────
            progress_bar.progress(88, text=steps[7])
            trend_result   = fus_mod.compute_trend_signal(current_price, predicted_price)
            trend_label    = trend_result["label"]
            trend_score    = trend_result["score"]
            fusion_score   = fus_mod.compute_fusion_score(trend_score, sentiment_norm, volatility_score)
            recommendation = fus_mod.apply_decision_rules(fusion_score)
            risk_level     = fus_mod.compute_risk_level(volatility_label, fusion_score)

            # ── 9: Reasons ─────────────────────────────────────
            progress_bar.progress(95, text=steps[8])
            reasons = fus_mod.build_reasons(
                trend_label, sentiment_label, volatility_label,
                current_price, predicted_price, recommendation,
            )

            progress_bar.progress(100, text="✅ Analysis complete!")
            time.sleep(0.4)
            progress_bar.empty()
            status_text.empty()

            # ══════════════════════════════════════════════════
            # DISPLAY RESULTS
            # ══════════════════════════════════════════════════

            # ── Company header ─────────────────────────────────
            st.markdown(f"""
            <div style="text-align:center; margin:10px 0 6px;">
                <span style="font-size:1.1rem; color:#888;">Analysing</span><br>
                <span style="font-size:1.9rem; font-weight:800; color:#fff;">{name}</span>
                <span style="font-size:1.1rem; color:#90caf9;"> ({symbol})</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Recommendation banner ──────────────────────────
            rec_cls = {"BUY": "rec-buy", "HOLD": "rec-hold", "SELL": "rec-sell"}[recommendation]
            st.markdown(
                f'<div class="rec-banner {rec_cls}">{recommendation}</div>',
                unsafe_allow_html=True,
            )

            # ── Top metrics: 4 columns ─────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            price_delta = current_price - previous_close
            price_pct   = (price_delta / previous_close) * 100 if previous_close else 0

            c1.metric("Current Price",
                      fmt(current_price, currency),
                      f"{price_delta:+.2f} ({price_pct:+.2f}%)")
            c2.metric("Predicted Price",
                      fmt(predicted_price, currency),
                      f"{predicted_price - current_price:+.2f}")
            c3.metric("Fusion Score",
                      f"{fusion_score:.4f}",
                      f"{fusion_score * 100:.1f}%")
            c4.metric("Risk Level",
                      f"{RISK_COLOUR[risk_level]} {risk_level}",
                      f"Model R² {train_result['r2']:.4f}")

            st.markdown("---")

            # ── Score bar ──────────────────────────────────────
            st.markdown("#### ⚖️ Fusion Score Breakdown")
            st.markdown(score_bar_html(fusion_score), unsafe_allow_html=True)
            col_t, col_s, col_v = st.columns(3)
            col_t.metric("Trend Score",     f"{trend_score:.2f}", trend_label)
            col_s.metric("Sentiment Score", f"{sentiment_norm:.4f}", sentiment_label)
            col_v.metric("Volatility Score",f"{volatility_score:.4f}", volatility_label)

            st.markdown("---")

            # ── Signal cards ───────────────────────────────────
            st.markdown("#### 🔍 Signal Summary")
            sig1, sig2, sig3 = st.columns(3)

            with sig1:
                trend_icon = "🔼" if trend_label == "UP" else "🔽"
                st.info(f"**Trend** {trend_icon}\n\n**{trend_label}**\n\n"
                        f"Predicted: {fmt(predicted_price, currency)}")
            with sig2:
                sent_icon = "😊" if sentiment_label == "Positive" else "😟" if sentiment_label == "Negative" else "😐"
                st.info(f"**Sentiment** {sent_icon}\n\n**{sentiment_label}**\n\n"
                        f"Compound: {avg_compound:.4f}")
            with sig3:
                vol_icon = "🟢" if volatility_label == "Low" else "🟡" if volatility_label == "Medium" else "🔴"
                st.info(f"**Volatility** {vol_icon}\n\n**{volatility_label}**\n\n"
                        f"Score: {volatility_score:.4f}")

            st.markdown("---")

            # ── Model performance ──────────────────────────────
            st.markdown("#### 🤖 ML Model Performance")

            dir_acc = train_result.get("dir_accuracy", None)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric(
                "R² Score",
                f"{train_result['r2']:.4f}",
                help="Predictive power on next-day closing price."
            )
            m2.metric(
                "Directional Accuracy",
                f"{dir_acc*100:.1f}%" if dir_acc is not None else "N/A",
                help="% of test days where we correctly predicted up/down trend."
            )
            m3.metric(
                "MAE",
                fmt(train_result['mae'], currency),
                help="Mean Absolute Error — average price prediction error."
            )
            m4.metric(
                "RMSE",
                fmt(train_result.get('rmse', 0), currency),
                help="Root Mean Squared Error."
            )
            m5.metric("Features", f"{len(train_result['features'])} indicators")

            st.markdown("---")

            # ── Reasons ────────────────────────────────────────
            st.markdown("#### 💡 Reasoning")
            for reason in reasons:
                st.markdown(f'<div class="reason-item">• {reason}</div>',
                            unsafe_allow_html=True)

            st.markdown("---")

            # ── Charts ─────────────────────────────────────────
            st.markdown("#### 📈 Visualisations")

            tab1, tab2, tab3, tab4 = st.tabs([
                "📉 Price Trend",
                "🎯 Pred vs Actual",
                "📰 Sentiment",
                "🌳 Feature Importance",
            ])

            with tab1:
                fig_pt = chart_price_trend(stock_df, symbol)
                st.image(fig_to_bytes(fig_pt), use_column_width=True)
                plt.close(fig_pt)

            with tab2:
                fig_pa = chart_pred_vs_actual(
                    train_result["y_test"], train_result["y_pred"], symbol)
                st.image(fig_to_bytes(fig_pa), use_column_width=True)
                plt.close(fig_pa)

            with tab3:
                fig_s = chart_sentiment(sentiment_result["individual"], symbol)
                if fig_s:
                    st.image(fig_to_bytes(fig_s), use_column_width=True)
                    plt.close(fig_s)
                else:
                    st.info("No news articles were fetched. Ensure NEWS_API_KEY is set in `.env`.")

            with tab4:
                fig_fi = chart_feature_importance(train_result["feature_importance"], symbol)
                st.image(fig_to_bytes(fig_fi), use_column_width=True)
                plt.close(fig_fi)

            st.markdown("---")

            # ── News table ─────────────────────────────────────
            if sentiment_result["individual"]:
                st.markdown("#### 📰 Analysed News Articles")
                news_df = pd.DataFrame([
                    {
                        "Title"   : r["title"][:80],
                        "Compound": r["compound"],
                        "Label"   : r["label"],
                    }
                    for r in sentiment_result["individual"]
                ])
                st.dataframe(news_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # ── Disclaimer ─────────────────────────────────────
            st.caption(
                "⚠️ This is a demonstration tool for educational purposes only. "
                "Do NOT use this as sole basis for real investment decisions. "
                "Always consult a qualified financial advisor."
            )

        # ── Error handling ────────────────────────────────────
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
    # ── Landing placeholder ────────────────────────────────────
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
            <tr><td>🇮🇳 NSE (India)</td><td>TCS &nbsp;·&nbsp; INFY &nbsp;·&nbsp; RELIANCE &nbsp;·&nbsp; WIPRO</td></tr>
            <tr><td>🇮🇳 BSE (India)</td><td>TCS.BO &nbsp;·&nbsp; INFY.BO &nbsp;·&nbsp; RELIANCE.BO</td></tr>
            <tr><td>🇺🇸 NASDAQ / NYSE</td><td>TSLA &nbsp;·&nbsp; AAPL &nbsp;·&nbsp; MSFT &nbsp;·&nbsp; NVDA</td></tr>
        </table>
        <p style="font-size:0.9rem; margin-top:18px; color:#666;">
            💡 No suffix needed for Indian stocks — the exchange is detected automatically.
        </p>
    </div>
    """, unsafe_allow_html=True)
