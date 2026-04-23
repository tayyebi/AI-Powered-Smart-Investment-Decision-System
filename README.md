# 📊 Smart Investment Decision Fusion System

A complete end-to-end Python project that integrates **real-time stock data**, **financial news sentiment**, and a **Machine Learning price predictor** into a unified investment decision engine.

---

## 🎯 Features

| Feature | Details |
|---|---|
| 📡 Live Market Data | yfinance – historical & real-time OHLCV |
| 📰 News Sentiment | NewsAPI + VADER compound scoring |
| 🤖 ML Prediction | RandomForestRegressor (200 trees, 6-month history) |
| ⚖️ Data Fusion | Weighted signal fusion: `0.5×Trend + 0.3×Sentiment + 0.2×Volatility` |
| 🚦 Decision Rules | BUY (>0.70) · HOLD (0.40–0.70) · SELL (<0.40) |
| 📊 Visualisations | Price trend · Predicted vs Actual · Sentiment bar chart |
| 🎛️ Streamlit UI | Full interactive web dashboard |
| 🔐 API Security | Secrets loaded from `.env`, never hardcoded |

---

## 📁 Project Structure

```
project/
├── app.py              ← CLI entry point
├── streamlit_app.py    ← Streamlit web UI
├── data_fetcher.py     ← yfinance + NewsAPI data retrieval
├── sentiment.py        ← VADER sentiment analysis
├── model.py            ← Feature engineering + RandomForest training
├── fusion.py           ← Data fusion, decision rules, risk level
├── utils.py            ← Formatting, console display, charts
├── requirements.txt    ← Python dependencies
├── .env.example        ← Template for environment variables
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone / open the project

```bash
cd "Mini project"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
# Copy the example file
copy .env.example .env

# Edit .env and paste your NewsAPI key
# Get a free key at https://newsapi.org/register
NEWS_API_KEY=paste_your_key_here
```

> **Note:** The system works without a NewsAPI key — sentiment will default to Neutral.

---

## 🖥️ Usage

### CLI (Command Line)

```bash
# Basic analysis
python app.py --symbol TCS.NS

# 1-year period
python app.py --symbol TSLA --period 1y

# Skip chart generation
python app.py --symbol INFY.NS --no-plots

# Help
python app.py --help
```

**NSE India stocks** need `.NS` suffix: `TCS.NS`, `INFY.NS`, `RELIANCE.NS`  
**US stocks** use bare symbols: `TSLA`, `AAPL`, `MSFT`, `NVDA`

### Streamlit Web UI

```bash
streamlit run streamlit_app.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Example Output

```
═══════════════════════════════════════════════════════
  SMART INVESTMENT DECISION SYSTEM
═══════════════════════════════════════════════════════

  Stock   : TCS.NS – Tata Consultancy Services Limited
  Currency: INR

  Current Price  : ₹4,012.50
  Previous Close : ₹3,988.00
  Predicted Price: ₹4,187.30

──────────────────────────────────────────────────────
  Trend      : UP
  Sentiment  : Positive  (Compound: 0.2341)
  Volatility : Low       (Score: 0.1823)
  Risk Level : Low
  Model R²   : 0.9412  |  MAE: 48.2100

──────────────────────────────────────────────────────
  Final Score    : 0.7813 (78.1%)
  Recommendation : BUY

──────────────────────────────────────────────────────
  Reasoning:
    • Predicted price is higher by 4.36% (upward trend)
    • News sentiment is positive, indicating market optimism
    • Low price volatility reduces investment risk
    • Strong bullish indicators support a BUY decision
═══════════════════════════════════════════════════════
```

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: Stock Symbol                      │
└─────────────────┬───────────────────────┬───────────────────┘
                  │                       │
          ┌───────▼──────┐      ┌─────────▼────────┐
          │  yfinance    │      │   NewsAPI         │
          │  OHLCV data  │      │   Headlines       │
          └───────┬──────┘      └─────────┬─────────┘
                  │                       │
     ┌────────────▼──────┐    ┌───────────▼──────────┐
     │ Feature           │    │ VADER Sentiment       │
     │ Engineering       │    │ Analysis              │
     │ + RandomForest    │    │ Compound → [0,1]      │
     └────────┬──────────┘    └──────────┬───────────┘
              │                          │
              │   ┌──────────────────────┘
              │   │
     ┌────────▼───▼──────────────────────────────────┐
     │   DATA FUSION ENGINE                           │
     │   Score = 0.5×Trend + 0.3×Sentiment + 0.2×Vol │
     └────────────────────────┬──────────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Decision Rules     │
                   │  >0.70 → BUY        │
                   │  0.40-0.70 → HOLD   │
                   │  <0.40 → SELL       │
                   └─────────────────────┘
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `yfinance` | Stock market data |
| `requests` | NewsAPI HTTP calls |
| `vaderSentiment` | News sentiment analysis |
| `scikit-learn` | ML model (RandomForestRegressor) |
| `matplotlib` | Chart generation |
| `python-dotenv` | Secure API key loading |
| `streamlit` | Interactive web dashboard |

---

## 🔐 Security

- **Never** commit real `.env` file (`.gitignore` excludes it)
- API key only lives in `.env` and is loaded via `python-dotenv`
- Rotate your NewsAPI key if accidentally exposed

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**.  
Do **NOT** use as the sole basis for real investment decisions.  
Always consult a qualified financial advisor.
