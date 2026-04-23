"""
src/data/stock.py
-----------------
Fetches historical OHLCV data and current ticker info.

Primary source: Alpha Vantage (when ALPHA_VANTAGE_API_KEY is set).
Fallback source: yfinance (always available, no key required).
"""

import os

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# days_map: approximate trading days per period label
_DAYS_MAP = {"3mo": 63, "6mo": 126, "1y": 252, "2y": 504}
_LOOKBACK = 50  # extra days needed for SMA_50 indicator


def fetch_stock_data(symbol: str, period: str) -> pd.DataFrame:
    """Return a DataFrame of OHLCV data for *symbol* covering *period*.

    Args:
        symbol: Ticker, e.g. ``"TCS.NS"`` or ``"TSLA"``.
        period: One of ``"3mo"``, ``"6mo"``, ``"1y"``, ``"2y"``.

    Returns:
        DataFrame with columns Open, High, Low, Close, Volume
        sorted by date ascending.

    Raises:
        ValueError: When no data can be retrieved from any source.
    """
    target_days = _DAYS_MAP.get(period, 126) + _LOOKBACK
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    df = None

    if api_key:
        df = _fetch_alpha_vantage(symbol, api_key, target_days)

    if df is None or df.empty:
        df = _fetch_yfinance(symbol, period, target_days)

    return df


def get_current_info(symbol: str) -> dict:
    """Return a snapshot dict: current_price, previous_close, currency, name.

    Tries Alpha Vantage first (if key present), falls back to yfinance.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if api_key:
        result = _current_info_alpha_vantage(symbol, api_key)
        if result:
            return result

    return _current_info_yfinance(symbol)


# ── Private helpers ─────────────────────────────────────────────────────────

def _fetch_alpha_vantage(symbol: str, api_key: str, target_days: int):
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "Time Series (Daily)" not in data:
            print("Alpha Vantage rate limit or unexpected response – falling back to yfinance.")
            return None
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df.index = pd.to_datetime(df.index)
        df.rename(columns={
            "1. open": "Open", "2. high": "High",
            "3. low": "Low",   "4. close": "Close", "5. volume": "Volume",
        }, inplace=True)
        df = df.apply(pd.to_numeric)
        df.sort_index(ascending=True, inplace=True)
        return df.tail(target_days)
    except Exception as exc:
        print(f"Alpha Vantage fetch failed: {exc}")
        return None


def _fetch_yfinance(symbol: str, period: str, target_days: int) -> pd.DataFrame:
    yf_period = "1y" if period in ("3mo", "6mo") else "2y" if period == "1y" else "5y"
    stock = yf.Ticker(symbol)
    df = stock.history(period=yf_period)
    if df.empty:
        raise ValueError(
            f"No data found for '{symbol}' via Alpha Vantage or yfinance."
        )
    return df.tail(target_days)


def _current_info_alpha_vantage(symbol: str, api_key: str):
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        quote = resp.json().get("Global Quote", {})
        if quote and quote.get("05. price"):
            return {
                "current_price":  float(quote["05. price"]),
                "previous_close": float(quote.get("08. previous close", 0)),
                "currency": "USD",
                "name": symbol,
            }
    except Exception as exc:
        print(f"Alpha Vantage GLOBAL_QUOTE failed: {exc}")
    return None


def _current_info_yfinance(symbol: str) -> dict:
    info = yf.Ticker(symbol).info
    return {
        "current_price":  info.get("currentPrice") or info.get("regularMarketPrice"),
        "previous_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
        "currency": info.get("currency", "USD"),
        "name": info.get("shortName") or info.get("longName") or symbol,
    }
