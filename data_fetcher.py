import yfinance as yf
import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_stock_data(symbol: str, period: str) -> pd.DataFrame:
    # 3mo = ~63 days, 6mo = ~126 days, 1y = ~252 days, 2y = ~504 days
    days_map = {"3mo": 63, "6mo": 126, "1y": 252, "2y": 504}
    lookback = 50 # Extra days needed for SMA_50
    target_days = days_map.get(period, 126) + lookback
    
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    df = None
    
    if api_key:
        try:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if "Time Series (Daily)" in data:
                df_av = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                df_av.index = pd.to_datetime(df_av.index)
                df_av.rename(columns={
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "4. close": "Close",
                    "5. volume": "Volume"
                }, inplace=True)
                for col in df_av.columns:
                    df_av[col] = pd.to_numeric(df_av[col])
                df_av = df_av.sort_index(ascending=True)
                df = df_av.tail(target_days)
            else:
                print("Alpha Vantage rate limit or issue. Falling back to yfinance.")
        except Exception as e:
            print(f"Alpha Vantage fetch failed: {e}")
            
    if df is None or df.empty:
        stock = yf.Ticker(symbol)
        yf_period = "1y" if period in ["3mo", "6mo"] else "2y" if period == "1y" else "5y"
        df_yf = stock.history(period=yf_period)
        if df_yf.empty:
            raise ValueError(f"No data found for symbol {symbol} using either Alpha Vantage or yfinance.")
        df = df_yf.tail(target_days)

    return df

def get_current_info(symbol: str) -> dict:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    result = None
    
    if api_key:
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            quote = data.get("Global Quote", {})
            if quote and quote.get("05. price"):
                result = {
                    "current_price": float(quote.get("05. price")),
                    "previous_close": float(quote.get("08. previous close", 0)),
                    "currency": "USD",
                    "name": symbol
                }
        except Exception as e:
            print(f"Alpha Vantage GLOBAL_QUOTE failed: {e}")
            
    if not result:
        stock = yf.Ticker(symbol)
        info = stock.info
        result = {
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "previous_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
            "currency": info.get("currency", "USD"),
            "name": info.get("shortName") or info.get("longName") or symbol
        }
        
    return result

def fetch_news(query: str) -> list:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        print("Warning: NEWS_API_KEY not found in environment.")
        return []
        
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=10"
    headers = {"X-Api-Key": api_key}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        return [
            {
                "title": art.get("title", ""), 
                "content": art.get("content", "") or art.get("description", "")
            } 
            for art in articles if art.get("title")
        ]
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []
