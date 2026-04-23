"""
pipeline.py
-----------
Core analysis pipeline for the Smart Investment Decision System.

Orchestrates data fetching, ML modelling, and signal fusion to produce
a single result dict that both the CLI and the Streamlit UI consume.
"""

from src.analysis import fusion, model, sentiment, volatility
from src.data import news as news_mod
from src.data import stock as stock_mod
from src.utils.printing import generate_all_plots


def run_analysis(
    symbol: str,
    period: str = "6mo",
    generate_plots: bool = True,
) -> dict:
    """Execute the full analysis pipeline for a given stock symbol.

    Steps:
        1. Fetch historical OHLCV data
        2. Fetch current ticker info (price, currency, name)
        3. Fetch news from all configured sources
        4. Perform VADER sentiment analysis
        5. Train XGBoost model
        6. Predict next-day price
        7. Compute annualised volatility signal
        8. Compute trend signal
        9. Run data fusion & apply decision rules
       10. Determine risk level and build explanatory reasons
       11. (Optional) Save visualisation charts

    Args:
        symbol:         Stock ticker, e.g. ``"TCS.NS"`` or ``"TSLA"``.
        period:         Historical period – ``"1y"``, ``"2y"``, …
        generate_plots: Whether to save PNG charts to the ``charts/`` folder.

    Returns:
        Dictionary with all analysis outputs (prices, signals, model metrics,
        recommendation, reasons, and optional chart paths).
    """
    print(f"\n{'=' * 55}")
    print(f"  Analysing: {symbol.upper()}  |  Period: {period}")
    print(f"{'=' * 55}\n")

    # 1. Historical stock data
    print("[1/9] Fetching historical stock data …")
    stock_df = stock_mod.fetch_stock_data(symbol, period=period)

    # 2. Current price snapshot
    print("[2/9] Fetching current ticker info …")
    info           = stock_mod.get_current_info(symbol)
    current_price  = info["current_price"]  or float(stock_df["Close"].iloc[-1])
    previous_close = info["previous_close"] or float(stock_df["Close"].iloc[-2])
    currency       = info["currency"]
    name           = info["name"]

    # 3. News (strip exchange suffix for cleaner search)
    print("[3/9] Fetching financial news …")
    news_query = symbol.split(".")[0]
    articles   = news_mod.fetch_news(f"{news_query} stock")

    # 4. Sentiment analysis
    print("[4/9] Performing sentiment analysis …")
    sentiment_result = sentiment.analyse_news(articles)
    sentiment_label  = sentiment_result["label"]
    avg_compound     = sentiment_result["average_compound"]
    sentiment_norm   = sentiment.normalise_sentiment(avg_compound)

    # 5. Train ML model
    print("[5/9] Training XGBoost model …")
    train_result = model.train_model(stock_df)
    r2  = train_result["r2"]
    mae = train_result["mae"]

    # 6. Predict next price
    print("[6/9] Predicting next trading day price …")
    predicted_price = model.predict_next_price(train_result)

    # 7. Volatility signal
    print("[7/9] Computing volatility signal …")
    vol_result       = volatility.compute_volatility_signal(stock_df)
    volatility_score = vol_result["score"]
    volatility_label = vol_result["label"]

    # 8. Trend signal & fusion
    print("[8/9] Running data fusion …")
    trend_result   = fusion.compute_trend_signal(current_price, predicted_price)
    trend_score    = trend_result["score"]
    trend_label    = trend_result["label"]
    fusion_score   = fusion.compute_fusion_score(trend_score, sentiment_norm, volatility_score)
    recommendation = fusion.apply_decision_rules(fusion_score)

    # 9. Risk level & reasons
    print("[9/9] Building explanation …")
    risk_level = fusion.compute_risk_level(volatility_label, fusion_score)
    reasons    = fusion.build_reasons(
        trend_label, sentiment_label, volatility_label,
        current_price, predicted_price, recommendation,
    )

    result = {
        # Identity
        "symbol"          : symbol.upper(),
        "name"            : name,
        "currency"        : currency,
        # Prices
        "current_price"   : current_price,
        "previous_close"  : previous_close,
        "predicted_price" : predicted_price,
        # Signals
        "trend_label"     : trend_label,
        "trend_score"     : trend_score,
        "sentiment_label" : sentiment_label,
        "sentiment_score" : avg_compound,
        "sentiment_norm"  : sentiment_norm,
        "volatility_label": volatility_label,
        "volatility_score": volatility_score,
        "risk_level"      : risk_level,
        # Model
        "r2"              : r2,
        "mae"             : mae,
        # Fusion
        "fusion_score"    : fusion_score,
        "recommendation"  : recommendation,
        "reasons"         : reasons,
        # Raw data (for the UI and plot generation)
        "_stock_df"       : stock_df,
        "_train_result"   : train_result,
        "_sent_individual": sentiment_result["individual"],
    }

    if generate_plots:
        print("\n[Charts] Generating visualisations …")
        result["chart_paths"] = generate_all_plots(
            stock_df,
            train_result,
            sentiment_result["individual"],
            symbol.upper(),
            out_dir="charts",
        )

    return result
