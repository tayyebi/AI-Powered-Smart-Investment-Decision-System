"""
app.py
------
Main entry point for the Smart Investment Decision Fusion System.

Usage (CLI):
    python app.py --symbol TCS.NS
    python app.py --symbol TSLA --period 1y
    python app.py --symbol INFY.NS --no-plots
"""

import argparse
import sys
from pathlib import Path

# ── Local modules ──────────────────────────────────────────────────────────────
import data_fetcher as df_mod
import sentiment     as sent_mod
import model         as mdl_mod
import fusion        as fus_mod
import utils         as util_mod


# ─────────────────────────────────────────────
# CORE ANALYSIS PIPELINE
# ─────────────────────────────────────────────

def run_analysis(symbol: str, period: str = "6mo", generate_plots: bool = True) -> dict:
    """
    Execute the full analysis pipeline for a given stock symbol.

    Steps:
        1. Fetch historical stock data
        2. Fetch company info (current price)
        3. Fetch financial news
        4. Perform sentiment analysis
        5. Train ML model & predict next price
        6. Compute volatility signal
        7. Run data fusion
        8. Apply decision rules
        9. Determine risk level
       10. Build explanatory reasons
       11. (Optional) Generate charts

    Args:
        symbol        : Stock ticker, e.g. "TCS.NS", "TSLA"
        period        : Historical period, e.g. "6mo", "1y"
        generate_plots: Whether to save visualisation charts.

    Returns:
        dict containing all analysis outputs.
    """
    print(f"\n{'=' * 55}")
    print(f"  Analysing: {symbol.upper()}  |  Period: {period}")
    print(f"{'=' * 55}\n")

    # ── Step 1: Historical stock data ─────────────────────────
    print("[1/9] Fetching historical stock data …")
    stock_df = df_mod.fetch_stock_data(symbol, period=period)

    # ── Step 2: Current price snapshot ────────────────────────
    print("[2/9] Fetching current ticker info …")
    info = df_mod.get_current_info(symbol)

    # Fallback: use last close from history if live price unavailable
    current_price  = info["current_price"]  or float(stock_df["Close"].iloc[-1])
    previous_close = info["previous_close"] or float(stock_df["Close"].iloc[-2])
    currency       = info["currency"]
    name           = info["name"]

    # ── Step 3: News ───────────────────────────────────────────
    print("[3/9] Fetching financial news …")
    # Use base ticker name without exchange suffix for better search results
    news_query = symbol.split(".")[0]
    articles   = df_mod.fetch_news(f"{news_query} stock")

    # ── Step 4: Sentiment analysis ─────────────────────────────
    print("[4/9] Performing sentiment analysis …")
    sentiment_result = sent_mod.analyse_news(articles)
    sentiment_label  = sentiment_result["label"]
    avg_compound     = sentiment_result["average_compound"]
    sentiment_norm   = sent_mod.normalise_sentiment(avg_compound)

    # ── Step 5: Train ML model ─────────────────────────────────
    print("[5/9] Training RandomForestRegressor …")
    train_result = mdl_mod.train_model(stock_df)
    r2  = train_result["r2"]
    mae = train_result["mae"]

    # ── Step 6: Predict next price ─────────────────────────────
    print("[6/9] Predicting next trading day price …")
    predicted_price = mdl_mod.predict_next_price(train_result)

    # ── Step 7: Volatility signal ──────────────────────────────
    print("[7/9] Computing volatility signal …")
    vol_result      = mdl_mod.compute_volatility_signal(stock_df)
    volatility_score = vol_result["score"]
    volatility_label = vol_result["label"]

    # ── Step 8: Trend signal & fusion score ────────────────────
    print("[8/9] Running data fusion …")
    trend_result  = fus_mod.compute_trend_signal(current_price, predicted_price)
    trend_score   = trend_result["score"]
    trend_label   = trend_result["label"]

    fusion_score  = fus_mod.compute_fusion_score(trend_score, sentiment_norm, volatility_score)
    recommendation= fus_mod.apply_decision_rules(fusion_score)

    # ── Step 9: Risk level & reasons ──────────────────────────
    print("[9/9] Building explanation …")
    risk_level = fus_mod.compute_risk_level(volatility_label, fusion_score)
    reasons    = fus_mod.build_reasons(
        trend_label, sentiment_label, volatility_label,
        current_price, predicted_price, recommendation,
    )

    # ── Compile result ─────────────────────────────────────────
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
        # Raw data (for plotting)
        "_stock_df"       : stock_df,
        "_train_result"   : train_result,
        "_sent_individual": sentiment_result["individual"],
    }

    # ── Optional charts ────────────────────────────────────────
    if generate_plots:
        print("\n[Charts] Generating visualisations …")
        chart_paths = util_mod.generate_all_plots(
            stock_df, train_result,
            sentiment_result["individual"],
            symbol.upper(),
            out_dir="charts",
        )
        result["chart_paths"] = chart_paths

    return result


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Investment Decision Fusion System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --symbol TCS.NS
  python app.py --symbol TSLA --period 1y
  python app.py --symbol INFY.NS --no-plots
        """,
    )
    parser.add_argument(
        "--symbol", "-s",
        required=True,
        help=(
            "Stock ticker symbol (e.g., TCS.NS for NSE, "
            "INFY.NS for NSE, TSLA for NASDAQ)."
        ),
    )
    parser.add_argument(
        "--period", "-p",
        default="2y",
        choices=["1y", "2y", "3y", "4y", "5y"],
        help="Historical data period (default: 2y).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip chart generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        result = run_analysis(
            symbol        = args.symbol,
            period        = args.period,
            generate_plots= not args.no_plots,
        )
        util_mod.print_result(result)

        if not args.no_plots and "chart_paths" in result:
            print("\n  📊 Charts saved:")
            for name, path in result["chart_paths"].items():
                if path:
                    print(f"     • {name}: {path}")

    except ValueError as exc:
        print(f"\n[ERROR] Invalid input: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\n[ERROR] Runtime failure: {exc}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
