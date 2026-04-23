"""
src/utils/printing.py
---------------------
CLI output formatting and chart-saving helpers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


# ── Currency / formatting ───────────────────────────────────────────────────

_CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}


def fmt(price: float, currency: str = "USD") -> str:
    """Format *price* with the appropriate currency symbol."""
    sym = _CURRENCY_SYMBOLS.get(currency, f"{currency} ")
    return f"{sym}{price:,.2f}"


# ── CLI result printer ──────────────────────────────────────────────────────

def print_result(result: dict) -> None:
    """Pretty-print the analysis result dict to stdout."""
    sep = "─" * 55
    currency = result.get("currency", "USD")
    print(f"\n{sep}")
    print(f"  {result['name']}  ({result['symbol']})")
    print(sep)
    print(f"  Current price   : {fmt(result['current_price'],  currency)}")
    print(f"  Predicted price : {fmt(result['predicted_price'], currency)}")
    print(f"  Trend           : {result['trend_label']}")
    print(f"  Sentiment       : {result['sentiment_label']}  (compound {result['sentiment_score']:.4f})")
    print(f"  Volatility      : {result['volatility_label']}  (score {result['volatility_score']:.4f})")
    print(f"  Risk level      : {result['risk_level']}")
    print(f"  Fusion score    : {result['fusion_score']:.4f}")
    print(f"  Recommendation  : {result['recommendation']}")
    print(f"  Model R²        : {result['r2']:.4f}")
    print(f"  Model MAE       : {fmt(result['mae'], currency)}")
    print(sep)
    print("  Reasons:")
    for reason in result.get("reasons", []):
        print(f"    • {reason}")
    print(sep)


# ── Chart saving ─────────────────────────────────────────────────────────────

def generate_all_plots(
    stock_df: pd.DataFrame,
    train_result: dict,
    individual_sentiments: list[dict],
    symbol: str,
    out_dir: str = "charts",
) -> dict[str, str | None]:
    """Save all analysis charts to *out_dir* and return a path dict.

    Args:
        stock_df:              OHLCV DataFrame.
        train_result:          Output from ``train_model()``.
        individual_sentiments: Per-article sentiment list.
        symbol:                Ticker symbol (used in filenames / titles).
        out_dir:               Directory where PNGs are written.

    Returns:
        Dict mapping chart name → file path (or ``None`` if skipped).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths: dict[str, str | None] = {}

    paths["price_trend"]        = _save_price_trend(stock_df, symbol, out_dir)
    paths["pred_vs_actual"]     = _save_pred_vs_actual(train_result, symbol, out_dir)
    paths["sentiment"]          = _save_sentiment(individual_sentiments, symbol, out_dir)
    paths["feature_importance"] = _save_feature_importance(train_result, symbol, out_dir)

    return paths


# ── Private chart helpers ────────────────────────────────────────────────────

def _save_price_trend(df: pd.DataFrame, symbol: str, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df.index, df["Close"],                label="Close",  linewidth=1.6)
    ax.plot(df.index, df["Close"].rolling(20).mean(), label="MA-20", linewidth=1.2, linestyle="--")
    ax.plot(df.index, df["Close"].rolling(50).mean(), label="MA-50", linewidth=1.2, linestyle=":")
    ax.set_title(f"{symbol} – Price History & Moving Averages")
    ax.set_xlabel("Date"); ax.set_ylabel("Price")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=25)
    plt.tight_layout()
    path = str(Path(out_dir) / f"{symbol}_price_trend.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_pred_vs_actual(train_result: dict, symbol: str, out_dir: str) -> str:
    y_test = train_result["y_test"]
    y_pred = train_result["y_pred"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, s=45)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_title(f"{symbol} – Predicted vs Actual (Test Set)")
    ax.set_xlabel("Actual Price"); ax.set_ylabel("Predicted Price")
    ax.legend()
    plt.tight_layout()
    path = str(Path(out_dir) / f"{symbol}_pred_vs_actual.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_sentiment(individual: list[dict], symbol: str, out_dir: str) -> str | None:
    if not individual:
        return None
    titles    = [r["title"][:35] + "…" if len(r["title"]) > 35 else r["title"] for r in individual]
    compounds = [r["compound"] for r in individual]
    colours   = ["#4CAF50" if c >= 0.05 else "#F44336" if c <= -0.05 else "#9E9E9E" for c in compounds]
    fig, ax   = plt.subplots(figsize=(11, max(5, len(titles) * 0.42)))
    ax.barh(range(len(titles)), compounds, color=colours)
    ax.set_yticks(range(len(titles)))
    ax.set_yticklabels(titles, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("VADER Compound Score")
    ax.set_title(f"{symbol} – News Sentiment Scores")
    plt.tight_layout()
    path = str(Path(out_dir) / f"{symbol}_sentiment.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_feature_importance(train_result: dict, symbol: str, out_dir: str) -> str:
    top = train_result["feature_importance"].head(10)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_title(f"{symbol} – Top Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = str(Path(out_dir) / f"{symbol}_feature_importance.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
