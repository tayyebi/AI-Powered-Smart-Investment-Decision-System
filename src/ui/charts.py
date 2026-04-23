"""
src/ui/charts.py
----------------
Dark-themed Matplotlib chart helpers for the Streamlit UI.

Each function returns a Matplotlib Figure that can be converted to bytes
with :func:`fig_to_bytes` and passed to ``st.image()``.
"""

from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ── Theme constants ──────────────────────────────────────────────────────────

DARK_BG   = "#0d1117"
DARK_GRID = "#1e2a38"
BLUE      = "#4F8EF7"
ORANGE    = "#F7A24F"
PURPLE    = "#A24FF7"


def dark_fig(w: float = 11, h: float = 5):
    """Return a (fig, ax) tuple styled with the dark theme."""
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


def fig_to_bytes(fig) -> bytes:
    """Serialise a Matplotlib figure to a PNG byte string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="none", transparent=True)
    buf.seek(0)
    return buf.read()


# ── Individual chart functions ───────────────────────────────────────────────

def chart_price_trend(df: pd.DataFrame, symbol: str):
    """Price history with MA-20 and MA-50 overlays."""
    fig, ax = dark_fig(11, 4.5)
    ax.plot(df.index, df["Close"],
            label="Close",  linewidth=1.6, color=BLUE)
    ax.plot(df.index, df["Close"].rolling(20).mean(),
            label="MA-20",  linewidth=1.2, linestyle="--", color=ORANGE)
    ax.plot(df.index, df["Close"].rolling(50).mean(),
            label="MA-50",  linewidth=1.2, linestyle=":", color=PURPLE)
    ax.set_title(f"{symbol} – Price History & Moving Averages",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(facecolor=DARK_BG, labelcolor="#ccc")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=25)
    plt.tight_layout()
    return fig


def chart_pred_vs_actual(y_test, y_pred, symbol: str):
    """Scatter plot of predicted vs actual prices on the test set."""
    fig, ax = dark_fig(6, 6)
    ax.scatter(y_test, y_pred, alpha=0.6, color=BLUE,
               edgecolors="white", linewidths=0.3, s=45)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_title(f"{symbol} – Predicted vs Actual (Test Set)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.legend(facecolor=DARK_BG, labelcolor="#ccc")
    plt.tight_layout()
    return fig


def chart_sentiment(individual: list[dict], symbol: str):
    """Horizontal bar chart of per-article VADER compound scores."""
    if not individual:
        return None
    titles    = [r["title"][:35] + "…" if len(r["title"]) > 35 else r["title"]
                 for r in individual]
    compounds = [r["compound"] for r in individual]
    colours   = ["#4CAF50" if c >= 0.05 else "#F44336" if c <= -0.05 else "#9E9E9E"
                 for c in compounds]
    fig, ax = dark_fig(11, max(5, len(titles) * 0.42))
    ax.barh(range(len(titles)), compounds, color=colours, edgecolor=DARK_BG)
    ax.set_yticks(range(len(titles)))
    ax.set_yticklabels(titles, fontsize=8, color="#ccc")
    ax.axvline(0,     color="#aaa",    linewidth=0.8)
    ax.axvline(0.05,  color="#4CAF50", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(-0.05, color="#F44336", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("VADER Compound Score")
    ax.set_title(f"{symbol} – News Sentiment Scores", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_feature_importance(importance: pd.Series, symbol: str):
    """Horizontal bar chart of top-10 XGBoost feature importances."""
    top = importance.head(10)
    fig, ax = dark_fig(9, 4)
    ax.barh(top.index[::-1], top.values[::-1], color=BLUE, edgecolor=DARK_BG)
    ax.set_title(f"{symbol} – Top Feature Importances",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig
