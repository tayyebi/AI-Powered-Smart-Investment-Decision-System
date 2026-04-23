"""
src/analysis/features.py
------------------------
Technical-indicator feature engineering for price-prediction models.

All indicator helpers produce scale-invariant outputs so the model can
generalise across different price levels.
"""

import numpy as np
import pandas as pd


# ── Indicator helpers ────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index scaled to [0, 1]."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))) / 100


def _macd_hist_norm(series: pd.Series) -> pd.Series:
    """MACD histogram normalised by price."""
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return (macd - signal) / (series + 1e-9)


def _bb_pct(series: pd.Series, window: int = 20) -> pd.Series:
    """Bollinger Band %B: position of price within the band."""
    mid   = series.rolling(window).mean()
    std   = series.rolling(window).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return (series - lower) / (upper - lower + 1e-9)


# ── Feature matrix construction ─────────────────────────────────────────────

#: Ordered list of feature column names expected by the model.
FEATURE_COLS: list[str] = [
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5",
    "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    "EMA_9", "EMA_21",
    "RSI_14", "MACD_Hist_N", "BB_Pct",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Augment *df* with technical indicators and a ``Target`` column.

    Args:
        df: Raw OHLCV DataFrame (must contain a ``"Close"`` column).

    Returns:
        New DataFrame with indicator columns and ``Target`` (next-day close),
        with rows containing NaN values dropped.
    """
    df = df.copy()
    cl = df["Close"]

    # Price lags
    for lag in [1, 2, 3, 5]:
        df[f"Close_Lag_{lag}"] = cl.shift(lag)

    # Simple moving averages
    for w in [5, 10, 20, 50]:
        df[f"SMA_{w}"] = cl.rolling(w).mean()

    # Exponential moving averages
    for span in [9, 21]:
        df[f"EMA_{span}"] = cl.ewm(span=span, adjust=False).mean()

    # Oscillators
    df["RSI_14"]      = _rsi(cl, 14)
    df["MACD_Hist_N"] = _macd_hist_norm(cl)
    df["BB_Pct"]      = _bb_pct(cl, 20)

    # Target: next-day absolute close price
    df["Target"] = cl.shift(-1)

    df.dropna(inplace=True)
    return df
