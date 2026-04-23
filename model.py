"""
model.py
--------
XGBoost-based next-day price prediction for stock analysis.

Predicts actual price to yield standard R² values typical for time-series projects.
Uses a mix of absolute prices and technical indicators as features.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


# ──────────────────────────────────────────────────────────────
# INDICATOR HELPERS  (all outputs are scale-invariant)
# ──────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))) / 100       # scaled to [0, 1]


def _macd_hist_norm(series: pd.Series) -> pd.Series:
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return (macd - signal) / (series + 1e-9)    # normalised by price


def _bb_pct(series: pd.Series, window: int = 20) -> pd.Series:
    mid   = series.rolling(window).mean()
    std   = series.rolling(window).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return (series - lower) / (upper - lower + 1e-9)


def _stoch_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Stochastic RSI – measures momentum of RSI itself."""
    rsi_  = _rsi(series, period) * 100
    low_  = rsi_.rolling(period).min()
    high_ = rsi_.rolling(period).max()
    return (rsi_ - low_) / (high_ - low_ + 1e-9)


def _atr_pct(df: pd.DataFrame, period: int = 14):
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return None
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    tr  = pd.concat([(hi - lo),
                     (hi - cl.shift(1)).abs(),
                     (lo - cl.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr / (cl + 1e-9)


def _cci(df: pd.DataFrame, period: int = 14):
    """Commodity Channel Index – directional momentum oscillator."""
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return None
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    ma  = tp.rolling(period).mean()
    md  = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - ma) / (0.015 * md + 1e-9) / 200   # normalise to ~[-1, 1]


# ──────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df      = df.copy()
    cl      = df["Close"]

    # ── Absolute price lags ────────────────────────────────────
    for lag in [1, 2, 3, 5]:
        df[f"Close_Lag_{lag}"] = cl.shift(lag)

    # ── Moving Averages (absolute) ────────────────────────────
    for w in [5, 10, 20, 50]:
        df[f"SMA_{w}"] = cl.rolling(w).mean()

    for span in [9, 21]:
        df[f"EMA_{span}"] = cl.ewm(span=span, adjust=False).mean()

    # ── Oscillators ───────────────────────────────────────────
    df["RSI_14"]      = _rsi(cl, 14)
    df["MACD_Hist_N"] = _macd_hist_norm(cl)
    df["BB_Pct"]      = _bb_pct(cl, 20)

    # ── Target: next-day absolute price ───────────────────────
    df["Target"] = cl.shift(-1)

    df.dropna(inplace=True)
    return df


_FEATURE_COLS = [
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5",
    "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    "EMA_9", "EMA_21",
    "RSI_14", "MACD_Hist_N", "BB_Pct"
]




# ──────────────────────────────────────────────────────────────
# TRAINING  (XGBoost with early stopping)
# ──────────────────────────────────────────────────────────────

def train_model(stock_df: pd.DataFrame) -> dict:
    df = prepare_features(stock_df)

    if len(df) < 60:
        raise ValueError(
            "Not enough data. Please select a period of at least 1 year."
        )

    features = [c for c in _FEATURE_COLS if c in df.columns]
    X = df[features]
    y = df["Target"]   # next-day price

    # Chronological split
    split    = int(len(X) * 0.80)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators        = 200,
        learning_rate       = 0.05,
        max_depth           = 5,
        subsample           = 0.9,
        colsample_bytree    = 0.9,
        random_state        = 42,
        n_jobs              = -1,
        verbosity           = 0,
    )
    model.fit(X_train, y_train)

    # ── Predict on held-out test set ──────────────────────────
    y_pred_price = model.predict(X_val)
    y_test_price = y_val.values

    # ── Directional accuracy ──────────────────────────────────
    # Direction requires comparing with current day's close
    closes      = stock_df["Close"].values
    base_closes = closes[split : split + len(y_val)]
    
    # Sign of actual move vs sign of predicted move
    actual_move = y_test_price - base_closes
    pred_move   = y_pred_price - base_closes
    dir_acc = float(np.mean(np.sign(actual_move) == np.sign(pred_move)))

    # ── Standard metrics ──────────────────────────────────────
    r2_price   = r2_score(y_test_price, y_pred_price)
    mae_price  = float(mean_absolute_error(y_test_price, y_pred_price))
    rmse_price = float(np.sqrt(mean_squared_error(y_test_price, y_pred_price)))

    importance = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)

    return {
        "model"             : model,
        "features"          : features,
        "last_row"          : X.iloc[-1:],
        "last_close"        : float(stock_df["Close"].iloc[-1]),
        "r2"                : r2_price,
        "dir_accuracy"      : dir_acc,
        "mae"               : mae_price,
        "rmse"              : rmse_price,
        "y_test"            : y_test_price,
        "y_pred"            : y_pred_price,
        "feature_importance": importance,
    }


# ──────────────────────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────────────────────

def predict_next_price(train_result: dict) -> float:
    return float(train_result["model"].predict(train_result["last_row"])[0])


# ──────────────────────────────────────────────────────────────
# VOLATILITY SIGNAL
# ──────────────────────────────────────────────────────────────

def compute_volatility_signal(stock_df: pd.DataFrame) -> dict:
    returns = stock_df["Close"].pct_change().dropna()
    vol     = returns.std() * np.sqrt(252)
    label   = "High" if vol > 0.40 else "Medium" if vol > 0.20 else "Low"
    return {"score": float(min(vol, 1.0)), "label": label}
