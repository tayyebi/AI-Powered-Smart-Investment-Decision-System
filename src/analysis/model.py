"""
src/analysis/model.py
---------------------
XGBoost next-day price prediction.

Trains an XGBRegressor on a chronological 80/20 split and reports
standard regression metrics plus directional accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from .features import FEATURE_COLS, prepare_features


def train_model(stock_df: pd.DataFrame) -> dict:
    """Train an XGBoost regressor to predict next-day closing price.

    Args:
        stock_df: Raw OHLCV DataFrame returned by the data layer.

    Returns:
        Dictionary containing:
          - ``model``              – fitted XGBRegressor
          - ``features``           – list of feature names used
          - ``last_row``           – feature row for the most recent day
          - ``last_close``         – most recent closing price (float)
          - ``r2``                 – R² on the held-out test set
          - ``dir_accuracy``       – directional accuracy on test set
          - ``mae``                – Mean Absolute Error
          - ``rmse``               – Root Mean Squared Error
          - ``y_test``             – actual prices on test set (ndarray)
          - ``y_pred``             – predicted prices on test set (ndarray)
          - ``feature_importance`` – pd.Series sorted by importance

    Raises:
        ValueError: When there are fewer than 60 training rows after
                    feature engineering.
    """
    df = prepare_features(stock_df)

    if len(df) < 60:
        raise ValueError(
            "Not enough data – please select a period of at least 1 year."
        )

    features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[features]
    y = df["Target"]

    split = int(len(X) * 0.80)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators     = 200,
        learning_rate    = 0.05,
        max_depth        = 5,
        subsample        = 0.9,
        colsample_bytree = 0.9,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
    )
    model.fit(X_train, y_train)

    y_pred_price = model.predict(X_val)
    y_test_price = y_val.values

    # Directional accuracy: compare sign of predicted move vs actual move
    closes      = stock_df["Close"].values
    base_closes = closes[split : split + len(y_val)]
    dir_acc = float(
        np.mean(np.sign(y_test_price - base_closes) == np.sign(y_pred_price - base_closes))
    )

    importance = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)

    return {
        "model"             : model,
        "features"          : features,
        "last_row"          : X.iloc[-1:],
        "last_close"        : float(stock_df["Close"].iloc[-1]),
        "r2"                : float(r2_score(y_test_price, y_pred_price)),
        "dir_accuracy"      : dir_acc,
        "mae"               : float(mean_absolute_error(y_test_price, y_pred_price)),
        "rmse"              : float(np.sqrt(mean_squared_error(y_test_price, y_pred_price))),
        "y_test"            : y_test_price,
        "y_pred"            : y_pred_price,
        "feature_importance": importance,
    }


def predict_next_price(train_result: dict) -> float:
    """Return the predicted next-day closing price.

    Args:
        train_result: Output dict from :func:`train_model`.
    """
    return float(train_result["model"].predict(train_result["last_row"])[0])
