"""
src/analysis/volatility.py
--------------------------
Annualised volatility signal derived from daily log returns.
"""

import numpy as np
import pandas as pd


def compute_volatility_signal(stock_df: pd.DataFrame) -> dict:
    """Compute annualised volatility from daily returns.

    Args:
        stock_df: OHLCV DataFrame with a ``"Close"`` column.

    Returns:
        Dictionary with:
          - ``"score"``  (float) – annualised volatility capped at 1.0
          - ``"label"``  (str)   – ``"High"`` / ``"Medium"`` / ``"Low"``
    """
    returns = stock_df["Close"].pct_change().dropna()
    vol     = float(returns.std() * np.sqrt(252))
    label   = "High" if vol > 0.40 else "Medium" if vol > 0.20 else "Low"
    return {"score": min(vol, 1.0), "label": label}
