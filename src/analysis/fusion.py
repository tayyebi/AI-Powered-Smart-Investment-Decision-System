"""
src/analysis/fusion.py
-----------------------
Multi-signal fusion: combines trend, sentiment, and volatility scores
to produce a single fusion score and an actionable recommendation.

Fusion formula:
    fusion = 0.5 × trend_score + 0.3 × sentiment_norm + 0.2 × (1 − volatility_score)

Decision thresholds:
    fusion ≥ 0.65  →  BUY
    fusion ≥ 0.40  →  HOLD
    fusion  < 0.40  →  SELL
"""


def compute_trend_signal(current_price: float, predicted_price: float) -> dict:
    """Translate a price prediction into a normalised trend score.

    Args:
        current_price:   Today's market price.
        predicted_price: Model's predicted next-day price.

    Returns:
        Dict with ``"label"`` (``"UP"`` or ``"DOWN"``) and
        ``"score"`` in [0, 1].
    """
    pct_change = (predicted_price - current_price) / current_price
    if predicted_price > current_price:
        score = min(0.5 + pct_change * 10, 1.0)
        label = "UP"
    else:
        score = max(0.5 + pct_change * 10, 0.0)
        label = "DOWN"
    return {"label": label, "score": score}


def compute_fusion_score(
    trend_score: float,
    sentiment_norm: float,
    volatility_score: float,
) -> float:
    """Compute the weighted fusion score.

    High volatility lowers the score (risk-adjusted weighting).

    Args:
        trend_score:      Trend signal in [0, 1].
        sentiment_norm:   Sentiment signal in [0, 1].
        volatility_score: Annualised volatility capped at 1.0.

    Returns:
        Fusion score in [0, 1].
    """
    vol_adj = max(1.0 - volatility_score, 0.0)
    return 0.5 * trend_score + 0.3 * sentiment_norm + 0.2 * vol_adj


def apply_decision_rules(fusion_score: float) -> str:
    """Map a fusion score to a trading recommendation.

    Returns:
        ``"BUY"``, ``"HOLD"``, or ``"SELL"``.
    """
    if fusion_score >= 0.65:
        return "BUY"
    if fusion_score >= 0.40:
        return "HOLD"
    return "SELL"


def compute_risk_level(volatility_label: str, fusion_score: float) -> str:
    """Derive a qualitative risk level primarily from volatility.

    Args:
        volatility_label: ``"High"``, ``"Medium"``, or ``"Low"``.
        fusion_score:     Currently unused; retained in the signature for
                          future composite risk models that factor in the
                          fusion score alongside volatility.

    Returns:
        ``"High"``, ``"Medium"``, or ``"Low"``.
    """
    _ = fusion_score  # reserved for future composite risk logic
    return volatility_label


def build_reasons(
    trend_label: str,
    sentiment_label: str,
    volatility_label: str,
    current_price: float,
    predicted_price: float,
    recommendation: str,
) -> list[str]:
    """Build a human-readable explanation of the recommendation.

    Returns:
        List of plain-English reason strings.
    """
    reasons: list[str] = []

    if trend_label == "UP":
        reasons.append(
            f"Model predicts a price increase from {current_price:.2f} to {predicted_price:.2f}."
        )
    else:
        reasons.append(
            f"Model predicts a price drop from {current_price:.2f} to {predicted_price:.2f}."
        )

    if sentiment_label == "Positive":
        reasons.append("Recent news sentiment is broadly positive, which may drive upward momentum.")
    elif sentiment_label == "Negative":
        reasons.append("Recent news sentiment is negative, indicating potential headwinds.")
    else:
        reasons.append("News sentiment is neutral, showing no strong emotional bias in media.")

    if volatility_label == "High":
        reasons.append("High volatility indicates significant price fluctuations and higher risk.")
    elif volatility_label == "Medium":
        reasons.append("Moderate volatility suggests normal market fluctuations.")
    else:
        reasons.append("Low volatility implies a stable asset.")

    if recommendation == "BUY":
        reasons.append("Overall metrics align favourably, leading to a BUY recommendation.")
    elif recommendation == "HOLD":
        reasons.append("Mixed signals suggest waiting; a HOLD strategy is recommended.")
    else:
        reasons.append("Negative indicators outweigh positives, leading to a SELL recommendation.")

    return reasons
