"""
src/analysis/sentiment.py
--------------------------
VADER-based sentiment scoring for a list of news articles.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def analyse_news(articles: list[dict]) -> dict:
    """Score a list of news articles with VADER.

    Args:
        articles: List of ``{"title": str, "content": str}`` dicts.

    Returns:
        Dictionary with:
          - ``"average_compound"`` (float) – mean compound score across articles
          - ``"label"``            (str)   – ``"Positive"`` / ``"Negative"`` / ``"Neutral"``
          - ``"individual"``       (list)  – per-article scores and labels
    """
    if not articles:
        return {"average_compound": 0.0, "label": "Neutral", "individual": []}

    individual = []
    total_compound = 0.0

    for art in articles:
        text  = f"{art.get('title', '')}. {art.get('content', '')}"
        score = _analyzer.polarity_scores(text)["compound"]
        label = _compound_label(score)
        total_compound += score
        individual.append({"title": art.get("title", ""), "compound": score, "label": label})

    avg_compound = total_compound / len(articles)
    return {
        "average_compound": avg_compound,
        "label": _compound_label(avg_compound),
        "individual": individual,
    }


def normalise_sentiment(compound: float) -> float:
    """Map a VADER compound score from [-1, 1] to [0, 1]."""
    return (compound + 1) / 2


def _compound_label(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"
