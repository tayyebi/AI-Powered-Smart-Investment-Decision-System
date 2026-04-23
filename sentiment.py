from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyse_news(articles: list) -> dict:
    analyzer = SentimentIntensityAnalyzer()
    
    if not articles:
        return {"average_compound": 0.0, "label": "Neutral", "individual": []}
        
    results = []
    total_compound = 0.0
    
    for art in articles:
        text = str(art.get("title", "")) + ". " + str(art.get("content", ""))
        scores = analyzer.polarity_scores(text)
        comp = scores["compound"]
        
        if comp >= 0.05:
            label = "Positive"
        elif comp <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
            
        total_compound += comp
        results.append({
            "title": art.get("title", ""),
            "compound": comp,
            "label": label
        })
        
    avg_compound = total_compound / len(articles)
    
    if avg_compound >= 0.05:
        overall_label = "Positive"
    elif avg_compound <= -0.05:
        overall_label = "Negative"
    else:
        overall_label = "Neutral"
        
    return {
        "average_compound": avg_compound,
        "label": overall_label,
        "individual": results
    }

def normalise_sentiment(compound: float) -> float:
    # Scale from [-1, 1] to [0, 1]
    return (compound + 1) / 2
