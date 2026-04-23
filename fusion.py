def compute_trend_signal(current_price: float, predicted_price: float) -> dict:
    pct_change = (predicted_price - current_price) / current_price
    
    if predicted_price > current_price:
        label = "UP"
        # Score from 0.5 to 1.0
        score = min(0.5 + (pct_change * 10), 1.0) 
    else:
        label = "DOWN"
        # Score from 0 to 0.5
        score = max(0.5 + (pct_change * 10), 0.0)
        
    return {
        "label": label,
        "score": score
    }

def compute_fusion_score(trend_score: float, sentiment_norm: float, volatility_score: float) -> float:
    # Base strategy:
    # High sentiment and High Trend pushes positive. 
    # High volatility gives some weight based on the instructions. 
    # Let's map high volatility to lower score for safety, meaning 1 - volatility.
    vol_adj = max(1.0 - volatility_score, 0.0)
    # 0.5 * Trend + 0.3 * Sent + 0.2 * Vol
    return (0.5 * trend_score) + (0.3 * sentiment_norm) + (0.2 * vol_adj)

def apply_decision_rules(fusion_score: float) -> str:
    if fusion_score >= 0.65:
        return "BUY"
    elif fusion_score >= 0.40:
        return "HOLD"
    else:
        return "SELL"

def compute_risk_level(volatility_label: str, fusion_score: float) -> str:
    if volatility_label == "High":
        return "High"
    elif volatility_label == "Medium":
        return "Medium"
    else:
        return "Low"

def build_reasons(trend_label: str, sentiment_label: str, volatility_label: str, current_price: float, predicted_price: float, recommendation: str) -> list:
    reasons = []
    
    if trend_label == "UP":
        reasons.append(f"Model predicts a price increase from {current_price:.2f} to {predicted_price:.2f}.")
    else:
        reasons.append(f"Model predicts a price drop from {current_price:.2f} to {predicted_price:.2f}.")
        
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
        reasons.append("Negative indicators outweigh positive ones, leading to a SELL recommendation.")
        
    return reasons
