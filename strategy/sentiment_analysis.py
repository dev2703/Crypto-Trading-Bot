"""
Sentiment Analysis Strategy Module
- Collects and processes sentiment data
- Computes sentiment scores
- Generates trading signals
- Includes backtesting and performance evaluation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from strategy.performance_evaluation import compute_performance_metrics

def compute_sentiment_score(texts: List[str]) -> List[float]:
    """Compute sentiment scores from a list of texts."""
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for text in texts:
        sentiment = analyzer.polarity_scores(text)
        scores.append(sentiment['compound'])
    return scores

def generate_signals(sentiment_scores: pd.Series, threshold: float = 0.2) -> pd.Series:
    """Generate trading signals based on sentiment scores."""
    signals = pd.Series(0, index=sentiment_scores.index)
    signals[sentiment_scores > threshold] = 1
    signals[sentiment_scores < -threshold] = -1
    return signals

def backtest_strategy(sentiment_scores: pd.Series, prices: pd.Series, threshold: float = 0.2) -> Dict:
    """Backtest a sentiment analysis strategy."""
    signals = generate_signals(sentiment_scores, threshold)
    returns = prices.pct_change().dropna()
    strategy_returns = signals.shift(1) * returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    metrics = compute_performance_metrics(strategy_returns)
    return {
        'signals': signals,
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'metrics': metrics
    }

# Example usage
if __name__ == "__main__":
    # Example: Load sample data and backtest strategy
    texts = [
        "Bitcoin is showing strong bullish momentum.",
        "Ethereum's price is expected to rise.",
        "Market sentiment is positive for crypto.",
        "Investors are optimistic about the future."
    ]
    sentiment_scores = pd.Series(compute_sentiment_score(texts))
    prices = pd.Series([100, 105, 110, 115, 120])
    result = backtest_strategy(sentiment_scores, prices)
    print(result) 