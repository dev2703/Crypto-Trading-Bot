import pytest
import pandas as pd
import numpy as np
from strategy.sentiment_analysis import compute_sentiment_score, generate_signals, backtest_strategy

def sample_sentiment():
    # Create a small sample sentiment DataFrame
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=20, freq='D'),
        'text': [
            'Bitcoin is soaring today!',
            'Ethereum looks bearish.',
            'Market is stable.',
            'Crypto is the future.',
            'Prices are dropping.',
            'Great news for investors.',
            'Uncertain market conditions.',
            'Strong buy signal.',
            'Sell everything!',
            'Hold your positions.',
            'Positive outlook.',
            'Negative sentiment.',
            'Neutral market.',
            'Bullish trend.',
            'Bearish trend.',
            'Mixed signals.',
            'Upward momentum.',
            'Downward pressure.',
            'Sideways movement.',
            'Breakout expected.'
        ],
        'close': np.linspace(100, 120, 20)
    }
    return pd.DataFrame(data)

def test_compute_sentiment_score():
    texts = [
        "Bitcoin is showing strong bullish momentum.",
        "Ethereum's price is expected to rise.",
        "Market sentiment is positive for crypto.",
        "Investors are optimistic about the future."
    ]
    scores = compute_sentiment_score(texts)
    assert len(scores) == 4
    assert all(isinstance(score, float) for score in scores)
    assert all(-1 <= score <= 1 for score in scores)

def test_generate_signals():
    sentiment_scores = pd.Series([0.5, 0.3, -0.2, -0.4])
    signals = generate_signals(sentiment_scores, threshold=0.2)
    assert len(signals) == 4
    assert signals.iloc[0] == 1
    assert signals.iloc[1] == 1
    assert signals.iloc[2] == 0
    assert signals.iloc[3] == -1

def test_backtest_strategy():
    sentiment_scores = pd.Series([0.5, 0.3, -0.2, -0.4])
    prices = pd.Series([100, 105, 110, 115, 120])
    result = backtest_strategy(sentiment_scores, prices)
    assert 'signals' in result
    assert 'strategy_returns' in result
    assert 'cumulative_returns' in result
    assert 'metrics' in result
    assert len(result['signals']) == 4
    assert len(result['strategy_returns']) == 4
    assert len(result['cumulative_returns']) == 4
    assert isinstance(result['metrics'], dict) 