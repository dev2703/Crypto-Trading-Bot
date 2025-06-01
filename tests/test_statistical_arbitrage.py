import pytest
import pandas as pd
import numpy as np
from strategy.statistical_arbitrage import (
    compute_spread,
    compute_zscore,
    generate_signals,
    position_sizing,
    backtest_strategy
)

def sample_prices():
    # Create a small sample price DataFrame
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=20, freq='D'),
        'asset1_close': np.linspace(100, 120, 20),
        'asset2_close': np.linspace(90, 110, 20)
    }
    return pd.DataFrame(data)

def test_compute_spread():
    df = sample_prices()
    spread = compute_spread(df['asset1_close'], df['asset2_close'])
    assert len(spread) == len(df)
    assert np.allclose(spread, np.log(df['asset1_close']) - np.log(df['asset2_close']))

def test_compute_zscore():
    df = sample_prices()
    spread = compute_spread(df['asset1_close'], df['asset2_close'])
    zscore = compute_zscore(spread, window=5)
    assert len(zscore) == len(df)
    assert zscore.isnull().sum() >= 4  # At least window-1 NaNs

def test_generate_signals():
    df = sample_prices()
    spread = compute_spread(df['asset1_close'], df['asset2_close'])
    zscore = compute_zscore(spread, window=5)
    signals = generate_signals(spread, zscore, threshold=1.0)
    assert len(signals) == len(df)
    assert signals.isin([-1, 0, 1]).all()

def test_position_sizing():
    df = sample_prices()
    spread = compute_spread(df['asset1_close'], df['asset2_close'])
    zscore = compute_zscore(spread, window=5)
    signals = generate_signals(spread, zscore, threshold=1.0)
    positions = position_sizing(signals, capital=10000.0, risk_per_trade=0.02)
    assert len(positions) == len(df)
    assert (positions == signals * 10000.0 * 0.02).all()

def test_backtest_strategy():
    df = sample_prices()
    results = backtest_strategy(df['asset1_close'], df['asset2_close'])
    assert 'signals' in results
    assert 'positions' in results
    assert 'returns' in results
    assert 'cumulative_returns' in results
    assert len(results['signals']) == len(df)
    assert len(results['positions']) == len(df)
    assert len(results['returns']) == len(df)
    assert len(results['cumulative_returns']) == len(df) 