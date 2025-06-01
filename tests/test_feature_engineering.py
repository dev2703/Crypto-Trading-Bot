import pytest
import pandas as pd
import numpy as np
from features.feature_engineering import (
    compute_returns,
    compute_volatility,
    compute_moving_average,
    compute_rsi,
    add_features
)

def sample_ohlcv():
    # Create a small sample OHLCV DataFrame
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=20, freq='D'),
        'open': np.linspace(100, 120, 20),
        'high': np.linspace(101, 121, 20),
        'low': np.linspace(99, 119, 20),
        'close': np.linspace(100, 120, 20),
        'volume': np.random.randint(1000, 2000, 20)
    }
    return pd.DataFrame(data)

def test_compute_returns():
    df = sample_ohlcv()
    returns = compute_returns(df)
    assert len(returns) == len(df)
    assert np.isnan(returns.iloc[0])  # First value should be NaN
    assert np.allclose(returns[1:], np.log(df['close'][1:].values) - np.log(df['close'][:-1].values))

def test_compute_volatility():
    df = sample_ohlcv()
    vol = compute_volatility(df, window=5)
    assert len(vol) == len(df)
    assert vol.isnull().sum() >= 4  # At least window-1 NaNs

def test_compute_moving_average():
    df = sample_ohlcv()
    ma = compute_moving_average(df, window=5)
    assert len(ma) == len(df)
    assert ma.isnull().sum() >= 4
    assert np.allclose(ma.dropna(), df['close'].rolling(window=5).mean().dropna())

def test_compute_rsi():
    df = sample_ohlcv()
    rsi = compute_rsi(df, window=5)
    assert len(rsi) == len(df)
    assert rsi.isnull().sum() >= 4
    assert (rsi.max() <= 100) and (rsi.min() >= 0)

def test_add_features():
    df = sample_ohlcv()
    df_feat = add_features(df)
    assert 'log_return' in df_feat.columns
    assert 'volatility_14' in df_feat.columns
    assert 'ma_14' in df_feat.columns
    assert 'rsi_14' in df_feat.columns
    # Check that feature columns have correct length
    for col in ['log_return', 'volatility_14', 'ma_14', 'rsi_14']:
        assert len(df_feat[col]) == len(df) 