import pytest
import pandas as pd
import numpy as np
from strategy.risk_management import (
    compute_var,
    compute_cvar,
    compute_drawdown,
    position_sizing,
    stop_loss,
    backtest_strategy
)

def sample_returns():
    # Create a small sample returns DataFrame
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=20, freq='D'),
        'returns': np.random.uniform(-0.01, 0.01, 20)
    }
    return pd.DataFrame(data)

def test_compute_var():
    df = sample_returns()
    var = compute_var(df['returns'])
    assert isinstance(var, float)
    assert var <= 0

def test_compute_cvar():
    df = sample_returns()
    cvar = compute_cvar(df['returns'])
    assert isinstance(cvar, float)
    assert cvar <= 0

def test_compute_drawdown():
    df = sample_returns()
    drawdown = compute_drawdown(df['returns'])
    assert len(drawdown) == len(df)
    assert drawdown.min() <= 0

def test_position_sizing():
    df = sample_returns()
    size = position_sizing(df['returns'])
    assert isinstance(size, float)
    assert size > 0

def test_stop_loss():
    price = 100.0
    var = -0.01
    stop = stop_loss(price, var)
    assert isinstance(stop, float)
    assert stop < price

def test_backtest_strategy():
    df = sample_returns()
    results = backtest_strategy(df['returns'])
    assert 'positions' in results
    assert 'returns' in results
    assert 'cumulative_returns' in results
    assert 'drawdown' in results
    assert 'var' in results
    assert 'cvar' in results
    assert 'max_drawdown' in results
    assert len(results['positions']) == len(df)
    assert len(results['returns']) == len(df)
    assert len(results['cumulative_returns']) == len(df)
    assert len(results['drawdown']) == len(df) 