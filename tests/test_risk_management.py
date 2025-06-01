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
    returns = pd.Series([0.05, 0.0476, 0.0455, 0.0435])
    var = compute_var(returns)
    assert isinstance(var, float)

def test_compute_cvar():
    returns = pd.Series([0.05, 0.0476, 0.0455, 0.0435])
    cvar = compute_cvar(returns)
    assert isinstance(cvar, float)

def test_compute_drawdown():
    returns = pd.Series([0.05, 0.0476, 0.0455, 0.0435])
    drawdown = compute_drawdown(returns)
    assert len(drawdown) == 4
    assert drawdown.iloc[0] == 0.0
    assert drawdown.iloc[1] == 0.0
    assert drawdown.iloc[2] == 0.0
    assert drawdown.iloc[3] == 0.0

def test_position_sizing():
    returns = pd.Series([0.05, 0.0476, 0.0455, 0.0435])
    position_size = position_sizing(returns)
    assert isinstance(position_size, float)
    assert 0 <= position_size <= 1.0

def test_stop_loss():
    returns = pd.Series([0.05, 0.0476, 0.0455, 0.0435])
    stop_loss_level = stop_loss(returns)
    assert isinstance(stop_loss_level, float)

def test_backtest_strategy():
    returns = pd.Series([0.05, 0.0476, 0.0455, 0.0435])
    result = backtest_strategy(returns)
    assert 'portfolio_values' in result
    assert 'strategy_returns' in result
    assert 'position_size' in result
    assert 'stop_loss_level' in result
    assert 'metrics' in result
    assert len(result['portfolio_values']) == 5
    assert len(result['strategy_returns']) == 4
    assert isinstance(result['position_size'], float)
    assert isinstance(result['stop_loss_level'], float)
    assert isinstance(result['metrics'], dict) 