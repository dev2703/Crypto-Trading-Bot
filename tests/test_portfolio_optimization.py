import pytest
import pandas as pd
import numpy as np
from strategy.portfolio_optimization import compute_optimal_weights, rebalance_portfolio, backtest_strategy

def sample_returns():
    # Create a small sample returns DataFrame
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=20, freq='D'),
        'asset1': np.random.uniform(-0.01, 0.01, 20),
        'asset2': np.random.uniform(-0.01, 0.01, 20),
        'asset3': np.random.uniform(-0.01, 0.01, 20),
        'asset4': np.random.uniform(-0.01, 0.01, 20)
    }
    return pd.DataFrame(data)

def test_compute_optimal_weights():
    returns = pd.DataFrame({
        'BTC': [0.05, 0.0476, 0.0455, 0.0435],
        'ETH': [0.04, 0.038, 0.036, 0.034]
    })
    weights = compute_optimal_weights(returns)
    assert isinstance(weights, pd.Series)
    assert len(weights) == 2
    assert 'BTC' in weights.index
    assert 'ETH' in weights.index
    assert np.isclose(weights.sum(), 1.0)

def test_rebalance_portfolio():
    current_weights = pd.Series({'BTC': 0.5, 'ETH': 0.5})
    target_weights = pd.Series({'BTC': 0.6, 'ETH': 0.4})
    assert rebalance_portfolio(current_weights, target_weights, threshold=0.1)
    assert not rebalance_portfolio(current_weights, target_weights, threshold=0.2)

def test_backtest_strategy():
    returns = pd.DataFrame({
        'BTC': [0.05, 0.0476, 0.0455, 0.0435],
        'ETH': [0.04, 0.038, 0.036, 0.034]
    })
    result = backtest_strategy(returns)
    assert 'portfolio_returns' in result
    assert 'cumulative_returns' in result
    assert 'metrics' in result
    assert len(result['portfolio_returns']) == 4
    assert len(result['cumulative_returns']) == 4
    assert isinstance(result['metrics'], dict) 