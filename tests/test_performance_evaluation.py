import pytest
import pandas as pd
import numpy as np
from strategy.performance_evaluation import compute_performance_metrics, plot_performance, compare_strategies

def test_compute_performance_metrics():
    returns = pd.Series([0.05, 0.0476, 0.0455, 0.0435])
    metrics = compute_performance_metrics(returns)
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'total_return' in metrics
    assert 'annualized_return' in metrics
    assert 'annualized_volatility' in metrics
    assert isinstance(metrics['sharpe_ratio'], float)
    assert isinstance(metrics['max_drawdown'], float)
    assert isinstance(metrics['total_return'], float)
    assert isinstance(metrics['annualized_return'], float)
    assert isinstance(metrics['annualized_volatility'], float)

def test_plot_performance():
    returns = pd.Series([0.05, 0.0476, 0.0455, 0.0435])
    plot_performance(returns)  # This will display a plot, but we can't assert on it

def test_compare_strategies():
    returns = pd.Series([0.05, 0.0476, 0.0455, 0.0435])
    strategy_returns = {
        'Strategy 1': returns,
        'Strategy 2': returns * 1.1
    }
    comparison = compare_strategies(strategy_returns)
    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) == 2
    assert 'Strategy 1' in comparison.index
    assert 'Strategy 2' in comparison.index
    assert 'sharpe_ratio' in comparison.columns
    assert 'max_drawdown' in comparison.columns
    assert 'total_return' in comparison.columns
    assert 'annualized_return' in comparison.columns
    assert 'annualized_volatility' in comparison.columns 