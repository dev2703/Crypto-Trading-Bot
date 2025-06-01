import pytest
import pandas as pd
import numpy as np
from strategy.backtesting import compute_returns, compute_sharpe_ratio, compute_drawdown, backtest_strategy

def test_compute_returns():
    prices = pd.Series([100, 105, 110, 115, 120])
    returns = compute_returns(prices)
    assert len(returns) == 4
    assert returns.iloc[0] == 0.05
    assert returns.iloc[1] == 0.0476
    assert returns.iloc[2] == 0.0455
    assert returns.iloc[3] == 0.0435

def test_compute_sharpe_ratio():
    returns = pd.Series([0.05, 0.0476, 0.0455, 0.0435])
    sharpe_ratio = compute_sharpe_ratio(returns)
    assert isinstance(sharpe_ratio, float)

def test_compute_drawdown():
    returns = pd.Series([0.05, 0.0476, 0.0455, 0.0435])
    drawdown = compute_drawdown(returns)
    assert len(drawdown) == 4
    assert drawdown.iloc[0] == 0.0
    assert drawdown.iloc[1] == 0.0
    assert drawdown.iloc[2] == 0.0
    assert drawdown.iloc[3] == 0.0

def test_backtest_strategy():
    prices = pd.Series([100, 105, 110, 115, 120])
    signals = pd.Series([0, 1, 0, -1, 0])
    result = backtest_strategy(prices, signals)
    assert 'portfolio_values' in result
    assert 'strategy_returns' in result
    assert 'sharpe_ratio' in result
    assert 'drawdown' in result
    assert len(result['portfolio_values']) == 5
    assert len(result['strategy_returns']) == 4
    assert isinstance(result['sharpe_ratio'], float)
    assert len(result['drawdown']) == 4 