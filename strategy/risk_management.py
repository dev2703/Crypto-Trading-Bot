"""
Risk Management Strategy Module
- Computes risk metrics (VaR, CVaR, drawdown)
- Implements position sizing and stop-loss mechanisms
- Includes risk-adjusted performance evaluation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from strategy.performance_evaluation import compute_performance_metrics

def compute_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Compute Value at Risk (VaR)."""
    return np.percentile(returns, (1 - confidence_level) * 100)

def compute_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Compute Conditional Value at Risk (CVaR)."""
    var = compute_var(returns, confidence_level)
    return returns[returns <= var].mean()

def compute_drawdown(returns: pd.Series) -> pd.Series:
    """Compute drawdown series."""
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    return drawdown

def position_sizing(returns: pd.Series, risk_free_rate: float = 0.0, max_position_size: float = 1.0) -> float:
    """Compute position size based on risk metrics."""
    sharpe_ratio = compute_performance_metrics(returns)['sharpe_ratio']
    position_size = min(max_position_size, sharpe_ratio / 2)
    return position_size

def stop_loss(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Compute stop-loss level based on VaR."""
    var = compute_var(returns, confidence_level)
    return -var

def backtest_strategy(returns: pd.Series, initial_capital: float = 100000.0, risk_free_rate: float = 0.0, max_position_size: float = 1.0) -> Dict:
    """Backtest a risk management strategy."""
    position_size = position_sizing(returns, risk_free_rate, max_position_size)
    stop_loss_level = stop_loss(returns)
    portfolio_value = initial_capital
    portfolio_values = [initial_capital]
    for i in range(1, len(returns)):
        if returns.iloc[i] < stop_loss_level:
            portfolio_value = portfolio_value * (1 + stop_loss_level)
        else:
            portfolio_value = portfolio_value * (1 + returns.iloc[i] * position_size)
        portfolio_values.append(portfolio_value)
    portfolio_values = pd.Series(portfolio_values, index=returns.index)
    strategy_returns = compute_returns(portfolio_values)
    metrics = compute_performance_metrics(strategy_returns)
    return {
        'portfolio_values': portfolio_values,
        'strategy_returns': strategy_returns,
        'position_size': position_size,
        'stop_loss_level': stop_loss_level,
        'metrics': metrics
    }

# Example usage
if __name__ == "__main__":
    # Example: Load sample data and backtest strategy
    returns = pd.Series([0.05, 0.0476, 0.0455, 0.0435])
    result = backtest_strategy(returns)
    print(result) 