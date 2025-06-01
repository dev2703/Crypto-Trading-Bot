"""
Performance Evaluation Strategy Module
- Computes performance metrics (returns, Sharpe ratio, drawdown)
- Visualizes performance results
- Compares different strategies
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from strategy.backtesting import compute_returns, compute_sharpe_ratio, compute_drawdown

def compute_performance_metrics(returns: pd.Series) -> Dict:
    """Compute performance metrics from returns."""
    sharpe_ratio = compute_sharpe_ratio(returns)
    drawdown = compute_drawdown(returns)
    max_drawdown = drawdown.min()
    cumulative_returns = (1 + returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_volatility = returns.std() * np.sqrt(252)
    return {
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility
    }

def plot_performance(returns: pd.Series, title: str = 'Strategy Performance'):
    """Plot performance results."""
    cumulative_returns = (1 + returns).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns.index, cumulative_returns.values)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.show()

def compare_strategies(strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compare different strategies."""
    metrics = {}
    for strategy_name, returns in strategy_returns.items():
        metrics[strategy_name] = compute_performance_metrics(returns)
    return pd.DataFrame(metrics).T

# Example usage
if __name__ == "__main__":
    # Example: Load sample data and evaluate performance
    returns = pd.Series([0.05, 0.0476, 0.0455, 0.0435])
    metrics = compute_performance_metrics(returns)
    print(metrics)
    plot_performance(returns)
    strategy_returns = {
        'Strategy 1': returns,
        'Strategy 2': returns * 1.1
    }
    comparison = compare_strategies(strategy_returns)
    print(comparison) 