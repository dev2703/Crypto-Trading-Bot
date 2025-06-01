"""
Portfolio Optimization Strategy Module
- Computes optimal asset weights using Modern Portfolio Theory
- Rebalances portfolios based on target weights
- Includes risk management and performance evaluation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from strategy.performance_evaluation import compute_performance_metrics

def compute_optimal_weights(returns: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.Series:
    """Compute optimal asset weights using Modern Portfolio Theory."""
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    ones = np.ones(len(mean_returns))
    numerator = inv_cov_matrix.dot(mean_returns - risk_free_rate)
    denominator = ones.dot(inv_cov_matrix).dot(mean_returns - risk_free_rate)
    weights = numerator / denominator
    return pd.Series(weights, index=returns.columns)

def rebalance_portfolio(current_weights: pd.Series, target_weights: pd.Series, threshold: float = 0.1) -> bool:
    """Rebalance portfolio if current weights deviate from target weights beyond threshold."""
    deviation = np.abs(current_weights - target_weights)
    return (deviation > threshold).any()

def backtest_strategy(returns: pd.DataFrame, rebalance_frequency: int = 20, risk_free_rate: float = 0.0) -> Dict:
    """Backtest a portfolio optimization strategy."""
    portfolio_returns = pd.Series(0.0, index=returns.index)
    current_weights = None
    for i in range(0, len(returns), rebalance_frequency):
        if i + rebalance_frequency <= len(returns):
            window_returns = returns.iloc[i:i+rebalance_frequency]
            target_weights = compute_optimal_weights(window_returns, risk_free_rate)
            if current_weights is None or rebalance_portfolio(current_weights, target_weights):
                current_weights = target_weights
            portfolio_returns.iloc[i:i+rebalance_frequency] = window_returns.dot(current_weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    metrics = compute_performance_metrics(portfolio_returns)
    return {
        'portfolio_returns': portfolio_returns,
        'cumulative_returns': cumulative_returns,
        'metrics': metrics
    }

# Example usage
if __name__ == "__main__":
    # Example: Load sample data and backtest strategy
    returns = pd.DataFrame({
        'BTC': [0.05, 0.0476, 0.0455, 0.0435],
        'ETH': [0.04, 0.038, 0.036, 0.034]
    })
    result = backtest_strategy(returns)
    print(result) 