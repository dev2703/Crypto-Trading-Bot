"""
Portfolio Optimization Strategy Module
- Computes optimal asset weights using Modern Portfolio Theory
- Rebalances portfolios based on target weights
- Includes risk management and performance evaluation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import minimize

def compute_optimal_weights(returns: pd.DataFrame, risk_free_rate: float = 0.0) -> np.ndarray:
    """Compute optimal asset weights based on historical returns."""
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(returns.columns)

    def neg_sharpe_ratio(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1 / num_assets] * num_assets)
    result = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def rebalance_portfolio(current_weights: np.ndarray, target_weights: np.ndarray, threshold: float = 0.1) -> bool:
    """Rebalance the portfolio if current weights deviate from target weights beyond a threshold."""
    deviation = np.abs(current_weights - target_weights)
    return np.any(deviation > threshold)

def backtest_strategy(returns: pd.DataFrame, rebalance_frequency: int = 20) -> Dict:
    """Backtest a portfolio optimization strategy."""
    portfolio_returns = pd.Series(0.0, index=returns.index)
    cumulative_returns = pd.Series(1.0, index=returns.index)
    for i in range(0, len(returns), rebalance_frequency):
        window_returns = returns.iloc[i:i + rebalance_frequency]
        if len(window_returns) < 2:
            continue
        weights = compute_optimal_weights(window_returns)
        portfolio_returns.iloc[i:i + rebalance_frequency] = (window_returns * weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return {
        'portfolio_returns': portfolio_returns,
        'cumulative_returns': cumulative_returns
    }

# Example usage
if __name__ == "__main__":
    # Example: Load sample data and backtest strategy
    returns = pd.DataFrame({
        'asset1': [0.01, -0.02, 0.03, 0.01, -0.01],
        'asset2': [0.02, -0.01, 0.02, 0.01, -0.02],
        'asset3': [0.01, 0.01, -0.01, 0.02, 0.01]
    })
    result = backtest_strategy(returns)
    print(result) 