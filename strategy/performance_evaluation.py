"""
Performance Evaluation Strategy Module
- Computes performance metrics
- Generates performance reports
- Includes visualization tools
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

def compute_performance_metrics(returns: pd.Series) -> Dict:
    """Compute performance metrics for a series of returns."""
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def generate_performance_report(metrics: Dict) -> str:
    """Generate a performance report."""
    report = f"""
    Performance Report:
    ----------------
    Total Return: {metrics['total_return']:.4f}
    Annualized Return: {metrics['annualized_return']:.4f}
    Volatility: {metrics['volatility']:.4f}
    Sharpe Ratio: {metrics['sharpe_ratio']:.4f}
    Maximum Drawdown: {metrics['max_drawdown']:.4f}
    """
    return report

def plot_performance(returns: pd.Series, title: str = 'Performance Plot'):
    """Plot the cumulative returns."""
    cumulative_returns = (1 + returns).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example: Compute performance metrics and generate a report
    returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    metrics = compute_performance_metrics(returns)
    report = generate_performance_report(metrics)
    print(report)
    plot_performance(returns) 