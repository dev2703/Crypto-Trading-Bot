"""
Statistical Arbitrage Strategy Module
- Computes cointegration between assets
- Generates trading signals based on mean reversion
- Includes backtesting and performance evaluation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from statsmodels.tsa.stattools import coint
from strategy.performance_evaluation import compute_performance_metrics

def compute_cointegration(price1: pd.Series, price2: pd.Series) -> Tuple[float, float]:
    """Compute cointegration between two price series."""
    score, pvalue = coint(price1, price2)
    return score, pvalue

def compute_spread(price1: pd.Series, price2: pd.Series, hedge_ratio: float) -> pd.Series:
    """Compute the spread between two price series."""
    return price1 - hedge_ratio * price2

def compute_hedge_ratio(price1: pd.Series, price2: pd.Series) -> float:
    """Compute the hedge ratio using OLS regression."""
    return np.cov(price1, price2)[0, 1] / np.var(price2)

def generate_signals(spread: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Generate trading signals based on mean reversion."""
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    z_score = (spread - mean) / std
    signals = pd.Series(0, index=spread.index)
    signals[z_score > num_std] = -1
    signals[z_score < -num_std] = 1
    return signals

def backtest_strategy(price1: pd.Series, price2: pd.Series, window: int = 20, num_std: float = 2.0) -> Dict:
    """Backtest a statistical arbitrage strategy."""
    hedge_ratio = compute_hedge_ratio(price1, price2)
    spread = compute_spread(price1, price2, hedge_ratio)
    signals = generate_signals(spread, window, num_std)
    returns1 = price1.pct_change().dropna()
    returns2 = price2.pct_change().dropna()
    strategy_returns = signals.shift(1) * (returns1 - hedge_ratio * returns2)
    cumulative_returns = (1 + strategy_returns).cumprod()
    metrics = compute_performance_metrics(strategy_returns)
    return {
        'signals': signals,
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'metrics': metrics,
        'hedge_ratio': hedge_ratio
    }

# Example usage
if __name__ == "__main__":
    # Example: Load sample data and backtest strategy
    price1 = pd.Series([100, 101, 102, 103, 104])
    price2 = pd.Series([200, 202, 204, 206, 208])
    result = backtest_strategy(price1, price2)
    print(result) 