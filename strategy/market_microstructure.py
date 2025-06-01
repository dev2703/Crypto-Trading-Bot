"""
Market Microstructure Analysis Strategy Module
- Computes bid-ask spread and order book imbalance
- Analyzes market depth and liquidity
- Generates trading signals
- Includes backtesting and performance evaluation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from strategy.performance_evaluation import compute_performance_metrics

def compute_bid_ask_spread(bid_prices: pd.Series, ask_prices: pd.Series) -> pd.Series:
    """Compute bid-ask spread."""
    return (ask_prices - bid_prices) / bid_prices

def compute_order_imbalance(bid_volumes: pd.Series, ask_volumes: pd.Series) -> pd.Series:
    """Compute order book imbalance."""
    return (bid_volumes - ask_volumes) / (bid_volumes + ask_volumes)

def compute_market_depth(bid_volumes: pd.Series, ask_volumes: pd.Series) -> pd.Series:
    """Compute market depth."""
    return bid_volumes + ask_volumes

def generate_signals(spread: pd.Series, imbalance: pd.Series, depth: pd.Series, spread_threshold: float = 0.001, imbalance_threshold: float = 0.2) -> pd.Series:
    """Generate trading signals based on market microstructure indicators."""
    signals = pd.Series(0, index=spread.index)
    signals[(spread < spread_threshold) & (imbalance > imbalance_threshold)] = 1
    signals[(spread < spread_threshold) & (imbalance < -imbalance_threshold)] = -1
    return signals

def backtest_strategy(bid_prices: pd.Series, ask_prices: pd.Series, bid_volumes: pd.Series, ask_volumes: pd.Series, mid_prices: pd.Series, spread_threshold: float = 0.001, imbalance_threshold: float = 0.2) -> Dict:
    """Backtest a market microstructure analysis strategy."""
    spread = compute_bid_ask_spread(bid_prices, ask_prices)
    imbalance = compute_order_imbalance(bid_volumes, ask_volumes)
    depth = compute_market_depth(bid_volumes, ask_volumes)
    signals = generate_signals(spread, imbalance, depth, spread_threshold, imbalance_threshold)
    returns = mid_prices.pct_change().dropna()
    strategy_returns = signals.shift(1) * returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    metrics = compute_performance_metrics(strategy_returns)
    return {
        'signals': signals,
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'metrics': metrics
    }

# Example usage
if __name__ == "__main__":
    # Example: Load sample data and backtest strategy
    bid_prices = pd.Series([100, 101, 102, 103, 104])
    ask_prices = pd.Series([101, 102, 103, 104, 105])
    bid_volumes = pd.Series([10, 12, 14, 16, 18])
    ask_volumes = pd.Series([8, 10, 12, 14, 16])
    mid_prices = (bid_prices + ask_prices) / 2
    result = backtest_strategy(bid_prices, ask_prices, bid_volumes, ask_volumes, mid_prices)
    print(result) 