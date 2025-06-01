"""
Backtesting Strategy Module
- Simulates trading strategies using historical data
- Evaluates performance metrics (returns, Sharpe ratio, drawdown)
- Includes transaction costs and slippage in the simulation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from strategy.execution import MarketOrder, compute_transaction_cost, compute_slippage

def compute_returns(prices: pd.Series) -> pd.Series:
    """Compute returns from price series."""
    return prices.pct_change().dropna()

def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute Sharpe ratio from returns."""
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def compute_drawdown(returns: pd.Series) -> pd.Series:
    """Compute drawdown series from returns."""
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    return drawdown

def backtest_strategy(prices: pd.Series, signals: pd.Series, initial_capital: float = 100000.0, fee_rate: float = 0.001, slippage_rate: float = 0.001) -> Dict:
    """Backtest a trading strategy using historical data."""
    returns = compute_returns(prices)
    portfolio_value = initial_capital
    portfolio_values = [initial_capital]
    for i in range(1, len(prices)):
        if signals.iloc[i] == 1:  # Buy signal
            order = MarketOrder('BTC', portfolio_value / prices.iloc[i])
            transaction_cost = compute_transaction_cost(order, prices.iloc[i], fee_rate)
            slippage = compute_slippage(order, prices.iloc[i], slippage_rate)
            portfolio_value = portfolio_value - transaction_cost - slippage
        elif signals.iloc[i] == -1:  # Sell signal
            order = MarketOrder('BTC', portfolio_value / prices.iloc[i])
            transaction_cost = compute_transaction_cost(order, prices.iloc[i], fee_rate)
            slippage = compute_slippage(order, prices.iloc[i], slippage_rate)
            portfolio_value = portfolio_value - transaction_cost - slippage
        portfolio_values.append(portfolio_value)
    portfolio_values = pd.Series(portfolio_values, index=prices.index)
    strategy_returns = compute_returns(portfolio_values)
    sharpe_ratio = compute_sharpe_ratio(strategy_returns)
    drawdown = compute_drawdown(strategy_returns)
    return {
        'portfolio_values': portfolio_values,
        'strategy_returns': strategy_returns,
        'sharpe_ratio': sharpe_ratio,
        'drawdown': drawdown
    }

# Example usage
if __name__ == "__main__":
    # Example: Load sample data and backtest strategy
    prices = pd.Series([100, 105, 110, 115, 120])
    signals = pd.Series([0, 1, 0, -1, 0])
    result = backtest_strategy(prices, signals)
    print(result) 