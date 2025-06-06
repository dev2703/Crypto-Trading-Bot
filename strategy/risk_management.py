"""
Risk Management Module
- Tracks key performance indicators (KPIs)
- Implements risk management rules
- Calculates portfolio metrics
- Monitors trading metrics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from scipy import stats
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskManager:
    """Risk management for trading strategies."""
    
    def __init__(self, initial_capital: float = 1000000.0):
        """Initialize risk manager."""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.daily_returns = pd.Series()
        self.risk_metrics = {}
        
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}
        
        # Basic returns
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (252/len(returns)) - 1
        
        # Risk-adjusted returns
        metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std()
        metrics['sortino_ratio'] = np.sqrt(252) * returns.mean() / returns[returns < 0].std()
        
        # Drawdown metrics
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        metrics['max_drawdown'] = drawdowns.min()
        metrics['avg_drawdown'] = drawdowns[drawdowns < 0].mean()
        
        # Win rate and profit factor
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        metrics['win_rate'] = len(winning_trades) / len(returns)
        metrics['profit_factor'] = abs(winning_trades.sum() / losing_trades.sum())
        
        return metrics
        
    def calculate_risk_metrics(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """Calculate risk metrics."""
        metrics = {}
        
        # Value at Risk
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['expected_shortfall'] = returns[returns <= metrics['var_95']].mean()
        
        # Volatility
        metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
        metrics['downside_volatility'] = returns[returns < 0].std() * np.sqrt(252)
        
        # Beta and Alpha (if benchmark provided)
        if benchmark_returns is not None:
            beta = np.cov(returns, benchmark_returns)[0,1] / np.var(benchmark_returns)
            alpha = returns.mean() - beta * benchmark_returns.mean()
            metrics['beta'] = beta
            metrics['alpha'] = alpha * 252  # Annualized
            metrics['information_ratio'] = alpha / (returns - benchmark_returns).std() * np.sqrt(252)
            
        return metrics
        
    def calculate_trading_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate trading metrics."""
        if not trades:
            return {}
            
        metrics = {}
        trade_df = pd.DataFrame(trades)
        
        # Turnover
        total_volume = trade_df['volume'].sum()
        metrics['turnover'] = total_volume / self.initial_capital
        
        # Slippage and market impact
        metrics['avg_slippage'] = (trade_df['execution_price'] - trade_df['intended_price']).mean()
        metrics['market_impact'] = (trade_df['execution_price'] - trade_df['prev_close']).mean()
        
        # Fill rate
        metrics['fill_rate'] = len(trade_df[trade_df['status'] == 'filled']) / len(trade_df)
        
        # Holding period
        trade_df['holding_period'] = (trade_df['exit_time'] - trade_df['entry_time']).dt.total_seconds() / 3600
        metrics['avg_holding_period'] = trade_df['holding_period'].mean()
        
        return metrics
        
    def calculate_portfolio_metrics(self, positions: Dict[str, float], prices: Dict[str, pd.Series]) -> Dict[str, float]:
        """Calculate portfolio metrics."""
        metrics = {}
        
        # Position sizes
        position_values = {coin: size * prices[coin].iloc[-1] for coin, size in positions.items()}
        total_value = sum(position_values.values())
        
        # Concentration
        weights = {coin: value/total_value for coin, value in position_values.items()}
        metrics['hhi'] = sum(w**2 for w in weights.values())  # Herfindahl-Hirschman Index
        
        # Correlation matrix
        returns = pd.DataFrame({coin: prices[coin].pct_change() for coin in positions.keys()})
        metrics['avg_correlation'] = returns.corr().mean().mean()
        
        # Liquidity risk
        metrics['liquidity_risk'] = {
            coin: positions[coin] / prices[coin].rolling(20).mean().iloc[-1]
            for coin in positions.keys()
        }
        
        return metrics
        
    def update_risk_limits(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Update risk limits based on current metrics."""
        limits = {
            'max_position_size': 0.1,  # 10% of portfolio
            'max_drawdown': 0.2,       # 20% maximum drawdown
            'min_sharpe': 1.0,         # Minimum Sharpe ratio
            'max_correlation': 0.7,    # Maximum correlation between assets
            'max_leverage': 2.0,       # Maximum leverage
            'min_liquidity': 0.01      # Minimum liquidity ratio
        }
        
        violations = {}
        for metric, value in metrics.items():
            if metric in limits:
                violations[metric] = value > limits[metric]
                
        return violations
        
    def calculate_position_size(self, 
                              signal_strength: float,
                              volatility: float,
                              market_regime: str,
                              portfolio_weight: float) -> float:
        """Calculate position size based on risk parameters."""
        # Base position size
        base_size = self.current_capital * portfolio_weight
        
        # Adjust for signal strength
        signal_factor = min(abs(signal_strength), 1.0)
        
        # Adjust for volatility
        vol_factor = 1 / (1 + volatility)
        
        # Adjust for market regime
        regime_factor = 1.2 if market_regime == 'trending' else 0.8
        
        # Calculate final position size
        position_size = base_size * signal_factor * vol_factor * regime_factor
        
        # Apply risk limits
        max_position = self.current_capital * 0.1  # 10% of portfolio
        position_size = min(position_size, max_position)
        
        return position_size
        
    def update_metrics(self, 
                      returns: pd.Series,
                      trades: List[Dict],
                      positions: Dict[str, float],
                      prices: Dict[str, pd.Series],
                      benchmark_returns: pd.Series = None) -> Dict[str, Dict[str, float]]:
        """Update all risk metrics."""
        self.risk_metrics = {
            'performance': self.calculate_performance_metrics(returns),
            'risk': self.calculate_risk_metrics(returns, benchmark_returns),
            'trading': self.calculate_trading_metrics(trades),
            'portfolio': self.calculate_portfolio_metrics(positions, prices)
        }
        
        # Update risk limits
        self.risk_metrics['violations'] = self.update_risk_limits(self.risk_metrics['performance'])
        
        return self.risk_metrics
        
    def generate_risk_report(self) -> str:
        """Generate a comprehensive risk report."""
        if not self.risk_metrics:
            return "No risk metrics available."
            
        report = "Risk Management Report\n"
        report += "=====================\n\n"
        
        # Performance metrics
        report += "Performance Metrics:\n"
        for metric, value in self.risk_metrics['performance'].items():
            report += f"{metric}: {value:.4f}\n"
            
        # Risk metrics
        report += "\nRisk Metrics:\n"
        for metric, value in self.risk_metrics['risk'].items():
            report += f"{metric}: {value:.4f}\n"
            
        # Trading metrics
        report += "\nTrading Metrics:\n"
        for metric, value in self.risk_metrics['trading'].items():
            report += f"{metric}: {value:.4f}\n"
            
        # Portfolio metrics
        report += "\nPortfolio Metrics:\n"
        for metric, value in self.risk_metrics['portfolio'].items():
            report += f"{metric}: {value:.4f}\n"
            
        # Risk limit violations
        report += "\nRisk Limit Violations:\n"
        for metric, violated in self.risk_metrics['violations'].items():
            report += f"{metric}: {'VIOLATED' if violated else 'OK'}\n"
            
        return report 