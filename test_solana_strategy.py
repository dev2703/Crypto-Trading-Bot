"""
Test script for Solana trading strategy
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy.feature_engineering import FeatureEngineer
from strategy.risk_management import RiskManager
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from strategy.data_collection import DataCollector
import os
from dotenv import load_dotenv
from typing import Tuple, List, Dict, Optional
import joblib
from strategy.advanced_indicators import AdvancedIndicators

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_solana_strategy(days: int = 365):
    """Run comprehensive strategy test for Solana with all indicators."""
    logger.info("Starting Solana strategy test with all indicators...")
    collector = DataCollector()
    try:
        # Fetch Solana data
        logger.info("Fetching Solana OHLCV data...")
        df = collector.collect_crypto_ohlcv_data("solana", days=days)
        if df is None or df.empty:
            logger.error("Failed to fetch Solana data.")
            return

        # Check data quality
        logger.info("Checking data quality...")
        quality_stats = check_data_quality(df)
        logger.info("\nData Quality Report:")
        logger.info(f"Missing Values: {quality_stats['missing_values']}")
        logger.info(f"Duplicate Timestamps: {quality_stats['duplicates']}")
        logger.info(f"Price Anomalies: {quality_stats['price_anomalies']}")
        logger.info(f"Volume Anomalies: {quality_stats['volume_anomalies']}")
        logger.info(f"Timestamp Gaps: {quality_stats['timestamp_gaps']}")
        logger.info(f"Price Gaps: {quality_stats['price_gaps']}")
        logger.info(f"Volume Spikes: {quality_stats['volume_spikes']}")

        # Clean data if necessary
        if quality_stats['duplicates'] > 0:
            df = df[~df.index.duplicated(keep='first')]
            logger.info("Removed duplicate timestamps")
        
        if quality_stats['missing_values']:
            df = df.fillna(method='ffill').fillna(method='bfill')
            logger.info("Filled missing values using forward and backward fill")

        # Feature engineering
        engineer = FeatureEngineer()
        logger.info("Generating features...")
        df = engineer.generate_features(df)

        # Calculate basic indicators
        logger.info("Calculating basic indicators...")
        df = calculate_additional_indicators(df)

        # Calculate advanced indicators
        logger.info("Calculating advanced indicators...")
        advanced = AdvancedIndicators()
        df = advanced.calculate_all_indicators(df)

        # Initialize risk manager
        risk_manager = RiskManager()
        
        # Generate trading signals
        logger.info("Generating trading signals...")
        signals = generate_trading_signals(df)
        
        # Simulate trading
        logger.info("Simulating trading...")
        results = simulate_trading(df, signals, risk_manager)
        
        # Print results
        logger.info("\nTrading Results:")
        logger.info(f"Total Return: {results['total_return']:.2%}")
        logger.info(f"Number of Trades: {results['num_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.2%}")
        logger.info(f"Average Win: {results['avg_win']:.2%}")
        logger.info(f"Average Loss: {results['avg_loss']:.2%}")
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
        
        # Plot results
        plot_results(df, results)
        
    except Exception as e:
        logger.error(f"Error in strategy test: {str(e)}")
        raise

def check_data_quality(df: pd.DataFrame) -> dict:
    """Check data quality and return statistics."""
    quality_stats = {
        'missing_values': {},
        'duplicates': 0,
        'price_anomalies': 0,
        'volume_anomalies': 0,
        'timestamp_gaps': 0,
        'price_gaps': 0,
        'volume_spikes': 0
    }
    
    # Check for missing values
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            quality_stats['missing_values'][col] = missing
    
    # Check for duplicate timestamps
    quality_stats['duplicates'] = df.index.duplicated().sum()
    
    # Check for price anomalies (using z-score)
    price_zscore = np.abs((df['close'] - df['close'].mean()) / df['close'].std())
    quality_stats['price_anomalies'] = (price_zscore > 3).sum()
    
    # Check for volume anomalies
    volume_zscore = np.abs((df['volume'] - df['volume'].mean()) / df['volume'].std())
    quality_stats['volume_anomalies'] = (volume_zscore > 3).sum()
    
    # Check for timestamp gaps
    expected_timestamps = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    quality_stats['timestamp_gaps'] = len(expected_timestamps) - len(df)
    
    # Check for price gaps (using percentage change)
    price_changes = df['close'].pct_change().abs()
    quality_stats['price_gaps'] = (price_changes > 0.1).sum()  # 10% price gaps
    
    # Check for volume spikes
    volume_changes = df['volume'].pct_change().abs()
    quality_stats['volume_spikes'] = (volume_changes > 2).sum()  # 200% volume spikes
    
    return quality_stats

def calculate_additional_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional technical indicators."""
    try:
        # Create a copy to avoid fragmentation
        df = df.copy()
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        raise

def generate_trading_signals(df: pd.DataFrame) -> pd.Series:
    """Generate trading signals based on technical indicators."""
    signals = pd.Series(0, index=df.index)
    
    # MACD crossover
    signals[df['macd'] > df['macd_signal']] = 1
    signals[df['macd'] < df['macd_signal']] = -1
    
    # RSI overbought/oversold
    signals[df['rsi'] > 70] = -1
    signals[df['rsi'] < 30] = 1
    
    # Bollinger Bands
    signals[df['close'] > df['bb_upper']] = -1
    signals[df['close'] < df['bb_lower']] = 1
    
    # Moving Average crossover
    signals[df['sma_20'] > df['sma_50']] = 1
    signals[df['sma_20'] < df['sma_50']] = -1
    
    return signals

def simulate_trading(df: pd.DataFrame, signals: pd.Series, risk_manager: RiskManager) -> dict:
    """Simulate trading with risk management."""
    position = 0
    trades = []
    portfolio_value = 1000  # Initial portfolio value
    portfolio_values = [portfolio_value]
    
    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        signal = signals.iloc[i]
        
        # Calculate position size based on risk management
        position_size = risk_manager.calculate_position_size(
            portfolio_value,
            current_price,
            df['close'].iloc[i-1:i+1].std()
        )
        
        if signal != 0 and signal != position:
            # Close existing position
            if position != 0:
                trade_return = (current_price - entry_price) / entry_price * position
                portfolio_value *= (1 + trade_return)
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': position,
                    'return': trade_return
                })
            
            # Open new position
            position = signal
            entry_price = current_price
            entry_date = df.index[i]
        
        portfolio_values.append(portfolio_value)
    
    # Calculate trading statistics
    if trades:
        returns = [trade['return'] for trade in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        results = {
            'total_return': (portfolio_value - 1000) / 1000,
            'num_trades': len(trades),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            'max_drawdown': calculate_max_drawdown(portfolio_values),
            'trades': trades,
            'portfolio_values': portfolio_values
        }
    else:
        results = {
            'total_return': 0,
            'num_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'trades': [],
            'portfolio_values': portfolio_values
        }
    
    return results

def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """Calculate maximum drawdown from portfolio values."""
    peak = portfolio_values[0]
    max_drawdown = 0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown

def plot_results(df: pd.DataFrame, results: dict):
    """Plot trading results."""
    plt.figure(figsize=(15, 10))
    
    # Plot price and portfolio value
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df['close'], label='Solana Price', alpha=0.5)
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(df.index, results['portfolio_values'], label='Portfolio Value', color='green')
    ax2.set_ylabel('Portfolio Value (USD)')
    ax2.legend(loc='upper right')
    
    # Plot trades
    for trade in results['trades']:
        if trade['position'] == 1:
            ax1.scatter(trade['entry_date'], trade['entry_price'], color='green', marker='^')
            ax1.scatter(trade['exit_date'], trade['exit_price'], color='red', marker='v')
        else:
            ax1.scatter(trade['entry_date'], trade['entry_price'], color='red', marker='v')
            ax1.scatter(trade['exit_date'], trade['exit_price'], color='green', marker='^')
    
    # Plot drawdown
    ax3 = plt.subplot(2, 1, 2)
    portfolio_values = np.array(results['portfolio_values'])
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    ax3.fill_between(df.index, drawdown, 0, color='red', alpha=0.3)
    ax3.set_ylabel('Drawdown')
    ax3.set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig('solana_trading_results.png')
    plt.close()

if __name__ == "__main__":
    run_solana_strategy() 