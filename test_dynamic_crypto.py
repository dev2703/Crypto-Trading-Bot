"""
Test script for dynamic cryptocurrency data collection
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

class DataQualityTracker:
    """Track data quality and imputation."""
    
    def __init__(self):
        self.original_data_points = 0
        self.imputed_data_points = 0
        self.imputed_columns = {}
        self.data_gaps = []
        self.quality_metrics = {}
    
    def track_imputation(self, df: pd.DataFrame, column: str, original_df: pd.DataFrame):
        """Track imputed values in a column."""
        if column not in self.imputed_columns:
            self.imputed_columns[column] = 0
        
        # Compare with original data to find imputed values
        imputed_mask = df[column] != original_df[column]
        imputed_count = imputed_mask.sum()
        self.imputed_columns[column] += imputed_count
        self.imputed_data_points += imputed_count
        
        # Track gaps
        if imputed_count > 0:
            gap_start = df.index[imputed_mask].min()
            gap_end = df.index[imputed_mask].max()
            self.data_gaps.append({
                'column': column,
                'start': gap_start,
                'end': gap_end,
                'count': imputed_count
            })
    
    def get_quality_report(self) -> Dict:
        """Generate data quality report."""
        total_points = self.original_data_points
        imputed_points = self.imputed_data_points
        
        return {
            'total_data_points': total_points,
            'imputed_data_points': imputed_points,
            'imputation_ratio': imputed_points / total_points if total_points > 0 else 0,
            'imputed_columns': self.imputed_columns,
            'data_gaps': self.data_gaps
        }

def run_crypto_strategy(coin_id: str, days: int = 365, interval: str = '1d'):
    """Run comprehensive strategy test for any cryptocurrency with all indicators."""
    logger.info(f"Starting {coin_id.capitalize()} strategy test with all indicators...")
    collector = DataCollector()
    quality_tracker = DataQualityTracker()
    
    try:
        # Fetch cryptocurrency data
        logger.info(f"Fetching {coin_id.capitalize()} OHLCV data...")
        df = collector.collect_crypto_ohlcv_data(coin_id, days=days, interval=interval)
        if df is None or df.empty:
            logger.error(f"Failed to fetch {coin_id.capitalize()} data.")
            return

        # Store original data for comparison
        original_df = df.copy()
        quality_tracker.original_data_points = df.size

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

        # Handle duplicates without imputation
        if quality_stats['duplicates'] > 0:
            df = df[~df.index.duplicated(keep='first')]
            logger.info("Removed duplicate timestamps")
        
        # Handle missing values with minimal imputation
        if quality_stats['missing_values']:
            # Only impute if gap is small (e.g., less than 3 periods)
            for col in df.columns:
                missing_mask = df[col].isna()
                if missing_mask.any():
                    # Find consecutive missing values
                    missing_groups = (missing_mask != missing_mask.shift()).cumsum()
                    for group in missing_groups.unique():
                        group_mask = missing_groups == group
                        if group_mask.sum() <= 3:  # Only impute small gaps
                            df.loc[group_mask, col] = df[col].ffill().bfill()
                            quality_tracker.track_imputation(df, col, original_df)
                        else:
                            logger.warning(f"Large gap detected in {col}: {group_mask.sum()} periods")
                            # Mark large gaps with NaN to avoid false signals
                            df.loc[group_mask, col] = np.nan

        # Feature engineering with quality tracking
        engineer = FeatureEngineer()
        logger.info("Generating features...")
        df = engineer.generate_features(df)

        # Calculate basic indicators
        logger.info("Calculating basic indicators...")
        df = calculate_additional_indicators(df)

        # Calculate advanced indicators (optional)
        # advanced = AdvancedIndicators()
        # Example: df['kama_20'] = advanced.calculate_kama(df['close'], period=20)

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
        
        # Print high-frequency trading metrics
        logger.info("\nHigh-Frequency Trading Metrics:")
        logger.info(f"Average Holding Periods: {results['avg_holding_periods']:.1f}")
        logger.info(f"Profit Target Hits: {results['profit_target_hits']} ({results['profit_target_hits']/results['num_trades']:.1%} of trades)")
        logger.info(f"Stop Loss Hits: {results['stop_loss_hits']} ({results['stop_loss_hits']/results['num_trades']:.1%} of trades)")
        logger.info(f"Time Exit Hits: {results['time_exit_hits']} ({results['time_exit_hits']/results['num_trades']:.1%} of trades)")
        
        # Print data quality report
        quality_report = quality_tracker.get_quality_report()
        logger.info("\nData Quality Report:")
        logger.info(f"Total Data Points: {quality_report['total_data_points']}")
        logger.info(f"Imputed Data Points: {quality_report['imputed_data_points']}")
        logger.info(f"Imputation Ratio: {quality_report['imputation_ratio']:.2%}")
        logger.info("\nImputed Columns:")
        for col, count in quality_report['imputed_columns'].items():
            logger.info(f"{col}: {count} points imputed")
        logger.info("\nData Gaps:")
        for gap in quality_report['data_gaps']:
            logger.info(f"Column: {gap['column']}, Period: {gap['start']} to {gap['end']}, Count: {gap['count']}")
        
        # Plot results
        plot_results(df, results, coin_id, quality_report)
        
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
    """Simulate trading with the given signals and risk management."""
    portfolio_value = 1000  # Initial portfolio value
    portfolio_weight = 0.02  # 2% risk per trade
    max_holding_periods = 4  # Maximum holding period (e.g., 4 hours for 1h timeframe)
    profit_threshold = 0.015  # 1.5% profit target
    stop_loss = 0.01  # 1% stop loss
    
    position = None
    entry_price = 0
    entry_time = None
    holding_periods = 0
    
    # Track metrics
    trades = []
    profit_target_hits = 0
    stop_loss_hits = 0
    time_exit_hits = 0
    total_holding_periods = 0
    
    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        current_time = df.index[i]
        
        # Check for exit conditions if in position
        if position is not None:
            holding_periods += 1
            pnl = (current_price - entry_price) / entry_price * position
            
            # Exit conditions
            if pnl >= profit_threshold:  # Take profit
                portfolio_value += portfolio_value * pnl
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'holding_periods': holding_periods,
                    'exit_reason': 'profit_target'
                })
                profit_target_hits += 1
                total_holding_periods += holding_periods
                position = None
                holding_periods = 0
                
            elif pnl <= -stop_loss:  # Stop loss
                portfolio_value += portfolio_value * pnl
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'holding_periods': holding_periods,
                    'exit_reason': 'stop_loss'
                })
                stop_loss_hits += 1
                total_holding_periods += holding_periods
                position = None
                holding_periods = 0
                
            elif holding_periods >= max_holding_periods:  # Time-based exit
                portfolio_value += portfolio_value * pnl
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'holding_periods': holding_periods,
                    'exit_reason': 'time_exit'
                })
                time_exit_hits += 1
                total_holding_periods += holding_periods
                position = None
                holding_periods = 0
                
            elif signals.iloc[i] == -position:  # Signal reversal
                portfolio_value += portfolio_value * pnl
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'holding_periods': holding_periods,
                    'exit_reason': 'signal_reversal'
                })
                total_holding_periods += holding_periods
                position = None
                holding_periods = 0
        
        # Enter new position if no current position
        if position is None and signals.iloc[i] != 0:
            position = signals.iloc[i]
            entry_price = current_price
            entry_time = current_time
            holding_periods = 0
    
    # Calculate metrics
    if trades:
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        total_return = (portfolio_value - 1000) / 1000
        win_rate = len(winning_trades) / len(trades)
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum([t['pnl'] for t in winning_trades]) / sum([t['pnl'] for t in losing_trades])) if losing_trades else float('inf')
        
        # Calculate drawdown
        portfolio_values = [1000]
        for trade in trades:
            portfolio_values.append(portfolio_values[-1] * (1 + trade['pnl']))
        portfolio_values = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Calculate high-frequency metrics
        avg_holding_periods = total_holding_periods / len(trades)
        
        return {
            'total_return': total_return,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'avg_holding_periods': avg_holding_periods,
            'profit_target_hits': profit_target_hits,
            'stop_loss_hits': stop_loss_hits,
            'time_exit_hits': time_exit_hits,
            'trades': trades
        }
    else:
        return {
            'total_return': 0,
            'num_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'avg_holding_periods': 0,
            'profit_target_hits': 0,
            'stop_loss_hits': 0,
            'time_exit_hits': 0,
            'trades': []
        }

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

def plot_results(df, results, coin_id, quality_report):
    """Plot trading results with high-frequency metrics."""
    plt.style.use('default')  # Use default style instead of seaborn
    fig = plt.figure(figsize=(15, 12))
    
    # Price and signals
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df.index, df['close'], label='Price', color='blue', alpha=0.7)
    
    # Plot entry and exit points
    for trade in results['trades']:
        if trade['pnl'] > 0:
            color = 'green'
        else:
            color = 'red'
        ax1.scatter(trade['entry_time'], trade['entry_price'], color=color, marker='^', s=100)
        ax1.scatter(trade['exit_time'], trade['exit_price'], color=color, marker='v', s=100)
    
    ax1.set_title(f'{coin_id.capitalize()} Trading Results')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Portfolio value
    ax2 = plt.subplot(3, 1, 2)
    portfolio_values = [1000]
    for trade in results['trades']:
        portfolio_values.append(portfolio_values[-1] * (1 + trade['pnl']))
    ax2.plot(df.index[:len(portfolio_values)], portfolio_values, label='Portfolio Value', color='green')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.legend()
    ax2.grid(True)
    
    # Trading metrics
    ax3 = plt.subplot(3, 1, 3)
    metrics = {
        'Total Return': f"{results['total_return']:.2%}",
        'Win Rate': f"{results['win_rate']:.2%}",
        'Avg Win': f"{results['avg_win']:.2%}",
        'Avg Loss': f"{results['avg_loss']:.2%}",
        'Profit Factor': f"{results['profit_factor']:.2f}",
        'Max Drawdown': f"{results['max_drawdown']:.2%}",
        'Avg Holding Periods': f"{results['avg_holding_periods']:.1f}",
        'Profit Target Hits': f"{results['profit_target_hits']} ({results['profit_target_hits']/results['num_trades']:.1%})",
        'Stop Loss Hits': f"{results['stop_loss_hits']} ({results['stop_loss_hits']/results['num_trades']:.1%})",
        'Time Exit Hits': f"{results['time_exit_hits']} ({results['time_exit_hits']/results['num_trades']:.1%})"
    }
    
    # Add data quality metrics
    metrics.update({
        'Total Data Points': f"{quality_report['total_data_points']}",
        'Imputed Points': f"{quality_report['imputed_data_points']} ({quality_report['imputation_ratio']:.2%})"
    })
    
    y_pos = np.arange(len(metrics))
    ax3.barh(y_pos, [1] * len(metrics), color='lightgray')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(metrics.keys())
    ax3.set_xlim(0, 1)
    ax3.set_xticks([])
    
    # Add metric values
    for i, (metric, value) in enumerate(metrics.items()):
        ax3.text(0.5, i, value, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{coin_id}_trading_results.png')
    plt.close()

def main(coin_id: str, interval: str = '1d'):
    """
    Main function to run the trading strategy for a specific cryptocurrency and interval.
    Args:
        coin_id (str): The ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum', 'solana')
        interval (str): The time frame/interval (e.g., '1d', '1h', '5m', etc.)
    """
    try:
        print(f"\nTesting strategy for {coin_id.capitalize()} [{interval}]...")
        run_crypto_strategy(coin_id, interval=interval)
        print(f"\nStrategy test completed for {coin_id.capitalize()} [{interval}]")
        print(f"Results have been saved to {coin_id}_trading_results.png")
    except Exception as e:
        print(f"Error testing {coin_id.capitalize()} [{interval}]: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        coin_id = sys.argv[1].lower()
        interval = sys.argv[2]
        main(coin_id, interval)
    elif len(sys.argv) > 1:
        coin_id = sys.argv[1].lower()
        main(coin_id)
    else:
        print("Please provide a cryptocurrency ID as an argument.")
        print("Example: python test_dynamic_crypto.py solana 1h")
        print("Available options: bitcoin, ethereum, solana, cardano, polkadot, etc.")
        print("You can also specify an interval, e.g. '1d', '1h', '5m', etc.") 