"""
Test script for feature engineering and risk management
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_data(days: int = 365) -> pd.DataFrame:
    """Generate synthetic test data."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate price data with trends and volatility
    np.random.seed(42)
    base_price = 50000
    trend = np.linspace(0, 0.5, days)  # Upward trend
    volatility = np.sin(np.linspace(0, 4*np.pi, days)) * 0.1  # Cyclical volatility
    
    prices = base_price * (1 + trend + volatility + np.random.normal(0, 0.02, days))
    prices = np.maximum(prices, 1000)  # Ensure minimum price
    
    # Generate volume data
    base_volume = 1e9
    volume = base_volume * (1 + 0.5 * np.sin(np.linspace(0, 8*np.pi, days)) + 
                           np.random.normal(0, 0.1, days))
    volume = np.maximum(volume, 1e8)  # Ensure minimum volume
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, days))),
        'close': prices,
        'volume': volume
    }, index=dates)
    
    return df

def simulate_trading_signals(df: pd.DataFrame, feature_importance: dict) -> list:
    """Simulate high-frequency trading signals with momentum, mean reversion, and noise."""
    signals = []
    np.random.seed(42)
    for i in range(10, len(df)):
        price = df['close'].iloc[i]
        if pd.isna(price) or price <= 0:
            continue
        # HFT-style features
        ret_1 = df['close'].pct_change(1).iloc[i]
        ret_3 = df['close'].pct_change(3).iloc[i]
        ret_5 = df['close'].pct_change(5).iloc[i]
        vol_5 = df['close'].pct_change().rolling(5).std().iloc[i]
        # Momentum signal
        momentum_signal = 1 if ret_1 > 0.001 else -1 if ret_1 < -0.001 else 0
        # Mean reversion signal
        mean_rev_signal = -1 if ret_5 > 0.01 else 1 if ret_5 < -0.01 else 0
        # Combine with some noise
        signal_strength = momentum_signal + mean_rev_signal
        if np.random.rand() < 0.1:
            signal_strength += np.random.choice([-1, 0, 1])
        # Cap signal strength to [-1, 1]
        signal_strength = max(-1, min(1, signal_strength))
        # Position size proportional to signal strength and volatility
        position_size = abs(signal_strength) * (0.03 + 0.07 * np.random.rand())
        # Simulate confidence as a function of volatility and randomness
        confidence = 0.3 + 0.7 * np.random.rand()
        signal = {
            'timestamp': df.index[i],
            'price': price,
            'feature_values': {
                'ret_1': ret_1,
                'ret_3': ret_3,
                'ret_5': ret_5,
                'vol_5': vol_5
            },
            'risk_metrics': {
                'volatility': vol_5,
                'drawdown': abs(df['close'].iloc[i] / df['close'].iloc[i-5:i].max() - 1),
                'correlation': np.random.uniform(0.1, 0.99)
            },
            'confidence': confidence,
            'signal_strength': signal_strength,
            'position_size': position_size
        }
        signals.append(signal)
    return signals

def plot_results(portfolio_metrics: dict, feature_importance: dict):
    """Plot test results."""
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot portfolio value
    ax1.plot(portfolio_metrics['portfolio_values'])
    ax1.set_title('Portfolio Value')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.grid(True)
    
    # Plot drawdown
    ax2.plot(portfolio_metrics['drawdowns'])
    ax2.set_title('Portfolio Drawdown')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True)
    
    # Plot position sizes
    ax3.bar(range(len(portfolio_metrics['position_sizes'])), 
            portfolio_metrics['position_sizes'])
    ax3.set_title('Position Sizes')
    ax3.set_xlabel('Trade')
    ax3.set_ylabel('Size')
    ax3.grid(True)
    
    # Plot feature importance
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    ax4.barh(features, importances)
    ax4.set_title('Feature Importance')
    ax4.set_xlabel('Importance')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()

def backtest(signals, df, initial_capital=100000, tx_cost_bps=5, slippage_bps=1):
    """Backtesting engine with transaction costs and slippage."""
    capital = initial_capital
    position = 0
    equity_curve = []
    trade_log = []
    last_price = None
    min_hold = 2  # Minimum holding period (bars)
    hold_counter = 0
    for i, signal in enumerate(signals):
        price = signal['price']
        if price <= 0 or pd.isna(price):
            continue
        # Apply slippage (random walk)
        slip = price * (slippage_bps / 10000.0) * (np.random.randn())
        exec_price = price + slip
        if signal.get('accepted', False):
            # Enter position (buy/long)
            if position == 0 and signal['signal_strength'] > 0:
                position = signal['position_size'] * capital / exec_price
                cost = exec_price * position * (tx_cost_bps / 10000.0)
                capital -= position * exec_price + cost
                trade_log.append({'type': 'BUY', 'price': exec_price, 'size': position, 'timestamp': signal['timestamp'], 'cost': cost})
                hold_counter = 0
            # Exit position (sell/close long)
            elif position > 0 and (signal['signal_strength'] < 0 or hold_counter >= min_hold):
                cost = exec_price * position * (tx_cost_bps / 10000.0)
                capital += position * exec_price - cost
                trade_log.append({'type': 'SELL', 'price': exec_price, 'size': position, 'timestamp': signal['timestamp'], 'cost': cost})
                position = 0
                hold_counter = 0
            else:
                hold_counter += 1
        last_price = exec_price
        equity = capital + (position * exec_price if position > 0 else 0)
        equity_curve.append(equity)
    # Liquidate at end
    if position > 0 and last_price is not None:
        cost = last_price * position * (tx_cost_bps / 10000.0)
        capital += position * last_price - cost
        trade_log.append({'type': 'SELL', 'price': last_price, 'size': position, 'timestamp': signals[-1]['timestamp'], 'cost': cost})
        position = 0
        equity_curve[-1] = capital
    return equity_curve, trade_log

def main():
    # Configurable asset and window
    asset_id = 'ethereum'  # Try 'bitcoin', 'ethereum', etc.
    window_days = 365      # 1 year (CoinGecko free API limit)
    logger.info(f"Fetching real {asset_id} OHLCV data from CoinGecko for {window_days} days...")
    load_dotenv()
    collector = DataCollector(api_key="")
    df = collector.collect_crypto_ohlcv_data(coin_id=asset_id, days=window_days)
    
    # Initialize feature engineer
    logger.info("Initializing feature engineer...")
    engineer = FeatureEngineer()
    
    # Generate features
    logger.info("Generating features...")
    features_df = engineer.generate_features(df)
    
    # Fine-tuned feature importance (not used in HFT signals, but kept for compatibility)
    feature_importance = {
        'ret_1': 0.3,
        'ret_3': 0.2,
        'ret_5': 0.2,
        'vol_5': 0.3
    }
    
    # Middle-ground risk manager thresholds for HFT
    logger.info("Initializing risk manager...")
    risk_manager = RiskManager(
        max_position_size=0.15,     # 15% of capital per trade
        max_volatility=2.0,         # High, but not extreme
        max_drawdown=0.3,           # 30% drawdown allowed
        max_correlation=0.95,       # Avoid highly correlated trades
        min_confidence=0.2,         # Filter out lowest-confidence signals
        feature_importance_threshold=0.01
    )
    risk_manager.update_feature_importance(feature_importance)
    
    # Simulate trading signals
    logger.info("Simulating trading signals...")
    signals = simulate_trading_signals(features_df, feature_importance)
    
    # Track portfolio metrics
    portfolio_metrics = {
        'portfolio_values': [],
        'drawdowns': [],
        'position_sizes': [],
        'risk_scores': []
    }
    
    # Process signals and mark accepted trades
    logger.info("Processing trading signals...")
    for i, signal in enumerate(signals):
        # Use position_size from signal (already HFT-style)
        market_conditions = {
            'market_volatility': signal['risk_metrics']['volatility'],
            'market_trend': np.random.uniform(-0.2, 0.2),
            'regime_volatility': np.random.uniform(0, 0.3)
        }
        passes_checks, risk_metrics = risk_manager.risk_check(signal, market_conditions)
        signal['accepted'] = passes_checks
        if passes_checks:
            risk_manager.update_portfolio(
                position_id=f"trade_{i}",
                size=signal['position_size'] * 1000000,
                price=signal['price'],
                timestamp=signal['timestamp'],
                risk_metrics=risk_metrics
            )
        portfolio_metrics['portfolio_values'].append(risk_manager.portfolio_value)
        portfolio_metrics['drawdowns'].append(risk_manager.calculate_portfolio_metrics()['drawdown'])
        portfolio_metrics['position_sizes'].append(signal['position_size'])
        portfolio_metrics['risk_scores'].append(risk_metrics.get('risk_score', 0))
    
    # Backtest for realistic P&L
    logger.info("Running backtest for realistic P&L simulation...")
    equity_curve, trade_log = backtest(signals, df)
    portfolio_metrics['equity_curve'] = equity_curve
    
    # Print full trade log
    print("\nFull Trade Log:")
    for t in trade_log:
        print(t)
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title('Backtest Equity Curve')
    plt.xlabel('Trade Index')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.savefig('equity_curve_debug.png')
    plt.close()
    
    # Generate final risk report
    logger.info("\nGenerating risk report...")
    print(risk_manager.generate_risk_report())
    
    # Plot results (add equity curve)
    logger.info("Plotting results...")
    plot_results(portfolio_metrics, feature_importance)
    
    logger.info("Test completed successfully!")
    print(f"Final equity: {equity_curve[-1]:.2f}")
    print(f"Number of trades: {len(trade_log)}")
    if trade_log:
        print("Sample trades:")
        for t in trade_log[:5]:
            print(t)

if __name__ == "__main__":
    main() 