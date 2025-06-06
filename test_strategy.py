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
from typing import Tuple, List, Dict, Optional
import joblib  # Add this import at the top of the file
import concurrent.futures
from scipy.optimize import minimize
import time
from strategy.advanced_indicators import AdvancedIndicators
import argparse

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

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value
    except Exception as e:
        logger.warning(f"Error calculating RSI: {str(e)}")
        return pd.Series(50, index=prices.index)  # Return neutral RSI on error

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD and Signal line."""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band

def calculate_additional_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional technical indicators (ATR removed)."""
    try:
        # Create a copy to avoid fragmentation
        df = df.copy()
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean().astype(float)
        df['sma_50'] = df['close'].rolling(window=50).mean().astype(float)
        
        # MACD and Signal line
        macd, signal = calculate_macd(df['close'])
        df['macd'] = macd.fillna(0).astype(float)
        df['macd_signal'] = signal.fillna(0).astype(float)
        
        # Bollinger Bands
        upper, lower = calculate_bollinger_bands(df['close'])
        df['bollinger_upper'] = upper.ffill().bfill().astype(float)
        df['bollinger_lower'] = lower.ffill().bfill().astype(float)
        
        # Ichimoku Cloud
        df['tenkan_sen'] = ((df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2).astype(float)
        df['kijun_sen'] = ((df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2).astype(float)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26).astype(float)
        df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26).astype(float)
        
        # Fill NaN values for Ichimoku using interpolation
        for col in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']:
            df[col] = df[col].interpolate(method='time').ffill().bfill().astype(float)
        
        # Stochastic Oscillator
        df['stoch_k'] = (100 * ((df['close'] - df['low'].rolling(window=14).min()) / 
                      (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()))).astype(float)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean().astype(float)
        df['stoch_k'] = df['stoch_k'].fillna(50).astype(float)
        df['stoch_d'] = df['stoch_d'].fillna(50).astype(float)
        
        # ADX for trend strength
        df['plus_dm'] = df['high'].diff().astype(float)
        df['minus_dm'] = df['low'].diff().astype(float)
        df['plus_dm'] = df['plus_dm'].where(df['plus_dm'] > df['minus_dm'].abs(), 0).astype(float)
        df['minus_dm'] = df['minus_dm'].where(df['minus_dm'] > df['plus_dm'].abs(), 0).astype(float)
        df['tr14'] = (df['high'] - df['low']).rolling(window=14).sum().astype(float)
        df['plus_di14'] = (100 * (df['plus_dm'].rolling(window=14).sum() / df['tr14'])).astype(float)
        df['minus_di14'] = (100 * (df['minus_dm'].rolling(window=14).sum() / df['tr14'])).astype(float)
        df['dx'] = (100 * abs(df['plus_di14'] - df['minus_di14']) / (df['plus_di14'] + df['minus_di14'])).astype(float)
        df['adx'] = df['dx'].rolling(window=14).mean().astype(float)
        
        # Fill NaN values for ADX using interpolation
        for col in ['plus_di14', 'minus_di14', 'dx', 'adx']:
            df[col] = df[col].interpolate(method='time').fillna(25).astype(float)  # Neutral ADX value
        
        # Volume indicators
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum().astype(float)
        df['vwap'] = ((df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()).astype(float)
        
        # Calculate RSI
        df['rsi'] = calculate_rsi(df['close']).astype(float)
        
        # Ensure all columns are float type
        for col in df.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        raise

def detect_market_regime(df: pd.DataFrame) -> pd.Series:
    """Detect market regime using multiple indicators."""
    try:
        # Volatility regime
        volatility = df['close'].pct_change().rolling(window=20).std()
        vol_regime = pd.Series(index=df.index, dtype='str')
        vol_regime[volatility > volatility.quantile(0.8)] = 'high_vol'
        vol_regime[volatility < volatility.quantile(0.2)] = 'low_vol'
        vol_regime[vol_regime == ''] = 'normal_vol'
        
        # Trend regime
        trend = pd.Series(index=df.index, dtype='str')
        sma_20 = df['close'].rolling(window=20).mean()
        sma_50 = df['close'].rolling(window=50).mean()
        trend[sma_20 > sma_50] = 'uptrend'
        trend[sma_20 < sma_50] = 'downtrend'
        trend[abs(sma_20 - sma_50) / sma_50 < 0.02] = 'sideways'
        
        # Momentum regime
        momentum = pd.Series(index=df.index, dtype='str')
        rsi = df['rsi']
        momentum[rsi > 70] = 'overbought'
        momentum[rsi < 30] = 'oversold'
        momentum[(rsi >= 30) & (rsi <= 70)] = 'neutral'
        
        # Combine regimes
        regime = pd.Series(index=df.index, dtype='str')
        for i in range(len(df)):
            regime.iloc[i] = f"{vol_regime.iloc[i]}_{trend.iloc[i]}_{momentum.iloc[i]}"
        
        return regime
    except Exception as e:
        logger.error(f"Error detecting market regime: {str(e)}")
        # Return neutral regime on error
        return pd.Series('normal_vol_sideways_neutral', index=df.index)

def calculate_position_size(signal_strength: float, volatility: float, regime: str, max_position: float = 0.15) -> float:
    """Calculate position size based on signal strength, volatility, and market regime."""
    # Base position size from signal strength (0 to 1)
    base_size = abs(signal_strength)
    
    # Adjust for volatility (reduce size in high volatility)
    vol_adjustment = 1.0 / (1.0 + volatility)
    
    # Adjust for market regime
    regime_multiplier = {
        'trending': 1.2,  # More aggressive in trending markets
        'ranging': 0.8,   # More conservative in ranging markets
        'volatile': 0.6,  # Most conservative in volatile markets
        'neutral': 1.0    # Neutral in other conditions
    }.get(regime, 1.0)
    
    # Calculate final position size
    position_size = base_size * vol_adjustment * regime_multiplier
    
    # Apply maximum position limit
    return min(position_size, max_position)

def simulate_trading_signals(df: pd.DataFrame, feature_importance: dict) -> list:
    """Simulate trading signals with enhanced risk management."""
    # Define threshold parameters
    base_threshold = 0.6  # Base threshold for signal strength
    confidence_threshold = 0.7  # Threshold for signal confidence
    volatility_threshold = 0.02  # Maximum acceptable volatility
    trend_strength_threshold = 0.5  # Minimum trend strength required
    
    signals = []
    position = 0  # Current position (1 for long, -1 for short, 0 for no position)
    np.random.seed(42)

    # Check for required indicators and their validation flags
    required_indicators = {
        'kama_20': 'kama_20_valid',
        'kama_50': 'kama_50_valid',
        'log_returns': 'log_returns_valid',
        'price_first_derivative': 'price_first_derivative_valid',
        'price_second_derivative': 'price_second_derivative_valid',
        'parkinson_vol': 'parkinson_vol_valid',
        'yang_zhang_vol': 'yang_zhang_vol_valid',
        'displacement': 'displacement_valid',
        'gaps': 'gaps_valid',
        'market_regime': 'market_regime_valid',
        'rsi': 'rsi_valid',
        'macd': 'macd_valid',
        'macd_signal': 'macd_signal_valid',
        'bollinger_upper': 'bollinger_upper_valid',
        'bollinger_lower': 'bollinger_lower_valid',
        'atr_20': 'atr_20_valid'
    }

    # Verify all required indicators and their validation flags exist
    missing_indicators = [ind for ind, valid_flag in required_indicators.items() 
                         if ind not in df.columns or valid_flag not in df.columns]
    if missing_indicators:
        logger.error(f"Missing required indicators or validation flags: {missing_indicators}")
        return signals

    # Counter for logging first few signals
    signal_count = 0
    max_signals_to_log = 5

    # Initialize market regime tracking
    current_regime = None
    regime_duration = 0
    min_regime_duration = 5

    # Initialize trailing stop tracking
    trailing_stops = {}
    max_trailing_stop_distance = 0.02  # 2% maximum trailing stop distance

    # Initialize trend strength tracking
    trend_strength = pd.Series(0.0, index=df.index)
    trend_window = 20

    # Debug: Log DataFrame columns
    logger.info(f"Available columns in DataFrame: {df.columns.tolist()}")

    for i in range(1, len(df)):
        # Skip if any required indicators are invalid
        if not all(df[valid_flag].iloc[i] for valid_flag in required_indicators.values()):
            logger.debug(f"Index {i}: Skipped - invalid indicators")
            continue

        # Get current price and indicators
        price = df['close'].iloc[i]
        volatility = df['yang_zhang_vol'].iloc[i]
        atr = df['atr_20'].iloc[i]
        
        # Calculate trend strength using multiple timeframes
        short_trend = df['close'].iloc[i] / df['close'].iloc[i-5] - 1  # 5-period trend
        medium_trend = df['close'].iloc[i] / df['close'].iloc[i-20] - 1  # 20-period trend
        long_trend = df['close'].iloc[i] / df['close'].iloc[i-50] - 1  # 50-period trend
        
        # Weighted trend strength
        trend_strength.iloc[i] = (short_trend * 0.5 + medium_trend * 0.3 + long_trend * 0.2)
        
        # Calculate KAMA trend and confidence with dynamic thresholds
        kama_trend = df['kama_20'].iloc[i] - df['kama_50'].iloc[i]
        kama_confidence = min(abs(kama_trend) / (df['yang_zhang_vol'].iloc[i] * 2), 1.0)
        
        # Calculate RSI signal and confidence with dynamic thresholds
        rsi = df['rsi'].iloc[i]
        rsi_upper = 65 + (volatility * 100)  # Lowered from 70
        rsi_lower = 35 - (volatility * 100)  # Raised from 30
        rsi_signal = 1 if rsi < rsi_lower else (-1 if rsi > rsi_upper else 0)
        rsi_confidence = abs(50 - rsi) / 50
        
        # Calculate MACD trend and confidence with volume confirmation
        macd_trend = df['macd'].iloc[i] - df['macd_signal'].iloc[i]
        # Fix volume ratio calculation
        volume_window = df['volume'].iloc[max(0, i-20):i+1]
        volume_mean = volume_window.mean()
        volume_ratio = df['volume'].iloc[i] / volume_mean if volume_mean > 0 else 1.0
        macd_confidence = min(abs(macd_trend) / (df['yang_zhang_vol'].iloc[i] * 2), 1.0) * min(volume_ratio, 2.0)
        
        # Calculate Bollinger Bands signal and confidence with dynamic bands
        bb_width = (df['bollinger_upper'].iloc[i] - df['bollinger_lower'].iloc[i]) / df['bollinger_upper'].iloc[i]
        bb_signal = 1 if price < df['bollinger_lower'].iloc[i] else (-1 if price > df['bollinger_upper'].iloc[i] else 0)
        bb_confidence = min(abs(price - df['bollinger_upper'].iloc[i]) / (df['yang_zhang_vol'].iloc[i] * 2), 1.0)
        
        # Calculate momentum signal and confidence with acceleration
        momentum_signal = df['price_first_derivative'].iloc[i]
        momentum_acceleration = df['price_second_derivative'].iloc[i]
        momentum_confidence = min(abs(momentum_signal) / (df['yang_zhang_vol'].iloc[i] * 2), 1.0) * (1 + abs(momentum_acceleration))
        
        # Calculate volatility signal with regime awareness
        vol_signal = -1 if df['yang_zhang_vol'].iloc[i] > df['yang_zhang_vol'].iloc[i-1] else 1
        vol_regime = 'high' if volatility > df['yang_zhang_vol'].rolling(window=20).mean().iloc[i] else 'low'
        
        # Determine market regime with trend confirmation
        regime = df['market_regime'].iloc[i]
        if current_regime != regime:
            if regime_duration >= min_regime_duration and abs(trend_strength.iloc[i]) > 0.01:  # Lowered from 0.02
                current_regime = regime
                regime_duration = 0
            else:
                regime_duration += 1
        else:
            regime_duration += 1
        
        # Calculate combined signal strength and confidence with dynamic weights
        signal_weights = {
            'trend': 0.3,
            'momentum': 0.2,
            'volatility': 0.15,
            'volume': 0.15,
            'mean_reversion': 0.2
        }
        
        # Adjust weights based on market regime
        if current_regime == 1:  # Trending market
            signal_weights['trend'] *= 1.2
            signal_weights['momentum'] *= 1.2
            signal_weights['mean_reversion'] *= 0.8
        else:  # Ranging market
            signal_weights['mean_reversion'] *= 1.2
            signal_weights['volatility'] *= 1.2
            signal_weights['trend'] *= 0.8
        
        signal_strength = (
            trend_strength.iloc[i] * signal_weights['trend'] +
            momentum_signal * momentum_confidence * signal_weights['momentum'] +
            vol_signal * signal_weights['volatility'] +
            volume_ratio * signal_weights['volume'] +
            bb_signal * bb_confidence * signal_weights['mean_reversion']
        )
        
        signal_confidence = (
            kama_confidence * signal_weights['trend'] +
            momentum_confidence * signal_weights['momentum'] +
            (1.0 if vol_regime == 'low' else 0.5) * signal_weights['volatility'] +
            min(volume_ratio, 2.0) * signal_weights['volume'] +
            bb_confidence * signal_weights['mean_reversion']
        )
        
        # Normalize signal strength and confidence
        norm_signal_strength = np.tanh(signal_strength / 10)  # Range [-1, 1]
        norm_signal_confidence = np.tanh(signal_confidence / 10)  # Range [-1, 1]

        # Use normalized values for thresholds and position sizing
        if abs(norm_signal_strength) > base_threshold and abs(norm_signal_confidence) > confidence_threshold:
            # Require multiple confirmations
            confirming_signals = sum([
                trend_strength.iloc[i] * signal_weights['trend'] > 0,
                momentum_signal * momentum_confidence * signal_weights['momentum'] > 0,
                vol_signal * signal_weights['volatility'] > 0,
                volume_ratio * signal_weights['volume'] > 0.2,  # Lowered from 0.3
                bb_signal * bb_confidence * signal_weights['mean_reversion'] > 0
            ])
            
            if confirming_signals >= 1:  # Lowered from 2
                # Calculate position size with enhanced risk management
                base_position_size = min(abs(norm_signal_strength), 1.0)
                volatility_adjustment = 1.0 / (1.0 + volatility * 10)  # Reduce position size in high volatility
                regime_adjustment = 0.8 if current_regime == 0 else 1.0  # Reduce position size in ranging markets
                trend_adjustment = 1.0 + abs(trend_strength.iloc[i])  # Increase size in strong trends
                
                position_size = base_position_size * volatility_adjustment * regime_adjustment * trend_adjustment
                
                # Calculate dynamic stop loss and take profit levels
                atr_multiplier = 2.0 if vol_regime == 'high' else 1.5
                stop_loss_distance = min(atr * atr_multiplier, price * max_trailing_stop_distance)
                take_profit_distance = stop_loss_distance * (2.0 if current_regime == 1 else 1.5)  # Higher R:R in trends
                
                signal = {
                    'timestamp': df.index[i],
                    'type': 'BUY' if norm_signal_strength > 0 else 'SELL',
                    'price': price,
                    'size': position_size,
                    'confidence': norm_signal_confidence,
                    'stop_loss': price - stop_loss_distance if norm_signal_strength > 0 else price + stop_loss_distance,
                    'take_profit': price + take_profit_distance if norm_signal_strength > 0 else price - take_profit_distance,
                    'features': {
                        'trend_strength': trend_strength.iloc[i],
                        'momentum': momentum_signal,
                        'volatility': vol_signal,
                        'market_regime': current_regime,
                        'atr': atr,
                        'volume_ratio': volume_ratio,
                        'bb_signal': bb_signal,
                        'rsi': rsi
                    },
                    'data_quality': {
                        'kama_valid': df['kama_20_valid'].iloc[i] and df['kama_50_valid'].iloc[i],
                        'rsi_valid': df['rsi_valid'].iloc[i],
                        'macd_valid': df['macd_valid'].iloc[i] and df['macd_signal_valid'].iloc[i],
                        'bb_valid': df['bollinger_upper_valid'].iloc[i] and df['bollinger_lower_valid'].iloc[i],
                        'derivatives_valid': df['price_first_derivative_valid'].iloc[i] and df['price_second_derivative_valid'].iloc[i],
                        'volatility_valid': df['parkinson_vol_valid'].iloc[i] and df['yang_zhang_vol_valid'].iloc[i],
                        'market_regime_valid': df['market_regime_valid'].iloc[i],
                        'atr_valid': df['atr_20_valid'].iloc[i]
                    },
                    'signal_strength': norm_signal_strength,
                    'signal_confidence': norm_signal_confidence
                }
                signals.append(signal)
                
                # Log first few signals for debugging
                if signal_count < max_signals_to_log:
                    logger.info(f"\nGenerated {signal['type']} signal at {signal['timestamp']}")
                    logger.info(f"Signal Strength: {norm_signal_strength:.4f}")
                    logger.info(f"Signal Confidence: {norm_signal_confidence:.4f}")
                    logger.info(f"Position Size: {position_size:.4f}")
                    logger.info(f"Market Regime: {'Trending' if current_regime == 1 else 'Ranging'}")
                    logger.info(f"Volatility Regime: {vol_regime}")
                    logger.info(f"Confirming Signals: {confirming_signals}")
                    logger.info(f"Stop Loss: ${signal['stop_loss']:.2f}")
                    logger.info(f"Take Profit: ${signal['take_profit']:.2f}")
                    signal_count += 1
            else:
                logger.debug(f"Index {i}: Skipped - not enough confirming signals ({confirming_signals})")
        else:
            logger.debug(f"Index {i}: Skipped - thresholds not met (|signal_strength|={abs(norm_signal_strength):.4f} vs {base_threshold:.4f}, signal_confidence={norm_signal_confidence:.4f} vs {confidence_threshold:.4f})")
    
    return signals

def calculate_correlation(series1: pd.Series, series2: pd.Series) -> float:
    """Calculate correlation between two series."""
    return series1.corr(series2)

def plot_results(results: Dict, feature_importance: Dict = None):
    """Plot test results for multiple coins."""
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot individual coin returns
    coin_returns = {coin: result['return'] for coin, result in results['individual_results'].items()}
    ax1.bar(coin_returns.keys(), [r * 100 for r in coin_returns.values()])
    ax1.set_title('Individual Coin Returns')
    ax1.set_xlabel('Coin')
    ax1.set_ylabel('Return (%)')
    ax1.grid(True)
    
    # Plot win rates
    win_rates = {coin: result['win_rate'] for coin, result in results['individual_results'].items()}
    ax2.bar(win_rates.keys(), [r * 100 for r in win_rates.values()])
    ax2.set_title('Individual Win Rates')
    ax2.set_xlabel('Coin')
    ax2.set_ylabel('Win Rate (%)')
    ax2.grid(True)
    
    # Plot number of trades
    num_trades = {coin: result['num_trades'] for coin, result in results['individual_results'].items()}
    ax3.bar(num_trades.keys(), num_trades.values())
    ax3.set_title('Number of Trades per Coin')
    ax3.set_xlabel('Coin')
    ax3.set_ylabel('Number of Trades')
    ax3.grid(True)
    
    # Plot portfolio values
    portfolio_values = {coin: result['portfolio_value'] for coin, result in results['individual_results'].items()}
    ax4.bar(portfolio_values.keys(), portfolio_values.values())
    ax4.set_title('Final Portfolio Value per Coin')
    ax4.set_xlabel('Coin')
    ax4.set_ylabel('Portfolio Value ($)')
    ax4.grid(True)
    
    # Add combined results as text
    plt.figtext(0.5, 0.01, 
                f"Combined Results:\n"
                f"Total Portfolio Value: ${results['total_portfolio_value']:,.2f}\n"
                f"Total Return: {results['total_return']*100:.2f}%\n"
                f"Total Trades: {results['total_trades']}\n"
                f"Combined Win Rate: {results['combined_win_rate']*100:.2f}%",
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()

def backtest(df: pd.DataFrame, signals: list, initial_capital: float = 100000.0) -> dict:
    """Run backtest with improved risk management and position sizing."""
    portfolio_value = initial_capital
    position = 0
    entry_price = 0
    trade_history = []
    portfolio_history = []
    open_trade = None
    open_trade_pnl = 0
    open_trade_entry_price = 0
    open_trade_position = 0
    open_trade_type = None
    open_trade_entry_time = None
    open_trade_partial_pnls = []
    
    # Initialize trade statistics
    trade_stats = {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_pnl': 0,
        'max_drawdown': 0,
        'max_drawdown_duration': 0,
        'current_drawdown': 0,
        'drawdown_start': None,
        'peak_portfolio': initial_capital,
        'consecutive_wins': 0,
        'consecutive_losses': 0,
        'max_consecutive_wins': 0,
        'max_consecutive_losses': 0,
        'avg_trade_duration': 0,
        'total_trade_duration': 0,
        'sharpe_ratio': 0,
        'sortino_ratio': 0,
        'calmar_ratio': 0
    }
    
    # Initialize risk management parameters
    max_position_size = 0.3  # Maximum 30% of portfolio per trade
    max_daily_loss = 0.05   # Maximum 5% daily loss
    max_drawdown_limit = 0.20  # Maximum 20% drawdown
    daily_pnl = 0
    last_trade_day = None
    
    # Initialize trailing stop tracking
    trailing_stop = None
    trailing_stop_distance = 0
    
    # Initialize profit taking levels
    profit_targets = []
    profit_target_weights = []
    
    # Initialize portfolio returns for ratio calculations
    portfolio_returns = []
    
    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        current_time = df.index[i]
        
        # Reset daily PnL tracking
        if last_trade_day is None or current_time.date() != last_trade_day:
            daily_pnl = 0
            last_trade_day = current_time.date()
        
        # Check for active trade
        if open_trade is not None:
            # Update trailing stop if in profit
            if trailing_stop is not None:
                if open_trade_type == 'BUY':
                    new_stop = current_price * (1 - trailing_stop_distance)
                    if new_stop > trailing_stop:
                        trailing_stop = new_stop
                else:  # SELL
                    new_stop = current_price * (1 + trailing_stop_distance)
                    if new_stop < trailing_stop:
                        trailing_stop = new_stop
            
            # Check stop loss (including trailing stop)
            stop_price = trailing_stop if trailing_stop is not None else open_trade['stop_loss']
            if (open_trade_type == 'BUY' and current_price <= stop_price) or \
               (open_trade_type == 'SELL' and current_price >= stop_price):
                # Close position at stop loss
                pnl = (current_price - open_trade_entry_price) * open_trade_position if open_trade_type == 'BUY' else \
                      (open_trade_entry_price - current_price) * open_trade_position
                open_trade_pnl += pnl
                trade_stats['total_trades'] += 1
                trade_stats['total_pnl'] += open_trade_pnl
                trade_duration = (current_time - open_trade_entry_time).total_seconds() / 3600
                trade_stats['total_trade_duration'] += trade_duration
                trade_history.append({
                    'entry_time': open_trade_entry_time,
                    'exit_time': current_time,
                    'type': open_trade_type,
                    'entry_price': open_trade_entry_price,
                    'exit_price': current_price,
                    'position_size': open_trade_position,
                    'pnl': open_trade_pnl,
                    'exit_reason': 'stop_loss',
                    'duration': trade_duration,
                    'market_regime': df['market_regime'].iloc[i],
                    'volatility': df['volatility'].iloc[i]
                })
                if open_trade_pnl > 0:
                    trade_stats['winning_trades'] += 1
                else:
                    trade_stats['losing_trades'] += 1
                open_trade = None
                open_trade_pnl = 0
                open_trade_entry_price = 0
                open_trade_position = 0
                open_trade_type = None
                open_trade_entry_time = None
                open_trade_partial_pnls = []
                trailing_stop = None
                profit_targets = []
                profit_target_weights = []
            # Check take profit levels
            elif profit_targets:
                for target, weight in zip(profit_targets, profit_target_weights):
                    if (open_trade_type == 'BUY' and current_price >= target) or \
                       (open_trade_type == 'SELL' and current_price <= target):
                        # Partial profit taking
                        partial_position = open_trade_position * weight
                        pnl = (current_price - open_trade_entry_price) * partial_position if open_trade_type == 'BUY' else \
                              (open_trade_entry_price - current_price) * partial_position
                        open_trade_pnl += pnl
                        open_trade_partial_pnls.append(pnl)
                        open_trade_position -= partial_position
                        # Remove taken profit target
                        profit_targets.remove(target)
                        profit_target_weights.remove(weight)
                        # If all profit targets hit, close remaining position
                        if not profit_targets:
                            pnl = (current_price - open_trade_entry_price) * open_trade_position if open_trade_type == 'BUY' else \
                                  (open_trade_entry_price - current_price) * open_trade_position
                            open_trade_pnl += pnl
                            trade_stats['total_trades'] += 1
                            trade_stats['total_pnl'] += open_trade_pnl
                            trade_duration = (current_time - open_trade_entry_time).total_seconds() / 3600
                            trade_stats['total_trade_duration'] += trade_duration
                            trade_history.append({
                                'entry_time': open_trade_entry_time,
                                'exit_time': current_time,
                                'type': open_trade_type,
                                'entry_price': open_trade_entry_price,
                                'exit_price': current_price,
                                'position_size': open_trade_position,
                                'pnl': open_trade_pnl,
                                'exit_reason': 'final_take_profit',
                                'duration': trade_duration,
                                'market_regime': df['market_regime'].iloc[i],
                                'volatility': df['volatility'].iloc[i]
                            })
                            if open_trade_pnl > 0:
                                trade_stats['winning_trades'] += 1
                            else:
                                trade_stats['losing_trades'] += 1
                            open_trade = None
                            open_trade_pnl = 0
                            open_trade_entry_price = 0
                            open_trade_position = 0
                            open_trade_type = None
                            open_trade_entry_time = None
                            open_trade_partial_pnls = []
                            trailing_stop = None
                            profit_targets = []
                            profit_target_weights = []
                        break
        
        # Check for new signals
        for signal in signals:
            if signal['timestamp'] == current_time:
                # Skip if we have an active position
                if position != 0:
                    continue
                
                # Check risk limits
                if daily_pnl <= -max_daily_loss * portfolio_value:
                    logger.info(f"Daily loss limit reached at {current_time}")
                    continue
                
                if trade_stats['current_drawdown'] >= max_drawdown_limit:
                    logger.info(f"Maximum drawdown limit reached at {current_time}")
                    continue
                
                # Get current market conditions
                current_volatility = df['volatility'].iloc[i]
                current_regime = df['market_regime'].iloc[i]
                current_trend_strength = df['trend_strength'].iloc[i]
                
                # Calculate position size with enhanced risk management
                base_size = signal['size']
                
                # Adjust for volatility
                vol_adjustment = 1.0 / (1.0 + current_volatility * 10)
                
                # Adjust for market regime
                regime_multiplier = {
                    'trending': 1.2,
                    'volatile': 0.6,
                    'ranging': 0.8
                }.get(current_regime, 1.0)
                
                # Adjust for trend strength
                trend_multiplier = min(1.0 + current_trend_strength, 1.5)
                
                # Adjust for recent performance
                performance_multiplier = 1.0
                if trade_stats['consecutive_losses'] >= 3:
                    performance_multiplier = 0.5
                elif trade_stats['consecutive_wins'] >= 3:
                    performance_multiplier = 1.2
                
                # Calculate final position size
                position_size = base_size * vol_adjustment * regime_multiplier * trend_multiplier * performance_multiplier
                position_size = min(position_size, max_position_size)
                
                # Calculate stop loss and take profit levels
                atr = df['atr_20'].iloc[i] if 'atr_20' in df.columns else current_volatility * current_price
                stop_distance = atr * (2.0 if current_regime == 'volatile' else 1.5)
                take_profit_distance = stop_distance * (2.5 if current_regime == 'trending' else 1.5)
                
                # Execute trade
                position = position_size * (1 if signal['type'] == 'BUY' else -1)
                entry_price = current_price
                current_trade = signal
                
                # Set up trailing stop
                trailing_stop_distance = stop_distance / current_price
                trailing_stop = current_price * (1 - trailing_stop_distance) if signal['type'] == 'BUY' else \
                               current_price * (1 + trailing_stop_distance)
                
                # Set up profit targets (3 levels)
                total_distance = take_profit_distance
                profit_targets = [
                    entry_price + total_distance * 0.5,  # 50% of range
                    entry_price + total_distance * 0.75,  # 75% of range
                    entry_price + total_distance  # Full target
                ]
                profit_target_weights = [0.3, 0.3, 0.4]  # Position size to close at each target
                
                logger.info(f"\nExecuting {signal['type']} trade at {current_time}")
                logger.info(f"Entry Price: ${entry_price:.2f}")
                logger.info(f"Position Size: {abs(position):.4f}")
                logger.info(f"Stop Loss: ${trailing_stop:.2f}")
                logger.info(f"Take Profit Levels: {[f'${p:.2f}' for p in profit_targets]}")
                logger.info(f"Market Regime: {current_regime}")
                logger.info(f"Volatility: {current_volatility:.4f}")
                logger.info(f"Trend Strength: {current_trend_strength:.4f}")
        
        # Update portfolio value
        if position != 0:
            portfolio_value = initial_capital + (current_price - entry_price) * position
        
        # Calculate daily return
        daily_return = (portfolio_value / initial_capital) - 1
        portfolio_returns.append(daily_return)
        
        # Update drawdown tracking
        if portfolio_value > trade_stats['peak_portfolio']:
            trade_stats['peak_portfolio'] = portfolio_value
            trade_stats['current_drawdown'] = 0
            trade_stats['drawdown_start'] = None
        else:
            current_drawdown = (trade_stats['peak_portfolio'] - portfolio_value) / trade_stats['peak_portfolio']
            trade_stats['current_drawdown'] = current_drawdown
            if trade_stats['drawdown_start'] is None:
                trade_stats['drawdown_start'] = current_time
            if current_drawdown > trade_stats['max_drawdown']:
                trade_stats['max_drawdown'] = current_drawdown
                trade_stats['max_drawdown_duration'] = (current_time - trade_stats['drawdown_start']).total_seconds() / 3600
        
        portfolio_history.append({
            'timestamp': current_time,
            'portfolio_value': portfolio_value,
            'position': position,
            'price': current_price,
            'daily_return': daily_return
        })
    
    # Calculate performance ratios
    if len(portfolio_returns) > 0:
        returns_series = pd.Series(portfolio_returns)
        risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
        
        # Sharpe Ratio
        excess_returns = returns_series - risk_free_rate
        trade_stats['sharpe_ratio'] = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
        
        # Sortino Ratio
        downside_returns = returns_series[returns_series < 0]
        trade_stats['sortino_ratio'] = np.sqrt(252) * excess_returns.mean() / downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Calmar Ratio
        trade_stats['calmar_ratio'] = (returns_series.mean() * 252) / trade_stats['max_drawdown'] if trade_stats['max_drawdown'] != 0 else 0
    
    # Calculate final statistics
    if trade_stats['total_trades'] > 0:
        win_rate = trade_stats['winning_trades'] / trade_stats['total_trades']
        avg_win = trade_stats['total_pnl'] / trade_stats['winning_trades'] if trade_stats['winning_trades'] > 0 else 0
        avg_loss = trade_stats['total_pnl'] / trade_stats['losing_trades'] if trade_stats['losing_trades'] > 0 else 0
        avg_trade_duration = trade_stats['total_trade_duration'] / trade_stats['total_trades']
    else:
        win_rate = avg_win = avg_loss = avg_trade_duration = 0
    
    return {
        'initial_capital': initial_capital,
        'final_portfolio_value': portfolio_value,
        'total_return': (portfolio_value - initial_capital) / initial_capital,
        'total_trades': trade_stats['total_trades'],
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_pnl': trade_stats['total_pnl'],
        'max_drawdown': trade_stats['max_drawdown'],
        'max_drawdown_duration': trade_stats['max_drawdown_duration'],
        'max_consecutive_wins': trade_stats['max_consecutive_wins'],
        'max_consecutive_losses': trade_stats['max_consecutive_losses'],
        'avg_trade_duration': avg_trade_duration,
        'sharpe_ratio': trade_stats['sharpe_ratio'],
        'sortino_ratio': trade_stats['sortino_ratio'],
        'calmar_ratio': trade_stats['calmar_ratio'],
        'trade_history': trade_history,
        'portfolio_history': portfolio_history
    }

def add_validation_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add validation flags for all indicators."""
    try:
        # Create a copy to avoid fragmentation
        df = df.copy()
        
        # Get all indicator columns (excluding price and volume data)
        indicator_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Create validation flags dictionary
        valid_flags = {}
        for col in indicator_columns:
            if col not in df.columns:
                logger.warning(f"Indicator column '{col}' not found in DataFrame. Skipping validation flag.")
                continue
                
            col_data = df[col]
            logger.debug(f"Processing indicator column: {col}, type: {type(col_data)}, shape: {getattr(col_data, 'shape', None)}")
            
            # Skip if column is not a Series or has wrong dimensions
            if not isinstance(col_data, pd.Series) or col_data.ndim != 1:
                logger.warning(f"Skipping validation flag for column '{col}' due to non-1D shape or type.")
                continue
            
            # Create validation flag name
            flag_name = f'{col}_valid'
            
            # Drop existing flag if present
            if flag_name in df.columns:
                logger.warning(f"Dropping existing column '{flag_name}' before adding new validation flag.")
                df = df.drop(columns=[flag_name])
            
            # Create validation flag based on data quality
            valid_flags[flag_name] = col_data.notna() & (col_data != np.inf) & (col_data != -np.inf)
            
            # Log validation statistics
            valid_count = valid_flags[flag_name].sum()
            total_count = len(valid_flags[flag_name])
            valid_percentage = (valid_count / total_count) * 100
            logger.debug(f"Validation stats for {col}: {valid_count}/{total_count} valid ({valid_percentage:.1f}%)")
        
        # Add all validation flags at once using pd.concat
        if valid_flags:
            df = pd.concat([df, pd.DataFrame(valid_flags, index=df.index)], axis=1)
            
            # Check for duplicate columns
            duplicates = df.columns[df.columns.duplicated()].unique()
            if len(duplicates) > 0:
                logger.error(f"Duplicate columns found after adding validation flags: {duplicates}")
                # Keep only the first occurrence of each duplicate
                df = df.loc[:, ~df.columns.duplicated()]
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding validation flags: {str(e)}")
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

def run_strategy(coin: str, timeframe: str = '1d', days: int = 365):
    """
    Run the trading strategy for any cryptocurrency and timeframe.
    
    Args:
        coin (str): The cryptocurrency symbol (e.g., 'bitcoin', 'ethereum')
        timeframe (str): The timeframe for analysis ('1d', '4h', '1h', etc.)
        days (int): Number of days of historical data to analyze
    """
    logger.info(f"Starting {coin} strategy test with {timeframe} timeframe...")
    collector = DataCollector()
    try:
        # Fetch cryptocurrency data
        logger.info(f"Fetching {coin} OHLCV data...")
        df = collector.collect_crypto_ohlcv_data(coin, interval=timeframe, days=days)
        if df is None or df.empty:
            logger.error(f"Failed to fetch {coin} data.")
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
        df = df.copy()
        
        # Add volatility-based position sizing
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        df['position_size'] = 1.0 / (1.0 + df['volatility'] * 10)
        
        # Add trend strength indicator
        df['trend_strength'] = abs(df['close'].pct_change(20)) / df['volatility']
        
        # Add market regime detection
        df['market_regime'] = np.where(df['trend_strength'] > 1.5, 'trending',
                                     np.where(df['volatility'] > df['volatility'].rolling(20).mean(), 'volatile', 'ranging'))
        
        # Calculate indicators
        df['kama_20'] = advanced.calculate_kama(df['close'], period=20)
        df['kama_50'] = advanced.calculate_kama(df['close'], period=50)
        df['log_returns'] = advanced.calculate_log_returns(df['close'])
        first_der, second_der = advanced.calculate_derivatives(df['close'])
        df['price_first_derivative'] = first_der
        df['price_second_derivative'] = second_der
        df['parkinson_vol'] = advanced.calculate_parkinson_volatility(df['high'], df['low'])
        df['yang_zhang_vol'] = advanced.calculate_yang_zhang_volatility(
            df['open'], df['high'], df['low'], df['close']
        )
        df['displacement'] = advanced.detect_displacement(df['high'], df['low'], df['close'])
        df['gaps'] = advanced.detect_gaps(df['open'], df['close'])

        # Add validation flags
        df = add_validation_flags(df)

        # Generate trading signals with enhanced risk management
        logger.info("Generating trading signals...")
        signals = simulate_trading_signals(df, {})

        # Run backtest with dynamic position sizing
        logger.info("Running backtest...")
        results = backtest(df, signals, initial_capital=100000)

        # Print detailed results
        logger.info(f"\n{coin.upper()} Backtest Results ({timeframe}):")
        logger.info(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
        logger.info(f"Total Return: {results['total_return']*100:.2f}%")
        logger.info(f"Number of Trades: {results['total_trades']}")
        
        if results['total_trades'] > 0:
            winning_trades = [t for t in results['trade_history'] if t['pnl'] > 0]
            win_rate = len(winning_trades) / results['total_trades']
            total_pnl = results['total_pnl']
            avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
            losing_trades = [t for t in results['trade_history'] if t['pnl'] <= 0]
            avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            logger.info(f"Win Rate: {win_rate:.2%}")
            logger.info(f"Total PnL: ${total_pnl:,.2f}")
            logger.info(f"Average Win: ${avg_win:,.2f}")
            logger.info(f"Average Loss: ${avg_loss:,.2f}")
            logger.info(f"Profit Factor: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "Profit Factor: N/A")
            logger.info(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
            logger.info(f"Max Drawdown Duration: {results['max_drawdown_duration']:.1f} hours")
            
            # Plot results
            logger.info("Plotting results...")
            plot_results({
                'individual_results': {
                    coin.upper(): {
                        'return': results['total_return'],
                        'win_rate': win_rate,
                        'num_trades': results['total_trades'],
                        'portfolio_value': results['final_portfolio_value']
                    }
                },
                'total_portfolio_value': results['final_portfolio_value'],
                'total_return': results['total_return'],
                'total_trades': results['total_trades'],
                'combined_win_rate': win_rate
            })
    except Exception as e:
        logger.error(f"Error during {coin} strategy: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run crypto trading strategy for any coin and timeframe.")
    parser.add_argument("coin", nargs="?", default="ethereum", help="Coin name (e.g. ethereum, bitcoin, solana)")
    parser.add_argument("timeframe", nargs="?", default="1d", help="Timeframe (e.g. 1d, 4h, 1h)")
    parser.add_argument("days", nargs="?", type=int, default=365, help="Number of days of historical data")
    args = parser.parse_args()

    run_strategy(args.coin, args.timeframe, args.days) 