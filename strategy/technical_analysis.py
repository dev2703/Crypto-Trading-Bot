"""
Technical Analysis Module
- Implements common technical indicators
- Generates trading signals based on technical analysis
- Includes RSI, MACD, Moving Averages, and Bollinger Bands
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Technical analysis for trading signals."""
    
    def __init__(self):
        """Initialize the technical analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: DataFrame with 'close' prices
            period: RSI period (default: 14)
            
        Returns:
            Series with RSI values
        """
        try:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(index=data.index)
    
    def calculate_macd(self, data: pd.DataFrame, 
                      fast_period: int = 12, 
                      slow_period: int = 26, 
                      signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: DataFrame with 'close' prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        try:
            # Calculate EMAs
            fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
            slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate Signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # Calculate Histogram
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            return pd.Series(), pd.Series(), pd.Series()
    
    def calculate_moving_averages(self, data: pd.DataFrame, 
                                short_period: int = 20, 
                                long_period: int = 50) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Simple Moving Averages.
        
        Args:
            data: DataFrame with 'close' prices
            short_period: Short-term MA period
            long_period: Long-term MA period
            
        Returns:
            Tuple of (Short MA, Long MA)
        """
        try:
            short_ma = data['close'].rolling(window=short_period).mean()
            long_ma = data['close'].rolling(window=long_period).mean()
            return short_ma, long_ma
            
        except Exception as e:
            self.logger.error(f"Error calculating Moving Averages: {str(e)}")
            return pd.Series(), pd.Series()
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, 
                                period: int = 20, 
                                std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with 'close' prices
            period: MA period
            std_dev: Number of standard deviations
            
        Returns:
            Tuple of (Middle Band, Upper Band, Lower Band)
        """
        try:
            middle_band = data['close'].rolling(window=period).mean()
            std = data['close'].rolling(window=period).std()
            
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            return middle_band, upper_band, lower_band
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return pd.Series(), pd.Series(), pd.Series()
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of signal values (-1 to 1)
        """
        try:
            signals = {}
            
            # RSI Signals
            rsi = self.calculate_rsi(data)
            if not rsi.empty:
                last_rsi = rsi.iloc[-1]
                if last_rsi > 70:
                    signals['rsi'] = -1  # Overbought
                elif last_rsi < 30:
                    signals['rsi'] = 1   # Oversold
                else:
                    signals['rsi'] = 0   # Neutral
            
            # MACD Signals
            macd_line, signal_line, _ = self.calculate_macd(data)
            if not macd_line.empty and not signal_line.empty:
                if macd_line.iloc[-1] > signal_line.iloc[-1]:
                    signals['macd'] = 1  # Bullish
                else:
                    signals['macd'] = -1 # Bearish
            
            # Moving Average Signals
            short_ma, long_ma = self.calculate_moving_averages(data)
            if not short_ma.empty and not long_ma.empty:
                if short_ma.iloc[-1] > long_ma.iloc[-1]:
                    signals['ma'] = 1    # Bullish
                else:
                    signals['ma'] = -1   # Bearish
            
            # Bollinger Bands Signals
            _, upper_band, lower_band = self.calculate_bollinger_bands(data)
            if not upper_band.empty and not lower_band.empty:
                last_close = data['close'].iloc[-1]
                if last_close > upper_band.iloc[-1]:
                    signals['bb'] = -1   # Overbought
                elif last_close < lower_band.iloc[-1]:
                    signals['bb'] = 1    # Oversold
                else:
                    signals['bb'] = 0    # Neutral
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return {}

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Initialize analyzer
    analyzer = TechnicalAnalyzer()
    
    # Generate signals
    signals = analyzer.generate_signals(data)
    print("\nTechnical Analysis Signals:")
    for indicator, signal in signals.items():
        print(f"{indicator.upper()}: {signal}") 