"""
Advanced Technical Indicators Module
- Kaufman's Adaptive Moving Average (KAMA)
- Log returns and derivatives
- Spread analysis
- Parkinson and Yang-Zhang volatility estimators
- Displacement detection
- Gap detection
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedIndicators:
    """Advanced technical indicators for trading strategies."""
    
    def __init__(self):
        """Initialize the advanced indicators calculator."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_kama(self, prices: pd.Series, period: int = 20, fast_ema: int = 2, slow_ema: int = 30) -> pd.Series:
        """
        Calculate Kaufman's Adaptive Moving Average (KAMA).
        
        Args:
            prices: Price series
            period: KAMA period
            fast_ema: Fast EMA period
            slow_ema: Slow EMA period
            
        Returns:
            Series with KAMA values
        """
        try:
            # Calculate efficiency ratio
            change = abs(prices - prices.shift(period))
            volatility = prices.diff().abs().rolling(window=period).sum()
            efficiency_ratio = change / volatility
            
            # Calculate smoothing constant
            sc_factor = (efficiency_ratio * (2.0/(fast_ema + 1) - 2.0/(slow_ema + 1)) + 2.0/(slow_ema + 1)) ** 2
            
            # Calculate KAMA
            kama = pd.Series(index=prices.index, dtype=float)
            kama.iloc[period-1] = prices.iloc[period-1]
            
            for i in range(period, len(prices)):
                kama.iloc[i] = kama.iloc[i-1] + sc_factor.iloc[i] * (prices.iloc[i] - kama.iloc[i-1])
            
            return kama
            
        except Exception as e:
            self.logger.error(f"Error calculating KAMA: {str(e)}")
            return pd.Series(index=prices.index)
    
    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns."""
        return np.log(prices / prices.shift(1))
    
    def calculate_derivatives(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate first and second derivatives of price.
        
        Returns:
            Tuple of (first_derivative, second_derivative)
        """
        # First derivative (rate of change)
        first_derivative = prices.diff()
        
        # Second derivative (acceleration)
        second_derivative = first_derivative.diff()
        
        return first_derivative, second_derivative
    
    def calculate_spread(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """Calculate spread between high and low prices."""
        return high - low
    
    def calculate_parkinson_volatility(self, high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate Parkinson volatility estimator.
        
        Parkinson volatility uses high-low range to estimate volatility,
        which is more efficient than close-to-close volatility.
        """
        try:
            # Calculate log of high-low range
            log_hl = np.log(high / low) ** 2
            
            # Calculate Parkinson volatility
            park_vol = np.sqrt(1 / (4 * window * np.log(2)) * log_hl.rolling(window=window).sum())
            
            return park_vol
            
        except Exception as e:
            self.logger.error(f"Error calculating Parkinson volatility: {str(e)}")
            return pd.Series(index=high.index)
    
    def calculate_yang_zhang_volatility(self, open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate Yang-Zhang volatility estimator.
        
        Yang-Zhang volatility combines overnight and intraday volatility
        to provide a more accurate estimate.
        """
        try:
            # Calculate overnight returns
            overnight_returns = np.log(open / close.shift(1))
            
            # Calculate intraday returns
            intraday_returns = np.log(close / open)
            
            # Calculate Parkinson volatility
            park_vol = self.calculate_parkinson_volatility(high, low, window)
            
            # Calculate Yang-Zhang volatility
            yz_vol = np.sqrt(
                (1 / window) * (
                    overnight_returns.rolling(window=window).var() +
                    intraday_returns.rolling(window=window).var() +
                    park_vol ** 2
                )
            )
            
            return yz_vol
            
        except Exception as e:
            self.logger.error(f"Error calculating Yang-Zhang volatility: {str(e)}")
            return pd.Series(index=open.index)
    
    def detect_displacement(self, high: pd.Series, low: pd.Series, close: pd.Series, threshold: float = 2.0) -> pd.Series:
        """
        Detect price displacement (big candles).
        
        Args:
            threshold: Number of standard deviations to consider a candle as big
            
        Returns:
            Series with displacement signals (1 for big up candle, -1 for big down candle, 0 otherwise)
        """
        try:
            # Calculate candle size
            candle_size = high - low
            
            # Calculate candle body
            body_size = abs(close - close.shift(1))
            
            # Calculate mean and std of candle sizes
            mean_size = candle_size.rolling(window=20).mean()
            std_size = candle_size.rolling(window=20).std()
            
            # Detect big candles
            is_big_candle = candle_size > (mean_size + threshold * std_size)
            
            # Determine direction
            displacement = pd.Series(0, index=close.index)
            displacement[is_big_candle & (close > close.shift(1))] = 1  # Big up candle
            displacement[is_big_candle & (close < close.shift(1))] = -1  # Big down candle
            
            return displacement
            
        except Exception as e:
            self.logger.error(f"Error detecting displacement: {str(e)}")
            return pd.Series(index=close.index)
    
    def detect_gaps(self, open: pd.Series, close: pd.Series, threshold: float = 0.01) -> pd.Series:
        """
        Detect price gaps.
        
        Args:
            threshold: Minimum gap size as a percentage
            
        Returns:
            Series with gap signals (1 for up gap, -1 for down gap, 0 otherwise)
        """
        try:
            # Calculate gap size
            gap_size = (open - close.shift(1)) / close.shift(1)
            
            # Detect gaps
            gaps = pd.Series(0, index=open.index)
            gaps[gap_size > threshold] = 1  # Up gap
            gaps[gap_size < -threshold] = -1  # Down gap
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"Error detecting gaps: {str(e)}")
            return pd.Series(index=open.index)
    
    def calculate_market_regime(self, prices: pd.Series, fast_kama_period: int = 10, slow_kama_period: int = 50) -> pd.Series:
        """
        Calculate market regime using KAMA difference.
        
        Returns:
            Series with market regime (1 for trending, 0 for ranging)
        """
        try:
            # Calculate fast and slow KAMA
            fast_kama = self.calculate_kama(prices, period=fast_kama_period)
            slow_kama = self.calculate_kama(prices, period=slow_kama_period)
            
            # Calculate KAMA difference
            kama_diff = (fast_kama - slow_kama) / slow_kama
            
            # Determine regime
            regime = pd.Series(0, index=prices.index)
            regime[abs(kama_diff) > 0.02] = 1  # Trending market
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Error calculating market regime: {str(e)}")
            return pd.Series(index=prices.index) 