"""
Trading Pipeline Module
- Integrates data collection, feature engineering, and signal generation
- Implements a complete trading pipeline for crypto assets
- Handles data preprocessing, feature generation, and trading decisions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import time

from .data_collection import DataCollector
from .feature_engineering import FeatureEngineer
from .technical_analysis import TechnicalAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingPipeline:
    """Trading pipeline for crypto assets."""
    
    def __init__(self, api_key: str):
        """
        Initialize the trading pipeline.
        
        Args:
            api_key: API key for data collection
        """
        self.collector = DataCollector(api_key=api_key)
        self.engineer = FeatureEngineer()
        self.analyzer = TechnicalAnalyzer()
        self.logger = logging.getLogger(__name__)
        
    def fetch_and_prepare_data(self, coin_id: str, days: int = 60) -> pd.DataFrame:
        """
        Fetch and prepare data for a given coin.
        
        Args:
            coin_id: CoinGecko coin ID
            days: Number of days of historical data
            
        Returns:
            DataFrame with prepared data
        """
        try:
            # Fetch OHLCV data
            df = self.collector.collect_crypto_ohlcv_data(coin_id=coin_id, days=days)
            if df is None or df.empty:
                raise ValueError(f"No data available for {coin_id}")
                
            # Generate features
            features = self.engineer.generate_features(df)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing data for {coin_id}: {str(e)}")
            return pd.DataFrame()
    
    def generate_trading_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signals based on technical analysis.
        
        Args:
            data: DataFrame with features
            
        Returns:
            Dictionary of signal values (-1 to 1)
        """
        try:
            # Generate technical signals
            signals = self.analyzer.generate_signals(data)
            
            # Calculate combined signal
            if signals:
                combined_signal = np.mean(list(signals.values()))
            else:
                combined_signal = 0
                
            return {
                'individual_signals': signals,
                'combined_signal': combined_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return {'individual_signals': {}, 'combined_signal': 0}
    
    def execute_trade(self, coin_id: str, signal: float, position: float = 0.0) -> Dict:
        """
        Execute a trade based on the signal (simulated for now).
        
        Args:
            coin_id: CoinGecko coin ID
            signal: Trading signal (-1 to 1)
            position: Current position size
            
        Returns:
            Dictionary with trade details
        """
        try:
            # Get current price
            data = self.collector.collect_crypto_ohlcv_data(coin_id=coin_id, days=1)
            current_price = data['close'].iloc[-1]
            
            # Determine trade action
            if signal > 0.5 and position <= 0:  # Strong buy signal
                action = 'BUY'
                size = 1.0  # Full position
            elif signal < -0.5 and position >= 0:  # Strong sell signal
                action = 'SELL'
                size = 1.0  # Full position
            else:
                action = 'HOLD'
                size = 0.0
                
            return {
                'timestamp': datetime.now(),
                'coin_id': coin_id,
                'action': action,
                'size': size,
                'price': current_price,
                'signal': signal
            }
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {coin_id}: {str(e)}")
            return {}
    
    def run_pipeline(self, coin_id: str, days: int = 60) -> Dict:
        """
        Run the complete trading pipeline for a coin.
        
        Args:
            coin_id: CoinGecko coin ID
            days: Number of days of historical data
            
        Returns:
            Dictionary with pipeline results
        """
        try:
            # Fetch and prepare data
            data = self.fetch_and_prepare_data(coin_id, days)
            if data.empty:
                return {'error': f'No data available for {coin_id}'}
                
            # Generate signals
            signals = self.generate_trading_signals(data)
            
            # Execute trade
            trade = self.execute_trade(coin_id, signals['combined_signal'])
            
            return {
                'data': data,
                'signals': signals,
                'trade': trade
            }
            
        except Exception as e:
            self.logger.error(f"Error running pipeline for {coin_id}: {str(e)}")
            return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = TradingPipeline(api_key="your_api_key")
    
    # Run pipeline for a coin
    results = pipeline.run_pipeline('bitcoin', days=60)
    
    # Print results
    if 'error' in results:
        print(f"Error: {results['error']}")
    else:
        print("\nTrading Signals:")
        print(results['signals'])
        print("\nTrade Details:")
        print(results['trade']) 