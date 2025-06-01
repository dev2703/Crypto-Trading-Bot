"""
Feature Engineering Module
- Implements comprehensive feature generation for trading strategies
- Includes price-based, volume-based, volatility, market structure features
- Handles temporal features and feature selection
"""
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for trading strategies."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added price features
        """
        try:
            # SMA
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['sma_200'] = ta.sma(df['close'], length=200)
            
            # EMA
            df['ema_20'] = ta.ema(df['close'], length=20)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['ema_200'] = ta.ema(df['close'], length=200)
            
            # Bollinger Bands
            bollinger = ta.bbands(df['close'], length=20)
            df['bb_upper'] = bollinger['BBU_20_2.0']
            df['bb_middle'] = bollinger['BBM_20_2.0']
            df['bb_lower'] = bollinger['BBL_20_2.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # MACD
            macd = ta.macd(df['close'])
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding price features: {str(e)}")
            return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added volume features
        """
        try:
            # On-Balance Volume (OBV)
            df['obv'] = ta.obv(df['close'], df['volume'])
            
            # Volume Weighted Average Price (VWAP)
            df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            
            # Money Flow Index (MFI)
            df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
            
            # Volume SMA
            df['volume_sma_20'] = ta.sma(df['volume'], length=20)
            
            # Volume EMA
            df['volume_ema_20'] = ta.ema(df['volume'], length=20)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding volume features: {str(e)}")
            return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added volatility features
        """
        try:
            # Average True Range (ATR)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # Historical Volatility (20-day)
            df['returns'] = df['close'].pct_change()
            df['historical_volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Bollinger Band Width (already added in price features)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding volatility features: {str(e)}")
            return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with added temporal features
        """
        try:
            # Extract time components
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Session encoding (0: Asian, 1: European, 2: American)
            df['session'] = pd.cut(df['hour'], 
                                 bins=[0, 8, 16, 24], 
                                 labels=[0, 1, 2], 
                                 include_lowest=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding temporal features: {str(e)}")
            return df
    
    def add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market structure features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added market structure features
        """
        try:
            # Support and Resistance levels (using recent lows and highs)
            df['support'] = df['low'].rolling(window=20).min()
            df['resistance'] = df['high'].rolling(window=20).max()
            
            # Price position relative to support/resistance
            df['price_to_support'] = (df['close'] - df['support']) / df['support']
            df['price_to_resistance'] = (df['resistance'] - df['close']) / df['close']
            
            # Trend strength (ADX)
            df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding market structure features: {str(e)}")
            return df
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features for the dataset.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features
        """
        try:
            # Add all feature groups
            df = self.add_price_features(df)
            df = self.add_volume_features(df)
            df = self.add_volatility_features(df)
            df = self.add_temporal_features(df)
            df = self.add_market_structure_features(df)
            
            # Drop NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating features: {str(e)}")
            return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select the most important features using PCA.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            DataFrame with selected features
        """
        try:
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Scale features
            scaled_features = self.scaler.fit_transform(df[numeric_cols])
            
            # Apply PCA
            pca_features = self.pca.fit_transform(scaled_features)
            
            # Create DataFrame with PCA features
            pca_df = pd.DataFrame(
                pca_features,
                columns=[f'PC{i+1}' for i in range(pca_features.shape[1])],
                index=df.index
            )
            
            return pca_df
            
        except Exception as e:
            logger.error(f"Error selecting features: {str(e)}")
            return df
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, save_path: str = None):
        """
        Plot correlation heatmap of features.
        
        Args:
            df: DataFrame with features
            save_path: Path to save the plot
        """
        try:
            plt.figure(figsize=(12, 8))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Heatmap')
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting correlation heatmap: {str(e)}")
    
    def plot_rolling_volatility(self, df: pd.DataFrame, window: int = 20, save_path: str = None):
        """
        Plot rolling volatility.
        
        Args:
            df: DataFrame with returns
            window: Rolling window size
            save_path: Path to save the plot
        """
        try:
            plt.figure(figsize=(12, 6))
            volatility = df['returns'].rolling(window=window).std() * np.sqrt(252)
            volatility.plot()
            plt.title(f'{window}-Day Rolling Volatility')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting rolling volatility: {str(e)}")
    
    def plot_returns_distribution(self, df: pd.DataFrame, save_path: str = None):
        """
        Plot returns distribution.
        
        Args:
            df: DataFrame with returns
            save_path: Path to save the plot
        """
        try:
            plt.figure(figsize=(12, 6))
            sns.histplot(df['returns'], kde=True)
            plt.title('Returns Distribution')
            plt.xlabel('Returns')
            plt.ylabel('Frequency')
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting returns distribution: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, len(dates)),
        'high': np.random.normal(101, 1, len(dates)),
        'low': np.random.normal(99, 1, len(dates)),
        'close': np.random.normal(100, 1, len(dates)),
        'volume': np.random.normal(1000, 100, len(dates))
    }, index=dates)
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Generate features
    features = engineer.generate_features(data)
    
    # Select important features
    selected_features = engineer.select_features(features)
    
    # Plot visualizations
    engineer.plot_correlation_heatmap(features, 'correlation_heatmap.png')
    engineer.plot_rolling_volatility(features, save_path='rolling_volatility.png')
    engineer.plot_returns_distribution(features, save_path='returns_distribution.png') 