"""
Feature Engineering Module for Crypto Trading
- Price-based features
- Volume-based features
- Volatility features
- Market structure features
- Temporal features
- No imputation to maintain data integrity
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import talib
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for crypto trading signals."""
    def __init__(self, window_sizes: List[int] = [5, 10, 20, 50, 100, 200]):
        """
        Args:
            window_sizes: List of window sizes for rolling features
            Using multiple timeframes: 5d (weekly), 10d (biweekly), 20d (monthly),
            50d (quarterly), 100d (half-yearly), 200d (yearly)
        """
        self.window_sizes = window_sizes
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.fitted = False

    def _safe_division(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Safely divide two series, handling division by zero."""
        result = np.where(b != 0, a / b, 0)
        result = np.where(np.isfinite(result), result, 0)
        return result

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features without imputation."""
        features = {}
        # Returns
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price momentum
        for window in self.window_sizes:
            features[f'momentum_{window}'] = df['close'].pct_change(window)
            features[f'roc_{window}'] = self._safe_division(
                df['close'] - df['close'].shift(window),
                df['close'].shift(window)
            )
            features[f'exp_momentum_{window}'] = df['close'].ewm(span=window, min_periods=1).mean().pct_change()
        
        # Price channels
        for window in self.window_sizes:
            features[f'donchian_high_{window}'] = df['high'].rolling(window, min_periods=1).max()
            features[f'donchian_low_{window}'] = df['low'].rolling(window, min_periods=1).min()
            features[f'donchian_mid_{window}'] = (features[f'donchian_high_{window}'] + features[f'donchian_low_{window}']) / 2
            channel_range = features[f'donchian_high_{window}'] - features[f'donchian_low_{window}']
            features[f'price_channel_pos_{window}'] = self._safe_division(
                df['close'] - features[f'donchian_low_{window}'],
                channel_range
            )
        
        # Price patterns
        features['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        features['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        features['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        
        return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features without imputation."""
        features = {}
        for window in self.window_sizes:
            features[f'volume_momentum_{window}'] = df['volume'].pct_change(window)
            features[f'volume_ma_{window}'] = df['volume'].rolling(window, min_periods=1).mean()
            features[f'volume_std_{window}'] = df['volume'].rolling(window, min_periods=1).std()
        
        features['volume_price_trend'] = df['volume'] * df['returns']
        features['volume_price_ratio'] = self._safe_division(df['volume'], df['close'])
        
        features['obv'] = talib.OBV(df['close'], df['volume'])
        features['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        features['adosc'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
        
        for window in self.window_sizes:
            features[f'volume_profile_{window}'] = df['volume'].rolling(window, min_periods=1).apply(
                lambda x: np.sum(x * (x > x.mean())) / np.sum(x) if np.sum(x) != 0 else 0
            )
        
        return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features without imputation."""
        features = {}
        for window in self.window_sizes:
            features[f'atr_{window}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=window)
            features[f'natr_{window}'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=window)
            features[f'hist_vol_{window}'] = df['returns'].rolling(window, min_periods=1).std() * np.sqrt(252)
            features[f'parkinson_vol_{window}'] = np.sqrt(
                (1.0 / (4.0 * np.log(2.0))) * ((np.log(df['high'] / df['low'])) ** 2)
            ).rolling(window, min_periods=1).mean() * np.sqrt(252)
            vol_mean = features[f'hist_vol_{window}'].rolling(window, min_periods=1).mean()
            features[f'vol_ratio_{window}'] = self._safe_division(features[f'hist_vol_{window}'], vol_mean)
        
        return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

    def _add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure features without imputation."""
        features = {}
        for window in self.window_sizes:
            features[f'pivot_{window}'] = (df['high'].rolling(window, min_periods=1).max() +
                                   df['low'].rolling(window, min_periods=1).min() +
                                   df['close'].rolling(window, min_periods=1).mean()) / 3
            features[f'support_{window}'] = 2 * features[f'pivot_{window}'] - df['high'].rolling(window, min_periods=1).max()
            features[f'resistance_{window}'] = 2 * features[f'pivot_{window}'] - df['low'].rolling(window, min_periods=1).min()
            features[f'price_to_support_{window}'] = self._safe_division(df['close'] - features[f'support_{window}'], df['close'])
            features[f'price_to_resistance_{window}'] = self._safe_division(df['close'] - features[f'resistance_{window}'], df['close'])
        
        for window in self.window_sizes:
            features[f'adx_{window}'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=window)
            features[f'dmi_plus_{window}'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=window)
            features[f'dmi_minus_{window}'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=window)
        
        for window in self.window_sizes:
            features[f'trend_strength_{window}'] = abs(df['close'].rolling(window, min_periods=1).mean() - 
                                               df['close'].rolling(window*2, min_periods=1).mean())
            features[f'range_ratio_{window}'] = self._safe_division(
                df['high'].rolling(window, min_periods=1).max() - df['low'].rolling(window, min_periods=1).min(),
                df['close'].rolling(window, min_periods=1).mean()
            )
        
        return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        features = {}
        features['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
        features['day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int) if 'day_of_week' in features else 0
        features['is_london_session'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int) if 'hour' in features else 0
        features['is_ny_session'] = ((features['hour'] >= 13) & (features['hour'] < 21)).astype(int) if 'hour' in features else 0
        features['is_tokyo_session'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int) if 'hour' in features else 0
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24) if 'hour' in features else 0
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24) if 'hour' in features else 0
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7) if 'day_of_week' in features else 0
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7) if 'day_of_week' in features else 0
        return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

    def _preprocess_for_pca(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for PCA without imputation."""
        # Remove columns with all NaN values
        df = df.dropna(axis=1, how='all')
        
        # Remove rows with any NaN values
        df = df.dropna(axis=0, how='any')
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Remove zero variance columns
        zero_var_cols = [col for col in numeric_cols if df[col].var() == 0]
        if zero_var_cols:
            logger.warning(f"Removing zero variance columns: {zero_var_cols}")
            df = df.drop(columns=zero_var_cols)
        
        # Recalculate numeric columns after dropping
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(df) == 0 or len(numeric_cols) == 0:
            logger.error("No data left after dropping NaNs and zero-variance columns.")
            return df
        
        # Scale features
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features without imputation."""
        logger.info("Starting feature generation...")
        
        # Add features
        df = self._add_price_features(df)
        logger.info("Added price features")
        
        df = self._add_volume_features(df)
        logger.info("Added volume features")
        
        df = self._add_volatility_features(df)
        logger.info("Added volatility features")
        
        df = self._add_market_structure_features(df)
        logger.info("Added market structure features")
        
        df = self._add_temporal_features(df)
        logger.info("Added temporal features")
        
        # Preprocess for PCA
        df = self._preprocess_for_pca(df)
        logger.info("Preprocessed data for PCA")
        
        # Apply PCA for feature reduction
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            try:
                pca_features = self.pca.fit_transform(df[numeric_cols])
                pca_cols = [f'pca_{i+1}' for i in range(pca_features.shape[1])]
                df_pca = pd.DataFrame(pca_features, index=df.index, columns=pca_cols)
                df = pd.concat([df, df_pca], axis=1)
                logger.info(f"Applied PCA, reduced to {len(pca_cols)} components")
            except Exception as e:
                logger.error(f"Error in PCA: {str(e)}")
                logger.info("Skipping PCA due to error")
        else:
            logger.warning("No numeric columns available for PCA")
        
        self.fitted = True
        return df

    def plot_correlation_heatmap(self, df: pd.DataFrame, save_path: str = 'correlation_heatmap.png'):
        """Plot correlation heatmap of features."""
        plt.figure(figsize=(20, 16))
        sns.heatmap(df.corr(), cmap='coolwarm', center=0, annot=False)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_rolling_volatility(self, df: pd.DataFrame, window: int = 20, save_path: str = 'rolling_volatility.png'):
        """Plot rolling volatility."""
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[f'hist_vol_{window}'])
        plt.title(f'{window}-Day Rolling Volatility')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def plot_returns_distribution(self, df: pd.DataFrame, save_path: str = 'returns_distribution.png'):
        """Plot returns distribution."""
        plt.figure(figsize=(10, 6))
        sns.histplot(df['returns'], kde=True)
        plt.title('Returns Distribution')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.savefig(save_path)
        plt.close()

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
    
    # Plot visualizations
    engineer.plot_correlation_heatmap(features, 'correlation_heatmap.png')
    engineer.plot_rolling_volatility(features, save_path='rolling_volatility.png')
    engineer.plot_returns_distribution(features, save_path='returns_distribution.png') 