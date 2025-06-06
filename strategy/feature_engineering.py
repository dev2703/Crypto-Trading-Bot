"""
Feature Engineering Module for Crypto Trading
- Price-based features
- Volume-based features
- Volatility features
- Market structure features
- Temporal features
- Conservative NaN handling with data quality checks
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import talib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from scipy import stats
from .advanced_indicators import AdvancedIndicators

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
        self.feature_importance = {}
        self.min_periods = 5  # Minimum periods for rolling calculations
        self.max_nan_ratio = 0.1  # Maximum allowed ratio of NaN values
        self.data_quality_threshold = 0.95  # Minimum required data quality score
        self.advanced_indicators = AdvancedIndicators()

    def _check_data_quality(self, df: pd.DataFrame) -> Tuple[bool, float, Dict[str, float]]:
        """
        Check data quality and return quality metrics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_acceptable, quality_score, column_quality)
        """
        total_rows = len(df)
        column_quality = {}
        for col in df.columns:
            try:
                nan_ratio = df[col].isna().sum() / total_rows
                if not isinstance(nan_ratio, float) and hasattr(nan_ratio, '__len__'):
                    logger.warning(f"Skipping column {col} in data quality check due to non-scalar nan_ratio.")
                    continue
                column_quality[col] = float(1 - nan_ratio)
                # Check for suspicious patterns
                if col in ['open', 'high', 'low', 'close']:
                    price_std = df[col].std()
                    price_mean = df[col].mean()
                    z_scores = np.abs((df[col] - price_mean) / price_std)
                    anomaly_ratio = (z_scores > 3).mean()
                    column_quality[col] *= (1 - anomaly_ratio)
                elif col == 'volume':
                    volume_std = df[col].std()
                    volume_mean = df[col].mean()
                    z_scores = np.abs((df[col] - volume_mean) / volume_std)
                    anomaly_ratio = (z_scores > 3).mean()
                    column_quality[col] *= (1 - anomaly_ratio)
            except Exception as e:
                logger.warning(f"Skipping column {col} in data quality check due to error: {e}")
                continue
        # Only use float values for quality_score
        float_qualities = [v for v in column_quality.values() if isinstance(v, float)]
        quality_score = np.mean(float_qualities) if float_qualities else 0.0
        is_acceptable = quality_score >= self.data_quality_threshold
        return is_acceptable, quality_score, column_quality

    def _handle_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle NaN values conservatively with data quality checks.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with NaN values handled appropriately
        """
        logger.info("Checking data quality...")
        is_acceptable, quality_score, column_quality = self._check_data_quality(df)
        
        if not is_acceptable:
            logger.warning(f"Data quality score {quality_score:.2f} below threshold {self.data_quality_threshold}")
            logger.warning("Column quality scores:")
            for col, score in column_quality.items():
                logger.warning(f"{col}: {score:.2f}")
        
        df_clean = df.copy()
        
        # 1. Handle price data
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df_clean.columns:
                nan_ratio = df_clean[col].isna().sum() / len(df_clean)
                if nan_ratio > self.max_nan_ratio:
                    logger.warning(f"High NaN ratio ({nan_ratio:.2f}) in {col}, marking as invalid")
                    df_clean[col] = np.nan
                else:
                    # Only fill small gaps (up to 3 periods)
                    df_clean[col] = df_clean[col].fillna(method='ffill', limit=3)
        
        # 2. Handle volume data
        if 'volume' in df_clean.columns:
            nan_ratio = df_clean['volume'].isna().sum() / len(df_clean)
            if nan_ratio > self.max_nan_ratio:
                logger.warning(f"High NaN ratio ({nan_ratio:.2f}) in volume, marking as invalid")
                df_clean['volume'] = np.nan
            else:
                # Use 0 for missing volume (more conservative than median)
                df_clean['volume'] = df_clean['volume'].fillna(0)
        
        # 3. Handle technical indicators
        for col in df_clean.columns:
            if col not in price_cols and col != 'volume':
                nan_ratio = df_clean[col].isna().sum() / len(df_clean)
                if nan_ratio > self.max_nan_ratio:
                    logger.warning(f"High NaN ratio ({nan_ratio:.2f}) in {col}, marking as invalid")
                    df_clean[col] = np.nan
                else:
                    # Don't fill NaN values for indicators
                    # Instead, mark them as invalid for signal generation
                    pass
        
        return df_clean

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
        
        # Trend features - Using pandas rolling mean for reliability
        for window in self.window_sizes:
            features[f'sma_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
            features[f'ema_{window}'] = df['close'].ewm(span=window, min_periods=1).mean()
        
        # MACD
        try:
            macd = MACD(close=df['close'])
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            features['macd_diff'] = macd.macd_diff()
        except Exception as e:
            logger.warning(f"Error calculating MACD: {str(e)}")
            # Fallback to manual calculation
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            features['macd'] = exp1 - exp2
            features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
            features['macd_diff'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        try:
            bb = BollingerBands(close=df['close'])
            features['bb_high'] = bb.bollinger_hband()
            features['bb_low'] = bb.bollinger_lband()
            features['bb_mid'] = bb.bollinger_mavg()
            features['bb_width'] = (features['bb_high'] - features['bb_low']) / features['bb_mid']
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {str(e)}")
            # Fallback to manual calculation
            features['bb_mid'] = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            features['bb_high'] = features['bb_mid'] + (std * 2)
            features['bb_low'] = features['bb_mid'] - (std * 2)
            features['bb_width'] = (features['bb_high'] - features['bb_low']) / features['bb_mid']
        
        # Ensure all features are properly filled
        for key in features:
            if isinstance(features[key], pd.Series):
                features[key] = features[key].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
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
        
        # Volume indicators
        features['volume_ma'] = df['volume'].rolling(window=20).mean()
        features['volume_std'] = df['volume'].rolling(window=20).std()
        features['volume_ratio'] = df['volume'] / features['volume_ma']
        
        # VWAP
        vwap = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        )
        features['vwap'] = vwap.volume_weighted_average_price()
        
        # Volume profile
        features['volume_price_ma'] = df['volume'] * df['close']
        features['volume_price_ratio'] = features['volume_price_ma'] / features['volume_ma']
        
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
        
        # Historical volatility
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
        
        # Volatility ratios
        features['volatility_ratio'] = features['volatility_20'] / features['volatility_50']
        features['atr_ratio'] = features['atr_20'] / df['close']
        
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
        
        # Ensure sma_20 and sma_50 are present
        if 'sma_20' not in features:
            features['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        if 'sma_50' not in features:
            features['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        # Market regime (vectorized, no ambiguous boolean)
        features['trend_strength'] = abs(features['sma_20'] - features['sma_50']) / features['sma_50']
        # Use np.where for vectorized assignment
        features['market_regime'] = np.where(features['trend_strength'].values > 0.02, 'trending', 'ranging')
        
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
        features['month'] = df.index.month if hasattr(df.index, 'month') else 0
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

    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add validation flags for each feature (True if not NaN, False if NaN).
        Only assign Series to each *_valid column, and skip columns that are not present or are not Series.
        """
        df_valid = df.copy()
        for col in df.columns:
            # Only validate if col is a Series (not a DataFrame) and not a multi-index
            if col in df and isinstance(df[col], pd.Series) and df[col].ndim == 1:
                df_valid[f'{col}_valid'] = ~df[col].isna()
        return df_valid

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features with data quality checks."""
        try:
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # Add basic price features
            df = self._add_price_features(df)
            
            # Add volume features
            df = self._add_volume_features(df)
            
            # Add volatility features
            df = self._add_volatility_features(df)
            
            # Add market structure features
            df = self._add_market_structure_features(df)
            
            # Add temporal features
            df = self._add_temporal_features(df)
            
            # Add advanced indicators
            df = self._add_advanced_indicators(df)
            
            # Add data quality flags
            df = self._add_data_quality_flags(df)
            
            # Fill NaN values with appropriate methods
            df = self._fill_missing_values(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating features: {str(e)}")
            raise

    def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators."""
        try:
            # Calculate KAMA
            df['kama_20'] = self.advanced_indicators.calculate_kama(df['close'], period=20)
            df['kama_50'] = self.advanced_indicators.calculate_kama(df['close'], period=50)
            
            # Calculate log returns
            df['log_returns'] = self.advanced_indicators.calculate_log_returns(df['close'])
            
            # Calculate derivatives
            first_der, second_der = self.advanced_indicators.calculate_derivatives(df['close'])
            df['price_first_derivative'] = first_der
            df['price_second_derivative'] = second_der
            
            # Calculate spread
            df['price_spread'] = self.advanced_indicators.calculate_spread(df['high'], df['low'])
            
            # Calculate volatility estimators
            df['parkinson_vol'] = self.advanced_indicators.calculate_parkinson_volatility(df['high'], df['low'])
            df['yang_zhang_vol'] = self.advanced_indicators.calculate_yang_zhang_volatility(
                df['open'], df['high'], df['low'], df['close']
            )
            
            # Detect displacement and gaps
            df['displacement'] = self.advanced_indicators.detect_displacement(df['high'], df['low'], df['close'])
            df['gaps'] = self.advanced_indicators.detect_gaps(df['open'], df['close'])
            
            # Calculate market regime using KAMA
            df['market_regime'] = self.advanced_indicators.calculate_market_regime(df['close'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding advanced indicators: {str(e)}")
            return df

    def _add_data_quality_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add data quality flags for all features."""
        try:
            # Create a copy to avoid fragmentation
            df = df.copy()
            
            # Ensure index is unique
            if not df.index.is_unique:
                logger.warning("Duplicate index values found. Resetting index to ensure uniqueness.")
                df = df.reset_index().drop_duplicates(subset='index').set_index('index')
            
            # Get all feature columns (excluding price and volume data)
            feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            # Create quality flags dictionary
            quality_flags = {}
            for col in feature_columns:
                if col not in df.columns:
                    logger.warning(f"Feature column '{col}' not found in DataFrame. Skipping quality flag.")
                    continue
                    
                col_data = df[col]
                logger.debug(f"Processing feature column: {col}, type: {type(col_data)}, shape: {getattr(col_data, 'shape', None)}")
                
                # Skip if column is not a Series or has wrong dimensions
                if not isinstance(col_data, pd.Series) or col_data.ndim != 1:
                    logger.warning(f"Skipping quality flag for column '{col}' due to non-1D shape or type.")
                    continue
                
                # Create quality flag name
                flag_name = f'{col}_valid'
                
                # Drop existing flag if present
                if flag_name in df.columns:
                    logger.warning(f"Dropping existing column '{flag_name}' before adding new quality flag.")
                    df = df.drop(columns=[flag_name])
                
                # Create quality flag based on data quality
                quality_flags[flag_name] = (
                    col_data.notna() & 
                    (col_data != np.inf) & 
                    (col_data != -np.inf) &
                    (col_data != 0)  # Exclude zero values as they might indicate missing data
                )
                
                # Log quality statistics
                valid_count = quality_flags[flag_name].sum()
                total_count = len(quality_flags[flag_name])
                valid_percentage = (valid_count / total_count) * 100
                logger.debug(f"Quality stats for {col}: {valid_count}/{total_count} valid ({valid_percentage:.1f}%)")
            
            # Add all quality flags at once using pd.concat
            if quality_flags:
                df = pd.concat([df, pd.DataFrame(quality_flags, index=df.index)], axis=1)
                
                # Check for duplicate columns
                duplicates = df.columns[df.columns.duplicated()].unique()
                if len(duplicates) > 0:
                    logger.error(f"Duplicate columns found after adding quality flags: {duplicates}")
                    # Keep only the first occurrence of each duplicate
                    df = df.loc[:, ~df.columns.duplicated()]
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding data quality flags: {str(e)}")
            raise

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate methods."""
        try:
            # Forward fill for price and volume data
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = df[price_cols].fillna(method='ffill')
            
            # Fill volume with 0
            df['volume'] = df['volume'].fillna(0)
            
            # Fill technical indicators with appropriate defaults
            indicator_defaults = {
                'rsi': 50,
                'macd': 0,
                'macd_signal': 0,
                'stoch_k': 50,
                'stoch_d': 50,
                'adx': 25,
                'plus_di14': 25,
                'minus_di14': 25,
                'dx': 25,
                'kama_20': df['close'],
                'kama_50': df['close'],
                'log_returns': 0,
                'price_first_derivative': 0,
                'price_second_derivative': 0,
                'parkinson_vol': df['close'].pct_change().rolling(20).std(),
                'yang_zhang_vol': df['close'].pct_change().rolling(20).std(),
                'displacement': 0,
                'gaps': 0,
                'market_regime': 0
            }
            
            for col, default in indicator_defaults.items():
                if col in df.columns:
                    df[col] = df[col].fillna(default)
            
            return df
            
        except Exception as e:
            logger.error(f"Error filling missing values: {str(e)}")
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

    def calculate_feature_importance(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """Calculate feature importance scores."""
        # Calculate correlation with target
        correlations = df.corr()[target].abs()
        
        # Calculate mutual information
        from sklearn.feature_selection import mutual_info_regression
        mi_scores = mutual_info_regression(df.drop(columns=[target]), df[target])
        mi_scores = pd.Series(mi_scores, index=df.drop(columns=[target]).columns)
        
        # Combine scores
        importance = (correlations + mi_scores) / 2
        self.feature_importance = importance.to_dict()
        
        return self.feature_importance

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