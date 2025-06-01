"""
Feature Engineering Strategy Module
- Defines the feature extraction methods
- Implements the feature engineering algorithm
- Generates feature reports
- Includes feature selection and validation tools
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

class FeatureEngineer:
    """Feature engineer for trading."""
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.features = pd.DataFrame()

    def extract_technical_indicators(self) -> pd.DataFrame:
        """Extract technical indicators from the data."""
        df = self.data.copy()
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        # MACD
        df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        self.features = df
        return df

    def select_features(self, target: pd.Series, k: int = 10) -> List[str]:
        """Select the top k features based on f-regression."""
        X = self.features.dropna()
        y = target[X.index]
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        return selected_features

    def scale_features(self) -> pd.DataFrame:
        """Scale the features using StandardScaler."""
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.features)
        return pd.DataFrame(scaled_features, columns=self.features.columns, index=self.features.index)

def generate_feature_report(features: pd.DataFrame) -> str:
    """Generate a feature report."""
    report = f"""
    Feature Report:
    -------------
    Number of Features: {len(features.columns)}
    Features: {', '.join(features.columns)}
    Data Types: {features.dtypes.to_dict()}
    Missing Values: {features.isnull().sum().to_dict()}
    """
    return report

# Example usage
if __name__ == "__main__":
    # Example: Load sample data and engineer features
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    engineer = FeatureEngineer(data)
    features = engineer.extract_technical_indicators()
    selected_features = engineer.select_features(data['close'])
    scaled_features = engineer.scale_features()
    report = generate_feature_report(features)
    print(report) 