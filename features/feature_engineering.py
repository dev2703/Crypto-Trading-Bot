"""
Feature Engineering Module for Crypto Trading Bot
- Computes features from raw OHLCV and market data
- Modular and extensible for new features
"""
import pandas as pd
import numpy as np

def compute_returns(df: pd.DataFrame, price_col: str = 'close', period: int = 1) -> pd.Series:
    """Compute log returns for a given price column."""
    return np.log(df[price_col]).diff(period)

def compute_volatility(df: pd.DataFrame, price_col: str = 'close', window: int = 14) -> pd.Series:
    """Compute rolling volatility (standard deviation of log returns)."""
    returns = compute_returns(df, price_col)
    return returns.rolling(window=window).std()

def compute_moving_average(df: pd.DataFrame, price_col: str = 'close', window: int = 14) -> pd.Series:
    """Compute simple moving average."""
    return df[price_col].rolling(window=window).mean()

def compute_rsi(df: pd.DataFrame, price_col: str = 'close', window: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to the DataFrame."""
    df = df.copy()
    df['log_return'] = compute_returns(df)
    df['volatility_14'] = compute_volatility(df)
    df['ma_14'] = compute_moving_average(df)
    df['rsi_14'] = compute_rsi(df)
    # Add more features as needed
    return df

# Example usage (for testing or pipeline integration)
if __name__ == "__main__":
    # Example: Load data from CSV (replace with actual data source)
    df = pd.read_csv('sample_ohlcv.csv', parse_dates=['timestamp'])
    df = add_features(df)
    print(df.tail()) 