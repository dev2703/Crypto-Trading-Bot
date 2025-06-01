"""
Data Collection Strategy Module
- Defines the data sources and collection methods
- Implements the data collection algorithm
- Generates data reports
- Includes data validation and cleaning tools
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import requests
import json
import time

class DataCollector:
    """Data collector for trading."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def collect_ohlcv_data(self, symbol: str, interval: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Collect OHLCV data from Alpha Vantage API."""
        params = {
            "function": "TIME_SERIES_INTRADAY" if interval == "1min" else "TIME_SERIES_DAILY",
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.text}")
        data = response.json()
        if "Error Message" in data:
            raise Exception(f"API Error: {data['Error Message']}")
        if "Note" in data:
            print(f"API Note: {data['Note']}")  # Handle rate limit warnings
        time_series_key = "Time Series (1min)" if interval == "1min" else "Time Series (Daily)"
        if time_series_key not in data:
            raise Exception(f"Unexpected API response: {data}")
        df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()  # Ensure the index is sorted
        df.columns = [col.split(".")[1] for col in df.columns]
        df = df.astype(float)
        # Only slice if the requested dates are within the available index
        available_start = df.index.min()
        available_end = df.index.max()
        req_start = pd.to_datetime(start_time)
        req_end = pd.to_datetime(end_time)
        # Adjust requested range to available range
        slice_start = max(available_start, req_start)
        slice_end = min(available_end, req_end)
        df = df.loc[slice_start:slice_end]
        return df

    def collect_order_book_data(self, symbol: str) -> Dict:
        """Collect order book data from Alpha Vantage API."""
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.text}")
        data = response.json()
        if "Error Message" in data:
            raise Exception(f"API Error: {data['Error Message']}")
        if "Note" in data:
            print(f"API Note: {data['Note']}")  # Handle rate limit warnings
        return data

    def collect_trade_data(self, symbol: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Collect trade data from Alpha Vantage API."""
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": "1min",
            "apikey": self.api_key,
            "outputsize": "full"
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.text}")
        data = response.json()
        if "Error Message" in data:
            raise Exception(f"API Error: {data['Error Message']}")
        if "Note" in data:
            print(f"API Note: {data['Note']}")  # Handle rate limit warnings
        time_series_key = "Time Series (1min)"
        if time_series_key not in data:
            raise Exception(f"Unexpected API response: {data}")
        df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
        df.index = pd.to_datetime(df.index)
        df.columns = [col.split(".")[1] for col in df.columns]
        df = df.astype(float)
        df = df.loc[start_time:end_time]
        return df

    def collect_crypto_ohlcv_data(self, coin_id: str, days: int = 60) -> pd.DataFrame:
        """Collect daily OHLCV data from CoinGecko API for a cryptocurrency."""
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily"
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.text}")
        data = response.json()
        if "prices" not in data or "total_volumes" not in data:
            raise Exception(f"Unexpected API response: {data}")
        # Extract prices and volumes
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
        # Merge prices and volumes
        df = pd.merge(prices, volumes, on="timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()
        # Compute OHLC from close prices (CoinGecko only provides close prices)
        df["open"] = df["close"].shift(1)
        df["high"] = df[["open", "close"]].max(axis=1)
        df["low"] = df[["open", "close"]].min(axis=1)
        df = df.dropna()
        return df

def validate_data(df: pd.DataFrame) -> bool:
    """Validate the collected data."""
    if df.empty:
        return False
    if df.isnull().any().any():
        return False
    if (df < 0).any().any():
        return False
    return True

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the collected data."""
    df = df.dropna()
    df = df[df > 0]
    return df

def generate_data_report(df: pd.DataFrame) -> str:
    """Generate a data report."""
    report = f"""
    Data Report:
    -----------
    Number of Rows: {len(df)}
    Number of Columns: {len(df.columns)}
    Columns: {', '.join(df.columns)}
    Data Types: {df.dtypes.to_dict()}
    Missing Values: {df.isnull().sum().to_dict()}
    """
    return report

# Example usage
if __name__ == "__main__":
    # Example: Collect and validate data
    collector = DataCollector(api_key="ARUSM71L0ISPNU4F")
    ohlcv_data = collector.collect_ohlcv_data("AAPL", "1min", "2023-01-01", "2023-01-31")
    if validate_data(ohlcv_data):
        ohlcv_data = clean_data(ohlcv_data)
        report = generate_data_report(ohlcv_data)
        print(report)
    else:
        print("Data validation failed.") 