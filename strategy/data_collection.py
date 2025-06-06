"""
Data Collection Strategy Module
- Defines the data sources and collection methods
- Implements the data collection algorithm
- Generates data reports
- Includes data validation and cleaning tools
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import requests
import json
import time
import yfinance as yf
import logging
from datetime import datetime, timedelta
import ccxt
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
import subprocess
import os
import tempfile
from binance.client import Client
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class DataCollector:
    """Data collection strategy class."""
    
    def __init__(self):
        """Initialize data collector with API keys."""
        load_dotenv()
        
        # Initialize API clients
        self.binance_client = Client(
            os.getenv('BINANCE_API_KEY', ''),
            os.getenv('BINANCE_API_SECRET', '')
        )
        
        # Define data sources
        self.sources = {
            'coingecko': self._fetch_coingecko,
            'yahoo': self._fetch_yahoo,
            'binance': self._fetch_binance
        }
        
        # Cache for coin mappings
        self._coin_mappings_cache = {}
        
    def _get_coin_mappings(self, coin_id: str) -> Dict[str, str]:
        """Get coin mappings for a given coin ID."""
        if coin_id in self._coin_mappings_cache:
            return self._coin_mappings_cache[coin_id]
            
        try:
            # Fetch coin info from CoinGecko
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Get symbol and name
            symbol = data['symbol'].upper()
            name = data['name'].lower()
            
            # Create mappings
            mappings = {
                'coingecko': coin_id,
                'yahoo': f"{symbol}-USD",
                'binance': f"{symbol}USDT"
            }
            
            # Cache the mappings
            self._coin_mappings_cache[coin_id] = mappings
            return mappings
            
        except Exception as e:
            logger.warning(f"Failed to get coin mappings for {coin_id}: {str(e)}")
            # Fallback mappings
            return {
                'coingecko': coin_id,
                'yahoo': f"{coin_id.upper()}-USD",
                'binance': f"{coin_id.upper()}USDT"
            }
    
    def _standardize_to_utc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame index is timezone-aware UTC."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        return df

    def _fetch_coingecko(self, coin_id: str, days: int = 365, interval: str = '1d') -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from CoinGecko."""
        try:
            # Map interval to CoinGecko's accepted values
            cg_interval = 'daily'
            if interval in ['1h', 'hourly']:
                cg_interval = 'hourly'
            elif interval in ['1m', 'minutely']:
                cg_interval = 'minutely'
            # CoinGecko only supports 'minutely', 'hourly', 'daily'
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': str(days),
                'interval': cg_interval
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add other OHLCV columns (CoinGecko only provides close prices)
            df['open'] = df['close'].shift(1)
            df['high'] = df['close']
            df['low'] = df['close']
            df['volume'] = 0  # CoinGecko doesn't provide volume in this endpoint
            
            df = self._standardize_to_utc(df)
            
            logger.info(f"Successfully fetched data from coingecko for {coin_id} [{cg_interval}]")
            return df
            
        except Exception as e:
            logger.warning(f"CoinGecko fetch failed for {coin_id}: {str(e)}")
            return None
    
    def _fetch_yahoo(self, coin_id: str, days: int = 365, interval: str = '1d') -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Yahoo Finance."""
        try:
            mappings = self._get_coin_mappings(coin_id)
            symbol = mappings['yahoo']
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # yfinance intervals: '1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'
            yf_interval = interval
            if interval == '1h':
                yf_interval = '60m'
            df = yf.download(symbol, start=start_date, end=end_date, interval=yf_interval)
            if not df.empty:
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() for col in df.columns.values]
                df = self._standardize_to_utc(df)
                logger.info(f"Successfully fetched data from yahoo for {coin_id} [{yf_interval}]")
                return df
            return None
            
        except Exception as e:
            logger.warning(f"Yahoo Finance fetch failed for {coin_id}: {str(e)}")
            return None
    
    def _fetch_binance(self, coin_id: str, days: int = 365, interval: str = '1d') -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Binance."""
        try:
            mappings = self._get_coin_mappings(coin_id)
            symbol = mappings['binance']
            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            
            # Map interval to Binance's accepted values
            binance_interval = interval
            # Acceptable: '1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d','1w','1M'
            klines = self.binance_client.get_historical_klines(
                symbol,
                binance_interval,
                start_time,
                end_time
            )
            
            if klines:
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignored'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Convert string values to float
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                df = self._standardize_to_utc(df)
                
                logger.info(f"Successfully fetched data from binance for {coin_id} [{binance_interval}]")
                return df
            return None
            
        except Exception as e:
            logger.warning(f"Binance fetch failed for {coin_id}: {str(e)}")
            return None
    
    def _merge_dataframes(self, dfs: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Merge dataframes from different sources, standardizing all indices to UTC."""
        if not dfs:
            return None
        # Standardize all indices to UTC before merging
        for k in dfs:
            dfs[k] = self._standardize_to_utc(dfs[k])
            
        # Use the source with the most data as base
        base_source = max(dfs.items(), key=lambda x: len(x[1]))[0]
        merged_df = dfs[base_source].copy()
        
        # Merge other sources
        for source, df in dfs.items():
            if source != base_source:
                # Merge on index (timestamp)
                merged_df = merged_df.join(df, how='outer', rsuffix=f'_{source}')
                
                # Fill missing values with data from other sources
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if f'{col}_{source}' in merged_df.columns:
                        merged_df[col] = merged_df[col].fillna(merged_df[f'{col}_{source}'])
                        merged_df.drop(f'{col}_{source}', axis=1, inplace=True)
        
        # Sort by timestamp
        merged_df.sort_index(inplace=True)
        
        # Fill any remaining NaN values
        merged_df.fillna(method='ffill', inplace=True)
        merged_df.fillna(method='bfill', inplace=True)
        
        return merged_df
    
    def collect_crypto_ohlcv_data(self, coin_id: str, days: int = 365, interval: str = '1d') -> Optional[pd.DataFrame]:
        """Collect OHLCV data from multiple sources."""
        dfs = {}
        for source, fetch_func in self.sources.items():
            try:
                df = fetch_func(coin_id, days, interval)
                if df is not None and not df.empty:
                    dfs[source] = df
                    logger.info(f"Successfully fetched data from {source} for {coin_id} [{interval}]")
                # Add delay between requests to avoid rate limiting
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Error fetching from {source} for {coin_id}: {str(e)}")
                continue
                
        if not dfs:
            logger.error(f"Failed to fetch data for {coin_id} from any source")
            return None
            
        merged_df = self._merge_dataframes(dfs)
        if merged_df is not None:
            logger.info(f"Successfully collected and merged data for {coin_id} [{interval}]")
            return merged_df
            
        return None

    async def collect_historical_data(self, coin_ids: List[str], days: int) -> Dict[str, pd.DataFrame]:
        """Collect historical data for multiple coins asynchronously."""
        async def fetch_coin_data(coin_id: str) -> Tuple[str, Optional[pd.DataFrame]]:
            return coin_id, self.collect_crypto_ohlcv_data(coin_id, days)
        
        tasks = [fetch_coin_data(coin_id) for coin_id in coin_ids]
        results = await asyncio.gather(*tasks)
        
        return {coin_id: df for coin_id, df in results if df is not None}

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
    collector = DataCollector()
    ohlcv_data = collector.collect_crypto_ohlcv_data("bitcoin", 60)
    if validate_data(ohlcv_data):
        ohlcv_data = clean_data(ohlcv_data)
        report = generate_data_report(ohlcv_data)
        print(report)
    else:
        print("Data validation failed.") 