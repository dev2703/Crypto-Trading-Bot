"""
Tests for the data collector module.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
from data_ingestion.crypto_data_collector import CoinGeckoCollector
from data_ingestion.rate_limiter import RateLimiter, Cache

@pytest.fixture
def collector():
    """Create a CoinGecko collector instance for testing."""
    return CoinGeckoCollector()

def test_historical_ohlcv(collector):
    """Test historical OHLCV data collection."""
    # Test with Bitcoin
    df = collector.get_historical_ohlcv(
        coin_id='bitcoin',
        vs_currency='usd',
        days='7',
        interval='daily'
    )
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    assert df['timestamp'].dtype == 'datetime64[ns]'
    
def test_market_data(collector):
    """Test market data collection."""
    # Test with multiple coins
    df = collector.get_market_data(
        coin_ids=['bitcoin', 'ethereum'],
        vs_currency='usd'
    )
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'coin_id' in df.columns
    assert len(df) == 2  # Should have data for both coins
    
def test_trending_coins(collector):
    """Test trending coins data collection."""
    trending = collector.get_trending_coins()
    
    assert isinstance(trending, list)
    assert len(trending) > 0
    assert all(isinstance(coin, dict) for coin in trending)
    
def test_coin_info(collector):
    """Test detailed coin information collection."""
    info = collector.get_coin_info('bitcoin')
    
    assert isinstance(info, dict)
    assert 'id' in info
    assert 'name' in info
    assert 'symbol' in info
    
def test_global_data(collector):
    """Test global market data collection."""
    data = collector.get_global_data()
    
    assert isinstance(data, dict)
    assert 'data' in data
    assert 'total_market_cap' in data['data']
    assert 'total_volume' in data['data']
    
def test_rate_limiter():
    """Test rate limiter functionality."""
    limiter = RateLimiter(requests_per_minute=2)
    
    # Should not wait on first request
    start_time = datetime.now()
    limiter.wait_if_needed('test_endpoint')
    assert (datetime.now() - start_time).total_seconds() < 1
    
    # Should not wait on second request
    start_time = datetime.now()
    limiter.wait_if_needed('test_endpoint')
    assert (datetime.now() - start_time).total_seconds() < 1
    
    # Should wait on third request
    start_time = datetime.now()
    limiter.wait_if_needed('test_endpoint')
    assert (datetime.now() - start_time).total_seconds() >= 1
    
def test_cache():
    """Test cache functionality."""
    cache = Cache(ttl_seconds=1)
    
    # Test setting and getting
    cache.set('test_key', {'data': 'test_value'})
    assert cache.get('test_key') == {'data': 'test_value'}
    
    # Test expiration
    import time
    time.sleep(1.1)  # Wait for cache to expire
    assert cache.get('test_key') is None
    
    # Test clearing
    cache.set('test_key', {'data': 'test_value'})
    cache.clear()
    assert cache.get('test_key') is None 