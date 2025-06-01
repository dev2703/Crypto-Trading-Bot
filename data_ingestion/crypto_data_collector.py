"""
Crypto Data Collector Module

This module handles data collection from various sources:
- CoinGecko (OHLCV, market data)
- On-chain data (Glassnode, Etherscan)
- Social sentiment (Twitter, Reddit, News)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import redis
import tweepy
import praw

from .rate_limiter import RateLimiter, Cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    """Base class for data collection with common utilities."""
    
    def __init__(self):
        self.engine = create_engine(os.getenv('DATABASE_URL'))
        self.Session = sessionmaker(bind=self.engine)
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=0
        )

    def _normalize_timestamp(self, timestamp: Union[int, str, datetime]) -> datetime:
        """Normalize various timestamp formats to datetime."""
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, int):
            return datetime.fromtimestamp(timestamp / 1000)
        elif isinstance(timestamp, str):
            return datetime.fromisoformat(timestamp)
        raise ValueError(f"Unsupported timestamp format: {timestamp}")

class CoinGeckoCollector(DataCollector):
    """Collects data from CoinGecko API."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'X-CG-API-KEY': api_key})
            
        # Initialize rate limiter and cache
        self.rate_limiter = RateLimiter(
            requests_per_minute=int(os.getenv('COINGECKO_RATE_LIMIT', 50))
        )
        self.cache = Cache(ttl_seconds=300)  # 5 minutes cache
        
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting and caching."""
        # Check cache first
        cache_key = f"{endpoint}:{str(params)}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            logger.info(f"Cache hit for {endpoint}")
            return cached_data
            
        # Rate limit check
        self.rate_limiter.wait_if_needed(endpoint)
        
        # Make request
        url = f"{self.base_url}/{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Cache response
        self.cache.set(cache_key, data)
        
        return data
        
    def get_historical_ohlcv(
        self,
        coin_id: str,
        vs_currency: str = 'usd',
        days: str = 'max',
        interval: str = 'daily'
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        try:
            params = {
                'vs_currency': vs_currency,
                'days': days
            }
            
            data = self._make_request(f"coins/{coin_id}/ohlc", params)
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add volume data
            volume_data = self.get_historical_volume(coin_id, vs_currency, days)
            df = df.merge(volume_data, on='timestamp', how='left')
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko API error: {e}")
            raise

    def get_historical_volume(
        self,
        coin_id: str,
        vs_currency: str = 'usd',
        days: str = 'max'
    ) -> pd.DataFrame:
        """Fetch historical volume data."""
        try:
            params = {
                'vs_currency': vs_currency,
                'days': days
            }
            
            data = self._make_request(f"coins/{coin_id}/market_chart", params)
            volume_data = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            volume_data['timestamp'] = pd.to_datetime(volume_data['timestamp'], unit='ms')
            
            return volume_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko API error: {e}")
            raise

    def get_market_data(
        self,
        coin_ids: List[str],
        vs_currency: str = 'usd'
    ) -> pd.DataFrame:
        """Fetch current market data for multiple coins."""
        try:
            params = {
                'ids': ','.join(coin_ids),
                'vs_currencies': vs_currency,
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            
            data = self._make_request("simple/price", params)
            df = pd.DataFrame.from_dict(data, orient='index')
            df.index.name = 'coin_id'
            df.reset_index(inplace=True)
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko API error: {e}")
            raise

    def get_trending_coins(self) -> List[Dict]:
        """Fetch trending coins data."""
        try:
            data = self._make_request("search/trending")
            return data['coins']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko API error: {e}")
            raise

    def get_coin_info(self, coin_id: str) -> Dict:
        """Fetch detailed information about a coin."""
        try:
            data = self._make_request(f"coins/{coin_id}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko API error: {e}")
            raise

    def get_global_data(self) -> Dict:
        """Fetch global cryptocurrency market data."""
        try:
            data = self._make_request("global")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko API error: {e}")
            raise

class OnChainCollector(DataCollector):
    """Collects on-chain data from various sources."""
    
    def __init__(self, glassnode_api_key: str):
        super().__init__()
        self.glassnode_api_key = glassnode_api_key
        self.base_url = "https://api.glassnode.com/v1"
        
    def get_glassnode_metrics(
        self,
        metric: str,
        asset: str = 'BTC',
        interval: str = '24h',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch metrics from Glassnode API."""
        try:
            params = {
                'a': asset,
                'i': interval,
                'api_key': self.glassnode_api_key
            }
            
            if start_time:
                params['s'] = int(start_time.timestamp())
            if end_time:
                params['u'] = int(end_time.timestamp())
                
            response = requests.get(
                f"{self.base_url}/{metric}",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['t'], unit='s')
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Glassnode API error: {e}")
            raise

class SentimentCollector(DataCollector):
    """Collects sentiment data from social media and news."""
    
    def __init__(
        self,
        twitter_api_key: str,
        twitter_api_secret: str,
        twitter_access_token: str,
        twitter_access_secret: str,
        reddit_client_id: str,
        reddit_client_secret: str,
        reddit_user_agent: str
    ):
        super().__init__()
        
        # Initialize Twitter client
        auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret)
        auth.set_access_token(twitter_access_token, twitter_access_secret)
        self.twitter_client = tweepy.API(auth)
        
        # Initialize Reddit client
        self.reddit_client = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )
        
    def get_twitter_sentiment(
        self,
        query: str,
        count: int = 100,
        lang: str = 'en'
    ) -> List[Dict]:
        """Fetch tweets and their sentiment."""
        try:
            tweets = self.twitter_client.search_tweets(
                q=query,
                count=count,
                lang=lang
            )
            
            return [{
                'timestamp': tweet.created_at,
                'text': tweet.text,
                'user': tweet.user.screen_name,
                'retweets': tweet.retweet_count,
                'favorites': tweet.favorite_count
            } for tweet in tweets]
            
        except tweepy.TweepyException as e:
            logger.error(f"Twitter API error: {e}")
            raise
            
    def get_reddit_sentiment(
        self,
        subreddit: str,
        limit: int = 100,
        time_filter: str = 'day'
    ) -> List[Dict]:
        """Fetch Reddit posts and their sentiment."""
        try:
            subreddit = self.reddit_client.subreddit(subreddit)
            posts = subreddit.top(time_filter=time_filter, limit=limit)
            
            return [{
                'timestamp': datetime.fromtimestamp(post.created_utc),
                'title': post.title,
                'text': post.selftext,
                'score': post.score,
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments
            } for post in posts]
            
        except praw.exceptions.PRAWException as e:
            logger.error(f"Reddit API error: {e}")
            raise

def main():
    """Example usage of the data collectors."""
    # Initialize collectors
    coingecko_collector = CoinGeckoCollector(
        api_key=os.getenv('COINGECKO_API_KEY')
    )
    
    onchain_collector = OnChainCollector(
        glassnode_api_key=os.getenv('GLASSNODE_API_KEY')
    )
    
    sentiment_collector = SentimentCollector(
        twitter_api_key=os.getenv('TWITTER_API_KEY'),
        twitter_api_secret=os.getenv('TWITTER_API_SECRET'),
        twitter_access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
        twitter_access_secret=os.getenv('TWITTER_ACCESS_SECRET'),
        reddit_client_id=os.getenv('REDDIT_CLIENT_ID'),
        reddit_client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        reddit_user_agent=os.getenv('REDDIT_USER_AGENT')
    )
    
    # Example: Collect BTC data
    btc_data = coingecko_collector.get_historical_ohlcv(
        coin_id='bitcoin',
        vs_currency='usd',
        days='30',
        interval='daily'
    )
    
    # Example: Get market data for multiple coins
    market_data = coingecko_collector.get_market_data(
        coin_ids=['bitcoin', 'ethereum', 'solana'],
        vs_currency='usd'
    )
    
    # Example: Get trending coins
    trending = coingecko_collector.get_trending_coins()
    
    # Example: Get global market data
    global_data = coingecko_collector.get_global_data()
    
    # Example: Get detailed BTC info
    btc_info = coingecko_collector.get_coin_info('bitcoin')
    
    # Example: Get on-chain metrics
    btc_metrics = onchain_collector.get_glassnode_metrics(
        metric='addresses/active_count',
        asset='BTC',
        interval='24h'
    )
    
    # Example: Get sentiment data
    crypto_tweets = sentiment_collector.get_twitter_sentiment(
        query='bitcoin OR ethereum',
        count=100
    )
    
    crypto_reddit = sentiment_collector.get_reddit_sentiment(
        subreddit='cryptocurrency',
        limit=100
    )

if __name__ == "__main__":
    main() 