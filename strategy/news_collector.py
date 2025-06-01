"""
News Collection Module
- Collects financial news from various sources
- Processes and filters news articles
- Integrates with sentiment analysis
"""
import requests
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsCollector:
    """Collects financial news from various sources."""
    
    def __init__(self):
        """Initialize the news collector with API keys."""
        load_dotenv()
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.newsapi_key = os.getenv('NEWSAPI_KEY')  # You'll need to sign up for NewsAPI
        self.finnhub_key = os.getenv('FINNHUB_KEY')  # You'll need to sign up for Finnhub
        
        if not all([self.alpha_vantage_key, self.newsapi_key, self.finnhub_key]):
            logger.warning("Some API keys are missing. Some news sources may not be available.")
    
    def get_alpha_vantage_news(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        Get news from Alpha Vantage.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            limit: Maximum number of news articles to return
        
        Returns:
            List of news articles
        """
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "apikey": self.alpha_vantage_key,
                "limit": limit
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "feed" not in data:
                logger.error(f"Unexpected API response: {data}")
                return []
            
            return data["feed"]
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {str(e)}")
            return []
    
    def get_newsapi_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Get news from NewsAPI.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            days: Number of days to look back
        
        Returns:
            List of news articles
        """
        try:
            url = "https://newsapi.org/v2/everything"
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            params = {
                "q": symbol,
                "from": from_date,
                "language": "en",
                "sortBy": "publishedAt",
                "apiKey": self.newsapi_key
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "articles" not in data:
                logger.error(f"Unexpected API response: {data}")
                return []
            
            return data["articles"]
        except Exception as e:
            logger.error(f"Error fetching NewsAPI news: {str(e)}")
            return []
    
    def get_finnhub_news(self, symbol: str, category: str = "general") -> List[Dict]:
        """
        Get news from Finnhub.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            category: News category (general, forex, crypto, merger)
        
        Returns:
            List of news articles
        """
        try:
            url = "https://finnhub.io/api/v1/news"
            params = {
                "category": category,
                "token": self.finnhub_key
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Filter news for the specific symbol
            return [article for article in data if symbol.lower() in article.get("category", "").lower()]
        except Exception as e:
            logger.error(f"Error fetching Finnhub news: {str(e)}")
            return []
    
    def collect_all_news(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """
        Collect news from all available sources.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            days: Number of days to look back
        
        Returns:
            DataFrame containing all news articles
        """
        all_news = []
        
        # Collect from Alpha Vantage
        alpha_vantage_news = self.get_alpha_vantage_news(symbol)
        for article in alpha_vantage_news:
            all_news.append({
                'source': 'Alpha Vantage',
                'title': article.get('title', ''),
                'text': article.get('summary', ''),
                'url': article.get('url', ''),
                'published_at': article.get('time_published', ''),
                'sentiment': article.get('overall_sentiment_score', 0)
            })
        
        # Collect from NewsAPI
        newsapi_news = self.get_newsapi_news(symbol, days)
        for article in newsapi_news:
            all_news.append({
                'source': 'NewsAPI',
                'title': article.get('title', ''),
                'text': article.get('description', ''),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'sentiment': None
            })
        
        # Collect from Finnhub
        finnhub_news = self.get_finnhub_news(symbol)
        for article in finnhub_news:
            all_news.append({
                'source': 'Finnhub',
                'title': article.get('headline', ''),
                'text': article.get('summary', ''),
                'url': article.get('url', ''),
                'published_at': article.get('datetime', ''),
                'sentiment': None
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(all_news)
        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'])
            df = df.sort_values('published_at', ascending=False)
        
        return df
    
    def filter_relevant_news(self, df: pd.DataFrame, min_sentiment: float = -1.0, 
                           max_sentiment: float = 1.0) -> pd.DataFrame:
        """
        Filter news articles based on relevance and sentiment.
        
        Args:
            df: DataFrame containing news articles
            min_sentiment: Minimum sentiment score
            max_sentiment: Maximum sentiment score
        
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        # Filter by sentiment if available
        if 'sentiment' in df.columns:
            df = df[
                (df['sentiment'].isna()) |
                ((df['sentiment'] >= min_sentiment) & (df['sentiment'] <= max_sentiment))
            ]
        
        # Remove duplicates based on title
        df = df.drop_duplicates(subset=['title'])
        
        return df

# Example usage
if __name__ == "__main__":
    # Initialize news collector
    collector = NewsCollector()
    
    # Collect news for Apple
    news_df = collector.collect_all_news("AAPL")
    
    # Filter relevant news
    filtered_news = collector.filter_relevant_news(news_df)
    
    # Display results
    print(f"Total news articles: {len(news_df)}")
    print(f"Filtered news articles: {len(filtered_news)}")
    print("\nLatest news:")
    print(filtered_news[['source', 'title', 'published_at']].head()) 