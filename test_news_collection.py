import os
from dotenv import load_dotenv
from strategy.news_collector import NewsCollector
import pandas as pd


def test_news_collection():
    # Load environment variables
    load_dotenv()
    print("Loaded environment variables.")
    
    # Initialize news collector
    collector = NewsCollector()
    print("Initialized NewsCollector.")
    
    # Test symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in symbols:
        print(f"\nFetching news for {symbol}...")
        try:
            # Collect news
            news_df = collector.collect_all_news(symbol)
            print(f"Fetched {len(news_df)} articles for {symbol}.")
        except Exception as e:
            print(f"Error collecting news for {symbol}: {e}")
            continue
        
        # Filter relevant news
        try:
            filtered_news = collector.filter_relevant_news(news_df)
            print(f"Filtered down to {len(filtered_news)} articles for {symbol}.")
        except Exception as e:
            print(f"Error filtering news for {symbol}: {e}")
            continue
        
        # Display results
        if not filtered_news.empty:
            print("\nLatest news articles:")
            for _, article in filtered_news.head(3).iterrows():
                print(f"\nTitle: {article['title']}")
                print(f"Source: {article['source']}")
                print(f"Published: {article['published_at']}")
                if article['sentiment'] is not None:
                    print(f"Sentiment: {article['sentiment']}")
                print(f"URL: {article['url']}")
        else:
            print("No news articles found.")

if __name__ == "__main__":
    test_news_collection() 