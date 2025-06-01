import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
from strategy.data_collection import DataCollector
from strategy.feature_engineering import FeatureEngineer

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# List of popular and liquid coins (CoinGecko IDs) - reduced set
coins = ['bitcoin', 'ethereum', 'binancecoin']

# Initialize DataCollector and FeatureEngineer
collector = DataCollector(api_key=API_KEY)
engineer = FeatureEngineer()

for coin_id in coins:
    print(f"\n=== {coin_id} ===")
    try:
        # Fetch daily OHLCV data from CoinGecko
        df = collector.collect_crypto_ohlcv_data(coin_id=coin_id, days=60)
        if df is None or df.empty:
            print(f"No data for {coin_id}")
            continue
        # Generate features
        features = engineer.generate_features(df)
        print(features.head())
        # Add a delay to avoid rate limits
        time.sleep(2)
    except Exception as e:
        print(f"Error processing {coin_id}: {e}") 