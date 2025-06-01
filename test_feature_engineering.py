"""
Test script for feature engineering with real cryptocurrency data.
"""
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from strategy.data_collection import DataCollector
from strategy.feature_engineering import FeatureEngineer

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

def main():
    # Initialize components
    collector = DataCollector(api_key=API_KEY)
    engineer = FeatureEngineer()
    
    # Fetch Bitcoin data for the last 365 days
    print("\nFetching Bitcoin data...")
    df = collector.collect_crypto_ohlcv_data(coin_id='bitcoin', days=365)
    
    if df is None or df.empty:
        print("Error: Failed to fetch data")
        return
    
    print("\nOriginal Data Sample:")
    print(df.head())
    
    # Generate features
    print("\nGenerating features...")
    features = engineer.generate_features(df)
    
    print("\nGenerated Features:")
    print("Total features:", len(features.columns))
    print("\nFeature Categories:")
    print("1. Price-based:", [col for col in features.columns if any(x in col for x in ['sma', 'ema', 'bb', 'rsi', 'macd'])])
    print("2. Volume-based:", [col for col in features.columns if any(x in col for x in ['obv', 'vwap', 'mfi', 'volume'])])
    print("3. Volatility:", [col for col in features.columns if any(x in col for x in ['atr', 'volatility', 'returns'])])
    print("4. Market Structure:", [col for col in features.columns if any(x in col for x in ['support', 'resistance', 'adx'])])
    print("5. Temporal:", [col for col in features.columns if any(x in col for x in ['hour', 'day', 'weekend', 'session'])])
    
    # Feature statistics
    print("\nFeature Statistics:")
    print(features.describe())
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    engineer.plot_correlation_heatmap(features, 'bitcoin_correlation_heatmap.png')
    engineer.plot_rolling_volatility(features, save_path='bitcoin_rolling_volatility.png')
    engineer.plot_returns_distribution(features, save_path='bitcoin_returns_distribution.png')
    
    # Feature selection
    print("\nPerforming feature selection...")
    selected_features = engineer.select_features(features)
    print("\nSelected Features (PCA):")
    print(f"Number of principal components: {selected_features.shape[1]}")
    print("\nPCA Feature Sample:")
    print(selected_features.head())
    
    print("\nTest completed successfully!")
    print("Visualizations saved as:")
    print("- bitcoin_correlation_heatmap.png")
    print("- bitcoin_rolling_volatility.png")
    print("- bitcoin_returns_distribution.png")

if __name__ == "__main__":
    main() 