"""
Test script for the integrated trading pipeline with real Bitcoin data.
"""
import os
from dotenv import load_dotenv
from strategy.trading_pipeline import TradingPipeline

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

def main():
    # Initialize pipeline
    pipeline = TradingPipeline(api_key=API_KEY)
    
    # Train ensemble stack on Bitcoin data
    print("\nTraining ensemble stack on Bitcoin data...")
    pipeline.train_ensemble(coin_id='bitcoin', days=365)
    
    # Predict trading signals
    print("\nPredicting trading signals...")
    signals = pipeline.predict_with_ensemble(coin_id='bitcoin', days=365)
    
    print("\nPredicted Signals:")
    print(signals)

if __name__ == "__main__":
    main() 