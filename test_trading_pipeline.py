import os
from dotenv import load_dotenv
from strategy.trading_pipeline import TradingPipeline
import time

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

def main():
    # Initialize pipeline
    pipeline = TradingPipeline(api_key=API_KEY)
    
    # Run pipeline for bitcoin
    print("\nRunning trading pipeline for Bitcoin...")
    results = pipeline.run_pipeline('bitcoin', days=60)
    
    # Print results
    if 'error' in results:
        print(f"Error: {results['error']}")
    else:
        print("\nTrading Signals:")
        print(results['signals'])
        print("\nTrade Details:")
        print(results['trade'])
        
        # Print some feature statistics
        data = results['data']
        print("\nFeature Statistics:")
        print(data.describe())

if __name__ == "__main__":
    main() 