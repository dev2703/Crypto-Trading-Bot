from backtest_engine import BacktestEngine
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def plot_results(results: dict, symbol: str, save_path: str = None):
    """Plot backtest results."""
    plt.figure(figsize=(15, 10))
    
    # Plot balance
    plt.subplot(2, 1, 1)
    plt.plot(results['balance'], label='Account Balance')
    plt.title(f'{symbol} Backtest Results - Account Balance')
    plt.xlabel('Time')
    plt.ylabel('Balance (USDT)')
    plt.legend()
    plt.grid(True)
    
    # Plot returns
    plt.subplot(2, 1, 2)
    plt.plot(results['returns'], label='Returns (%)')
    plt.title('Returns Over Time')
    plt.xlabel('Time')
    plt.ylabel('Returns (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    # Create results directory if it doesn't exist
    results_dir = 'backtest_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize backtest engine
    backtest = BacktestEngine(initial_balance=10000.0)
    
    # Define trading pairs to test
    pairs = ['ETH/USDT', 'BTC/USDT']
    
    # Define timeframes to test
    timeframes = ['1h', '4h', '1d']
    
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 90 days of historical data
    
    # Run backtests
    for pair in pairs:
        print(f"\nRunning backtests for {pair}")
        print("-" * 50)
        
        for timeframe in timeframes:
            print(f"\nTimeframe: {timeframe}")
            
            # Run backtest
            results = backtest.run_backtest(
                symbol=pair,
                timeframe=timeframe,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # Calculate metrics
            metrics = backtest.calculate_metrics(results)
            
            # Print metrics
            print("\nPerformance Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")
            
            # Plot results
            plot_path = os.path.join(results_dir, f"{pair.replace('/', '_')}_{timeframe}_results.png")
            plot_results(results, pair, plot_path)
            print(f"\nResults plot saved to: {plot_path}")
            
            # Save detailed results to CSV
            results_df = pd.DataFrame({
                'timestamp': pd.date_range(start=start_date, end=end_date, periods=len(results['balance'])),
                'balance': results['balance'],
                'returns': results['returns'],
                'position': results['positions']
            })
            
            csv_path = os.path.join(results_dir, f"{pair.replace('/', '_')}_{timeframe}_results.csv")
            results_df.to_csv(csv_path, index=False)
            print(f"Detailed results saved to: {csv_path}")

if __name__ == "__main__":
    main() 