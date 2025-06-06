import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import os
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: Dict[str, float] = {}
        self.trades: List[Dict] = []
        self.current_prices: Dict[str, float] = {}
        
        # Initialize exchange
        load_dotenv()
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'enableRateLimit': True
        })
    
    def fetch_historical_data(self, symbol: str, timeframe: str = '1h', 
                            start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch historical OHLCV data for backtesting."""
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Convert dates to timestamps
            start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=start_timestamp,
                limit=1000
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for strategy."""
        # Simple Moving Averages
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['SMA50'] = df['close'].rolling(window=50).mean()
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on technical indicators."""
        df['signal'] = 0
        
        # Buy signals
        df.loc[(df['SMA20'] > df['SMA50']) & 
               (df['RSI'] < 70) & 
               (df['MACD'] > df['Signal_Line']), 'signal'] = 1
        
        # Sell signals
        df.loc[(df['SMA20'] < df['SMA50']) & 
               (df['RSI'] > 30) & 
               (df['MACD'] < df['Signal_Line']), 'signal'] = -1
        
        return df
    
    def run_backtest(self, symbol: str, timeframe: str = '1h',
                    start_date: str = None, end_date: str = None) -> Dict:
        """Run backtest for a given symbol and timeframe."""
        # Fetch and prepare data
        df = self.fetch_historical_data(symbol, timeframe, start_date, end_date)
        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        
        # Initialize backtest results
        results = {
            'trades': [],
            'balance': [self.initial_balance],
            'returns': [0],
            'positions': []
        }
        
        current_position = 0
        entry_price = 0
        
        # Iterate through data
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]
            
            # Update current balance
            if current_position != 0:
                pnl = (current_price - entry_price) * current_position
                self.balance += pnl
            
            # Execute trades based on signals
            if signal == 1 and current_position <= 0:  # Buy signal
                if current_position < 0:  # Close short position
                    self.balance += abs(current_position) * (entry_price - current_price)
                
                # Open long position
                current_position = self.balance / current_price
                entry_price = current_price
                
                results['trades'].append({
                    'timestamp': df.index[i],
                    'type': 'BUY',
                    'price': current_price,
                    'size': current_position
                })
                
            elif signal == -1 and current_position >= 0:  # Sell signal
                if current_position > 0:  # Close long position
                    self.balance += current_position * (current_price - entry_price)
                
                # Open short position
                current_position = -self.balance / current_price
                entry_price = current_price
                
                results['trades'].append({
                    'timestamp': df.index[i],
                    'type': 'SELL',
                    'price': current_price,
                    'size': abs(current_position)
                })
            
            # Record results
            results['balance'].append(self.balance)
            results['returns'].append((self.balance - self.initial_balance) / self.initial_balance * 100)
            results['positions'].append(current_position)
        
        return results
    
    def calculate_metrics(self, results: Dict) -> Dict:
        """Calculate performance metrics for the backtest."""
        returns = pd.Series(results['returns'])
        
        metrics = {
            'Total Return (%)': returns.iloc[-1],
            'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(252),
            'Max Drawdown (%)': (returns - returns.cummax()).min(),
            'Win Rate (%)': len(returns[returns > 0]) / len(returns) * 100,
            'Number of Trades': len(results['trades'])
        }
        
        return metrics

if __name__ == "__main__":
    # Example usage
    backtest = BacktestEngine(initial_balance=10000.0)
    
    # Run backtest for ETH/USDT
    eth_results = backtest.run_backtest(
        symbol='ETH/USDT',
        timeframe='1h',
        start_date='2024-01-01',
        end_date='2024-03-01'
    )
    
    # Calculate and print metrics
    eth_metrics = backtest.calculate_metrics(eth_results)
    print("\nETH/USDT Backtest Results:")
    for metric, value in eth_metrics.items():
        print(f"{metric}: {value:.2f}") 