# Crypto Trading Bot

A modular, Python-based trading bot for cryptocurrency and stock markets. This project includes various trading strategies, data collection, signal generation, and a Streamlit dashboard for monitoring.

## Features

- **Data Collection**: Fetches historical and real-time data using Alpha Vantage API.
- **Trading Strategies**:
  - Market Microstructure Analysis
  - Statistical Arbitrage
  - Sentiment Analysis
  - Reinforcement Learning
  - Portfolio Optimization
  - Risk Management
- **Signal Combiner**: Aggregates signals from multiple strategies.
- **Backtesting**: Simulates trading strategies on historical data.
- **Performance Evaluation**: Computes metrics like Sharpe ratio, drawdown, and returns.
- **Streamlit Dashboard**: Monitors signals, historical data, and performance metrics.

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/crypto-trading-bot.git
   cd crypto-trading-bot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Alpha Vantage API key**:
   - Sign up for a free API key at [Alpha Vantage](https://www.alphavantage.co/).
   - Replace `ARUSM71L0ISPNU4F` in `strategy/data_collection.py` and `app.py` with your API key.
   - Note: The free tier has a rate limit of 5 API calls per minute and 500 calls per day.

## Usage

### Running the Streamlit Dashboard

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the dashboard**:
   - Open your browser and go to `http://localhost:8501`.
   - Use the sidebar to input parameters (symbol, interval, start/end time).
   - Click "Fetch Data" to load historical data and view signals.

### Running Backtests

- Use the `backtest_strategy` functions in each strategy module to simulate trading on historical data.
- Example:
  ```python
  from strategy.statistical_arbitrage import backtest_strategy
  result = backtest_strategy(price1, price2)
  print(result)
  ```

## Next Steps

- **Train a custom LLM** for sentiment analysis (e.g., fine-tune FinBERT).
- **Implement real-time data streaming** using WebSocket APIs.
- **Deploy the bot on a server** (e.g., AWS, GCP) for continuous operation.
- **Build a more advanced frontend** for real-time monitoring and manual overrides.
- **Parameter tuning and optimization** using real historical data.
- **Deploy in a sandbox/paper trading environment** before going live.

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. 