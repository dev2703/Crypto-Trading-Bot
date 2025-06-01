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
- **Signal Combiner**: Aggregates signals from multiple strategies using majority voting.
- **Backtesting**: Simulates trading strategies on historical data.
- **Performance Evaluation**: Computes metrics like Sharpe ratio, drawdown, and returns.
- **Streamlit Dashboard**: Monitors signals, historical data, and performance metrics.
- **Real-time Data Streaming**: WebSocket support for live market data.

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

3. **Set up environment variables**:
   - Create a `.env` file in the project root directory
   - Add your Alpha Vantage API key:
     ```
     ALPHA_VANTAGE_API_KEY=your_api_key_here
     ```
   - Note: Never commit the `.env` file to version control
   - The free tier has a rate limit of 5 API calls per minute and 500 calls per day

## Usage

### Running the Streamlit Dashboard

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the dashboard**:
   - Open your browser and go to `http://localhost:8501`
   - Use the sidebar to input parameters (symbol, interval, start/end time)
   - Click "Start Streaming" to receive real-time price updates
   - Click "Fetch Historical Data" to load historical data and view signals

### Running Backtests

- Use the `backtest_strategy` functions in each strategy module to simulate trading on historical data
- Example:
  ```python
  from strategy.statistical_arbitrage import backtest_strategy
  result = backtest_strategy(price1, price2)
  print(result)
  ```

## Next Steps

- **Train a custom LLM** for sentiment analysis (e.g., fine-tune FinBERT)
- **Implement real-time data streaming** using WebSocket APIs
- **Deploy the bot on a server** (e.g., AWS, GCP) for continuous operation
- **Build a more advanced frontend** for real-time monitoring and manual overrides
- **Parameter tuning and optimization** using real historical data
- **Deploy in a sandbox/paper trading environment** before going live

## Security Notes

- Never commit API keys or sensitive credentials to version control
- Use environment variables for all sensitive information
- Keep your `.env` file secure and never share it
- Regularly rotate API keys and credentials

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. 