# Crypto Trading Bot

A sophisticated crypto quant trading system designed to generate alpha using advanced multi-modal ML strategies. This bot serves as the IP backbone for a crypto quant hedge fund or asset management service.

## Features

- Statistical Arbitrage on Crypto Pairs
- Reinforcement Learning Agent (PPO)
- Market Microstructure Alpha
- Sentiment + On-chain Hybrid Signals
- Ensemble Meta-Strategy

## Technical Stack

- Python (Pandas, NumPy, SciPy, Scikit-learn, PyTorch)
- Machine Learning: Stable-Baselines3, XGBoost, statsmodels
- Data: Binance API, Twitter API, Glassnode, Etherscan, Pushshift, RSS
- Backend: FastAPI
- Frontend: React + Tailwind + Recharts
- Database: PostgreSQL + TimescaleDB
- Cache: Redis
- Live Trading: Binance Spot/Margin API (REST + WebSocket)
- Deployment: Docker + GitHub Actions CI/CD

## Project Structure

```
crypto-trading-bot/
├── data_ingestion/     # Data collection from various sources
├── features/          # Feature engineering
├── strategy/          # Trading strategies
├── execution/         # Live trade execution
├── risk/             # Risk management
├── backtest/         # Backtesting engine
├── ui/               # Dashboard
├── api/              # FastAPI endpoints
└── devops/           # Docker and deployment
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/dev2703/Crypto-Trading-Bot.git
cd Crypto-Trading-Bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Set up the database:
```bash
# Install PostgreSQL and TimescaleDB
# Create a new database and enable the TimescaleDB extension
```

## Usage

1. Start the data collection:
```bash
python data_ingestion/crypto_data_collector.py
```

2. Run the trading bot:
```bash
python strategy/main.py
```

3. Access the dashboard:
```bash
cd ui
npm install
npm start
```

## Development

- Follow PEP 8 style guide
- Write tests for new features
- Use meaningful commit messages
- Keep the codebase modular and well-documented

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Security

- Never commit API keys or sensitive information
- Use environment variables for configuration
- Implement proper error handling and logging
- Follow security best practices for API usage

## Support

For support, please open an issue in the GitHub repository. 