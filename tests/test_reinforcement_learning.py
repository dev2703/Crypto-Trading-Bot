import pytest
import pandas as pd
import numpy as np
from strategy.reinforcement_learning import TradingEnv, train_agent, generate_signals, backtest_strategy

def sample_ohlcv():
    # Create a small sample OHLCV DataFrame
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=20, freq='D'),
        'open': np.linspace(100, 120, 20),
        'high': np.linspace(101, 121, 20),
        'low': np.linspace(99, 119, 20),
        'close': np.linspace(100, 120, 20),
        'volume': np.random.randint(1000, 2000, 20)
    }
    return pd.DataFrame(data)

def test_trading_env():
    df = sample_ohlcv()
    env = TradingEnv(df, window_size=5)
    assert env.action_space.n == 3
    assert env.observation_space.shape == (5, 5)
    obs = env.reset()
    assert obs.shape == (5, 5)
    action = 1
    obs, reward, done, _ = env.step(action)
    assert obs.shape == (5, 5)
    assert isinstance(reward, float)
    assert isinstance(done, bool)

def test_train_agent():
    df = sample_ohlcv()
    env = TradingEnv(df, window_size=5)
    model = train_agent(env, total_timesteps=100)
    assert model is not None

def test_generate_signals():
    df = sample_ohlcv()
    env = TradingEnv(df, window_size=5)
    model = train_agent(env, total_timesteps=100)
    signals = generate_signals(model, env)
    assert len(signals) == len(df) - 5
    assert signals.isin([0, 1, 2]).all()

def test_backtest_strategy():
    df = sample_ohlcv()
    env = TradingEnv(df, window_size=5)
    model = train_agent(env, total_timesteps=100)
    signals = generate_signals(model, env)
    results = backtest_strategy(signals, df)
    assert 'signals' in results
    assert 'positions' in results
    assert 'returns' in results
    assert 'cumulative_returns' in results
    assert len(results['signals']) == len(signals)
    assert len(results['positions']) == len(signals)
    assert len(results['returns']) == len(signals)
    assert len(results['cumulative_returns']) == len(signals) 