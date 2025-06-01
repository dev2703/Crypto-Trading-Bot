import pytest
import pandas as pd
import numpy as np
from strategy.market_microstructure import (
    compute_bid_ask_spread,
    compute_amihud_illiquidity,
    compute_order_imbalance,
    compute_market_depth,
    generate_signals,
    backtest_strategy
)

def sample_orderbook():
    # Create a small sample order book DataFrame
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=20, freq='D'),
        'bid': np.linspace(100, 120, 20),
        'ask': np.linspace(101, 121, 20),
        'returns': np.random.uniform(-0.01, 0.01, 20),
        'volume': np.random.randint(1000, 2000, 20),
        'bid_volume': np.random.randint(500, 1000, 20),
        'ask_volume': np.random.randint(500, 1000, 20)
    }
    return pd.DataFrame(data)

def test_compute_bid_ask_spread():
    bid_prices = pd.Series([100, 101, 102, 103, 104])
    ask_prices = pd.Series([101, 102, 103, 104, 105])
    spread = compute_bid_ask_spread(bid_prices, ask_prices)
    assert len(spread) == 5
    assert spread.iloc[0] == 0.01
    assert spread.iloc[1] == 0.0099
    assert spread.iloc[2] == 0.0098
    assert spread.iloc[3] == 0.0097
    assert spread.iloc[4] == 0.0096

def test_compute_amihud_illiquidity():
    df = sample_orderbook()
    illiquidity = compute_amihud_illiquidity(df['returns'], df['volume'])
    assert len(illiquidity) == len(df)
    assert np.allclose(illiquidity, abs(df['returns']) / df['volume'])

def test_compute_order_imbalance():
    bid_volumes = pd.Series([10, 12, 14, 16, 18])
    ask_volumes = pd.Series([8, 10, 12, 14, 16])
    imbalance = compute_order_imbalance(bid_volumes, ask_volumes)
    assert len(imbalance) == 5
    assert imbalance.iloc[0] == 0.1111
    assert imbalance.iloc[1] == 0.0909
    assert imbalance.iloc[2] == 0.0769
    assert imbalance.iloc[3] == 0.0667
    assert imbalance.iloc[4] == 0.0588

def test_compute_market_depth():
    bid_volumes = pd.Series([10, 12, 14, 16, 18])
    ask_volumes = pd.Series([8, 10, 12, 14, 16])
    depth = compute_market_depth(bid_volumes, ask_volumes)
    assert len(depth) == 5
    assert depth.iloc[0] == 18
    assert depth.iloc[1] == 22
    assert depth.iloc[2] == 26
    assert depth.iloc[3] == 30
    assert depth.iloc[4] == 34

def test_generate_signals():
    spread = pd.Series([0.001, 0.002, 0.001, 0.002, 0.001])
    imbalance = pd.Series([0.3, 0.1, -0.3, 0.1, -0.3])
    depth = pd.Series([18, 22, 26, 30, 34])
    signals = generate_signals(spread, imbalance, depth, spread_threshold=0.0015, imbalance_threshold=0.2)
    assert len(signals) == 5
    assert signals.iloc[0] == 1
    assert signals.iloc[1] == 0
    assert signals.iloc[2] == -1
    assert signals.iloc[3] == 0
    assert signals.iloc[4] == -1

def test_backtest_strategy():
    bid_prices = pd.Series([100, 101, 102, 103, 104])
    ask_prices = pd.Series([101, 102, 103, 104, 105])
    bid_volumes = pd.Series([10, 12, 14, 16, 18])
    ask_volumes = pd.Series([8, 10, 12, 14, 16])
    mid_prices = (bid_prices + ask_prices) / 2
    result = backtest_strategy(bid_prices, ask_prices, bid_volumes, ask_volumes, mid_prices)
    assert 'signals' in result
    assert 'strategy_returns' in result
    assert 'cumulative_returns' in result
    assert 'metrics' in result
    assert len(result['signals']) == 5
    assert len(result['strategy_returns']) == 4
    assert len(result['cumulative_returns']) == 4
    assert isinstance(result['metrics'], dict) 