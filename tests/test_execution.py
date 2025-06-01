import pytest
from strategy.execution import MarketOrder, LimitOrder, StopOrder, compute_transaction_cost, compute_slippage, execute_order
import unittest
import pandas as pd
import numpy as np
from strategy.execution import ExecutionEnvironment, ExecutionAgent, backtest_strategy

def test_market_order():
    order = MarketOrder('BTC', 1.0)
    assert order.symbol == 'BTC'
    assert order.quantity == 1.0
    assert order.order_type == 'market'

def test_limit_order():
    order = LimitOrder('ETH', 10.0, 2000.0)
    assert order.symbol == 'ETH'
    assert order.quantity == 10.0
    assert order.order_type == 'limit'
    assert order.limit_price == 2000.0

def test_stop_order():
    order = StopOrder('XRP', 100.0, 0.5)
    assert order.symbol == 'XRP'
    assert order.quantity == 100.0
    assert order.order_type == 'stop'
    assert order.stop_price == 0.5

def test_compute_transaction_cost():
    order = MarketOrder('BTC', 1.0)
    cost = compute_transaction_cost(order, 50000.0, 0.001)
    assert cost == 50.0

def test_compute_slippage():
    order = MarketOrder('BTC', 1.0)
    slippage = compute_slippage(order, 50000.0, 0.001)
    assert slippage == 50.0

def test_execute_order():
    order = MarketOrder('BTC', 1.0)
    result = execute_order(order, 50000.0, 0.001, 0.001)
    assert result['order'] == order
    assert result['market_price'] == 50000.0
    assert result['transaction_cost'] == 50.0
    assert result['slippage'] == 50.0
    assert result['total_cost'] == 100.0

class TestExecution(unittest.TestCase):
    def setUp(self):
        self.prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        self.volume = pd.Series([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000])
        self.env = ExecutionEnvironment(self.prices, self.volume, window_size=5)
        self.agent = ExecutionAgent(state_size=4, action_size=2)

    def test_environment_initialization(self):
        self.assertEqual(self.env.window_size, 5)
        self.assertEqual(self.env.current_step, 5)
        self.assertEqual(self.env.max_steps, len(self.prices) - 5)
        self.assertEqual(self.env.action_space, [0, 1])

    def test_environment_step(self):
        state, reward, done = self.env.step(1)
        self.assertEqual(self.env.current_step, 6)
        self.assertIsInstance(state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertFalse(done)

    def test_environment_reset(self):
        self.env.current_step = 10
        state = self.env.reset()
        self.assertEqual(self.env.current_step, 5)
        self.assertIsInstance(state, np.ndarray)

    def test_agent_initialization(self):
        self.assertEqual(self.agent.state_size, 4)
        self.assertEqual(self.agent.action_size, 2)
        self.assertEqual(self.agent.learning_rate, 0.1)
        self.assertEqual(self.agent.discount_factor, 0.99)
        self.assertEqual(self.agent.exploration_rate, 0.1)
        self.assertEqual(self.agent.q_table.shape, (4, 2))

    def test_agent_get_action(self):
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = self.agent.get_action(state)
        self.assertIn(action, [0, 1])

    def test_agent_update(self):
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = 1
        reward = 0.5
        next_state = np.array([0.2, 0.3, 0.4, 0.5])
        self.agent.update(state, action, reward, next_state)
        self.assertNotEqual(self.agent.q_table[state, action], 0)

    def test_backtest_strategy(self):
        result = backtest_strategy(self.prices, self.volume)
        self.assertIn('signals', result)
        self.assertIn('strategy_returns', result)
        self.assertIn('cumulative_returns', result)
        self.assertIn('metrics', result)
        self.assertEqual(len(result['signals']), len(self.prices))
        self.assertEqual(len(result['strategy_returns']), len(self.prices) - 1)
        self.assertEqual(len(result['cumulative_returns']), len(self.prices) - 1)

if __name__ == '__main__':
    unittest.main() 