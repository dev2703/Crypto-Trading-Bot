import unittest
import pandas as pd
import numpy as np
from strategy.reinforcement_learning import TradingEnvironment, QLearningAgent, backtest_strategy

class TestReinforcementLearning(unittest.TestCase):
    def setUp(self):
        self.prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        self.env = TradingEnvironment(self.prices, window_size=5)
        self.agent = QLearningAgent(state_size=4, action_size=3)

    def test_environment_initialization(self):
        self.assertEqual(self.env.window_size, 5)
        self.assertEqual(self.env.current_step, 5)
        self.assertEqual(self.env.max_steps, len(self.prices) - 5)
        self.assertEqual(self.env.action_space, [-1, 0, 1])

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
        self.assertEqual(self.agent.action_size, 3)
        self.assertEqual(self.agent.learning_rate, 0.1)
        self.assertEqual(self.agent.discount_factor, 0.99)
        self.assertEqual(self.agent.exploration_rate, 0.1)
        self.assertEqual(self.agent.q_table.shape, (4, 3))

    def test_agent_get_action(self):
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = self.agent.get_action(state)
        self.assertIn(action, [0, 1, 2])

    def test_agent_update(self):
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = 1
        reward = 0.5
        next_state = np.array([0.2, 0.3, 0.4, 0.5])
        self.agent.update(state, action, reward, next_state)
        self.assertNotEqual(self.agent.q_table[state, action], 0)

    def test_backtest_strategy(self):
        result = backtest_strategy(self.prices)
        self.assertIn('signals', result)
        self.assertIn('strategy_returns', result)
        self.assertIn('cumulative_returns', result)
        self.assertIn('metrics', result)
        self.assertEqual(len(result['signals']), len(self.prices))
        self.assertEqual(len(result['strategy_returns']), len(self.prices) - 1)
        self.assertEqual(len(result['cumulative_returns']), len(self.prices) - 1)

if __name__ == '__main__':
    unittest.main() 