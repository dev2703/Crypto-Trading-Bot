import unittest
import pandas as pd
import numpy as np
from strategy.portfolio_optimization import compute_optimal_weights, rebalance_portfolio, backtest_strategy

class TestPortfolioOptimization(unittest.TestCase):
    def setUp(self):
        self.returns = pd.DataFrame({
            'asset1': [0.01, -0.02, 0.03, 0.01, -0.01],
            'asset2': [0.02, -0.01, 0.02, 0.01, -0.02],
            'asset3': [0.01, 0.01, -0.01, 0.02, 0.01]
        })

    def test_compute_optimal_weights(self):
        weights = compute_optimal_weights(self.returns)
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(len(weights), len(self.returns.columns))
        self.assertAlmostEqual(np.sum(weights), 1.0)
        self.assertTrue(np.all(weights >= 0))

    def test_rebalance_portfolio(self):
        current_weights = np.array([0.4, 0.3, 0.3])
        target_weights = np.array([0.5, 0.3, 0.2])
        self.assertTrue(rebalance_portfolio(current_weights, target_weights, threshold=0.1))
        self.assertFalse(rebalance_portfolio(current_weights, target_weights, threshold=0.2))

    def test_backtest_strategy(self):
        result = backtest_strategy(self.returns)
        self.assertIn('portfolio_returns', result)
        self.assertIn('cumulative_returns', result)
        self.assertIsInstance(result['portfolio_returns'], pd.Series)
        self.assertIsInstance(result['cumulative_returns'], pd.Series)
        self.assertEqual(len(result['portfolio_returns']), len(self.returns))
        self.assertEqual(len(result['cumulative_returns']), len(self.returns))

if __name__ == '__main__':
    unittest.main() 