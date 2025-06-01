import unittest
import pandas as pd
import numpy as np
from strategy.risk_management import (
    compute_var,
    compute_cvar,
    compute_drawdown,
    position_sizing,
    stop_loss,
    backtest_strategy
)

class TestRiskManagement(unittest.TestCase):
    def setUp(self):
        self.returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])

    def test_compute_var(self):
        var = compute_var(self.returns)
        self.assertIsInstance(var, float)
        self.assertLess(var, 0)

    def test_compute_cvar(self):
        cvar = compute_cvar(self.returns)
        self.assertIsInstance(cvar, float)
        self.assertLess(cvar, 0)

    def test_compute_drawdown(self):
        drawdown = compute_drawdown(self.returns)
        self.assertIsInstance(drawdown, pd.Series)
        self.assertEqual(len(drawdown), len(self.returns))
        self.assertLessEqual(drawdown.min(), 0)

    def test_position_sizing(self):
        size = position_sizing(self.returns)
        self.assertIsInstance(size, float)
        self.assertGreater(size, 0)

    def test_stop_loss(self):
        level = stop_loss(self.returns)
        self.assertIsInstance(level, float)
        self.assertLess(level, 0)

    def test_backtest_strategy(self):
        position_size = position_sizing(self.returns)
        stop_loss_level = stop_loss(self.returns)
        result = backtest_strategy(self.returns, position_size, stop_loss_level)
        self.assertIn('portfolio_returns', result)
        self.assertIn('cumulative_returns', result)
        self.assertIn('drawdown', result)
        self.assertIsInstance(result['portfolio_returns'], pd.Series)
        self.assertIsInstance(result['cumulative_returns'], pd.Series)
        self.assertIsInstance(result['drawdown'], pd.Series)
        self.assertEqual(len(result['portfolio_returns']), len(self.returns))
        self.assertEqual(len(result['cumulative_returns']), len(self.returns))
        self.assertEqual(len(result['drawdown']), len(self.returns))

if __name__ == '__main__':
    unittest.main() 