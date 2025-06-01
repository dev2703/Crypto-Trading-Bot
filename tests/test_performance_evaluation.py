import unittest
import pandas as pd
import numpy as np
from strategy.performance_evaluation import compute_performance_metrics, generate_performance_report, analyze_performance

class TestPerformanceEvaluation(unittest.TestCase):
    def setUp(self):
        self.returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])

    def test_compute_performance_metrics(self):
        metrics = compute_performance_metrics(self.returns)
        self.assertIn('total_return', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIsInstance(metrics['total_return'], float)
        self.assertIsInstance(metrics['annualized_return'], float)
        self.assertIsInstance(metrics['volatility'], float)
        self.assertIsInstance(metrics['sharpe_ratio'], float)
        self.assertIsInstance(metrics['max_drawdown'], float)

    def test_generate_performance_report(self):
        report = generate_performance_report(self.returns)
        self.assertIsInstance(report, str)
        self.assertIn('Performance Report', report)
        self.assertIn('Total Return', report)
        self.assertIn('Annualized Return', report)
        self.assertIn('Volatility', report)
        self.assertIn('Sharpe Ratio', report)
        self.assertIn('Maximum Drawdown', report)

    def test_analyze_performance(self):
        result = analyze_performance(self.returns)
        self.assertIn('metrics', result)
        self.assertIn('report', result)
        self.assertIsInstance(result['metrics'], dict)
        self.assertIsInstance(result['report'], str)

if __name__ == '__main__':
    unittest.main() 