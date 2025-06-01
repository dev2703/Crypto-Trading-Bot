import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from strategy.model_evaluation import ModelEvaluator, generate_evaluation_report

class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        self.model = LinearRegression()
        self.X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        self.y = pd.Series([2, 3, 4, 5, 6])
        self.model.fit(self.X, self.y)
        self.evaluator = ModelEvaluator(self.model)

    def test_evaluate_model(self):
        metrics = self.evaluator.evaluate_model(self.X, self.y)
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        self.assertIsInstance(metrics['mse'], float)
        self.assertIsInstance(metrics['mae'], float)
        self.assertIsInstance(metrics['r2'], float)

    def test_plot_predictions(self):
        y_pred = self.model.predict(self.X)
        self.evaluator.plot_predictions(self.y, y_pred)

    def test_plot_residuals(self):
        y_pred = self.model.predict(self.X)
        self.evaluator.plot_residuals(self.y, y_pred)

    def test_generate_evaluation_report(self):
        report = generate_evaluation_report(self.evaluator, self.X, self.y)
        self.assertIsInstance(report, str)
        self.assertIn('Evaluation Report', report)
        self.assertIn('Mean Squared Error', report)
        self.assertIn('Mean Absolute Error', report)
        self.assertIn('R-squared Score', report)

if __name__ == '__main__':
    unittest.main() 