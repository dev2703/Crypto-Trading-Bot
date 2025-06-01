import unittest
import pandas as pd
import numpy as np
from strategy.model_training import ModelTrainer, generate_training_report
import os

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6]
        })
        self.target = pd.Series([2, 3, 4, 5, 6])
        self.trainer = ModelTrainer(self.features, self.target)

    def test_train_linear_regression(self):
        model = self.trainer.train_linear_regression()
        self.assertIsNotNone(model)
        self.assertEqual(type(model).__name__, 'LinearRegression')

    def test_train_random_forest(self):
        model = self.trainer.train_random_forest()
        self.assertIsNotNone(model)
        self.assertEqual(type(model).__name__, 'RandomForestRegressor')

    def test_evaluate_model(self):
        self.trainer.train_linear_regression()
        evaluation = self.trainer.evaluate_model()
        self.assertIn('mse', evaluation)
        self.assertIn('r2', evaluation)
        self.assertIsInstance(evaluation['mse'], float)
        self.assertIsInstance(evaluation['r2'], float)

    def test_save_and_load_model(self):
        self.trainer.train_linear_regression()
        filepath = 'test_model.joblib'
        self.trainer.save_model(filepath)
        self.assertTrue(os.path.exists(filepath))
        self.trainer.load_model(filepath)
        self.assertIsNotNone(self.trainer.model)
        os.remove(filepath)

    def test_generate_training_report(self):
        self.trainer.train_linear_regression()
        evaluation = self.trainer.evaluate_model()
        report = generate_training_report(self.trainer.model, evaluation)
        self.assertIsInstance(report, str)
        self.assertIn('Training Report', report)
        self.assertIn('Model Type', report)
        self.assertIn('Mean Squared Error', report)
        self.assertIn('R-squared Score', report)

if __name__ == '__main__':
    unittest.main() 