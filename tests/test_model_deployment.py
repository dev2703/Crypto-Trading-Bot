import unittest
import pandas as pd
import numpy as np
from strategy.model_deployment import ModelDeployer, generate_deployment_report
import os
import joblib
from sklearn.linear_model import LinearRegression

class TestModelDeployment(unittest.TestCase):
    def setUp(self):
        self.model_path = 'test_model.joblib'
        self.model = LinearRegression()
        self.model.fit(np.array([[1], [2], [3]]), np.array([2, 3, 4]))
        joblib.dump(self.model, self.model_path)
        self.deployer = ModelDeployer(self.model_path)

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists('deployment.log'):
            os.remove('deployment.log')

    def test_load_model(self):
        self.deployer.load_model()
        self.assertIsNotNone(self.deployer.model)
        self.assertEqual(type(self.deployer.model).__name__, 'LinearRegression')

    def test_deploy_model(self):
        self.deployer.load_model()
        features = pd.DataFrame({'feature1': [1, 2, 3]})
        predictions = self.deployer.deploy_model(features)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(features))

    def test_monitor_performance(self):
        self.deployer.load_model()
        features = pd.DataFrame({'feature1': [1, 2, 3]})
        predictions = self.deployer.deploy_model(features)
        actual = np.array([2, 3, 4])
        performance = self.deployer.monitor_performance(predictions, actual)
        self.assertIn('mse', performance)
        self.assertIn('mae', performance)
        self.assertIsInstance(performance['mse'], float)
        self.assertIsInstance(performance['mae'], float)

    def test_generate_deployment_report(self):
        self.deployer.load_model()
        features = pd.DataFrame({'feature1': [1, 2, 3]})
        predictions = self.deployer.deploy_model(features)
        actual = np.array([2, 3, 4])
        performance = self.deployer.monitor_performance(predictions, actual)
        report = generate_deployment_report(self.deployer, performance)
        self.assertIsInstance(report, str)
        self.assertIn('Deployment Report', report)
        self.assertIn('Model Path', report)
        self.assertIn('Model Type', report)
        self.assertIn('Mean Squared Error', report)
        self.assertIn('Mean Absolute Error', report)

if __name__ == '__main__':
    unittest.main() 