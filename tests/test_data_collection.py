import unittest
import pandas as pd
import numpy as np
from strategy.data_collection import DataCollector, validate_data, clean_data, generate_data_report

class TestDataCollection(unittest.TestCase):
    def setUp(self):
        self.collector = DataCollector(api_key="test_api_key")
        self.sample_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

    def test_collect_ohlcv_data(self):
        # This test is a placeholder for actual API testing
        # In a real scenario, you would mock the API response
        pass

    def test_collect_order_book_data(self):
        # This test is a placeholder for actual API testing
        # In a real scenario, you would mock the API response
        pass

    def test_collect_trade_data(self):
        # This test is a placeholder for actual API testing
        # In a real scenario, you would mock the API response
        pass

    def test_validate_data(self):
        self.assertTrue(validate_data(self.sample_data))
        invalid_data = pd.DataFrame({
            'open': [100, np.nan, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        self.assertFalse(validate_data(invalid_data))

    def test_clean_data(self):
        dirty_data = pd.DataFrame({
            'open': [100, np.nan, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        cleaned_data = clean_data(dirty_data)
        self.assertEqual(len(cleaned_data), 4)
        self.assertTrue(cleaned_data.isnull().sum().sum() == 0)

    def test_generate_data_report(self):
        report = generate_data_report(self.sample_data)
        self.assertIsInstance(report, str)
        self.assertIn('Data Report', report)
        self.assertIn('Number of Rows', report)
        self.assertIn('Number of Columns', report)
        self.assertIn('Columns', report)
        self.assertIn('Data Types', report)
        self.assertIn('Missing Values', report)

if __name__ == '__main__':
    unittest.main() 