"""
Model Evaluation Strategy Module
- Defines the evaluation metrics and criteria
- Implements the evaluation algorithm
- Generates evaluation reports
- Includes visualization and analysis tools
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class ModelEvaluator:
    """Model evaluator for trading."""
    def __init__(self, model: object):
        self.model = model

    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate the model using various metrics."""
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }

    def plot_predictions(self, y_true: pd.Series, y_pred: np.ndarray, title: str = 'Model Predictions'):
        """Plot the actual vs. predicted values."""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.grid(True)
        plt.show()

    def plot_residuals(self, y_true: pd.Series, y_pred: np.ndarray, title: str = 'Residuals Plot'):
        """Plot the residuals of the model."""
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(title)
        plt.grid(True)
        plt.show()

def generate_evaluation_report(evaluator: ModelEvaluator, X: pd.DataFrame, y: pd.Series) -> str:
    """Generate an evaluation report."""
    metrics = evaluator.evaluate_model(X, y)
    report = f"""
    Evaluation Report:
    ----------------
    Model Type: {type(evaluator.model).__name__}
    Mean Squared Error: {metrics['mse']:.4f}
    Mean Absolute Error: {metrics['mae']:.4f}
    R-squared Score: {metrics['r2']:.4f}
    """
    return report

# Example usage
if __name__ == "__main__":
    # Example: Evaluate a model and generate a report
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
    y = pd.Series([2, 3, 4, 5, 6])
    model.fit(X, y)
    evaluator = ModelEvaluator(model)
    report = generate_evaluation_report(evaluator, X, y)
    print(report)
    evaluator.plot_predictions(y, model.predict(X))
    evaluator.plot_residuals(y, model.predict(X)) 