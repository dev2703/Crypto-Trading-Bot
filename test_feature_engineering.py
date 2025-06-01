"""
Test script for feature engineering with walk-forward validation, purging, and feature importance analysis.
"""
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategy.data_collection import DataCollector
from strategy.feature_engineering import FeatureEngineer
from strategy.ensemble_stack import EnsembleStack
from sklearn.metrics import classification_report

# --- Walk-forward validation with purging ---
def walk_forward_validation(X, y, model_class, window=60, purge=5):
    """
    Walk-forward validation with purging.
    Args:
        X: Features DataFrame
        y: Target Series
        model_class: Model class to instantiate
        window: Training window size
        purge: Number of samples to purge between train/test
    Returns:
        List of predictions, true values, and feature importances
    """
    preds = []
    trues = []
    importances = []
    for start in range(0, len(X) - window - purge - 1, window):
        train_X = X.iloc[start:start+window]
        train_y = y.iloc[start:start+window]
        test_X = X.iloc[start+window+purge:start+window+purge+1]
        test_y = y.iloc[start+window+purge:start+window+purge+1]
        if len(test_X) == 0:
            break
        model = model_class()
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
        preds.extend(pred)
        trues.extend(test_y)
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances.append(model.feature_importances_)
        elif hasattr(model, 'coef_'):
            importances.append(np.abs(model.coef_).flatten())
        else:
            importances.append(np.zeros(train_X.shape[1]))
    return np.array(preds), np.array(trues), np.array(importances)

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('COINGECKO_API_KEY')
    
    # Initialize components
    collector = DataCollector(api_key)
    engineer = FeatureEngineer(window_sizes=[5, 10, 20, 50, 100, 200])
    
    # Fetch Bitcoin data for the last 365 days
    print("\nFetching Bitcoin data...")
    df = collector.collect_crypto_ohlcv_data(coin_id='bitcoin', days=365)
    
    print("\nOriginal Data Sample:")
    print(df.head())
    
    # Generate features
    print("\nGenerating features...")
    features = engineer.generate_features(df)
    
    # Target: next day return (classification: up/down/flat)
    y = features['returns'].shift(-1)
    threshold = 0.001
    y_cls = np.where(y > threshold, 1, np.where(y < -threshold, -1, 0))
    y_cls = pd.Series(y_cls, index=features.index)
    X = features.dropna().drop(columns=[col for col in features.columns if 'pca_' in col])
    y_cls = y_cls.loc[X.index]
    
    # Walk-forward validation with purging
    print("\nRunning walk-forward validation with purging...")
    from sklearn.ensemble import RandomForestClassifier
    preds, trues, importances = walk_forward_validation(X, y_cls, RandomForestClassifier, window=60, purge=5)
    
    print("\nClassification Report:")
    print(classification_report(trues, preds, labels=[-1, 0, 1], target_names=['SELL', 'HOLD', 'BUY']))
    
    # Feature importance analysis
    mean_importance = np.mean(importances, axis=0)
    importance_df = pd.DataFrame({'feature': X.columns, 'importance': mean_importance})
    importance_df = importance_df.sort_values('importance', ascending=False)
    print("\nTop 10 Feature Importances:")
    print(importance_df.head(10))
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'][:15][::-1], importance_df['importance'][:15][::-1])
    plt.title('Top 15 Feature Importances (Walk-Forward)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_walkforward.png')
    plt.close()
    print("\nFeature importance plot saved as feature_importance_walkforward.png")
    
    print("\nFeature engineering and walk-forward validation completed successfully!")

if __name__ == "__main__":
    main() 