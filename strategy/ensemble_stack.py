"""
Ensemble Stack Module for Trading Signals
- Level 1: ML base models (XGB, LGBM, CatBoost, RF, ET)
- Level 2: Deep learning models (LSTM, CNN-LSTM, Transformer)
- Level 3: Meta-learner (Ridge, MLP)
- RL: DQN/PPO agent (stable-baselines3)
- Custom loss and metrics
- Adaptive threshold adjustment
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gym

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveThreshold:
    """Adaptive threshold adjustment based on market conditions."""
    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window
        self.volatility_threshold = 0.02  # Base volatility threshold
        self.trend_threshold = 0.01  # Base trend threshold
        
    def calculate_thresholds(self, returns: pd.Series, close: pd.Series) -> tuple:
        """
        Calculate adaptive thresholds based on market conditions.
        
        Args:
            returns: Series of returns
            close: Series of closing prices
            
        Returns:
            tuple: (buy_threshold, sell_threshold)
        """
        # Calculate rolling volatility
        rolling_vol = returns.rolling(self.lookback_window).std()
        
        # Calculate trend strength using ADX-like measure
        price_change = close.pct_change(self.lookback_window)
        trend_strength = abs(price_change) / rolling_vol
        
        # Adjust thresholds based on market conditions
        volatility_factor = rolling_vol / rolling_vol.mean()
        trend_factor = trend_strength / trend_strength.mean()
        
        # Dynamic threshold adjustment
        buy_threshold = self.volatility_threshold * volatility_factor * (1 + trend_factor)
        sell_threshold = -self.volatility_threshold * volatility_factor * (1 + trend_factor)
        
        return buy_threshold, sell_threshold

class EnsembleStack:
    """Ensemble stack for regression and classification trading signals."""
    def __init__(self, task: str = 'classification', n_folds: int = 5, random_state: int = 42):
        """
        Args:
            task: 'regression' or 'classification'
            n_folds: Number of folds for stacking
            random_state: Seed
        """
        self.task = task
        self.n_folds = n_folds
        self.random_state = random_state
        self.base_models = self._init_base_models()
        self.meta_model = self._init_meta_model()
        self.scaler = StandardScaler()
        self.adaptive_threshold = AdaptiveThreshold()
        self.fitted = False

    def _init_base_models(self):
        if self.task == 'regression':
            return [
                xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.01,
                    max_depth=5,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    verbosity=0
                ),
                lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.01,
                    max_depth=5,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    verbose=-1
                ),
                cb.CatBoostRegressor(
                    n_estimators=100,
                    learning_rate=0.01,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bylevel=0.8,
                    random_state=self.random_state,
                    verbose=0
                ),
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.random_state
                ),
                ExtraTreesRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.random_state
                )
            ]
        else:
            return [
                xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.01,
                    max_depth=5,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    verbosity=0,
                    use_label_encoder=False
                ),
                lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.01,
                    max_depth=5,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    verbose=-1
                ),
                cb.CatBoostClassifier(
                    n_estimators=100,
                    learning_rate=0.01,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bylevel=0.8,
                    random_state=self.random_state,
                    verbose=0
                ),
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.random_state
                ),
                ExtraTreesClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.random_state
                )
            ]

    def _init_meta_model(self):
        if self.task == 'regression':
            return Ridge(alpha=1.0)
        else:
            return LogisticRegression(max_iter=1000, C=1.0)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit all base models and the meta-learner using adaptive thresholds.
        """
        # Convert categorical/object columns to integer codes
        X = X.copy()
        for col in X.select_dtypes(include=['category', 'object']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        # Fit and store the scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate adaptive thresholds
        buy_threshold, sell_threshold = self.adaptive_threshold.calculate_thresholds(
            y, X['close'] if 'close' in X.columns else X.iloc[:, 0]
        )
        
        # Discretize y using adaptive thresholds
        y_discrete = np.where(y > buy_threshold, 1, np.where(y < sell_threshold, -1, 0))
        
        # Remap: -1 -> 0, 0 -> 1, 1 -> 2
        y_mapped = np.where(y_discrete == -1, 0, np.where(y_discrete == 0, 1, 2))
        self.y_discrete_ = y_mapped
        
        # Fit base models
        for model in self.base_models:
            model.fit(X_scaled, y_mapped)
        
        # Generate base model predictions for meta-learner
        base_preds = np.column_stack([
            model.predict(X_scaled) for model in self.base_models
        ])
        
        # Fit meta-learner
        self.meta_model.fit(base_preds, y_mapped)
        self.fitted = True
        logger.info("Ensemble stack fitted.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the ensemble stack."""
        if not self.fitted:
            raise Exception("Model not fitted!")
        
        X = X.copy()
        for col in X.select_dtypes(include=['category', 'object']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        X_scaled = self.scaler.transform(X)
        
        meta_features = np.column_stack([
            model.predict(X_scaled) if self.task == 'regression' else (
                model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_scaled)
            )
            for model in self.base_models
        ])
        
        preds = self.meta_model.predict(meta_features)
        
        # Remap back: 0 -> -1, 1 -> 0, 2 -> 1
        if self.task == 'classification':
            preds = np.where(preds == 0, -1, np.where(preds == 1, 0, 1))
        
        return preds

    def predict_signal(self, X: pd.DataFrame) -> List[str]:
        """
        Output Buy/Sell/Hold signals based on classification or regression output.
        """
        preds = self.predict(X)
        
        # Calculate adaptive thresholds for prediction
        buy_threshold, sell_threshold = self.adaptive_threshold.calculate_thresholds(
            pd.Series(preds), X['close'] if 'close' in X.columns else X.iloc[:, 0]
        )
        
        if self.task == 'regression':
            # Map regression output to signals using adaptive thresholds
            signals = np.where(preds > buy_threshold, 'BUY', 
                             np.where(preds < sell_threshold, 'SELL', 'HOLD'))
        else:
            # For classification: +1=BUY, 0=HOLD, -1=SELL
            signals = np.where(preds == 1, 'BUY', 
                             np.where(preds == -1, 'SELL', 'HOLD'))
        
        # Log signal distribution
        signal_counts = pd.Series(signals).value_counts()
        logger.info(f"Signal distribution: {signal_counts.to_dict()}")
        
        return signals.tolist()

# --- Deep Learning Models ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(CNNLSTMModel, self).__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead), num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

# --- Reinforcement Learning Agent ---
class RLAgent:
    def __init__(self, env_id, model_type='dqn'):
        self.env = make_vec_env(env_id, n_envs=1, vec_env_cls=DummyVecEnv)
        if model_type == 'dqn':
            self.model = DQN('MlpPolicy', self.env, verbose=1)
        elif model_type == 'ppo':
            self.model = PPO('MlpPolicy', self.env, verbose=1)
        else:
            raise ValueError("Model type must be 'dqn' or 'ppo'")

    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, observation):
        return self.model.predict(observation)

# --- Custom Loss and Metrics ---
def asymmetric_loss(y_pred, y_true, alpha=0.5):
    """
    Asymmetric loss function penalizing wrong direction more.
    """
    loss = torch.where(y_true * y_pred > 0, torch.abs(y_true - y_pred), alpha * torch.abs(y_true - y_pred))
    return loss.mean()

def directional_accuracy(y_pred, y_true):
    """
    Calculate directional accuracy.
    """
    return np.mean(np.sign(y_pred) == np.sign(y_true))

# Example usage
if __name__ == "__main__":
    # Generate dummy data for demonstration
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(500, 10), columns=[f'f{i}' for i in range(10)])
    y_reg = pd.Series(np.random.randn(500))
    y_cls = pd.Series(np.random.choice([-1, 0, 1], size=500))
    
    # Regression stack
    print("\n--- Regression Ensemble ---")
    stack_reg = EnsembleStack(task='regression')
    stack_reg.fit(X, y_reg)
    stack_reg.evaluate(X, y_reg)
    print("Signals:", stack_reg.predict_signal(X[:10]))
    
    # Classification stack
    print("\n--- Classification Ensemble ---")
    stack_cls = EnsembleStack(task='classification')
    stack_cls.fit(X, y_cls)
    stack_cls.evaluate(X, y_cls)
    print("Signals:", stack_cls.predict_signal(X[:10])) 