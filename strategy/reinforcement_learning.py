"""
Reinforcement Learning Strategy Module
- Defines the trading environment and state space
- Implements the Q-learning algorithm
- Generates trading signals
- Includes backtesting and performance evaluation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from strategy.performance_evaluation import compute_performance_metrics

class TradingEnvironment:
    """Trading environment for reinforcement learning."""
    def __init__(self, prices: pd.Series, window_size: int = 10):
        self.prices = prices
        self.window_size = window_size
        self.current_step = window_size
        self.max_steps = len(prices) - window_size
        self.state = self._get_state()
        self.action_space = [-1, 0, 1]  # Sell, Hold, Buy

    def _get_state(self) -> np.ndarray:
        """Get the current state."""
        window = self.prices[self.current_step - self.window_size:self.current_step]
        returns = window.pct_change().dropna()
        return np.array([
            returns.mean(),
            returns.std(),
            returns.skew(),
            returns.kurtosis()
        ])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Take a step in the environment."""
        self.current_step += 1
        next_state = self._get_state()
        reward = self._compute_reward(action)
        done = self.current_step >= self.max_steps
        return next_state, reward, done

    def _compute_reward(self, action: int) -> float:
        """Compute the reward for an action."""
        if self.current_step >= len(self.prices):
            return 0.0
        returns = self.prices.pct_change().dropna()
        if action == 0:
            return 0.0
        elif action == 1:
            return returns.iloc[self.current_step - 1]
        else:
            return -returns.iloc[self.current_step - 1]

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.current_step = self.window_size
        self.state = self._get_state()
        return self.state

class QLearningAgent:
    """Q-learning agent for trading."""
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1, discount_factor: float = 0.99, exploration_rate: float = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((state_size, action_size))

    def get_action(self, state: np.ndarray) -> int:
        """Get the action for a state."""
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state])

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Update the Q-table."""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state, action] = new_value

def backtest_strategy(prices: pd.Series, window_size: int = 10, learning_rate: float = 0.1, discount_factor: float = 0.99, exploration_rate: float = 0.1) -> Dict:
    """Backtest a reinforcement learning strategy."""
    env = TradingEnvironment(prices, window_size)
    agent = QLearningAgent(env.state_size, len(env.action_space), learning_rate, discount_factor, exploration_rate)
    signals = pd.Series(0, index=prices.index)
    for _ in range(1000):  # Training episodes
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        signals.iloc[env.current_step] = env.action_space[action]
        next_state, _, done = env.step(action)
        state = next_state
    returns = prices.pct_change().dropna()
    strategy_returns = signals.shift(1) * returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    metrics = compute_performance_metrics(strategy_returns)
    return {
        'signals': signals,
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'metrics': metrics
    }

# Example usage
if __name__ == "__main__":
    # Example: Load sample data and backtest strategy
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    result = backtest_strategy(prices)
    print(result) 