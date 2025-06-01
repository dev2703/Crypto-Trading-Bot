import numpy as np
from typing import Dict, List, Tuple

def combine_signals(signals: Dict[str, int]) -> int:
    """
    Combine signals from multiple strategies using majority voting.
    
    Args:
        signals: Dictionary of signals from each strategy (e.g., {'sentiment': 1, 'statistical_arbitrage': 0, 'market_microstructure': -1}).
    
    Returns:
        Final signal: 1 (buy), 0 (hold), or -1 (sell) based on majority vote.
        If there's a tie, returns 0 (hold).
    """
    # Count votes for each signal type
    vote_counts = {1: 0, 0: 0, -1: 0}
    for signal in signals.values():
        vote_counts[signal] += 1
    
    # Find the signal with the most votes
    max_votes = max(vote_counts.values())
    winning_signals = [signal for signal, votes in vote_counts.items() if votes == max_votes]
    
    # If there's a tie, return hold (0)
    if len(winning_signals) > 1:
        return 0
    
    return winning_signals[0]

# Example usage
if __name__ == "__main__":
    # Example 1: Clear majority
    signals1 = {
        'sentiment': 1,
        'statistical_arbitrage': 1,
        'market_microstructure': -1
    }
    final_signal1 = combine_signals(signals1)
    print(f"Example 1 - Final Signal: {final_signal1}")  # Should return 1 (buy)
    
    # Example 2: Tie between buy and sell
    signals2 = {
        'sentiment': 1,
        'statistical_arbitrage': -1,
        'market_microstructure': 0
    }
    final_signal2 = combine_signals(signals2)
    print(f"Example 2 - Final Signal: {final_signal2}")  # Should return 0 (hold) 