"""
Random policy for testing environments.

This module provides a simple random action policy that samples
actions uniformly from the action space.
"""

import numpy as np
from typing import Optional


class RandomPolicy:
    """
    Policy that samples random actions from the action space.
    
    Useful for testing environments and as a baseline.
    """
    
    def __init__(self, action_space, seed: Optional[int] = None):
        """
        Initialize random policy.
        
        Args:
            action_space: Gymnasium action space
            seed: Random seed for reproducibility
        """
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)
    
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Select a random action.
        
        Args:
            observation: Current observation (unused for random policy)
            
        Returns:
            action: Random action sampled from action space
        """
        return self.action_space.sample()
    
    def reset(self):
        """Reset policy state (no-op for random policy)."""
        pass
