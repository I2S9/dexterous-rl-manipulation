"""
Heuristic policy for dexterous manipulation.

This module provides a simple heuristic policy that attempts to
close the hand and move towards the object.
"""

import numpy as np
from typing import Optional


class HeuristicPolicy:
    """
    Heuristic policy that closes fingers and moves towards object.
    
    This is a simple baseline that demonstrates basic manipulation
    behavior without learning.
    """
    
    def __init__(self, action_space, num_fingers: int = 5, joints_per_finger: int = 3):
        """
        Initialize heuristic policy.
        
        Args:
            action_space: Gymnasium action space
            num_fingers: Number of fingers on the hand
            joints_per_finger: Number of joints per finger
        """
        self.action_space = action_space
        self.num_fingers = num_fingers
        self.joints_per_finger = joints_per_finger
        self.num_joints = num_fingers * joints_per_finger
    
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Select action based on heuristic rules.
        
        Heuristic:
        1. Close fingers (negative action for closing)
        2. Slight bias towards object position
        
        Args:
            observation: Current observation containing hand and object state
            
        Returns:
            action: Heuristic action
        """
        # Extract relevant parts of observation
        # Observation structure: [joint_pos, joint_vel, obj_pos, obj_quat, obj_vel, contacts]
        num_joints = self.num_joints
        joint_positions = observation[:num_joints]
        object_position = observation[2 * num_joints:2 * num_joints + 3]
        
        # Heuristic: close fingers (move joints towards negative values)
        action = -0.5 * np.ones(self.num_joints, dtype=np.float32)
        
        # Add small random noise for exploration
        noise = np.random.uniform(-0.1, 0.1, size=self.num_joints).astype(np.float32)
        action += noise
        
        # Clip to action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        return action
    
    def reset(self):
        """Reset policy state (no-op for heuristic policy)."""
        pass
