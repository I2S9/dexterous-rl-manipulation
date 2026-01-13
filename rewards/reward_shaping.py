"""
Reward shaping functions for dexterous manipulation.

This module provides dense reward formulations to guide learning
more effectively than sparse rewards.
"""

import numpy as np
from typing import Dict, Optional


class RewardShaping:
    """
    Base class for reward shaping functions.
    
    Provides dense reward signals to guide policy learning
    towards successful manipulation behaviors.
    """
    
    def __init__(
        self,
        distance_weight: float = 1.0,
        contact_weight: float = 0.5,
        closure_weight: float = 0.3,
        stability_weight: float = 0.2,
    ):
        """
        Initialize reward shaping parameters.
        
        Args:
            distance_weight: Weight for distance-to-object reward
            contact_weight: Weight for contact establishment reward
            closure_weight: Weight for grasp closure reward
            stability_weight: Weight for contact stability reward
        """
        self.distance_weight = distance_weight
        self.contact_weight = contact_weight
        self.closure_weight = closure_weight
        self.stability_weight = stability_weight
        
        # Track previous state for stability computation
        self.prev_contacts = None
        self.prev_distances = None
    
    def reset(self):
        """Reset internal state tracking."""
        self.prev_contacts = None
        self.prev_distances = None
    
    def compute(
        self,
        joint_positions: np.ndarray,
        finger_tips: np.ndarray,
        object_position: np.ndarray,
        contacts: np.ndarray,
        num_fingers: int,
        joints_per_finger: int,
    ) -> Dict[str, float]:
        """
        Compute dense reward components.
        
        Args:
            joint_positions: Current joint positions
            finger_tips: Positions of finger tips
            object_position: Current object position
            contacts: Binary contact array per finger
            num_fingers: Number of fingers
            joints_per_finger: Number of joints per finger
            
        Returns:
            Dictionary with reward components and total reward
        """
        # Compute distance to object reward
        distance_reward = self._compute_distance_reward(finger_tips, object_position)
        
        # Compute contact reward
        contact_reward = self._compute_contact_reward(contacts, num_fingers)
        
        # Compute grasp closure reward
        closure_reward = self._compute_closure_reward(joint_positions, num_fingers, joints_per_finger)
        
        # Compute contact stability reward
        stability_reward = self._compute_stability_reward(contacts)
        
        # Weighted sum
        total_reward = (
            self.distance_weight * distance_reward +
            self.contact_weight * contact_reward +
            self.closure_weight * closure_reward +
            self.stability_weight * stability_reward
        )
        
        return {
            "total": total_reward,
            "distance": distance_reward,
            "contact": contact_reward,
            "closure": closure_reward,
            "stability": stability_reward,
        }
    
    def _compute_distance_reward(
        self,
        finger_tips: np.ndarray,
        object_position: np.ndarray
    ) -> float:
        """
        Reward for minimizing distance between fingers and object.
        
        Uses exponential decay to provide smooth gradient.
        """
        distances = np.linalg.norm(finger_tips - object_position, axis=1)
        min_distance = np.min(distances)
        
        # Exponential reward: closer = higher reward
        # Max reward at distance 0, decays with distance
        distance_reward = np.exp(-5.0 * min_distance)
        
        return float(distance_reward)
    
    def _compute_contact_reward(
        self,
        contacts: np.ndarray,
        num_fingers: int
    ) -> float:
        """
        Reward for establishing contacts with the object.
        
        Encourages multiple fingers to make contact.
        """
        num_contacts = np.sum(contacts > 0.5)
        
        # Linear reward for number of contacts, normalized
        # Optimal: all fingers in contact
        contact_reward = num_contacts / num_fingers
        
        return float(contact_reward)
    
    def _compute_closure_reward(
        self,
        joint_positions: np.ndarray,
        num_fingers: int,
        joints_per_finger: int
    ) -> float:
        """
        Reward for closing the hand (grasp closure).
        
        Measures how closed the hand is, encouraging grasping motion.
        """
        closure_scores = []
        
        for i in range(num_fingers):
            finger_joints = joint_positions[i * joints_per_finger:(i + 1) * joints_per_finger]
            # Negative joint positions indicate closing (in our convention)
            # Sum of negative joint positions = closure measure
            closure = -np.sum(finger_joints[finger_joints < 0])
            closure_scores.append(closure)
        
        # Normalize by number of fingers
        avg_closure = np.mean(closure_scores) if closure_scores else 0.0
        
        # Normalize to [0, 1] range (assuming max closure around 1.0)
        closure_reward = np.clip(avg_closure / num_fingers, 0.0, 1.0)
        
        return float(closure_reward)
    
    def _compute_stability_reward(self, contacts: np.ndarray) -> float:
        """
        Reward for maintaining stable contacts over time.
        
        Penalizes contact changes (fingers losing/gaining contact).
        """
        if self.prev_contacts is None:
            # First step: no stability penalty
            self.prev_contacts = contacts.copy()
            return 0.0
        
        # Count contact changes
        contact_changes = np.sum(np.abs(contacts - self.prev_contacts))
        
        # Reward stability: fewer changes = higher reward
        # Normalize by number of fingers
        stability_reward = 1.0 - (contact_changes / len(contacts))
        stability_reward = np.clip(stability_reward, 0.0, 1.0)
        
        self.prev_contacts = contacts.copy()
        
        return float(stability_reward)


class SparseReward:
    """
    Sparse reward function (baseline).
    
    Only provides reward upon task completion.
    """
    
    def __init__(self):
        """Initialize sparse reward function."""
        pass
    
    def reset(self):
        """Reset internal state (no-op for sparse reward)."""
        pass
    
    def compute(
        self,
        joint_positions: np.ndarray,
        finger_tips: np.ndarray,
        object_position: np.ndarray,
        contacts: np.ndarray,
        num_fingers: int,
        joints_per_finger: int,
    ) -> Dict[str, float]:
        """
        Compute sparse reward.
        
        Args:
            joint_positions: Current joint positions (unused)
            finger_tips: Positions of finger tips (unused)
            object_position: Current object position (unused)
            contacts: Binary contact array per finger
            num_fingers: Number of fingers
            joints_per_finger: Number of joints per finger (unused)
            
        Returns:
            Dictionary with reward components and total reward
        """
        num_contacts = np.sum(contacts > 0.5)
        
        # Sparse reward: success if 3+ fingers in contact
        if num_contacts >= 3:
            total_reward = 1.0
        else:
            total_reward = -0.01  # Small negative reward for exploration
        
        return {
            "total": total_reward,
            "distance": 0.0,
            "contact": 0.0,
            "closure": 0.0,
            "stability": 0.0,
        }
