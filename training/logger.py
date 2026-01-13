"""
Logging utilities for training metrics and convergence tracking.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class TrainingLogger:
    """
    Logger for tracking training metrics and convergence.
    
    Logs episode rewards, success rates, and convergence statistics
    to JSON files for analysis.
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "experiment"):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.success_rates: List[float] = []
        self.reward_components: List[Dict[str, float]] = []
        
        # Convergence tracking
        self.convergence_step: Optional[int] = None
        self.convergence_threshold: float = 0.5
        
    def log_episode(
        self,
        episode: int,
        reward: float,
        steps: int,
        success: bool,
        reward_components: Optional[Dict[str, float]] = None
    ):
        """
        Log statistics for a single episode.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            steps: Number of steps in episode
            success: Whether episode was successful
            reward_components: Optional reward component breakdown
        """
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.success_rates.append(1.0 if success else 0.0)
        
        if reward_components:
            self.reward_components.append(reward_components.copy())
        
        # Check for convergence
        if self.convergence_step is None and reward > self.convergence_threshold:
            self.convergence_step = episode
    
    def get_statistics(self, window_size: int = 10) -> Dict[str, float]:
        """
        Get current training statistics.
        
        Args:
            window_size: Size of rolling window for recent statistics
            
        Returns:
            Dictionary with training statistics
        """
        if len(self.episode_rewards) == 0:
            return {}
        
        recent_rewards = self.episode_rewards[-window_size:] if len(self.episode_rewards) >= window_size else self.episode_rewards
        recent_success = self.success_rates[-window_size:] if len(self.success_rates) >= window_size else self.success_rates
        
        stats = {
            "total_episodes": len(self.episode_rewards),
            "mean_reward": float(np.mean(self.episode_rewards)),
            "std_reward": float(np.std(self.episode_rewards)),
            "recent_mean_reward": float(np.mean(recent_rewards)),
            "recent_std_reward": float(np.std(recent_rewards)),
            "overall_success_rate": float(np.mean(self.success_rates)),
            "recent_success_rate": float(np.mean(recent_success)),
            "convergence_step": self.convergence_step,
            "mean_episode_steps": float(np.mean(self.episode_steps)) if self.episode_steps else 0.0,
        }
        
        return stats
    
    def save(self, filename: Optional[str] = None):
        """
        Save logged data to JSON file.
        
        Args:
            filename: Optional custom filename (default: experiment_name_log.json)
        """
        if filename is None:
            filename = f"{self.experiment_name}_log.json"
        
        filepath = self.log_dir / filename
        
        data = {
            "experiment_name": self.experiment_name,
            "statistics": self.get_statistics(),
            "episode_rewards": [float(r) for r in self.episode_rewards],
            "episode_steps": self.episode_steps,
            "success_rates": [float(s) for s in self.success_rates],
            "reward_components": self.reward_components,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def reset(self):
        """Reset logger state."""
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rates = []
        self.reward_components = []
        self.convergence_step = None
