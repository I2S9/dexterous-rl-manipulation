"""
Compare sparse vs dense reward formulations.

This script runs training simulations to compare convergence speed
and stability between sparse and dense reward shaping.
"""

import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs import DexterousManipulationEnv
from policies import RandomPolicy


class SimpleLearner:
    """
    Simple learning policy that improves over time.
    
    This is a simplified learning mechanism to demonstrate
    reward shaping effects without full RL implementation.
    """
    
    def __init__(self, action_space, learning_rate: float = 0.01):
        """
        Initialize simple learner.
        
        Args:
            action_space: Gymnasium action space
            learning_rate: Learning rate for policy updates
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        
        # Simple policy: mean action that gets updated
        self.mean_action = np.zeros(action_space.shape[0], dtype=np.float32)
        self.best_reward = -np.inf
        
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Select action with exploration.
        
        Args:
            observation: Current observation
            
        Returns:
            action: Action with exploration noise
        """
        # Add exploration noise
        noise = np.random.normal(0, 0.3, size=self.mean_action.shape).astype(np.float32)
        action = self.mean_action + noise
        
        # Clip to action space
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        return action
    
    def update(self, reward: float):
        """
        Update policy based on reward.
        
        Simple update: if reward improved, slightly adjust mean action.
        
        Args:
            reward: Reward received
        """
        if reward > self.best_reward:
            # Small random adjustment towards better performance
            adjustment = np.random.normal(0, self.learning_rate, size=self.mean_action.shape)
            self.mean_action += adjustment
            self.mean_action = np.clip(self.mean_action, -0.5, 0.5)
            self.best_reward = reward
    
    def reset(self):
        """Reset policy state."""
        self.best_reward = -np.inf


def run_training_episode(
    env: DexterousManipulationEnv,
    policy,
    max_steps: int = 200
) -> Dict[str, float]:
    """
    Run a single training episode.
    
    Args:
        env: Environment to run
        policy: Policy to use
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary with episode statistics
    """
    obs, info = env.reset()
    policy.reset()
    
    episode_reward = 0.0
    episode_steps = 0
    reward_history = []
    
    for step in range(max_steps):
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_steps += 1
        reward_history.append(reward)
        
        # Update policy if it's a learner
        if hasattr(policy, 'update'):
            policy.update(reward)
        
        if terminated or truncated:
            break
    
    # Extract reward components if available
    reward_components = info.get("reward_components", {})
    
    return {
        "episode_reward": episode_reward,
        "episode_steps": episode_steps,
        "num_contacts": info.get("num_contacts", 0),
        "terminated": terminated,
        "truncated": truncated,
        "reward_components": reward_components,
        "reward_history": reward_history,
    }


def run_training_comparison(
    reward_type: str,
    num_episodes: int = 100,
    max_steps: int = 200,
    seed: int = 42
) -> Dict[str, List]:
    """
    Run training with a specific reward type.
    
    Args:
        reward_type: "sparse" or "dense"
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        seed: Random seed
        
    Returns:
        Dictionary with training statistics
    """
    np.random.seed(seed)
    
    env = DexterousManipulationEnv(
        num_fingers=5,
        joints_per_finger=3,
        max_episode_steps=max_steps,
        reward_type=reward_type
    )
    
    policy = SimpleLearner(env.action_space, learning_rate=0.01)
    
    episode_rewards = []
    episode_steps = []
    success_rates = []
    convergence_step = None
    
    # Track convergence: first episode where average reward over window exceeds threshold
    convergence_threshold = 0.5
    convergence_window = 10
    
    for episode in range(num_episodes):
        stats = run_training_episode(env, policy, max_steps)
        
        episode_rewards.append(stats["episode_reward"])
        episode_steps.append(stats["episode_steps"])
        success_rates.append(1.0 if stats["terminated"] else 0.0)
        
        # Check for convergence: average reward over recent window
        if convergence_step is None and len(episode_rewards) >= convergence_window:
            recent_avg = np.mean(episode_rewards[-convergence_window:])
            if recent_avg > convergence_threshold:
                convergence_step = episode - convergence_window + 1
    
    env.close()
    
    return {
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "success_rates": success_rates,
        "convergence_step": convergence_step,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "final_success_rate": np.mean(success_rates[-10:]) if len(success_rates) >= 10 else np.mean(success_rates),
    }


def compare_rewards(
    num_episodes: int = 100,
    max_steps: int = 200,
    output_dir: str = "logs"
) -> Dict[str, any]:
    """
    Compare sparse vs dense reward formulations.
    
    Args:
        num_episodes: Number of episodes per reward type
        max_steps: Maximum steps per episode
        output_dir: Directory to save results
        
    Returns:
        Dictionary with comparison results
    """
    print("=" * 60)
    print("Reward Shaping Comparison: Sparse vs Dense")
    print("=" * 60)
    
    # Run sparse reward training
    print("\nRunning sparse reward training...")
    sparse_results = run_training_comparison("sparse", num_episodes, max_steps, seed=42)
    
    # Run dense reward training
    print("Running dense reward training...")
    dense_results = run_training_comparison("dense", num_episodes, max_steps, seed=42)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("Results Comparison")
    print("=" * 60)
    print(f"\nSparse Reward:")
    print(f"  Mean reward: {sparse_results['mean_reward']:.3f} ± {sparse_results['std_reward']:.3f}")
    print(f"  Final success rate: {sparse_results['final_success_rate']:.3f}")
    print(f"  Convergence step: {sparse_results['convergence_step']}")
    
    print(f"\nDense Reward:")
    print(f"  Mean reward: {dense_results['mean_reward']:.3f} ± {dense_results['std_reward']:.3f}")
    print(f"  Final success rate: {dense_results['final_success_rate']:.3f}")
    print(f"  Convergence step: {dense_results['convergence_step']}")
    
    # Compute improvement
    if sparse_results['convergence_step'] is not None and dense_results['convergence_step'] is not None:
        improvement = ((sparse_results['convergence_step'] - dense_results['convergence_step']) / 
                      sparse_results['convergence_step']) * 100
        print(f"\nConvergence improvement: {improvement:.1f}% faster with dense rewards")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "sparse": {
            "mean_reward": float(sparse_results['mean_reward']),
            "std_reward": float(sparse_results['std_reward']),
            "final_success_rate": float(sparse_results['final_success_rate']),
            "convergence_step": sparse_results['convergence_step'],
            "episode_rewards": [float(r) for r in sparse_results['episode_rewards']],
        },
        "dense": {
            "mean_reward": float(dense_results['mean_reward']),
            "std_reward": float(dense_results['std_reward']),
            "final_success_rate": float(dense_results['final_success_rate']),
            "convergence_step": dense_results['convergence_step'],
            "episode_rewards": [float(r) for r in dense_results['episode_rewards']],
        },
    }
    
    output_path = Path(output_dir) / "reward_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    compare_rewards(num_episodes=100, max_steps=200)
