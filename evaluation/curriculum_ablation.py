"""
Curriculum learning ablation study.

This script compares training with and without curriculum learning
to measure the benefits in terms of convergence speed and variance.
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs import DexterousManipulationEnv
from policies import RandomPolicy
from experiments import CurriculumConfig, CurriculumScheduler, CurriculumLogger
from training.logger import TrainingLogger


class SimpleLearner:
    """
    Simple learning policy that improves over time.
    
    This is a simplified learning mechanism to demonstrate
    curriculum effects without full RL implementation.
    """
    
    def __init__(self, action_space, learning_rate: float = 0.01):
        """Initialize simple learner."""
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.mean_action = np.zeros(action_space.shape[0], dtype=np.float32)
        self.best_reward = -np.inf
    
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """Select action with exploration."""
        noise = np.random.normal(0, 0.3, size=self.mean_action.shape).astype(np.float32)
        action = self.mean_action + noise
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action
    
    def update(self, reward: float):
        """Update policy based on reward."""
        if reward > self.best_reward:
            adjustment = np.random.normal(0, self.learning_rate, size=self.mean_action.shape)
            self.mean_action += adjustment
            self.mean_action = np.clip(self.mean_action, -0.5, 0.5)
            self.best_reward = reward
    
    def reset(self):
        """Reset policy state."""
        self.best_reward = -np.inf


def run_episode(env, policy, max_steps=200):
    """Run a single episode."""
    obs, info = env.reset()
    policy.reset()
    
    episode_reward = 0.0
    episode_steps = 0
    success = False
    
    for step in range(max_steps):
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_steps += 1
        
        if terminated or truncated:
            success = terminated
            break
    
    if hasattr(policy, 'update'):
        policy.update(episode_reward)
    
    return success, episode_steps, episode_reward


def train_with_curriculum(
    num_episodes: int = 200,
    seed: int = 42
) -> Dict:
    """
    Train with curriculum learning.
    
    Returns:
        Dictionary with training statistics
    """
    np.random.seed(seed)
    
    # Initialize curriculum
    initial_config = CurriculumConfig.easy()
    target_config = CurriculumConfig.hard()
    
    scheduler = CurriculumScheduler(
        initial_config=initial_config,
        target_config=target_config,
        success_rate_threshold=0.3,  # Lower for demonstration
        window_size=15,
        min_episodes_before_progression=20,
        progression_steps=5,
    )
    
    env = DexterousManipulationEnv(
        curriculum_config=scheduler.get_current_config(),
        reward_type="dense"
    )
    
    policy = SimpleLearner(env.action_space, learning_rate=0.01)
    
    episode_rewards = []
    episode_steps = []
    success_rates = []
    difficulty_levels = []
    
    for episode in range(num_episodes):
        success, steps, reward = run_episode(env, policy)
        
        progression = scheduler.update(success, steps)
        
        if progression:
            env.curriculum_config = scheduler.get_current_config()
        
        episode_rewards.append(reward)
        episode_steps.append(steps)
        success_rates.append(1.0 if success else 0.0)
        difficulty_levels.append(scheduler.get_difficulty_level())
    
    env.close()
    
    return {
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "success_rates": success_rates,
        "difficulty_levels": difficulty_levels,
        "scheduler_stats": scheduler.get_statistics(),
    }


def train_without_curriculum(
    num_episodes: int = 200,
    seed: int = 42,
    use_hard: bool = True
) -> Dict:
    """
    Train without curriculum learning (fixed difficulty).
    
    Args:
        num_episodes: Number of training episodes
        seed: Random seed
        use_hard: If True, use hard config; if False, use easy config
        
    Returns:
        Dictionary with training statistics
    """
    np.random.seed(seed)
    
    # Use fixed configuration (hard for fair comparison)
    if use_hard:
        config = CurriculumConfig.hard()
    else:
        config = CurriculumConfig.easy()
    
    env = DexterousManipulationEnv(
        curriculum_config=config,
        reward_type="dense"
    )
    
    policy = SimpleLearner(env.action_space, learning_rate=0.01)
    
    episode_rewards = []
    episode_steps = []
    success_rates = []
    
    for episode in range(num_episodes):
        success, steps, reward = run_episode(env, policy)
        
        episode_rewards.append(reward)
        episode_steps.append(steps)
        success_rates.append(1.0 if success else 0.0)
    
    env.close()
    
    return {
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "success_rates": success_rates,
    }


def compute_convergence_metrics(
    episode_rewards: List[float],
    success_rates: List[float],
    window_size: int = 20,
    threshold: float = 0.5
) -> Dict:
    """
    Compute convergence metrics.
    
    Args:
        episode_rewards: List of episode rewards
        success_rates: List of success rates
        window_size: Window size for moving average
        threshold: Threshold for convergence
        
    Returns:
        Dictionary with convergence metrics
    """
    episode_rewards = np.array(episode_rewards)
    success_rates = np.array(success_rates)
    
    # Compute moving averages
    if len(episode_rewards) >= window_size:
        reward_ma = np.convolve(
            episode_rewards, 
            np.ones(window_size) / window_size, 
            mode='valid'
        )
        success_ma = np.convolve(
            success_rates,
            np.ones(window_size) / window_size,
            mode='valid'
        )
    else:
        reward_ma = episode_rewards
        success_ma = success_rates
    
    # Find convergence point (first episode where MA exceeds threshold)
    convergence_episode = None
    for i, ma_val in enumerate(reward_ma):
        if ma_val >= threshold:
            convergence_episode = i + window_size - 1
            break
    
    # Compute variance metrics
    reward_variance = np.var(episode_rewards)
    reward_std = np.std(episode_rewards)
    
    # Final performance
    final_reward = np.mean(episode_rewards[-window_size:]) if len(episode_rewards) >= window_size else np.mean(episode_rewards)
    final_success_rate = np.mean(success_rates[-window_size:]) if len(success_rates) >= window_size else np.mean(success_rates)
    
    # Stability (coefficient of variation)
    reward_mean = np.mean(episode_rewards)
    coefficient_of_variation = reward_std / reward_mean if reward_mean != 0 else np.inf
    
    return {
        "convergence_episode": convergence_episode,
        "convergence_steps": convergence_episode * np.mean(episode_rewards) if convergence_episode else None,
        "reward_variance": float(reward_variance),
        "reward_std": float(reward_std),
        "coefficient_of_variation": float(coefficient_of_variation),
        "final_reward": float(final_reward),
        "final_success_rate": float(final_success_rate),
        "mean_reward": float(reward_mean),
        "overall_success_rate": float(np.mean(success_rates)),
    }


def run_ablation_study(
    num_episodes: int = 200,
    num_runs: int = 5,
    output_dir: str = "logs"
) -> Dict:
    """
    Run ablation study comparing with and without curriculum.
    
    Args:
        num_episodes: Number of episodes per run
        num_runs: Number of independent runs for statistical significance
        output_dir: Directory to save results
        
    Returns:
        Dictionary with ablation results
    """
    print("=" * 60)
    print("Curriculum Learning Ablation Study")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run multiple times for statistical significance
    curriculum_results = []
    no_curriculum_results = []
    
    print(f"\nRunning {num_runs} independent runs...")
    print("Training with curriculum...")
    
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...", end="\r")
        result = train_with_curriculum(num_episodes=num_episodes, seed=42 + run)
        curriculum_results.append(result)
    
    print("\nTraining without curriculum (fixed hard)...")
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...", end="\r")
        result = train_without_curriculum(num_episodes=num_episodes, seed=42 + run, use_hard=True)
        no_curriculum_results.append(result)
    
    print("\n" + "=" * 60)
    print("Computing metrics...")
    print("=" * 60)
    
    # Compute metrics for each run
    curriculum_metrics = []
    no_curriculum_metrics = []
    
    for result in curriculum_results:
        metrics = compute_convergence_metrics(
            result["episode_rewards"],
            result["success_rates"]
        )
        curriculum_metrics.append(metrics)
    
    for result in no_curriculum_results:
        metrics = compute_convergence_metrics(
            result["episode_rewards"],
            result["success_rates"]
        )
        no_curriculum_metrics.append(metrics)
    
    # Aggregate statistics
    def aggregate_metrics(metrics_list):
        """Aggregate metrics across runs."""
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if m[key] is not None]
            if values:
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                }
            else:
                aggregated[key] = None
        return aggregated
    
    curriculum_agg = aggregate_metrics(curriculum_metrics)
    no_curriculum_agg = aggregate_metrics(no_curriculum_metrics)
    
    # Compute improvements
    improvements = {}
    for key in curriculum_agg.keys():
        if curriculum_agg[key] is not None and no_curriculum_agg[key] is not None:
            cur_mean = curriculum_agg[key]["mean"]
            no_cur_mean = no_curriculum_agg[key]["mean"]
            
            if no_cur_mean != 0:
                improvement_pct = ((cur_mean - no_cur_mean) / abs(no_cur_mean)) * 100
            else:
                improvement_pct = 0.0
            
            improvements[key] = {
                "absolute": float(cur_mean - no_cur_mean),
                "percentage": float(improvement_pct),
            }
    
    # Print results
    print("\n" + "=" * 60)
    print("Ablation Study Results")
    print("=" * 60)
    
    print("\nWITH CURRICULUM:")
    print(f"  Convergence episode: {curriculum_agg['convergence_episode']['mean']:.1f} ± {curriculum_agg['convergence_episode']['std']:.1f}")
    print(f"  Reward variance: {curriculum_agg['reward_variance']['mean']:.4f} ± {curriculum_agg['reward_variance']['std']:.4f}")
    print(f"  Reward std: {curriculum_agg['reward_std']['mean']:.4f} ± {curriculum_agg['reward_std']['std']:.4f}")
    print(f"  Coefficient of variation: {curriculum_agg['coefficient_of_variation']['mean']:.4f} ± {curriculum_agg['coefficient_of_variation']['std']:.4f}")
    print(f"  Final reward: {curriculum_agg['final_reward']['mean']:.4f} ± {curriculum_agg['final_reward']['std']:.4f}")
    print(f"  Final success rate: {curriculum_agg['final_success_rate']['mean']:.4f} ± {curriculum_agg['final_success_rate']['std']:.4f}")
    
    print("\nWITHOUT CURRICULUM (Fixed Hard):")
    print(f"  Convergence episode: {no_curriculum_agg['convergence_episode']['mean']:.1f} ± {no_curriculum_agg['convergence_episode']['std']:.1f}")
    print(f"  Reward variance: {no_curriculum_agg['reward_variance']['mean']:.4f} ± {no_curriculum_agg['reward_variance']['std']:.4f}")
    print(f"  Reward std: {no_curriculum_agg['reward_std']['mean']:.4f} ± {no_curriculum_agg['reward_std']['std']:.4f}")
    print(f"  Coefficient of variation: {no_curriculum_agg['coefficient_of_variation']['mean']:.4f} ± {no_curriculum_agg['coefficient_of_variation']['std']:.4f}")
    print(f"  Final reward: {no_curriculum_agg['final_reward']['mean']:.4f} ± {no_curriculum_agg['final_reward']['std']:.4f}")
    print(f"  Final success rate: {no_curriculum_agg['final_success_rate']['mean']:.4f} ± {no_curriculum_agg['final_success_rate']['std']:.4f}")
    
    print("\nIMPROVEMENTS (Curriculum vs No Curriculum):")
    for key, imp in improvements.items():
        print(f"  {key}:")
        print(f"    Absolute: {imp['absolute']:.4f}")
        print(f"    Percentage: {imp['percentage']:.1f}%")
    
    # Save results
    results = {
        "num_runs": num_runs,
        "num_episodes": num_episodes,
        "with_curriculum": {
            "aggregated_metrics": curriculum_agg,
            "individual_metrics": curriculum_metrics,
        },
        "without_curriculum": {
            "aggregated_metrics": no_curriculum_agg,
            "individual_metrics": no_curriculum_metrics,
        },
        "improvements": improvements,
    }
    
    output_path = Path(output_dir) / "curriculum_ablation.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = run_ablation_study(num_episodes=150, num_runs=3)
