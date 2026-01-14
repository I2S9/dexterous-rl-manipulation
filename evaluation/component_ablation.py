"""
Component ablation study for dexterous manipulation.

This module systematically ablates key training components to isolate
their individual contributions to performance:
- Curriculum learning
- Dense reward shaping

Tests all combinations to quantify the impact of each component.
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs import DexterousManipulationEnv
from policies import RandomPolicy
from experiments import CurriculumConfig, CurriculumScheduler
from rewards import RewardShaping, SparseReward


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    use_curriculum: bool
    use_dense_reward: bool
    name: str = ""
    
    def __post_init__(self):
        """Generate name if not provided."""
        if not self.name:
            parts = []
            if self.use_curriculum:
                parts.append("curriculum")
            else:
                parts.append("no-curriculum")
            if self.use_dense_reward:
                parts.append("dense-reward")
            else:
                parts.append("sparse-reward")
            self.name = "_".join(parts)


@dataclass
class TrainingResults:
    """Results from a training run."""
    config: AblationConfig
    episode_rewards: List[float]
    episode_steps: List[int]
    success_rates: List[float]
    final_success_rate: float
    mean_episode_length: float
    convergence_step: Optional[int]
    total_episodes: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": asdict(self.config),
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "success_rates": self.success_rates,
            "final_success_rate": float(self.final_success_rate),
            "mean_episode_length": float(self.mean_episode_length),
            "convergence_step": self.convergence_step,
            "total_episodes": self.total_episodes,
        }


class SimpleLearner:
    """
    Simple learning policy that improves over time.
    
    This is a simplified learning mechanism to demonstrate
    component effects without full RL implementation.
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


def run_episode(
    env: DexterousManipulationEnv,
    policy,
    max_steps: int = 200
) -> Tuple[bool, int, float]:
    """
    Run a single episode.
    
    Args:
        env: Environment to run
        policy: Policy to use
        max_steps: Maximum steps per episode
        
    Returns:
        success: Whether the episode was successful
        steps: Number of steps taken
        total_reward: Total reward accumulated
    """
    obs, info = env.reset()
    policy.reset()
    
    total_reward = 0.0
    success = False
    
    for step in range(max_steps):
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        policy.update(reward)
        
        if terminated or truncated:
            success = info.get("success", False)
            break
    
    return success, step + 1, total_reward


def train_with_config(
    config: AblationConfig,
    num_episodes: int = 200,
    seed: int = 42
) -> TrainingResults:
    """
    Train with a specific ablation configuration.
    
    Args:
        config: Ablation configuration
        num_episodes: Number of training episodes
        seed: Random seed
        
    Returns:
        Training results
    """
    np.random.seed(seed)
    
    # Setup curriculum if enabled
    scheduler = None
    if config.use_curriculum:
        initial_config = CurriculumConfig.easy()
        target_config = CurriculumConfig.hard()
        
        scheduler = CurriculumScheduler(
            initial_config=initial_config,
            target_config=target_config,
            success_rate_threshold=0.3,
            window_size=15,
            min_episodes_before_progression=20,
            progression_steps=5,
        )
        curriculum_config = scheduler.get_current_config()
    else:
        # Use fixed hard configuration for fair comparison
        curriculum_config = CurriculumConfig.hard()
    
    # Setup environment
    reward_type = "dense" if config.use_dense_reward else "sparse"
    env = DexterousManipulationEnv(
        curriculum_config=curriculum_config,
        reward_type=reward_type
    )
    
    policy = SimpleLearner(env.action_space, learning_rate=0.01)
    
    episode_rewards = []
    episode_steps = []
    success_rates = []
    
    for episode in range(num_episodes):
        success, steps, reward = run_episode(env, policy)
        
        # Update curriculum if enabled
        if scheduler is not None:
            progression = scheduler.update(success, steps)
            if progression:
                env.curriculum_config = scheduler.get_current_config()
        
        episode_rewards.append(reward)
        episode_steps.append(steps)
        success_rates.append(1.0 if success else 0.0)
    
    env.close()
    
    # Compute convergence step (first episode where success rate >= 0.5 for window)
    convergence_step = None
    window_size = 20
    for i in range(window_size, len(success_rates)):
        window_success = np.mean(success_rates[i - window_size:i])
        if window_success >= 0.5:
            convergence_step = i
            break
    
    final_success_rate = np.mean(success_rates[-window_size:]) if len(success_rates) >= window_size else np.mean(success_rates)
    mean_episode_length = np.mean(episode_steps)
    
    return TrainingResults(
        config=config,
        episode_rewards=episode_rewards,
        episode_steps=episode_steps,
        success_rates=success_rates,
        final_success_rate=final_success_rate,
        mean_episode_length=mean_episode_length,
        convergence_step=convergence_step,
        total_episodes=num_episodes,
    )


def run_component_ablation(
    num_episodes: int = 200,
    seeds: List[int] = [42, 123, 456],
    output_dir: str = "logs"
) -> Dict[str, List[TrainingResults]]:
    """
    Run complete component ablation study.
    
    Tests all combinations of:
    - Curriculum learning (on/off)
    - Dense reward (on/off)
    
    Args:
        num_episodes: Number of episodes per configuration
        seeds: List of random seeds for multiple runs
        output_dir: Directory to save results
        
    Returns:
        Dictionary mapping config names to lists of results (one per seed)
    """
    # Create all ablation configurations
    configs = [
        AblationConfig(use_curriculum=True, use_dense_reward=True, name="baseline"),
        AblationConfig(use_curriculum=False, use_dense_reward=True, name="no_curriculum"),
        AblationConfig(use_curriculum=True, use_dense_reward=False, name="no_dense_reward"),
        AblationConfig(use_curriculum=False, use_dense_reward=False, name="minimal"),
    ]
    
    all_results = {}
    
    print("=" * 80)
    print("Component Ablation Study")
    print("=" * 80)
    print(f"Testing {len(configs)} configurations with {len(seeds)} seeds each")
    print(f"Episodes per run: {num_episodes}")
    print("=" * 80)
    
    for config in configs:
        print(f"\nTesting: {config.name}")
        print(f"  Curriculum: {config.use_curriculum}")
        print(f"  Dense reward: {config.use_dense_reward}")
        
        config_results = []
        
        for seed_idx, seed in enumerate(seeds):
            print(f"  Seed {seed_idx + 1}/{len(seeds)} (seed={seed})...", end=" ", flush=True)
            
            results = train_with_config(config, num_episodes=num_episodes, seed=seed)
            config_results.append(results)
            
            print(f"Success rate: {results.final_success_rate:.3f}")
        
        all_results[config.name] = config_results
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / "component_ablation_results.json"
    
    # Convert to JSON-serializable format
    output_data = {}
    for config_name, results_list in all_results.items():
        output_data[config_name] = [r.to_dict() for r in results_list]
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return all_results


def compute_ablation_statistics(
    all_results: Dict[str, List[TrainingResults]]
) -> Dict[str, Dict]:
    """
    Compute statistics across seeds for each configuration.
    
    Args:
        all_results: Dictionary mapping config names to lists of results
        
    Returns:
        Dictionary with statistics for each configuration
    """
    stats = {}
    
    for config_name, results_list in all_results.items():
        final_success_rates = [r.final_success_rate for r in results_list]
        mean_episode_lengths = [r.mean_episode_length for r in results_list]
        convergence_steps = [r.convergence_step for r in results_list if r.convergence_step is not None]
        
        stats[config_name] = {
            "final_success_rate": {
                "mean": float(np.mean(final_success_rates)),
                "std": float(np.std(final_success_rates)),
                "min": float(np.min(final_success_rates)),
                "max": float(np.max(final_success_rates)),
            },
            "mean_episode_length": {
                "mean": float(np.mean(mean_episode_lengths)),
                "std": float(np.std(mean_episode_lengths)),
            },
            "convergence_step": {
                "mean": float(np.mean(convergence_steps)) if convergence_steps else None,
                "std": float(np.std(convergence_steps)) if convergence_steps else None,
                "num_converged": len(convergence_steps),
            },
        }
    
    return stats


def print_ablation_report(
    all_results: Dict[str, List[TrainingResults]],
    stats: Dict[str, Dict]
):
    """
    Print formatted ablation report.
    
    Args:
        all_results: Dictionary mapping config names to lists of results
        stats: Statistics dictionary
    """
    print("\n" + "=" * 80)
    print("Component Ablation Report")
    print("=" * 80)
    
    # Baseline for comparison
    baseline_name = "baseline"
    if baseline_name not in stats:
        baseline_name = list(stats.keys())[0]
    
    baseline_success = stats[baseline_name]["final_success_rate"]["mean"]
    
    print(f"\nBaseline (Curriculum + Dense Reward):")
    print(f"  Final Success Rate: {baseline_success:.3f} +/- {stats[baseline_name]['final_success_rate']['std']:.3f}")
    
    print("\n" + "-" * 80)
    print("Component Contributions:")
    print("-" * 80)
    
    for config_name, config_stats in stats.items():
        if config_name == baseline_name:
            continue
        
        success_mean = config_stats["final_success_rate"]["mean"]
        success_std = config_stats["final_success_rate"]["std"]
        diff = success_mean - baseline_success
        diff_pct = (diff / baseline_success * 100) if baseline_success > 0 else 0
        
        print(f"\n{config_name.replace('_', ' ').title()}:")
        print(f"  Final Success Rate: {success_mean:.3f} +/- {success_std:.3f}")
        print(f"  Difference from baseline: {diff:+.3f} ({diff_pct:+.1f}%)")
        
        if config_stats["convergence_step"]["mean"] is not None:
            conv_mean = config_stats["convergence_step"]["mean"]
            conv_std = config_stats["convergence_step"]["std"]
            print(f"  Convergence step: {conv_mean:.0f} +/- {conv_std:.0f}")
        else:
            print(f"  Convergence: Not reached")
    
    print("\n" + "-" * 80)
    print("Key Insights:")
    print("-" * 80)
    
    # Compare no curriculum vs baseline
    if "no_curriculum" in stats:
        no_curriculum_success = stats["no_curriculum"]["final_success_rate"]["mean"]
        curriculum_impact = baseline_success - no_curriculum_success
        curriculum_impact_pct = (curriculum_impact / baseline_success * 100) if baseline_success > 0 else 0
        print(f"\n1. Curriculum Learning Impact:")
        print(f"   Without curriculum: {no_curriculum_success:.3f}")
        print(f"   With curriculum: {baseline_success:.3f}")
        print(f"   Improvement: {curriculum_impact:+.3f} ({curriculum_impact_pct:+.1f}%)")
    
    # Compare no dense reward vs baseline
    if "no_dense_reward" in stats:
        no_dense_success = stats["no_dense_reward"]["final_success_rate"]["mean"]
        dense_reward_impact = baseline_success - no_dense_success
        dense_reward_impact_pct = (dense_reward_impact / baseline_success * 100) if baseline_success > 0 else 0
        print(f"\n2. Dense Reward Impact:")
        print(f"   Without dense reward: {no_dense_success:.3f}")
        print(f"   With dense reward: {baseline_success:.3f}")
        print(f"   Improvement: {dense_reward_impact:+.3f} ({dense_reward_impact_pct:+.1f}%)")
    
    # Compare minimal vs baseline
    if "minimal" in stats:
        minimal_success = stats["minimal"]["final_success_rate"]["mean"]
        total_impact = baseline_success - minimal_success
        total_impact_pct = (total_impact / baseline_success * 100) if baseline_success > 0 else 0
        print(f"\n3. Combined Impact:")
        print(f"   Minimal (no curriculum, sparse reward): {minimal_success:.3f}")
        print(f"   Full system (curriculum + dense reward): {baseline_success:.3f}")
        print(f"   Total improvement: {total_impact:+.3f} ({total_impact_pct:+.1f}%)")
    
    print("\n" + "=" * 80)
