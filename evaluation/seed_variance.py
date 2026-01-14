"""
Seed variance analysis for stability evaluation.

This module evaluates the variance across different random seeds
to assess the stability and reproducibility of results.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from evaluation.evaluator import Evaluator
from evaluation.metrics import EvaluationMetrics
from evaluation.heldout_objects import HeldOutObjectSet


class SeedVarianceAnalyzer:
    """
    Analyzer for evaluating variance across different random seeds.
    """
    
    def __init__(
        self,
        policy,
        heldout_set: HeldOutObjectSet,
        reward_type: str = "dense",
        max_episode_steps: int = 200
    ):
        """
        Initialize seed variance analyzer.
        
        Args:
            policy: Policy to evaluate
            heldout_set: Held-out object set
            reward_type: Type of reward
            max_episode_steps: Maximum steps per episode
        """
        self.policy = policy
        self.heldout_set = heldout_set
        self.reward_type = reward_type
        self.max_episode_steps = max_episode_steps
    
    def evaluate_multiple_seeds(
        self,
        seeds: List[int],
        num_episodes_per_object: int = 5
    ) -> Dict:
        """
        Evaluate policy with multiple seeds.
        
        Args:
            seeds: List of random seeds to test
            num_episodes_per_object: Number of episodes per held-out object
            
        Returns:
            Dictionary with results for each seed
        """
        results_by_seed = {}
        
        for seed in seeds:
            print(f"  Evaluating with seed {seed}...")
            
            evaluator = Evaluator(
                policy=self.policy,
                heldout_set=self.heldout_set,
                reward_type=self.reward_type,
                max_episode_steps=self.max_episode_steps
            )
            
            results = evaluator.evaluate_heldout_set(
                num_episodes_per_object=num_episodes_per_object,
                seed=seed
            )
            
            results_by_seed[seed] = results
        
        return results_by_seed
    
    def compute_variance_statistics(
        self,
        results_by_seed: Dict
    ) -> Dict:
        """
        Compute variance statistics across seeds.
        
        Args:
            results_by_seed: Dictionary mapping seeds to evaluation results
            
        Returns:
            Dictionary with variance statistics
        """
        seeds = list(results_by_seed.keys())
        
        # Extract metrics for each seed
        metrics_by_seed = {}
        for seed, results in results_by_seed.items():
            metrics = results.get("metrics", {})
            metrics_by_seed[seed] = metrics
        
        # Aggregate metrics across seeds
        variance_stats = {}
        
        # Success rate
        success_rates = [m.get("grasp_success_rate", 0.0) for m in metrics_by_seed.values()]
        variance_stats["grasp_success_rate"] = {
            "mean": float(np.mean(success_rates)),
            "std": float(np.std(success_rates)),
            "min": float(np.min(success_rates)),
            "max": float(np.max(success_rates)),
            "cv": float(np.std(success_rates) / np.mean(success_rates)) if np.mean(success_rates) > 0 else 0.0,  # Coefficient of variation
            "values": success_rates,
        }
        
        # Mean episode length
        episode_lengths = [m.get("mean_episode_length", 0.0) for m in metrics_by_seed.values()]
        variance_stats["mean_episode_length"] = {
            "mean": float(np.mean(episode_lengths)),
            "std": float(np.std(episode_lengths)),
            "min": float(np.min(episode_lengths)),
            "max": float(np.max(episode_lengths)),
            "cv": float(np.std(episode_lengths) / np.mean(episode_lengths)) if np.mean(episode_lengths) > 0 else 0.0,
            "values": episode_lengths,
        }
        
        # Overall success rate
        overall_success_rates = [m.get("overall_success_rate", 0.0) for m in metrics_by_seed.values()]
        variance_stats["overall_success_rate"] = {
            "mean": float(np.mean(overall_success_rates)),
            "std": float(np.std(overall_success_rates)),
            "min": float(np.min(overall_success_rates)),
            "max": float(np.max(overall_success_rates)),
            "cv": float(np.std(overall_success_rates) / np.mean(overall_success_rates)) if np.mean(overall_success_rates) > 0 else 0.0,
            "values": overall_success_rates,
        }
        
        # Mean reward
        mean_rewards = [m.get("mean_reward", 0.0) for m in metrics_by_seed.values()]
        variance_stats["mean_reward"] = {
            "mean": float(np.mean(mean_rewards)),
            "std": float(np.std(mean_rewards)),
            "min": float(np.min(mean_rewards)),
            "max": float(np.max(mean_rewards)),
            "cv": float(np.std(mean_rewards) / np.mean(mean_rewards)) if np.mean(mean_rewards) > 0 else 0.0,
            "values": mean_rewards,
        }
        
        return {
            "seeds": seeds,
            "num_seeds": len(seeds),
            "variance_stats": variance_stats,
            "individual_results": results_by_seed,
        }
    
    def validate_variance(
        self,
        variance_stats: Dict,
        max_cv_threshold: float = 0.2
    ) -> Dict[str, bool]:
        """
        Validate that variance is controlled.
        
        Args:
            variance_stats: Dictionary with variance statistics
            max_cv_threshold: Maximum coefficient of variation threshold
            
        Returns:
            Dictionary with validation results for each metric
        """
        validation = {}
        
        for metric_name, stats in variance_stats.items():
            cv = stats.get("cv", 0.0)
            std = stats.get("std", 0.0)
            mean = stats.get("mean", 0.0)
            
            # Check coefficient of variation
            cv_ok = cv <= max_cv_threshold
            
            # Check relative standard deviation
            relative_std = (std / abs(mean)) if mean != 0 else 0.0
            relative_std_ok = relative_std <= max_cv_threshold
            
            # Overall validation
            is_controlled = cv_ok and relative_std_ok
            
            validation[metric_name] = {
                "is_controlled": is_controlled,
                "cv": cv,
                "cv_threshold": max_cv_threshold,
                "cv_ok": cv_ok,
                "relative_std": relative_std,
                "relative_std_ok": relative_std_ok,
            }
        
        return validation


def plot_seed_variance(
    variance_analysis: Dict,
    output_path: str = "logs/seed_variance.png"
) -> None:
    """
    Plot seed variance analysis.
    
    Args:
        variance_analysis: Dictionary with variance analysis results
        output_path: Path to save the plot
    """
    variance_stats = variance_analysis["variance_stats"]
    seeds = variance_analysis["seeds"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Success rate across seeds
    ax1 = axes[0, 0]
    success_rates = variance_stats["grasp_success_rate"]["values"]
    ax1.plot(seeds, success_rates, 'o-', linewidth=2, markersize=8, label="Success Rate")
    ax1.axhline(variance_stats["grasp_success_rate"]["mean"], color='r', 
                linestyle='--', alpha=0.7, label=f"Mean: {variance_stats['grasp_success_rate']['mean']:.3f}")
    ax1.fill_between(seeds, 
                    variance_stats["grasp_success_rate"]["mean"] - variance_stats["grasp_success_rate"]["std"],
                    variance_stats["grasp_success_rate"]["mean"] + variance_stats["grasp_success_rate"]["std"],
                    alpha=0.2, color='red', label=f"±1 std: {variance_stats['grasp_success_rate']['std']:.3f}")
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Grasp Success Rate")
    ax1.set_title("Success Rate Variance Across Seeds")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Plot 2: Mean episode length across seeds
    ax2 = axes[0, 1]
    episode_lengths = variance_stats["mean_episode_length"]["values"]
    ax2.plot(seeds, episode_lengths, 's-', linewidth=2, markersize=8, 
            color='orange', label="Mean Episode Length")
    ax2.axhline(variance_stats["mean_episode_length"]["mean"], color='r', 
                linestyle='--', alpha=0.7, label=f"Mean: {variance_stats['mean_episode_length']['mean']:.1f}")
    ax2.fill_between(seeds,
                     variance_stats["mean_episode_length"]["mean"] - variance_stats["mean_episode_length"]["std"],
                     variance_stats["mean_episode_length"]["mean"] + variance_stats["mean_episode_length"]["std"],
                     alpha=0.2, color='red', label=f"±1 std: {variance_stats['mean_episode_length']['std']:.1f}")
    ax2.set_xlabel("Seed")
    ax2.set_ylabel("Mean Episode Length (steps)")
    ax2.set_title("Episode Length Variance Across Seeds")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot of all metrics
    ax3 = axes[1, 0]
    metrics_to_plot = ["grasp_success_rate", "mean_episode_length", "overall_success_rate"]
    data_to_plot = [variance_stats[m]["values"] for m in metrics_to_plot]
    bp = ax3.boxplot(data_to_plot, labels=["Success Rate", "Episode Length", "Overall Success"], 
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    bp['boxes'][2].set_facecolor('lightgreen')
    ax3.set_ylabel("Value")
    ax3.set_title("Metric Distribution Across Seeds")
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Coefficient of variation
    ax4 = axes[1, 1]
    metric_names = list(variance_stats.keys())
    cvs = [variance_stats[m]["cv"] for m in metric_names]
    colors = ['green' if cv <= 0.2 else 'orange' if cv <= 0.5 else 'red' for cv in cvs]
    bars = ax4.bar(metric_names, cvs, color=colors, alpha=0.7)
    ax4.axhline(0.2, color='r', linestyle='--', label="Threshold (0.2)", alpha=0.7)
    ax4.set_xlabel("Metric")
    ax4.set_ylabel("Coefficient of Variation")
    ax4.set_title("Variance Stability (CV)")
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, cv in zip(bars, cvs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{cv:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Seed variance plot saved to {output_path}")


def print_variance_report(variance_analysis: Dict, validation: Dict):
    """
    Print formatted variance analysis report.
    
    Args:
        variance_analysis: Dictionary with variance analysis results
        validation: Dictionary with validation results
    """
    print("\n" + "=" * 60)
    print("Seed Variance Analysis Report")
    print("=" * 60)
    
    seeds = variance_analysis["seeds"]
    variance_stats = variance_analysis["variance_stats"]
    
    print(f"\nSeeds tested: {seeds}")
    print(f"Number of seeds: {len(seeds)}")
    
    print(f"\nVariance Statistics:")
    for metric_name, stats in variance_stats.items():
        print(f"\n  {metric_name}:")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Std:  {stats['std']:.4f}")
        print(f"    Min:  {stats['min']:.4f}")
        print(f"    Max:  {stats['max']:.4f}")
        print(f"    CV:   {stats['cv']:.4f} (Coefficient of Variation)")
        
        # Show validation status
        if metric_name in validation:
            val = validation[metric_name]
            status = "[PASS]" if val["is_controlled"] else "[FAIL]"
            print(f"    Status: {status} (CV threshold: {val['cv_threshold']})")
    
    print(f"\nValidation Summary:")
    all_controlled = all(v["is_controlled"] for v in validation.values())
    for metric_name, val in validation.items():
        status = "CONTROLLED" if val["is_controlled"] else "HIGH VARIANCE"
        print(f"  {metric_name}: {status} (CV: {val['cv']:.4f})")
    
    if all_controlled:
        print(f"\n[PASS] All metrics show controlled variance across seeds")
    else:
        print(f"\n[WARNING] Some metrics show high variance across seeds")
    
    print("=" * 60)


def analyze_seed_variance(
    policy,
    heldout_set: HeldOutObjectSet,
    seeds: List[int],
    num_episodes_per_object: int = 5,
    reward_type: str = "dense",
    max_episode_steps: int = 200,
    output_dir: str = "logs",
    max_cv_threshold: float = 0.2
) -> Dict:
    """
    Perform comprehensive seed variance analysis.
    
    Args:
        policy: Policy to evaluate
        heldout_set: Held-out object set
        seeds: List of random seeds to test (≥ 3)
        num_episodes_per_object: Number of episodes per held-out object
        reward_type: Type of reward
        max_episode_steps: Maximum steps per episode
        output_dir: Directory to save results
        max_cv_threshold: Maximum coefficient of variation threshold
        
    Returns:
        Dictionary with complete analysis results
    """
    print("=" * 60)
    print("Seed Variance Analysis")
    print("=" * 60)
    
    if len(seeds) < 3:
        raise ValueError("At least 3 seeds required for variance analysis")
    
    print(f"\nEvaluating with {len(seeds)} seeds: {seeds}")
    
    # Create analyzer
    analyzer = SeedVarianceAnalyzer(
        policy=policy,
        heldout_set=heldout_set,
        reward_type=reward_type,
        max_episode_steps=max_episode_steps
    )
    
    # Evaluate with multiple seeds
    print("\n1. Running evaluations with multiple seeds...")
    results_by_seed = analyzer.evaluate_multiple_seeds(
        seeds=seeds,
        num_episodes_per_object=num_episodes_per_object
    )
    
    # Compute variance statistics
    print("\n2. Computing variance statistics...")
    variance_analysis = analyzer.compute_variance_statistics(results_by_seed)
    
    # Validate variance
    print("\n3. Validating variance...")
    validation = analyzer.validate_variance(
        variance_analysis["variance_stats"],
        max_cv_threshold=max_cv_threshold
    )
    
    # Print report
    print_variance_report(variance_analysis, validation)
    
    # Generate plots
    print("\n4. Generating visualizations...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_path / "seed_variance.png"
    plot_seed_variance(variance_analysis, str(plot_path))
    
    # Save results
    print("\n5. Saving results...")
    results_path = output_path / "seed_variance_analysis.json"
    
    # Convert to JSON-serializable format
    def convert_to_json(obj):
        if isinstance(obj, (np.integer, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return [convert_to_json(item) for item in obj]
        return obj
    
    json_data = {
        "variance_analysis": convert_to_json(variance_analysis),
        "validation": convert_to_json(validation),
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"   Results saved to {results_path}")
    
    print("\n" + "=" * 60)
    print("Seed Variance Analysis Complete")
    print("=" * 60)
    
    return {
        "variance_analysis": variance_analysis,
        "validation": validation,
    }
