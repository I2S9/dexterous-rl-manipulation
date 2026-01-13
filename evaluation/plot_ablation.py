"""
Plot curriculum ablation study results.

This script visualizes the comparison between training with and without
curriculum learning, showing convergence speed and variance differences.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


def load_ablation_results(log_dir: str = "logs") -> Dict:
    """
    Load ablation study results.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        Dictionary with ablation results
    """
    log_path = Path(log_dir) / "curriculum_ablation.json"
    
    if not log_path.exists():
        raise FileNotFoundError(f"Ablation results not found at {log_path}")
    
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_ablation_comparison(results: Dict, output_path: str = "logs/curriculum_ablation.png"):
    """
    Plot ablation study comparison.
    
    Args:
        results: Dictionary with ablation results
        output_path: Path to save the plot
    """
    curriculum_metrics = results["with_curriculum"]["individual_metrics"]
    no_curriculum_metrics = results["without_curriculum"]["individual_metrics"]
    
    # Extract convergence episodes
    curriculum_convergence = [
        m["convergence_episode"] 
        for m in curriculum_metrics 
        if m["convergence_episode"] is not None
    ]
    no_curriculum_convergence = [
        m["convergence_episode"] 
        for m in no_curriculum_metrics 
        if m["convergence_episode"] is not None
    ]
    
    # Extract variances
    curriculum_variance = [m["reward_variance"] for m in curriculum_metrics]
    no_curriculum_variance = [m["reward_variance"] for m in no_curriculum_metrics]
    
    # Extract coefficients of variation
    curriculum_cv = [m["coefficient_of_variation"] for m in curriculum_metrics]
    no_curriculum_cv = [m["coefficient_of_variation"] for m in no_curriculum_metrics]
    
    # Extract final rewards
    curriculum_final = [m["final_reward"] for m in curriculum_metrics]
    no_curriculum_final = [m["final_reward"] for m in no_curriculum_metrics]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Convergence speed comparison
    ax1 = axes[0, 0]
    if curriculum_convergence and no_curriculum_convergence:
        data = [curriculum_convergence, no_curriculum_convergence]
        bp1 = ax1.boxplot(data, tick_labels=["With Curriculum", "Without Curriculum"], 
                         patch_artist=True)
        bp1['boxes'][0].set_facecolor('lightblue')
        bp1['boxes'][1].set_facecolor('lightcoral')
        ax1.set_ylabel("Convergence Episode")
        ax1.set_title("Convergence Speed Comparison")
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Variance comparison
    ax2 = axes[0, 1]
    data = [curriculum_variance, no_curriculum_variance]
    bp2 = ax2.boxplot(data, tick_labels=["With Curriculum", "Without Curriculum"],
                     patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel("Reward Variance")
    ax2.set_title("Variance Comparison (Lower is Better)")
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Coefficient of variation (stability)
    ax3 = axes[1, 0]
    data = [curriculum_cv, no_curriculum_cv]
    bp3 = ax3.boxplot(data, tick_labels=["With Curriculum", "Without Curriculum"],
                     patch_artist=True)
    bp3['boxes'][0].set_facecolor('lightblue')
    bp3['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel("Coefficient of Variation")
    ax3.set_title("Stability Comparison (Lower is Better)")
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Final performance
    ax4 = axes[1, 1]
    data = [curriculum_final, no_curriculum_final]
    bp4 = ax4.boxplot(data, tick_labels=["With Curriculum", "Without Curriculum"],
                     patch_artist=True)
    bp4['boxes'][0].set_facecolor('lightblue')
    bp4['boxes'][1].set_facecolor('lightcoral')
    ax4.set_ylabel("Final Reward (Moving Average)")
    ax4.set_title("Final Performance Comparison")
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Ablation comparison plot saved to {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Ablation Study Summary")
    print("=" * 60)
    
    curriculum_agg = results["with_curriculum"]["aggregated_metrics"]
    no_curriculum_agg = results["without_curriculum"]["aggregated_metrics"]
    improvements = results["improvements"]
    
    print("\nConvergence Speed:")
    if curriculum_agg["convergence_episode"] and no_curriculum_agg["convergence_episode"]:
        cur_conv = curriculum_agg["convergence_episode"]["mean"]
        no_cur_conv = no_curriculum_agg["convergence_episode"]["mean"]
        speedup = ((no_cur_conv - cur_conv) / no_cur_conv) * 100 if no_cur_conv > 0 else 0
        print(f"  With curriculum: {cur_conv:.1f} episodes")
        print(f"  Without curriculum: {no_cur_conv:.1f} episodes")
        print(f"  Speedup: {speedup:.1f}%")
    
    print("\nVariance:")
    cur_var = curriculum_agg["reward_variance"]["mean"]
    no_cur_var = no_curriculum_agg["reward_variance"]["mean"]
    var_reduction = ((no_cur_var - cur_var) / no_cur_var) * 100 if no_cur_var > 0 else 0
    print(f"  With curriculum: {cur_var:.4f}")
    print(f"  Without curriculum: {no_cur_var:.4f}")
    print(f"  Variance reduction: {var_reduction:.1f}%")
    
    print("\nStability (Coefficient of Variation):")
    cur_cv = curriculum_agg["coefficient_of_variation"]["mean"]
    no_cur_cv = no_curriculum_agg["coefficient_of_variation"]["mean"]
    stability_improvement = ((no_cur_cv - cur_cv) / no_cur_cv) * 100 if no_cur_cv > 0 else 0
    print(f"  With curriculum: {cur_cv:.4f}")
    print(f"  Without curriculum: {no_cur_cv:.4f}")
    print(f"  Stability improvement: {stability_improvement:.1f}%")
    
    print("\nFinal Performance:")
    cur_final = curriculum_agg["final_reward"]["mean"]
    no_cur_final = no_curriculum_agg["final_reward"]["mean"]
    perf_improvement = ((cur_final - no_cur_final) / abs(no_cur_final)) * 100 if no_cur_final != 0 else 0
    print(f"  With curriculum: {cur_final:.4f}")
    print(f"  Without curriculum: {no_cur_final:.4f}")
    print(f"  Performance improvement: {perf_improvement:.1f}%")
    
    print("=" * 60)


if __name__ == "__main__":
    try:
        results = load_ablation_results()
        plot_ablation_comparison(results)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run evaluation/curriculum_ablation.py first to generate results.")
