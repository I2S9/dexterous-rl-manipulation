"""
Plot component ablation study results.

This script visualizes the comparison between different component
configurations to isolate the impact of each design choice.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.component_ablation import TrainingResults, compute_ablation_statistics


def load_component_ablation_results(log_dir: str = "logs") -> Dict:
    """
    Load component ablation study results.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        Dictionary with ablation results
    """
    log_path = Path(log_dir) / "component_ablation_results.json"
    
    if not log_path.exists():
        raise FileNotFoundError(f"Component ablation results not found at {log_path}")
    
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_component_ablation(
    results: Dict,
    output_path: str = "logs/component_ablation.png"
):
    """
    Plot component ablation study comparison.
    
    Args:
        results: Dictionary with ablation results (from JSON)
        output_path: Path to save the plot
    """
    # Convert JSON back to TrainingResults objects for processing
    all_results = {}
    for config_name, results_list in results.items():
        all_results[config_name] = []
        for r_dict in results_list:
            # Reconstruct TrainingResults (simplified)
            from evaluation.component_ablation import AblationConfig, TrainingResults
            config = AblationConfig(**r_dict["config"])
            all_results[config_name].append(TrainingResults(
                config=config,
                episode_rewards=r_dict["episode_rewards"],
                episode_steps=r_dict["episode_steps"],
                success_rates=r_dict["success_rates"],
                final_success_rate=r_dict["final_success_rate"],
                mean_episode_length=r_dict["mean_episode_length"],
                convergence_step=r_dict["convergence_step"],
                total_episodes=r_dict["total_episodes"],
            ))
    
    stats = compute_ablation_statistics(all_results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data for plotting
    config_names = list(stats.keys())
    success_means = [stats[name]["final_success_rate"]["mean"] for name in config_names]
    success_stds = [stats[name]["final_success_rate"]["std"] for name in config_names]
    
    # Plot 1: Final success rate comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(config_names))
    bars = ax1.bar(x_pos, success_means, yerr=success_stds, capsize=5, 
                   color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db'])
    ax1.set_ylabel("Final Success Rate")
    ax1.set_title("Final Success Rate by Configuration")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([name.replace('_', '\n') for name in config_names], rotation=0, ha='center')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(success_means, success_stds)):
        ax1.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Success rate evolution over episodes
    ax2 = axes[0, 1]
    for config_name, results_list in all_results.items():
        # Average across seeds
        all_success_rates = np.array([r.success_rates for r in results_list])
        mean_success = np.mean(all_success_rates, axis=0)
        std_success = np.std(all_success_rates, axis=0)
        
        episodes = np.arange(len(mean_success))
        ax2.plot(episodes, mean_success, label=config_name.replace('_', ' ').title(), linewidth=2)
        ax2.fill_between(episodes, mean_success - std_success, mean_success + std_success, alpha=0.2)
    
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate")
    ax2.set_title("Success Rate Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.0])
    
    # Plot 3: Convergence step comparison
    ax3 = axes[1, 0]
    convergence_data = []
    convergence_labels = []
    for config_name in config_names:
        conv_mean = stats[config_name]["convergence_step"]["mean"]
        if conv_mean is not None:
            convergence_data.append(conv_mean)
            convergence_labels.append(config_name.replace('_', '\n'))
    
    if convergence_data:
        x_pos_conv = np.arange(len(convergence_data))
        bars = ax3.bar(x_pos_conv, convergence_data, color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db'][:len(convergence_data)])
        ax3.set_ylabel("Convergence Episode")
        ax3.set_title("Convergence Speed (Lower is Better)")
        ax3.set_xticks(x_pos_conv)
        ax3.set_xticklabels(convergence_labels, rotation=0, ha='center')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, val in enumerate(convergence_data):
            ax3.text(i, val + max(convergence_data) * 0.02, f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, "No convergence reached", ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("Convergence Speed")
    
    # Plot 4: Component impact visualization
    ax4 = axes[1, 1]
    
    # Extract baseline
    baseline_name = "baseline"
    if baseline_name not in stats:
        baseline_name = config_names[0]
    
    baseline_success = stats[baseline_name]["final_success_rate"]["mean"]
    
    # Compute impacts
    impacts = []
    impact_labels = []
    
    if "no_curriculum" in stats:
        no_curriculum_success = stats["no_curriculum"]["final_success_rate"]["mean"]
        curriculum_impact = baseline_success - no_curriculum_success
        impacts.append(curriculum_impact)
        impact_labels.append("Curriculum\nLearning")
    
    if "no_dense_reward" in stats:
        no_dense_success = stats["no_dense_reward"]["final_success_rate"]["mean"]
        dense_reward_impact = baseline_success - no_dense_success
        impacts.append(dense_reward_impact)
        impact_labels.append("Dense\nReward")
    
    if impacts:
        x_pos_impact = np.arange(len(impacts))
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in impacts]
        bars = ax4.bar(x_pos_impact, impacts, color=colors)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax4.set_ylabel("Success Rate Impact")
        ax4.set_title("Individual Component Impact")
        ax4.set_xticks(x_pos_impact)
        ax4.set_xticklabels(impact_labels, rotation=0, ha='center')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, val in enumerate(impacts):
            y_pos = val + (0.02 if val > 0 else -0.02)
            ax4.text(i, y_pos, f'{val:+.3f}', ha='center', 
                    va='bottom' if val > 0 else 'top', fontsize=9)
    else:
        ax4.text(0.5, 0.5, "No impact data", ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("Component Impact")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Component ablation plot saved to {output_path}")


if __name__ == "__main__":
    try:
        results = load_component_ablation_results()
        plot_component_ablation(results)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run evaluation/run_component_ablation.py first to generate results.")
