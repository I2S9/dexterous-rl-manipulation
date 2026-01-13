"""
Plot training curves and convergence comparison.

This script visualizes the difference between sparse and dense rewards
in terms of learning curves and convergence speed.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


def load_comparison_results(log_dir: str = "logs") -> Dict:
    """
    Load reward comparison results.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        Dictionary with comparison results
    """
    log_path = Path(log_dir) / "reward_comparison.json"
    
    if not log_path.exists():
        raise FileNotFoundError(f"Comparison results not found at {log_path}")
    
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_comparison(results: Dict, output_path: str = "logs/convergence_comparison.png"):
    """
    Plot comparison of sparse vs dense rewards.
    
    Args:
        results: Dictionary with comparison results
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    sparse_rewards = results["sparse"]["episode_rewards"]
    dense_rewards = results["dense"]["episode_rewards"]
    
    episodes = np.arange(len(sparse_rewards))
    
    # Plot 1: Episode rewards
    ax1 = axes[0]
    ax1.plot(episodes, sparse_rewards, label="Sparse Reward", alpha=0.7, linewidth=1)
    ax1.plot(episodes, dense_rewards, label="Dense Reward", alpha=0.7, linewidth=1)
    
    # Add convergence markers
    if results["sparse"]["convergence_step"] is not None:
        ax1.axvline(results["sparse"]["convergence_step"], color='blue', 
                   linestyle='--', alpha=0.5, label='Sparse Convergence')
    if results["dense"]["convergence_step"] is not None:
        ax1.axvline(results["dense"]["convergence_step"], color='orange', 
                   linestyle='--', alpha=0.5, label='Dense Convergence')
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Learning Curves: Sparse vs Dense Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving average
    ax2 = axes[1]
    window_size = 10
    
    sparse_ma = np.convolve(sparse_rewards, np.ones(window_size)/window_size, mode='valid')
    dense_ma = np.convolve(dense_rewards, np.ones(window_size)/window_size, mode='valid')
    
    ax2.plot(episodes[window_size-1:], sparse_ma, label="Sparse Reward (MA)", 
            alpha=0.8, linewidth=2)
    ax2.plot(episodes[window_size-1:], dense_ma, label="Dense Reward (MA)", 
            alpha=0.8, linewidth=2)
    
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Moving Average Reward (window=10)")
    ax2.set_title("Smoothed Learning Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Convergence Summary")
    print("=" * 60)
    print(f"Sparse Reward:")
    print(f"  Convergence step: {results['sparse']['convergence_step']}")
    print(f"  Mean reward: {results['sparse']['mean_reward']:.3f} ± {results['sparse']['std_reward']:.3f}")
    print(f"  Final success rate: {results['sparse']['final_success_rate']:.3f}")
    print(f"\nDense Reward:")
    print(f"  Convergence step: {results['dense']['convergence_step']}")
    print(f"  Mean reward: {results['dense']['mean_reward']:.3f} ± {results['dense']['std_reward']:.3f}")
    print(f"  Final success rate: {results['dense']['final_success_rate']:.3f}")
    
    if results['sparse']['convergence_step'] is not None and results['dense']['convergence_step'] is not None:
        improvement = ((results['sparse']['convergence_step'] - results['dense']['convergence_step']) / 
                      results['sparse']['convergence_step']) * 100
        print(f"\nConvergence improvement: {improvement:.1f}% faster with dense rewards")


if __name__ == "__main__":
    try:
        results = load_comparison_results()
        plot_comparison(results)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run training/reward_comparison.py first to generate results.")
