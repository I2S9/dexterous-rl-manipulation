"""
Run component ablation study.

This script executes the complete component ablation study,
testing all combinations of curriculum learning and dense reward shaping.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.component_ablation import (
    run_component_ablation,
    compute_ablation_statistics,
    print_ablation_report
)
from evaluation.plot_component_ablation import plot_component_ablation, load_component_ablation_results


def main():
    """Run component ablation study."""
    print("=" * 80)
    print("Component Ablation Study")
    print("=" * 80)
    print("\nThis study tests all combinations of:")
    print("  - Curriculum learning (on/off)")
    print("  - Dense reward shaping (on/off)")
    print("\nConfigurations tested:")
    print("  1. Baseline: Curriculum + Dense Reward")
    print("  2. No Curriculum: Fixed difficulty + Dense Reward")
    print("  3. No Dense Reward: Curriculum + Sparse Reward")
    print("  4. Minimal: Fixed difficulty + Sparse Reward")
    print("=" * 80)
    
    # Run ablation study
    seeds = [42, 123, 456, 789, 1000]  # 5 seeds for robust statistics
    num_episodes = 200
    
    all_results = run_component_ablation(
        num_episodes=num_episodes,
        seeds=seeds,
        output_dir="logs"
    )
    
    # Compute statistics
    stats = compute_ablation_statistics(all_results)
    
    # Print report
    print_ablation_report(all_results, stats)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        results_dict = load_component_ablation_results()
        plot_component_ablation(results_dict)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    print("\n" + "=" * 80)
    print("Component Ablation Study Complete")
    print("=" * 80)
    print("\nResults saved to: logs/component_ablation_results.json")
    print("Plots saved to: logs/component_ablation.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
