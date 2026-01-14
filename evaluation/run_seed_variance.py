"""
Example script for seed variance analysis.

This script demonstrates how to evaluate stability across
different random seeds to show that results are reproducible.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments import CurriculumConfig
from evaluation import (
    HeldOutObjectSet,
    analyze_seed_variance,
)
from policies import RandomPolicy
from envs import DexterousManipulationEnv


def main():
    """Run seed variance analysis example."""
    print("=" * 60)
    print("Seed Variance Analysis Example")
    print("=" * 60)
    
    # Step 1: Setup
    print("\n1. Setting up evaluation...")
    train_config = CurriculumConfig.medium()
    heldout_set = HeldOutObjectSet(train_config, num_heldout_objects=10, seed=42)
    
    env = DexterousManipulationEnv(curriculum_config=train_config)
    policy = RandomPolicy(env.action_space, seed=42)
    env.close()
    
    # Step 2: Define seeds to test (≥ 3 required)
    seeds = [42, 123, 456, 789, 1000, 2024, 3000]  # 7 seeds for robust analysis
    print(f"\n2. Testing with {len(seeds)} seeds: {seeds}")
    print(f"   (At least 3 seeds required for variance analysis)")
    
    # Step 3: Run analysis
    print("\n3. Running seed variance analysis...")
    print("   This will evaluate the policy with each seed and compare results")
    
    results = analyze_seed_variance(
        policy=policy,
        heldout_set=heldout_set,
        seeds=seeds,
        num_episodes_per_object=5,
        reward_type="dense",
        max_episode_steps=200,
        output_dir="logs",
        max_cv_threshold=0.2  # Coefficient of variation threshold
    )
    
    # Step 4: Summary
    print("\n" + "=" * 60)
    print("Analysis Summary")
    print("=" * 60)
    
    variance_analysis = results["variance_analysis"]
    validation = results["validation"]
    
    print(f"\nSeeds tested: {len(variance_analysis['seeds'])}")
    
    # Show key metrics
    variance_stats = variance_analysis["variance_stats"]
    print(f"\nKey Metrics:")
    print(f"  Grasp Success Rate:")
    print(f"    Mean: {variance_stats['grasp_success_rate']['mean']:.3f}")
    print(f"    Std:  {variance_stats['grasp_success_rate']['std']:.3f}")
    print(f"    CV:   {variance_stats['grasp_success_rate']['cv']:.4f}")
    
    print(f"\n  Mean Episode Length:")
    print(f"    Mean: {variance_stats['mean_episode_length']['mean']:.1f}")
    print(f"    Std:  {variance_stats['mean_episode_length']['std']:.1f}")
    print(f"    CV:   {variance_stats['mean_episode_length']['cv']:.4f}")
    
    # Validation summary
    print(f"\nVariance Validation:")
    controlled_count = sum(1 for v in validation.values() if v["is_controlled"])
    total_count = len(validation)
    print(f"  {controlled_count}/{total_count} metrics show controlled variance")
    
    if controlled_count == total_count:
        print(f"  [PASS] All metrics show stable results across seeds")
    else:
        print(f"  [NOTE] Some metrics show higher variance")
        print(f"         This indicates areas where results may vary")
    
    print("\n" + "=" * 60)
    print("Seed Variance Analysis Complete")
    print("=" * 60)
    print("\nKey points:")
    print("  - Evaluated with multiple seeds (>= 3)")
    print("  - Computed mean and std for all metrics")
    print("  - Validated variance is controlled (CV < threshold)")
    print("  - Results saved for reproducibility")
    print("=" * 60)


if __name__ == "__main__":
    main()
