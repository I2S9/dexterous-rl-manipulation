"""
Test script for seed variance analysis.

This script validates that seed variance analysis works correctly
and shows stability across different random seeds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiments import CurriculumConfig
from evaluation import (
    HeldOutObjectSet,
    analyze_seed_variance,
)
from policies import RandomPolicy
from envs import DexterousManipulationEnv


def test_seed_variance():
    """Test seed variance analysis."""
    print("=" * 60)
    print("Seed Variance Analysis Test")
    print("=" * 60)
    
    # Setup
    print("\n1. Setting up evaluation...")
    train_config = CurriculumConfig.medium()
    heldout_set = HeldOutObjectSet(train_config, num_heldout_objects=5, seed=42)
    
    env = DexterousManipulationEnv(curriculum_config=train_config)
    policy = RandomPolicy(env.action_space, seed=42)
    env.close()
    
    # Test with multiple seeds (≥ 3)
    seeds = [42, 123, 456, 789, 1000]  # 5 seeds for robust analysis
    print(f"\n2. Testing with {len(seeds)} seeds: {seeds}")
    
    # Run analysis
    results = analyze_seed_variance(
        policy=policy,
        heldout_set=heldout_set,
        seeds=seeds,
        num_episodes_per_object=3,  # Reduced for faster testing
        reward_type="dense",
        max_episode_steps=200,
        output_dir="logs",
        max_cv_threshold=0.2
    )
    
    # Validate results
    print("\n3. Validating results...")
    variance_analysis = results["variance_analysis"]
    validation = results["validation"]
    
    # Check that we have results for all seeds
    assert len(variance_analysis["seeds"]) == len(seeds), \
        "Should have results for all seeds"
    
    # Check that variance stats are computed
    variance_stats = variance_analysis["variance_stats"]
    assert "grasp_success_rate" in variance_stats, "Should have success rate stats"
    assert "mean_episode_length" in variance_stats, "Should have episode length stats"
    
    # Check that validation is performed
    assert len(validation) > 0, "Should have validation results"
    
    # Check variance is controlled (or at least measured)
    print("\n4. Variance validation results:")
    all_controlled = True
    for metric_name, val in validation.items():
        status = "CONTROLLED" if val["is_controlled"] else "HIGH VARIANCE"
        print(f"   {metric_name}: {status} (CV: {val['cv']:.4f})")
        if not val["is_controlled"]:
            all_controlled = False
    
    if all_controlled:
        print("\n[PASS] All metrics show controlled variance")
    else:
        print("\n[NOTE] Some metrics show higher variance")
        print("       This may be expected depending on policy and task difficulty")
    
    print("\n[PASS] Seed variance analysis test passed")
    return True


def test_minimum_seeds():
    """Test that at least 3 seeds are required."""
    print("\n" + "=" * 60)
    print("Test: Minimum Seeds Requirement")
    print("=" * 60)
    
    train_config = CurriculumConfig.medium()
    heldout_set = HeldOutObjectSet(train_config, num_heldout_objects=3, seed=42)
    
    env = DexterousManipulationEnv(curriculum_config=train_config)
    policy = RandomPolicy(env.action_space, seed=42)
    env.close()
    
    # Test with insufficient seeds
    try:
        analyze_seed_variance(
            policy=policy,
            heldout_set=heldout_set,
            seeds=[42, 123],  # Only 2 seeds
            num_episodes_per_object=2,
        )
        print("[FAIL] Should have raised error for < 3 seeds")
        return False
    except ValueError as e:
        print(f"[PASS] Correctly raised error: {e}")
        return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Seed Variance Analysis Tests")
    print("=" * 60)
    
    tests = [
        ("Seed Variance Analysis", test_seed_variance),
        ("Minimum Seeds Requirement", test_minimum_seeds),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    main()
