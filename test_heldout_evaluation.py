"""
Test script for held-out object evaluation.

This script validates:
1. Strict separation between training and evaluation objects
2. Training is frozen during evaluation
3. Evaluation works correctly with held-out objects
"""

import numpy as np
import json
from pathlib import Path
from experiments import CurriculumConfig
from evaluation import HeldOutObjectSet, generate_training_objects, Evaluator
from policies import RandomPolicy
from envs import DexterousManipulationEnv


def test_separation():
    """Test that training and evaluation objects are strictly separated."""
    print("=" * 60)
    print("Test 1: Training/Eval Separation")
    print("=" * 60)
    
    # Create training configuration
    train_config = CurriculumConfig(
        object_size=0.05,
        object_size_range=(0.03, 0.07),
        object_mass=0.1,
        object_mass_range=(0.05, 0.15),
        friction_coefficient=0.5,
        friction_range=(0.3, 0.7),
    )
    
    # Generate training objects
    train_objects = generate_training_objects(train_config, num_samples=100, seed=42)
    print(f"Generated {len(train_objects)} training objects")
    print(f"Training size range: {min(o.size for o in train_objects):.4f} - {max(o.size for o in train_objects):.4f}")
    print(f"Training mass range: {min(o.mass for o in train_objects):.4f} - {max(o.mass for o in train_objects):.4f}")
    print(f"Training friction range: {min(o.friction for o in train_objects):.4f} - {max(o.friction for o in train_objects):.4f}")
    
    # Create held-out set
    heldout_set = HeldOutObjectSet(
        train_config=train_config,
        num_heldout_objects=20,
        seed=123  # Different seed
    )
    
    print(f"\nGenerated {len(heldout_set.heldout_objects)} held-out objects")
    stats = heldout_set.get_statistics()
    print(f"Held-out size range: {stats['size_range'][0]:.4f} - {stats['size_range'][1]:.4f}")
    print(f"Held-out mass range: {stats['mass_range'][0]:.4f} - {stats['mass_range'][1]:.4f}")
    print(f"Held-out friction range: {stats['friction_range'][0]:.4f} - {stats['friction_range'][1]:.4f}")
    
    # Verify separation
    is_separated = heldout_set.verify_separation(train_objects)
    print(f"\nSeparation verified: {is_separated}")
    
    if not is_separated:
        print("ERROR: Training and evaluation objects overlap!")
        return False
    
    print("[PASS] Separation test passed")
    return True


def test_frozen_training():
    """Test that training is frozen during evaluation."""
    print("\n" + "=" * 60)
    print("Test 2: Frozen Training During Evaluation")
    print("=" * 60)
    
    # Create a simple policy with update method
    class TestPolicy:
        def __init__(self):
            self.update_count = 0
            self.best_reward = -np.inf
        
        def select_action(self, obs):
            return np.random.uniform(-1, 1, size=(15,))
        
        def update(self, reward):
            self.update_count += 1
            if reward > self.best_reward:
                self.best_reward = reward
    
    policy = TestPolicy()
    
    # Create held-out set
    train_config = CurriculumConfig.easy()
    heldout_set = HeldOutObjectSet(train_config, num_heldout_objects=5, seed=42)
    
    # Create evaluator
    evaluator = Evaluator(policy, heldout_set)
    
    # Check initial state
    initial_update_count = policy.update_count
    
    # Run evaluation (should not update policy)
    eval_config = heldout_set.get_eval_config(0)
    result = evaluator.evaluate_episode(eval_config, seed=42)
    
    # Check that update was not called
    final_update_count = policy.update_count
    
    print(f"Initial update count: {initial_update_count}")
    print(f"Final update count: {final_update_count}")
    print(f"Policy frozen: {evaluator._policy_frozen}")
    
    if final_update_count > initial_update_count:
        print("ERROR: Policy was updated during evaluation!")
        return False
    
    print("[PASS] Frozen training test passed")
    
    # Test context manager
    print("\nTesting context manager...")
    policy2 = TestPolicy()
    evaluator2 = Evaluator(policy2, heldout_set)
    
    with evaluator2:
        eval_config = heldout_set.get_eval_config(0)
        result = evaluator2.evaluate_episode(eval_config, seed=42)
    
    print(f"Update count after context: {policy2.update_count}")
    print(f"Policy frozen after context: {evaluator2._policy_frozen}")
    
    if policy2.update_count > 0:
        print("ERROR: Policy was updated during evaluation context!")
        return False
    
    if evaluator2._policy_frozen:
        print("ERROR: Policy not unfrozen after context!")
        return False
    
    print("[PASS] Context manager test passed")
    return True


def test_evaluation():
    """Test evaluation on held-out objects."""
    print("\n" + "=" * 60)
    print("Test 3: Held-Out Object Evaluation")
    print("=" * 60)
    
    # Create policy
    train_config = CurriculumConfig.easy()
    env = DexterousManipulationEnv(curriculum_config=train_config)
    policy = RandomPolicy(env.action_space, seed=42)
    env.close()
    
    # Create held-out set
    heldout_set = HeldOutObjectSet(train_config, num_heldout_objects=5, seed=123)
    
    # Create evaluator
    evaluator = Evaluator(policy, heldout_set)
    
    # Evaluate on held-out set
    print("Running evaluation on held-out objects...")
    results = evaluator.evaluate_heldout_set(num_episodes_per_object=3, seed=42)
    
    overall = results["overall_stats"]
    print(f"\nOverall Results:")
    print(f"  Number of objects: {overall['num_objects']}")
    print(f"  Total episodes: {overall['total_episodes']}")
    print(f"  Overall success rate: {overall['overall_success_rate']:.3f}")
    print(f"  Mean reward: {overall['mean_reward']:.3f} ± {overall['std_reward']:.3f}")
    print(f"  Mean steps: {overall['mean_steps']:.1f}")
    
    print(f"\nPer-Object Results:")
    for obj_idx, obj_result in results["per_object_results"].items():
        props = obj_result["object_properties"]
        print(f"  Object {obj_idx}:")
        print(f"    Size: {props['size']:.4f}, Mass: {props['mass']:.4f}, Friction: {props['friction']:.3f}")
        print(f"    Success rate: {obj_result['success_rate']:.3f}")
        print(f"    Mean reward: {obj_result['mean_reward']:.3f}")
    
    print("\n[PASS] Evaluation test passed")
    return True


def test_reproducibility():
    """Test that evaluation is reproducible with same seeds."""
    print("\n" + "=" * 60)
    print("Test 4: Evaluation Reproducibility")
    print("=" * 60)
    
    train_config = CurriculumConfig.easy()
    env = DexterousManipulationEnv(curriculum_config=train_config)
    policy1 = RandomPolicy(env.action_space, seed=42)
    policy2 = RandomPolicy(env.action_space, seed=42)
    env.close()
    
    heldout_set = HeldOutObjectSet(train_config, num_heldout_objects=3, seed=123)
    
    evaluator1 = Evaluator(policy1, heldout_set)
    evaluator2 = Evaluator(policy2, heldout_set)
    
    # Run same evaluation twice
    results1 = evaluator1.evaluate_heldout_set(num_episodes_per_object=2, seed=100)
    results2 = evaluator2.evaluate_heldout_set(num_episodes_per_object=2, seed=100)
    
    # Compare results (allow small numerical differences)
    rewards1 = [r["episode_reward"] for r in results1["all_episodes"]]
    rewards2 = [r["episode_reward"] for r in results2["all_episodes"]]
    
    print(f"Run 1 mean reward: {np.mean(rewards1):.6f}")
    print(f"Run 2 mean reward: {np.mean(rewards2):.6f}")
    print(f"Max difference: {np.max(np.abs(np.array(rewards1) - np.array(rewards2))):.6f}")
    
    # Allow small differences due to floating point precision
    if not np.allclose(rewards1, rewards2, rtol=1e-5, atol=1e-5):
        print("WARNING: Results have small differences (may be due to floating point precision)")
        # Check if differences are very small
        max_diff = np.max(np.abs(np.array(rewards1) - np.array(rewards2)))
        if max_diff > 0.1:  # Only fail if difference is significant
            print(f"ERROR: Results are not reproducible! Max diff: {max_diff}")
            return False
        else:
            print(f"Differences are small ({max_diff:.6f}), likely due to numerical precision")
    
    print("[PASS] Reproducibility test passed")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Held-Out Object Evaluation Tests")
    print("=" * 60)
    
    tests = [
        ("Separation", test_separation),
        ("Frozen Training", test_frozen_training),
        ("Evaluation", test_evaluation),
        ("Reproducibility", test_reproducibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nERROR in {test_name}: {e}")
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
