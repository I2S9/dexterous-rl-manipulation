"""
Example script for running held-out object evaluation.

This demonstrates how to evaluate a trained policy on held-out objects
with strict train/eval separation and frozen training.
"""

import json
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments import CurriculumConfig
from evaluation import HeldOutObjectSet, Evaluator, generate_training_objects
from policies import RandomPolicy
from envs import DexterousManipulationEnv


def main():
    """Run held-out object evaluation example."""
    print("=" * 60)
    print("Held-Out Object Evaluation Example")
    print("=" * 60)
    
    # Step 1: Define training configuration
    print("\n1. Defining training configuration...")
    train_config = CurriculumConfig(
        object_size=0.05,
        object_size_range=(0.03, 0.07),
        object_mass=0.1,
        object_mass_range=(0.05, 0.15),
        friction_coefficient=0.5,
        friction_range=(0.3, 0.7),
    )
    
    # Step 2: Generate training objects (for verification)
    print("2. Generating training objects...")
    train_objects = generate_training_objects(train_config, num_samples=100, seed=42)
    print(f"   Generated {len(train_objects)} training object samples")
    
    # Step 3: Create held-out object set
    print("3. Creating held-out object set...")
    heldout_set = HeldOutObjectSet(
        train_config=train_config,
        num_heldout_objects=20,
        seed=123  # Different seed ensures separation
    )
    
    stats = heldout_set.get_statistics()
    print(f"   Generated {stats['num_objects']} held-out objects")
    print(f"   Size range: {stats['size_range'][0]:.4f} - {stats['size_range'][1]:.4f}")
    print(f"   Mass range: {stats['mass_range'][0]:.4f} - {stats['mass_range'][1]:.4f}")
    print(f"   Friction range: {stats['friction_range'][0]:.4f} - {stats['friction_range'][1]:.4f}")
    
    # Step 4: Verify separation
    print("4. Verifying train/eval separation...")
    is_separated = heldout_set.verify_separation(train_objects)
    print(f"   Separation verified: {is_separated}")
    
    if not is_separated:
        print("   ERROR: Training and evaluation objects overlap!")
        return
    
    # Step 5: Load or create policy (in real scenario, this would be a trained policy)
    print("5. Loading policy...")
    env = DexterousManipulationEnv(curriculum_config=train_config)
    policy = RandomPolicy(env.action_space, seed=42)
    env.close()
    print("   Policy loaded (using random policy for demonstration)")
    
    # Step 6: Create evaluator
    print("6. Creating evaluator...")
    evaluator = Evaluator(
        policy=policy,
        heldout_set=heldout_set,
        reward_type="dense",
        max_episode_steps=200
    )
    print("   Evaluator created (training will be frozen during evaluation)")
    
    # Step 7: Run evaluation
    print("7. Running evaluation on held-out objects...")
    print("   (Training is frozen - no policy updates)")
    
    results = evaluator.evaluate_heldout_set(
        num_episodes_per_object=5,
        seed=42
    )
    
    # Step 8: Display results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    overall = results["overall_stats"]
    print(f"\nOverall Statistics:")
    print(f"  Number of held-out objects: {overall['num_objects']}")
    print(f"  Total evaluation episodes: {overall['total_episodes']}")
    print(f"  Overall success rate: {overall['overall_success_rate']:.3f}")
    print(f"  Mean reward: {overall['mean_reward']:.3f} ± {overall['std_reward']:.3f}")
    print(f"  Mean steps per episode: {overall['mean_steps']:.1f}")
    
    print(f"\nPer-Object Results (first 5):")
    for obj_idx in range(min(5, len(results["per_object_results"]))):
        obj_result = results["per_object_results"][obj_idx]
        props = obj_result["object_properties"]
        print(f"  Object {obj_idx}:")
        print(f"    Properties: size={props['size']:.4f}, mass={props['mass']:.4f}, friction={props['friction']:.3f}")
        print(f"    Success rate: {obj_result['success_rate']:.3f}")
        print(f"    Mean reward: {obj_result['mean_reward']:.3f}")
        print(f"    Mean steps: {obj_result['mean_steps']:.1f}")
    
    # Step 9: Save results
    print("\n8. Saving evaluation results...")
    output_path = Path("logs") / "heldout_evaluation.json"
    output_path.parent.mkdir(exist_ok=True)
    
    # Convert numpy types to native Python types for JSON
    def convert_to_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json(item) for item in obj]
        return obj
    
    json_results = convert_to_json(results)
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"   Results saved to {output_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)
    print("\nKey points:")
    print("  - Training and evaluation objects are strictly separated")
    print("  - Training is frozen during evaluation (no policy updates)")
    print("  - Evaluation uses held-out objects not seen during training")
    print("  - Results are reproducible with fixed seeds")
    print("=" * 60)


if __name__ == "__main__":
    main()
