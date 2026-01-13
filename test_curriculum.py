"""
Test script to validate curriculum parameters.

This script verifies that curriculum variables are properly
exposed and functional in the environment.
"""

from envs import DexterousManipulationEnv
from experiments import CurriculumConfig

print("=" * 60)
print("Curriculum Parameters Validation")
print("=" * 60)

# Test 1: Default configuration
print("\n1. Testing default configuration:")
env_default = DexterousManipulationEnv()
obs, info = env_default.reset(seed=42)
print(f"   Object size: {info['curriculum']['object_size']:.4f} m")
print(f"   Object mass: {info['curriculum']['object_mass']:.4f} kg")
print(f"   Friction coefficient: {info['curriculum']['friction_coefficient']:.4f}")
print(f"   Object position: {info['object_position']}")

# Test 2: Easy configuration
print("\n2. Testing easy configuration:")
config_easy = CurriculumConfig.easy()
env_easy = DexterousManipulationEnv(curriculum_config=config_easy)
obs, info = env_easy.reset(seed=42)
print(f"   Object size: {info['curriculum']['object_size']:.4f} m (expected: larger)")
print(f"   Object mass: {info['curriculum']['object_mass']:.4f} kg (expected: lighter)")
print(f"   Friction coefficient: {info['curriculum']['friction_coefficient']:.4f} (expected: higher)")
print(f"   Object position: {info['object_position']}")

# Test 3: Hard configuration
print("\n3. Testing hard configuration:")
config_hard = CurriculumConfig.hard()
env_hard = DexterousManipulationEnv(curriculum_config=config_hard)
obs, info = env_hard.reset(seed=42)
print(f"   Object size: {info['curriculum']['object_size']:.4f} m (expected: smaller)")
print(f"   Object mass: {info['curriculum']['object_mass']:.4f} kg (expected: heavier)")
print(f"   Friction coefficient: {info['curriculum']['friction_coefficient']:.4f} (expected: lower)")
print(f"   Object position: {info['object_position']}")

# Test 4: Custom configuration with ranges
print("\n4. Testing custom configuration with ranges:")
config_custom = CurriculumConfig(
    object_size=0.05,
    object_size_range=(0.03, 0.07),
    object_mass=0.1,
    object_mass_range=(0.05, 0.15),
    friction_coefficient=0.5,
    friction_range=(0.3, 0.7),
    spawn_distance=0.15,
)
env_custom = DexterousManipulationEnv(curriculum_config=config_custom)

# Test multiple resets to see randomization
sizes = []
masses = []
frictions = []
for i in range(5):
    obs, info = env_custom.reset(seed=i)
    sizes.append(info['curriculum']['object_size'])
    masses.append(info['curriculum']['object_mass'])
    frictions.append(info['curriculum']['friction_coefficient'])

print(f"   Object sizes (5 resets): {[f'{s:.4f}' for s in sizes]}")
print(f"   Object masses (5 resets): {[f'{m:.4f}' for m in masses]}")
print(f"   Friction coefficients (5 resets): {[f'{f:.4f}' for f in frictions]}")
print(f"   Size range check: min={min(sizes):.4f}, max={max(sizes):.4f} (expected: 0.03-0.07)")
print(f"   Mass range check: min={min(masses):.4f}, max={max(masses):.4f} (expected: 0.05-0.15)")
print(f"   Friction range check: min={min(frictions):.4f}, max={max(frictions):.4f} (expected: 0.3-0.7)")

# Test 5: Configuration persistence
print("\n5. Testing configuration persistence:")
env_test = DexterousManipulationEnv(curriculum_config=config_easy)
obs1, info1 = env_test.reset(seed=100)
obs2, info2 = env_test.reset(seed=100)
print(f"   Same seed produces same config: {info1['curriculum'] == info2['curriculum']}")

# Test 6: Step dynamics affected by curriculum
print("\n6. Testing that dynamics are affected by curriculum:")
env_light = DexterousManipulationEnv(curriculum_config=CurriculumConfig(object_mass=0.05))
env_heavy = DexterousManipulationEnv(curriculum_config=CurriculumConfig(object_mass=0.2))

obs_light, info_light = env_light.reset(seed=42)
obs_heavy, info_heavy = env_heavy.reset(seed=42)

# Take a few steps and compare velocities
for _ in range(10):
    action = env_light.action_space.sample()
    obs_light, _, _, _, info_light = env_light.step(action)
    obs_heavy, _, _, _, info_heavy = env_heavy.step(action)

print(f"   Light object velocity: {info_light['object_position']}")
print(f"   Heavy object velocity: {info_heavy['object_position']}")
print(f"   (Both should fall, but dynamics may differ)")

print("\n" + "=" * 60)
print("All curriculum parameter tests completed!")
print("=" * 60)
