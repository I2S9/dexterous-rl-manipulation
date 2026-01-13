"""
Test loading curriculum configurations from JSON files.
"""

from experiments import CurriculumConfig
from envs import DexterousManipulationEnv

print("Testing configuration loading from JSON files...")
print("=" * 60)

# Test loading easy config
print("\n1. Loading easy configuration:")
config_easy = CurriculumConfig.from_json("experiments/config_easy.json")
print(f"   Object size: {config_easy.object_size}")
print(f"   Object mass: {config_easy.object_mass}")
print(f"   Friction: {config_easy.friction_coefficient}")

# Test loading variable config
print("\n2. Loading variable configuration:")
config_var = CurriculumConfig.from_json("experiments/config_variable.json")
print(f"   Object size range: {config_var.object_size_range}")
print(f"   Object mass range: {config_var.object_mass_range}")
print(f"   Friction range: {config_var.friction_range}")

# Test using loaded config in environment
print("\n3. Using loaded config in environment:")
env = DexterousManipulationEnv(curriculum_config=config_easy)
obs, info = env.reset(seed=42)
print(f"   Environment object size: {info['curriculum']['object_size']}")
print(f"   Environment object mass: {info['curriculum']['object_mass']}")

# Test saving config
print("\n4. Testing config save/load:")
test_config = CurriculumConfig(
    object_size=0.06,
    object_mass=0.12,
    friction_coefficient=0.6
)
test_config.to_json("experiments/test_config.json")
loaded_config = CurriculumConfig.from_json("experiments/test_config.json")
print(f"   Saved and loaded config matches: {test_config.object_size == loaded_config.object_size}")

print("\n" + "=" * 60)
print("Configuration loading tests completed!")
print("=" * 60)
