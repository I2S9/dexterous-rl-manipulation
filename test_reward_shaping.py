"""
Quick test script for reward shaping functionality.
"""

from envs import DexterousManipulationEnv
from rewards import RewardShaping, SparseReward

print("Testing reward shaping...")

# Test dense reward
print("\n1. Testing dense reward:")
env_dense = DexterousManipulationEnv(reward_type='dense')
obs, info = env_dense.reset()
action = env_dense.action_space.sample()
obs, reward, term, trunc, info = env_dense.step(action)
print(f"   Reward: {reward:.3f}")
print(f"   Components: {info.get('reward_components', {})}")

# Test sparse reward
print("\n2. Testing sparse reward:")
env_sparse = DexterousManipulationEnv(reward_type='sparse')
obs, info = env_sparse.reset()
action = env_sparse.action_space.sample()
obs, reward, term, trunc, info = env_sparse.step(action)
print(f"   Reward: {reward:.3f}")
print(f"   Components: {info.get('reward_components', {})}")

print("\nReward shaping test completed successfully!")
