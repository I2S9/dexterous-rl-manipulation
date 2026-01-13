"""
Test script to verify environment and policy rollouts.

This script runs a few episodes with random and heuristic policies
to ensure the environment is functioning correctly.
"""

import numpy as np
from envs import DexterousManipulationEnv
from policies import RandomPolicy, HeuristicPolicy


def test_rollout(policy, env, num_episodes: int = 3, max_steps: int = 50):
    """
    Run rollouts with a given policy and environment.
    
    Args:
        policy: Policy to use for action selection
        env: Environment to run
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    print(f"\nTesting {policy.__class__.__name__}...")
    print("-" * 50)
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        policy.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        
        for step in range(max_steps):
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1}:")
        print(f"  Steps: {episode_steps}")
        print(f"  Total reward: {episode_reward:.3f}")
        print(f"  Object position: {info['object_position']}")
        print(f"  Number of contacts: {info['num_contacts']}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print()


def main():
    """Run rollout tests."""
    print("=" * 50)
    print("Dexterous Manipulation Environment - Rollout Test")
    print("=" * 50)
    
    # Create environment
    env = DexterousManipulationEnv(
        num_fingers=5,
        joints_per_finger=3,
        max_episode_steps=200
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()
    
    # Test with random policy
    random_policy = RandomPolicy(env.action_space, seed=42)
    test_rollout(random_policy, env, num_episodes=3, max_steps=50)
    
    # Test with heuristic policy
    heuristic_policy = HeuristicPolicy(
        env.action_space,
        num_fingers=5,
        joints_per_finger=3
    )
    test_rollout(heuristic_policy, env, num_episodes=3, max_steps=50)
    
    # Test observation and action consistency
    print("=" * 50)
    print("Consistency checks:")
    print("-" * 50)
    
    obs, info = env.reset(seed=123)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    action = env.action_space.sample()
    print(f"Sample action shape: {action.shape}")
    print(f"Sample action range: [{action.min():.3f}, {action.max():.3f}]")
    
    obs_next, reward, terminated, truncated, info_next = env.step(action)
    print(f"Next observation shape: {obs_next.shape}")
    print(f"Reward: {reward:.3f}")
    print(f"Observation changed: {not np.allclose(obs, obs_next)}")
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)
    
    env.close()


if __name__ == "__main__":
    main()
