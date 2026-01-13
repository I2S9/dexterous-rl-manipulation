"""
Test script for curriculum scheduler with logging.

This script demonstrates curriculum progression based on success rate
and logs the evolution of difficulty.
"""

import numpy as np
from envs import DexterousManipulationEnv
from policies import RandomPolicy
from experiments import CurriculumConfig, CurriculumScheduler, CurriculumLogger


def simulate_episode(env, policy, max_steps=200):
    """
    Simulate a single episode.
    
    Returns:
        Tuple of (success, episode_steps, episode_reward)
    """
    obs, info = env.reset()
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
    
    success = info.get("num_contacts", 0) >= 3
    return success, episode_steps, episode_reward


def run_curriculum_training(
    num_episodes: int = 200,
    success_rate_threshold: float = 0.7,
    window_size: int = 20,
    min_episodes: int = 50,
):
    """
    Run training with curriculum scheduling.
    
    Args:
        num_episodes: Number of training episodes
        success_rate_threshold: Success rate needed to progress
        window_size: Window size for success rate calculation
        min_episodes: Minimum episodes before first progression
    """
    print("=" * 60)
    print("Curriculum Learning Test")
    print("=" * 60)
    
    # Initialize curriculum
    initial_config = CurriculumConfig.easy()
    target_config = CurriculumConfig.hard()
    
    scheduler = CurriculumScheduler(
        initial_config=initial_config,
        target_config=target_config,
        success_rate_threshold=success_rate_threshold,
        window_size=window_size,
        min_episodes_before_progression=min_episodes,
        progression_steps=5,
    )
    
    logger = CurriculumLogger()
    
    # Create environment with initial config
    env = DexterousManipulationEnv(
        curriculum_config=scheduler.get_current_config(),
        reward_type="dense"
    )
    
    # Use a simple policy (random for demonstration)
    policy = RandomPolicy(env.action_space, seed=42)
    
    print(f"\nInitial configuration:")
    print(f"  Object size: {initial_config.object_size:.4f} m")
    print(f"  Object mass: {initial_config.object_mass:.4f} kg")
    print(f"  Friction: {initial_config.friction_coefficient:.3f}")
    print(f"\nTarget configuration:")
    print(f"  Object size: {target_config.object_size:.4f} m")
    print(f"  Object mass: {target_config.object_mass:.4f} kg")
    print(f"  Friction: {target_config.friction_coefficient:.3f}")
    print(f"\nProgression conditions:")
    print(f"  Success rate threshold: {success_rate_threshold}")
    print(f"  Window size: {window_size}")
    print(f"  Min episodes before progression: {min_episodes}")
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    
    # Training loop
    for episode in range(num_episodes):
        # Run episode
        success, episode_steps, episode_reward = simulate_episode(env, policy)
        
        # Update scheduler
        progression_occurred = scheduler.update(success, episode_steps)
        
        # Log episode
        logger.log_episode(episode, scheduler, success, episode_steps)
        logger.log_progression(scheduler, progression_occurred)
        
        # Update environment config if progression occurred
        if progression_occurred:
            env.curriculum_config = scheduler.get_current_config()
            stats = scheduler.get_statistics()
            latest_prog = scheduler.progression_history[-1]
            
            print(f"\n[Episode {episode}] PROGRESSION!")
            print(f"  Difficulty level: {latest_prog['difficulty_level']:.2f}")
            print(f"  Success rate: {latest_prog.get('success_rate', 0):.3f}")
            print(f"  Total steps: {latest_prog['total_steps']}")
            print(f"  New object size: {latest_prog['object_size']:.4f} m")
            print(f"  New object mass: {latest_prog['object_mass']:.4f} kg")
            print(f"  New friction: {latest_prog['friction_coefficient']:.3f}")
        
        # Print progress every 20 episodes
        if (episode + 1) % 20 == 0:
            stats = scheduler.get_statistics()
            print(f"\n[Episode {episode + 1}]")
            print(f"  Difficulty level: {stats['current_difficulty_level']:.2f}")
            print(f"  Recent success rate: {stats['recent_success_rate']:.3f}")
            print(f"  Overall success rate: {stats['overall_success_rate']:.3f}")
            print(f"  Total steps: {stats['total_steps']}")
            print(f"  Progressions: {stats['num_progressions']}")
    
    # Save logs
    log_path = logger.save("curriculum_progression.json")
    print(f"\n" + "=" * 60)
    print(f"Logs saved to: {log_path}")
    print("=" * 60)
    
    # Print summary
    logger.print_progression_summary(scheduler)
    
    return scheduler, logger


if __name__ == "__main__":
    # Run with lower threshold for demonstration (since random policy won't achieve 70%)
    scheduler, logger = run_curriculum_training(
        num_episodes=150,
        success_rate_threshold=0.3,  # Lower for demonstration
        window_size=15,
        min_episodes=20,
    )
