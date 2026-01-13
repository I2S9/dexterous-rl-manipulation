"""
Evaluation system with frozen training.

This module provides evaluation functionality that ensures
training is frozen during evaluation runs.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from envs import DexterousManipulationEnv
from experiments.config import CurriculumConfig
from evaluation.heldout_objects import HeldOutObjectSet


class Evaluator:
    """
    Evaluator for held-out objects with frozen training.
    
    Ensures that no learning occurs during evaluation.
    """
    
    def __init__(
        self,
        policy,
        heldout_set: HeldOutObjectSet,
        reward_type: str = "dense",
        max_episode_steps: int = 200,
    ):
        """
        Initialize evaluator.
        
        Args:
            policy: Policy to evaluate (must not be updated during eval)
            heldout_set: Held-out object set
            reward_type: Type of reward to use
            max_episode_steps: Maximum steps per episode
        """
        self.policy = policy
        self.heldout_set = heldout_set
        self.reward_type = reward_type
        self.max_episode_steps = max_episode_steps
        
        # Track if policy has been frozen
        self._policy_frozen = False
    
    def freeze_policy(self):
        """Freeze policy to prevent updates during evaluation."""
        # Disable any update mechanisms
        if hasattr(self.policy, 'update'):
            original_update = self.policy.update
            
            def frozen_update(*args, **kwargs):
                """Frozen update that does nothing."""
                pass
            
            self.policy.update = frozen_update
            self._original_update = original_update
            self._policy_frozen = True
    
    def unfreeze_policy(self):
        """Unfreeze policy (restore update functionality)."""
        if self._policy_frozen and hasattr(self, '_original_update'):
            self.policy.update = self._original_update
            self._policy_frozen = False
    
    def evaluate_episode(
        self,
        eval_config: CurriculumConfig,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Evaluate a single episode with a held-out object.
        
        Args:
            eval_config: Curriculum configuration for evaluation object
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with episode evaluation results
        """
        # Ensure policy is frozen
        if not self._policy_frozen:
            self.freeze_policy()
        
        # Create environment with eval config
        env = DexterousManipulationEnv(
            curriculum_config=eval_config,
            reward_type=self.reward_type,
            max_episode_steps=self.max_episode_steps
        )
        
        # Run episode
        obs, info = env.reset(seed=seed)
        if hasattr(self.policy, 'reset'):
            self.policy.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        success = False
        
        for step in range(self.max_episode_steps):
            action = self.policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            if terminated or truncated:
                success = terminated
                break
        
        env.close()
        
        return {
            "episode_reward": float(episode_reward),
            "episode_steps": int(episode_steps),
            "success": bool(success),
            "num_contacts": int(info.get("num_contacts", 0)),
            "object_size": float(info["curriculum"]["object_size"]),
            "object_mass": float(info["curriculum"]["object_mass"]),
            "friction_coefficient": float(info["curriculum"]["friction_coefficient"]),
        }
    
    def evaluate_heldout_set(
        self,
        num_episodes_per_object: int = 5,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Evaluate on all held-out objects.
        
        Args:
            num_episodes_per_object: Number of episodes per held-out object
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with evaluation results
        """
        # Ensure policy is frozen
        if not self._policy_frozen:
            self.freeze_policy()
        
        all_results = []
        object_results = {}
        
        rng = np.random.default_rng(seed)
        
        for obj_idx in range(len(self.heldout_set.heldout_objects)):
            obj = self.heldout_set.heldout_objects[obj_idx]
            eval_config = self.heldout_set.get_eval_config(obj_idx)
            
            obj_results = []
            
            for episode in range(num_episodes_per_object):
                episode_seed = rng.integers(0, 2**31) if seed is None else seed + episode
                result = self.evaluate_episode(eval_config, seed=episode_seed)
                
                result["object_idx"] = obj_idx
                result["episode"] = episode
                
                all_results.append(result)
                obj_results.append(result)
            
            # Aggregate per object
            object_results[obj_idx] = {
                "object_properties": {
                    "size": obj.size,
                    "mass": obj.mass,
                    "friction": obj.friction,
                },
                "episodes": obj_results,
                "mean_reward": float(np.mean([r["episode_reward"] for r in obj_results])),
                "mean_steps": float(np.mean([r["episode_steps"] for r in obj_results])),
                "success_rate": float(np.mean([1.0 if r["success"] else 0.0 for r in obj_results])),
            }
        
        # Overall statistics
        overall_stats = {
            "num_objects": len(self.heldout_set.heldout_objects),
            "total_episodes": len(all_results),
            "overall_success_rate": float(np.mean([1.0 if r["success"] else 0.0 for r in all_results])),
            "mean_reward": float(np.mean([r["episode_reward"] for r in all_results])),
            "std_reward": float(np.std([r["episode_reward"] for r in all_results])),
            "mean_steps": float(np.mean([r["episode_steps"] for r in all_results])),
        }
        
        return {
            "overall_stats": overall_stats,
            "per_object_results": object_results,
            "all_episodes": all_results,
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.freeze_policy()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unfreeze_policy()
