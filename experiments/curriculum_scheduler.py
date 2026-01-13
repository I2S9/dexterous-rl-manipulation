"""
Curriculum scheduler for progressive difficulty adjustment.

This module implements a scheduler that adjusts task difficulty based on
training metrics such as success rate and number of steps.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from experiments.config import CurriculumConfig


class CurriculumScheduler:
    """
    Scheduler that progressively increases task difficulty based on performance.
    
    The scheduler tracks training metrics and adjusts curriculum parameters
    when certain conditions are met (e.g., success rate threshold, step count).
    """
    
    def __init__(
        self,
        initial_config: CurriculumConfig,
        target_config: CurriculumConfig,
        success_rate_threshold: float = 0.7,
        min_episodes_before_progression: int = 50,
        window_size: int = 20,
        progression_steps: int = 5,
    ):
        """
        Initialize curriculum scheduler.
        
        Args:
            initial_config: Starting curriculum configuration (easy)
            target_config: Target curriculum configuration (hard)
            success_rate_threshold: Success rate needed to progress (default: 0.7)
            min_episodes_before_progression: Minimum episodes before first progression
            window_size: Number of recent episodes to consider for metrics
            progression_steps: Number of difficulty levels between initial and target
        """
        self.initial_config = initial_config
        self.target_config = target_config
        self.success_rate_threshold = success_rate_threshold
        self.min_episodes_before_progression = min_episodes_before_progression
        self.window_size = window_size
        self.progression_steps = progression_steps
        
        # Current configuration (starts at initial)
        self.current_config = self._copy_config(initial_config)
        self.current_difficulty_level = 0.0  # 0.0 = initial, 1.0 = target
        
        # Training metrics
        self.episode_successes: List[bool] = []
        self.episode_steps: List[int] = []
        self.total_steps = 0
        self.total_episodes = 0
        
        # Progression history
        self.progression_history: List[Dict] = []
        
    def _copy_config(self, config: CurriculumConfig) -> CurriculumConfig:
        """Create a copy of a curriculum configuration."""
        return CurriculumConfig(
            object_size=config.object_size,
            object_size_range=config.object_size_range,
            object_mass=config.object_mass,
            object_mass_range=config.object_mass_range,
            friction_coefficient=config.friction_coefficient,
            friction_range=config.friction_range,
            spawn_distance=config.spawn_distance,
            spawn_distance_range=config.spawn_distance_range,
            spawn_x_range=config.spawn_x_range,
            spawn_y_range=config.spawn_y_range,
            spawn_z_range=config.spawn_z_range,
        )
    
    def _interpolate_config(
        self,
        difficulty: float
    ) -> CurriculumConfig:
        """
        Interpolate between initial and target configurations.
        
        Args:
            difficulty: Difficulty level between 0.0 (initial) and 1.0 (target)
            
        Returns:
            Interpolated curriculum configuration
        """
        difficulty = np.clip(difficulty, 0.0, 1.0)
        
        # Interpolate each parameter
        object_size = (
            self.initial_config.object_size * (1 - difficulty) +
            self.target_config.object_size * difficulty
        )
        
        object_mass = (
            self.initial_config.object_mass * (1 - difficulty) +
            self.target_config.object_mass * difficulty
        )
        
        friction_coefficient = (
            self.initial_config.friction_coefficient * (1 - difficulty) +
            self.target_config.friction_coefficient * difficulty
        )
        
        spawn_distance = (
            self.initial_config.spawn_distance * (1 - difficulty) +
            self.target_config.spawn_distance * difficulty
        )
        
        # Interpolate ranges if they exist
        object_size_range = None
        if (self.initial_config.object_size_range is not None and
            self.target_config.object_size_range is not None):
            min_size = (
                self.initial_config.object_size_range[0] * (1 - difficulty) +
                self.target_config.object_size_range[0] * difficulty
            )
            max_size = (
                self.initial_config.object_size_range[1] * (1 - difficulty) +
                self.target_config.object_size_range[1] * difficulty
            )
            object_size_range = (min_size, max_size)
        
        return CurriculumConfig(
            object_size=object_size,
            object_size_range=object_size_range,
            object_mass=object_mass,
            object_mass_range=self.initial_config.object_mass_range,  # Keep initial for now
            friction_coefficient=friction_coefficient,
            friction_range=self.initial_config.friction_range,  # Keep initial for now
            spawn_distance=spawn_distance,
            spawn_distance_range=self.initial_config.spawn_distance_range,  # Keep initial for now
            spawn_x_range=self.initial_config.spawn_x_range,
            spawn_y_range=self.initial_config.spawn_y_range,
            spawn_z_range=self.initial_config.spawn_z_range,
        )
    
    def update(self, success: bool, episode_steps: int) -> bool:
        """
        Update scheduler with episode results and check for progression.
        
        Args:
            success: Whether the episode was successful
            episode_steps: Number of steps in the episode
            
        Returns:
            True if difficulty was increased, False otherwise
        """
        self.episode_successes.append(success)
        self.episode_steps.append(episode_steps)
        self.total_steps += episode_steps
        self.total_episodes += 1
        
        # Check if we should progress
        if self._should_progress():
            return self._progress()
        
        return False
    
    def _should_progress(self) -> bool:
        """
        Check if conditions are met for difficulty progression.
        
        Returns:
            True if progression conditions are met
        """
        # Need minimum episodes before first progression
        if self.total_episodes < self.min_episodes_before_progression:
            return False
        
        # Already at maximum difficulty
        if self.current_difficulty_level >= 1.0:
            return False
        
        # Check success rate in recent window
        if len(self.episode_successes) < self.window_size:
            return False
        
        recent_successes = self.episode_successes[-self.window_size:]
        success_rate = np.mean(recent_successes)
        
        # Condition: success rate above threshold
        return success_rate >= self.success_rate_threshold
    
    def _progress(self) -> bool:
        """
        Increase difficulty level.
        
        Returns:
            True if progression occurred
        """
        # Calculate new difficulty level
        step_size = 1.0 / self.progression_steps
        new_difficulty = min(
            self.current_difficulty_level + step_size,
            1.0
        )
        
        if new_difficulty > self.current_difficulty_level:
            self.current_difficulty_level = new_difficulty
            self.current_config = self._interpolate_config(new_difficulty)
            
            # Log progression
            recent_successes = self.episode_successes[-self.window_size:]
            success_rate = np.mean(recent_successes)
            
            self.progression_history.append({
                "episode": self.total_episodes,
                "total_steps": self.total_steps,
                "difficulty_level": float(self.current_difficulty_level),
                "success_rate": float(success_rate),
                "object_size": float(self.current_config.object_size),
                "object_mass": float(self.current_config.object_mass),
                "friction_coefficient": float(self.current_config.friction_coefficient),
            })
            
            return True
        
        return False
    
    def get_current_config(self) -> CurriculumConfig:
        """
        Get current curriculum configuration.
        
        Returns:
            Current curriculum configuration
        """
        return self.current_config
    
    def get_difficulty_level(self) -> float:
        """
        Get current difficulty level (0.0 to 1.0).
        
        Returns:
            Current difficulty level
        """
        return self.current_difficulty_level
    
    def get_statistics(self) -> Dict:
        """
        Get scheduler statistics.
        
        Returns:
            Dictionary with scheduler statistics
        """
        recent_successes = (
            self.episode_successes[-self.window_size:]
            if len(self.episode_successes) >= self.window_size
            else self.episode_successes
        )
        
        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "current_difficulty_level": float(self.current_difficulty_level),
            "recent_success_rate": float(np.mean(recent_successes)) if recent_successes else 0.0,
            "overall_success_rate": float(np.mean(self.episode_successes)) if self.episode_successes else 0.0,
            "num_progressions": len(self.progression_history),
            "progression_history": self.progression_history.copy(),
        }
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_config = self._copy_config(self.initial_config)
        self.current_difficulty_level = 0.0
        self.episode_successes = []
        self.episode_steps = []
        self.total_steps = 0
        self.total_episodes = 0
        self.progression_history = []


class StepBasedScheduler(CurriculumScheduler):
    """
    Scheduler that progresses based on step count milestones.
    
    Increases difficulty after reaching certain step milestones,
    regardless of success rate.
    """
    
    def __init__(
        self,
        initial_config: CurriculumConfig,
        target_config: CurriculumConfig,
        step_milestones: List[int],
        **kwargs
    ):
        """
        Initialize step-based scheduler.
        
        Args:
            initial_config: Starting curriculum configuration
            target_config: Target curriculum configuration
            step_milestones: List of step counts at which to progress
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(initial_config, target_config, **kwargs)
        self.step_milestones = sorted(step_milestones)
        self.current_milestone_idx = 0
    
    def _should_progress(self) -> bool:
        """Check if step milestone is reached."""
        if self.current_milestone_idx >= len(self.step_milestones):
            return False
        
        next_milestone = self.step_milestones[self.current_milestone_idx]
        return self.total_steps >= next_milestone
    
    def _progress(self) -> bool:
        """Progress to next milestone."""
        if self.current_milestone_idx < len(self.step_milestones):
            self.current_milestone_idx += 1
            
            # Calculate difficulty based on milestone progress
            progress = self.current_milestone_idx / len(self.step_milestones)
            self.current_difficulty_level = min(progress, 1.0)
            self.current_config = self._interpolate_config(self.current_difficulty_level)
            
            # Log progression
            self.progression_history.append({
                "episode": self.total_episodes,
                "total_steps": self.total_steps,
                "milestone": self.step_milestones[self.current_milestone_idx - 1],
                "difficulty_level": float(self.current_difficulty_level),
                "object_size": float(self.current_config.object_size),
                "object_mass": float(self.current_config.object_mass),
                "friction_coefficient": float(self.current_config.friction_coefficient),
            })
            
            return True
        
        return False
