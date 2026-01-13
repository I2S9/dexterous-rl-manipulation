"""
Logging utilities for curriculum progression tracking.
"""

import json
from pathlib import Path
from typing import Dict, List
from experiments.curriculum_scheduler import CurriculumScheduler


class CurriculumLogger:
    """
    Logger for tracking curriculum progression and difficulty evolution.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize curriculum logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logged data
        self.episode_logs: List[Dict] = []
        self.progression_logs: List[Dict] = []
    
    def log_episode(
        self,
        episode: int,
        scheduler: CurriculumScheduler,
        success: bool,
        episode_steps: int
    ):
        """
        Log episode with curriculum information.
        
        Args:
            episode: Episode number
            scheduler: Curriculum scheduler instance
            success: Whether episode was successful
            episode_steps: Number of steps in episode
        """
        stats = scheduler.get_statistics()
        
        log_entry = {
            "episode": episode,
            "total_steps": scheduler.total_steps,
            "difficulty_level": stats["current_difficulty_level"],
            "success": success,
            "episode_steps": episode_steps,
            "recent_success_rate": stats["recent_success_rate"],
            "object_size": scheduler.current_config.object_size,
            "object_mass": scheduler.current_config.object_mass,
            "friction_coefficient": scheduler.current_config.friction_coefficient,
        }
        
        self.episode_logs.append(log_entry)
    
    def log_progression(
        self,
        scheduler: CurriculumScheduler,
        progression_occurred: bool
    ):
        """
        Log curriculum progression event.
        
        Args:
            scheduler: Curriculum scheduler instance
            progression_occurred: Whether progression occurred in this update
        """
        if progression_occurred and scheduler.progression_history:
            latest = scheduler.progression_history[-1]
            self.progression_logs.append(latest.copy())
    
    def save(self, filename: str = "curriculum_progression.json"):
        """
        Save curriculum logs to JSON file.
        
        Args:
            filename: Name of log file
        """
        filepath = self.log_dir / filename
        
        data = {
            "episode_logs": self.episode_logs,
            "progression_logs": self.progression_logs,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def print_progression_summary(self, scheduler: CurriculumScheduler):
        """
        Print summary of curriculum progression.
        
        Args:
            scheduler: Curriculum scheduler instance
        """
        stats = scheduler.get_statistics()
        
        print("\n" + "=" * 60)
        print("Curriculum Progression Summary")
        print("=" * 60)
        print(f"Total episodes: {stats['total_episodes']}")
        print(f"Total steps: {stats['total_steps']}")
        print(f"Current difficulty level: {stats['current_difficulty_level']:.2f}")
        print(f"Recent success rate: {stats['recent_success_rate']:.3f}")
        print(f"Overall success rate: {stats['overall_success_rate']:.3f}")
        print(f"Number of progressions: {stats['num_progressions']}")
        
        if self.progression_logs:
            print("\nProgression History:")
            print("-" * 60)
            for i, prog in enumerate(self.progression_logs, 1):
                print(f"Progression {i}:")
                print(f"  Episode: {prog.get('episode', 'N/A')}")
                print(f"  Total steps: {prog.get('total_steps', 'N/A')}")
                print(f"  Difficulty level: {prog.get('difficulty_level', 0):.2f}")
                print(f"  Success rate: {prog.get('success_rate', 0):.3f}")
                print(f"  Object size: {prog.get('object_size', 0):.4f} m")
                print(f"  Object mass: {prog.get('object_mass', 0):.4f} kg")
                print(f"  Friction: {prog.get('friction_coefficient', 0):.3f}")
                print()
        
        print("=" * 60)
