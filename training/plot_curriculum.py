"""
Plot curriculum progression and difficulty evolution.

This script visualizes how task difficulty evolves over training
based on the curriculum scheduler logs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def load_curriculum_logs(log_dir: str = "logs") -> Dict:
    """
    Load curriculum progression logs.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        Dictionary with curriculum logs
    """
    log_path = Path(log_dir) / "curriculum_progression.json"
    
    if not log_path.exists():
        raise FileNotFoundError(f"Curriculum logs not found at {log_path}")
    
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_curriculum_evolution(logs: Dict, output_path: str = "logs/curriculum_evolution.png"):
    """
    Plot curriculum evolution over training.
    
    Args:
        logs: Dictionary with curriculum logs
        output_path: Path to save the plot
    """
    episode_logs = logs["episode_logs"]
    progression_logs = logs["progression_logs"]
    
    if not episode_logs:
        print("No episode logs found.")
        return
    
    # Extract data
    episodes = [log["episode"] for log in episode_logs]
    difficulty_levels = [log["difficulty_level"] for log in episode_logs]
    success_rates = [log["recent_success_rate"] for log in episode_logs]
    object_sizes = [log["object_size"] for log in episode_logs]
    object_masses = [log["object_mass"] for log in episode_logs]
    friction_coeffs = [log["friction_coefficient"] for log in episode_logs]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Plot 1: Difficulty level over episodes
    ax1 = axes[0, 0]
    ax1.plot(episodes, difficulty_levels, linewidth=2, label="Difficulty Level")
    
    # Mark progression points
    if progression_logs:
        prog_episodes = [p.get("episode", 0) for p in progression_logs]
        prog_difficulties = [p.get("difficulty_level", 0) for p in progression_logs]
        ax1.scatter(prog_episodes, prog_difficulties, color='red', s=100, 
                   marker='*', zorder=5, label="Progression")
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Difficulty Level")
    ax1.set_title("Curriculum Difficulty Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Plot 2: Success rate over episodes
    ax2 = axes[0, 1]
    ax2.plot(episodes, success_rates, linewidth=2, label="Recent Success Rate", alpha=0.7)
    
    # Mark progression points
    if progression_logs:
        prog_episodes = [p.get("episode", 0) for p in progression_logs]
        prog_success = [p.get("success_rate", 0) for p in progression_logs]
        ax2.scatter(prog_episodes, prog_success, color='red', s=100, 
                   marker='*', zorder=5, label="Progression")
    
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate")
    ax2.set_title("Success Rate Over Training")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    # Plot 3: Object size evolution
    ax3 = axes[1, 0]
    ax3.plot(episodes, object_sizes, linewidth=2, label="Object Size", color='green')
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Object Size (m)")
    ax3.set_title("Object Size Evolution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Object mass evolution
    ax4 = axes[1, 1]
    ax4.plot(episodes, object_masses, linewidth=2, label="Object Mass", color='orange')
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Object Mass (kg)")
    ax4.set_title("Object Mass Evolution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Friction coefficient evolution
    ax5 = axes[2, 0]
    ax5.plot(episodes, friction_coeffs, linewidth=2, label="Friction Coefficient", color='purple')
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Friction Coefficient")
    ax5.set_title("Friction Coefficient Evolution")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Combined curriculum parameters
    ax6 = axes[2, 1]
    ax6_twin = ax6.twinx()
    
    line1 = ax6.plot(episodes, object_sizes, linewidth=2, label="Object Size", color='green')
    line2 = ax6_twin.plot(episodes, object_masses, linewidth=2, label="Object Mass", color='orange')
    line3 = ax6_twin.plot(episodes, friction_coeffs, linewidth=2, label="Friction", color='purple')
    
    ax6.set_xlabel("Episode")
    ax6.set_ylabel("Object Size (m)", color='green')
    ax6_twin.set_ylabel("Mass (kg) / Friction", color='orange')
    ax6.set_title("All Curriculum Parameters")
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper left')
    
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Curriculum evolution plot saved to {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Curriculum Evolution Summary")
    print("=" * 60)
    print(f"Total episodes: {len(episodes)}")
    print(f"Initial difficulty: {difficulty_levels[0]:.2f}")
    print(f"Final difficulty: {difficulty_levels[-1]:.2f}")
    print(f"Number of progressions: {len(progression_logs)}")
    
    if progression_logs:
        print("\nProgression milestones:")
        for i, prog in enumerate(progression_logs, 1):
            print(f"  {i}. Episode {prog.get('episode', 'N/A')}, "
                  f"Steps: {prog.get('total_steps', 'N/A')}, "
                  f"Difficulty: {prog.get('difficulty_level', 0):.2f}")
    
    print("=" * 60)


if __name__ == "__main__":
    try:
        logs = load_curriculum_logs()
        plot_curriculum_evolution(logs)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run test_curriculum_scheduler.py first to generate logs.")
