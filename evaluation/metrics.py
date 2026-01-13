"""
Evaluation metrics for dexterous manipulation.

This module computes key metrics for evaluating manipulation performance:
- Grasp success rate
- Mean episode length
- Failure type frequency
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum


class FailureType(Enum):
    """Types of manipulation failures."""
    SLIPPAGE = "slippage"  # Object slips from fingers
    UNSTABLE_CONTACTS = "unstable_contacts"  # Contacts are unstable
    MISALIGNED_GRASP = "misaligned_grasp"  # Misaligned grasp points
    TIMEOUT = "timeout"  # Episode reached max steps without success
    OBJECT_DROPPED = "object_dropped"  # Object fell/dropped
    INSUFFICIENT_CONTACTS = "insufficient_contacts"  # Not enough fingers in contact


class EvaluationMetrics:
    """
    Computes evaluation metrics for manipulation tasks.
    """
    
    def __init__(self, success_threshold: int = 3):
        """
        Initialize metrics calculator.
        
        Args:
            success_threshold: Minimum number of contacts for successful grasp
        """
        self.success_threshold = success_threshold
    
    def classify_failure(
        self,
        episode_data: Dict,
        max_steps: int = 200
    ) -> Optional[FailureType]:
        """
        Classify the type of failure for an unsuccessful episode.
        
        Args:
            episode_data: Dictionary with episode information
            max_steps: Maximum steps per episode
            
        Returns:
            FailureType if episode failed, None if successful
        """
        success = episode_data.get("success", False)
        if success:
            return None
        
        episode_steps = episode_data.get("episode_steps", 0)
        num_contacts = episode_data.get("num_contacts", 0)
        final_contacts = episode_data.get("final_contacts", num_contacts)
        contact_history = episode_data.get("contact_history", [])
        
        # Timeout: reached max steps without success
        if episode_steps >= max_steps:
            return FailureType.TIMEOUT
        
        # Object dropped (no contacts at end) - check this first
        if final_contacts == 0:
            return FailureType.OBJECT_DROPPED
        
        # Analyze contact stability if history available
        if contact_history:
            contact_counts = [len([c for c in contacts if c > 0.5]) for contacts in contact_history]
            
            if len(contact_counts) > 5:
                # Check for unstable contacts (high variance in contact count)
                contact_variance = np.var(contact_counts)
                if contact_variance > 2.0:  # High variance indicates instability
                    return FailureType.UNSTABLE_CONTACTS
                
                # Check for slippage (contacts decreasing over time)
                if len(contact_counts) > 10:
                    recent_trend = np.mean(contact_counts[-5:]) - np.mean(contact_counts[:5])
                    if recent_trend < -1.0:  # Contacts decreasing
                        return FailureType.SLIPPAGE
        
        # Misaligned grasp (contacts exist but not sufficient)
        if num_contacts > 0 and num_contacts < self.success_threshold:
            return FailureType.MISALIGNED_GRASP
        
        # Insufficient contacts: never reached minimum contacts
        if num_contacts < self.success_threshold and final_contacts < self.success_threshold:
            return FailureType.INSUFFICIENT_CONTACTS
        
        # Default classification
        return FailureType.INSUFFICIENT_CONTACTS
    
    def compute_episode_metrics(
        self,
        episode_data: Dict,
        max_steps: int = 200
    ) -> Dict:
        """
        Compute metrics for a single episode.
        
        Args:
            episode_data: Dictionary with episode information
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary with episode metrics
        """
        success = episode_data.get("success", False)
        episode_steps = episode_data.get("episode_steps", 0)
        num_contacts = episode_data.get("num_contacts", 0)
        
        failure_type = None if success else self.classify_failure(episode_data, max_steps)
        
        return {
            "success": success,
            "episode_length": episode_steps,
            "num_contacts": num_contacts,
            "failure_type": failure_type.value if failure_type else None,
        }
    
    def compute_aggregate_metrics(
        self,
        all_episodes: List[Dict],
        max_steps: int = 200
    ) -> Dict:
        """
        Compute aggregate metrics across all episodes.
        
        Args:
            all_episodes: List of episode data dictionaries
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary with aggregate metrics
        """
        if not all_episodes:
            return {}
        
        # Compute episode metrics
        episode_metrics = [
            self.compute_episode_metrics(ep, max_steps) for ep in all_episodes
        ]
        
        # Grasp success rate
        successes = [m["success"] for m in episode_metrics]
        grasp_success_rate = float(np.mean(successes))
        
        # Mean episode length
        episode_lengths = [m["episode_length"] for m in episode_metrics]
        mean_episode_length = float(np.mean(episode_lengths))
        std_episode_length = float(np.std(episode_lengths))
        
        # Failure type frequency
        failure_types = [m["failure_type"] for m in episode_metrics if m["failure_type"] is not None]
        failure_type_counts = {}
        for ft in FailureType:
            count = failure_types.count(ft.value)
            failure_type_counts[ft.value] = {
                "count": count,
                "frequency": float(count / len(all_episodes)) if all_episodes else 0.0,
            }
        
        # Additional statistics
        successful_episodes = [m for m in episode_metrics if m["success"]]
        failed_episodes = [m for m in episode_metrics if not m["success"]]
        
        mean_contacts = float(np.mean([m["num_contacts"] for m in episode_metrics]))
        
        # Success episode statistics
        if successful_episodes:
            mean_success_length = float(np.mean([m["episode_length"] for m in successful_episodes]))
            mean_success_contacts = float(np.mean([m["num_contacts"] for m in successful_episodes]))
        else:
            mean_success_length = None
            mean_success_contacts = None
        
        # Failure episode statistics
        if failed_episodes:
            mean_failure_length = float(np.mean([m["episode_length"] for m in failed_episodes]))
            mean_failure_contacts = float(np.mean([m["num_contacts"] for m in failed_episodes]))
        else:
            mean_failure_length = None
            mean_failure_contacts = None
        
        return {
            "grasp_success_rate": grasp_success_rate,
            "mean_episode_length": mean_episode_length,
            "std_episode_length": std_episode_length,
            "failure_type_frequency": failure_type_counts,
            "total_episodes": len(all_episodes),
            "successful_episodes": len(successful_episodes),
            "failed_episodes": len(failed_episodes),
            "mean_contacts": mean_contacts,
            "mean_success_length": mean_success_length,
            "mean_success_contacts": mean_success_contacts,
            "mean_failure_length": mean_failure_length,
            "mean_failure_contacts": mean_failure_contacts,
        }
    
    def compute_per_object_metrics(
        self,
        per_object_results: Dict,
        max_steps: int = 200
    ) -> Dict:
        """
        Compute metrics per object.
        
        Args:
            per_object_results: Dictionary with per-object episode results
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary with per-object metrics
        """
        per_object_metrics = {}
        
        for obj_idx, obj_data in per_object_results.items():
            episodes = obj_data.get("episodes", [])
            
            if episodes:
                metrics = self.compute_aggregate_metrics(episodes, max_steps)
                per_object_metrics[obj_idx] = metrics
        
        return per_object_metrics


def format_metrics_report(metrics: Dict) -> str:
    """
    Format metrics as a readable report.
    
    Args:
        metrics: Dictionary with metrics
        
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Evaluation Metrics Report")
    lines.append("=" * 60)
    
    # Overall statistics
    lines.append("\nOverall Statistics:")
    lines.append(f"  Total episodes: {metrics.get('total_episodes', 0)}")
    lines.append(f"  Successful episodes: {metrics.get('successful_episodes', 0)}")
    lines.append(f"  Failed episodes: {metrics.get('failed_episodes', 0)}")
    
    # Grasp success rate
    success_rate = metrics.get('grasp_success_rate', 0.0)
    lines.append(f"\nGrasp Success Rate: {success_rate:.1%}")
    if success_rate >= 0.70:
        lines.append(f"  Status: PASS (>= 70%)")
    elif success_rate >= 0.50:
        lines.append(f"  Status: MARGINAL (50-70%)")
    else:
        lines.append(f"  Status: FAIL (< 50%)")
    
    # Mean episode length
    mean_length = metrics.get('mean_episode_length', 0.0)
    std_length = metrics.get('std_episode_length', 0.0)
    lines.append(f"\nMean Episode Length: {mean_length:.1f} ± {std_length:.1f} steps")
    
    if metrics.get('mean_success_length'):
        lines.append(f"  Success episodes: {metrics['mean_success_length']:.1f} steps")
    if metrics.get('mean_failure_length'):
        lines.append(f"  Failure episodes: {metrics['mean_failure_length']:.1f} steps")
    
    # Failure type frequency
    lines.append(f"\nFailure Type Frequency:")
    failure_freq = metrics.get('failure_type_frequency', {})
    for failure_type, data in sorted(failure_freq.items(), key=lambda x: x[1]['count'], reverse=True):
        count = data['count']
        frequency = data['frequency']
        if count > 0:
            lines.append(f"  {failure_type}: {count} ({frequency:.1%})")
    
    # Contact statistics
    lines.append(f"\nContact Statistics:")
    lines.append(f"  Mean contacts: {metrics.get('mean_contacts', 0.0):.2f}")
    if metrics.get('mean_success_contacts'):
        lines.append(f"  Success episodes: {metrics['mean_success_contacts']:.2f} contacts")
    if metrics.get('mean_failure_contacts'):
        lines.append(f"  Failure episodes: {metrics['mean_failure_contacts']:.2f} contacts")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
