"""
Failure analysis and reporting for dexterous manipulation.

This module provides tools for analyzing and reporting failure modes.
"""

from typing import Dict, List
from evaluation.failure_taxonomy import FailureClassifier, FailureMode, FAILURE_MODE_DEFINITIONS


def print_failure_taxonomy():
    """Print the failure taxonomy with definitions."""
    print("=" * 60)
    print("Dexterous Manipulation Failure Taxonomy")
    print("=" * 60)
    
    for mode, definition in FAILURE_MODE_DEFINITIONS.items():
        print(f"\n{mode.value.upper()}: {definition.name}")
        print(f"  Description: {definition.description}")
        print(f"  Key Indicators:")
        for indicator in definition.key_indicators:
            print(f"    - {indicator}")
    
    print("\n" + "=" * 60)


def print_failure_statistics(statistics: Dict):
    """
    Print formatted failure statistics.
    
    Args:
        statistics: Dictionary with failure statistics
    """
    print("\n" + "=" * 60)
    print("Failure Mode Statistics")
    print("=" * 60)
    
    print(f"\nOverall:")
    print(f"  Total episodes: {statistics['total_episodes']}")
    print(f"  Successful: {statistics['successful_episodes']} ({statistics['success_rate']:.1%})")
    print(f"  Failed: {statistics['failed_episodes']}")
    
    print(f"\nFailure Mode Distribution:")
    failure_counts = statistics['failure_counts']
    failure_frequencies = statistics['failure_frequencies']
    
    # Sort by frequency
    sorted_failures = sorted(
        failure_frequencies.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for mode, frequency in sorted_failures:
        count = failure_counts[mode]
        print(f"  {mode}: {count} episodes ({frequency:.1%})")
    
    print("=" * 60)


def analyze_failure_modes(
    episodes: List[Dict],
    max_steps: int = 200,
    success_threshold: int = 3
) -> Dict:
    """
    Analyze failure modes in a set of episodes.
    
    Args:
        episodes: List of episode data dictionaries
        max_steps: Maximum steps per episode
        success_threshold: Success threshold for contacts
        
    Returns:
        Dictionary with analysis results
    """
    classifier = FailureClassifier(success_threshold=success_threshold)
    statistics = classifier.get_failure_statistics(episodes, max_steps)
    
    # Add detailed analysis per failure mode
    detailed_analysis = {}
    for mode, mode_episodes in statistics["classified_episodes"].items():
        if mode is None or len(mode_episodes) == 0:
            continue
        
        # Compute statistics for this failure mode
        episode_lengths = [e["episode_steps"] for e in mode_episodes]
        contact_counts = [e.get("num_contacts", 0) for e in mode_episodes]
        
        detailed_analysis[mode.value] = {
            "count": len(mode_episodes),
            "mean_episode_length": float(sum(episode_lengths) / len(episode_lengths)) if episode_lengths else 0.0,
            "mean_contacts": float(sum(contact_counts) / len(contact_counts)) if contact_counts else 0.0,
            "episodes": mode_episodes,
        }
    
    statistics["detailed_analysis"] = detailed_analysis
    
    return statistics
