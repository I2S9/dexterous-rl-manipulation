"""
Failure taxonomy for dexterous manipulation.

This module defines a comprehensive taxonomy of failure modes
for dexterous manipulation tasks with automatic classification.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class FailureMode(Enum):
    """
    Taxonomy of failure modes for dexterous manipulation.
    
    Each failure mode represents a distinct way in which
    a manipulation attempt can fail.
    """
    SLIPPAGE = "slippage"
    UNSTABLE_GRASP = "unstable_grasp"
    MISALIGNMENT = "misalignment"
    TIMEOUT = "timeout"
    OBJECT_DROPPED = "object_dropped"
    INSUFFICIENT_CONTACTS = "insufficient_contacts"


@dataclass
class FailureModeDefinition:
    """
    Definition of a failure mode with characteristics and detection criteria.
    """
    mode: FailureMode
    name: str
    description: str
    key_indicators: List[str]
    detection_criteria: Dict


# Failure mode definitions
FAILURE_MODE_DEFINITIONS = {
    FailureMode.SLIPPAGE: FailureModeDefinition(
        mode=FailureMode.SLIPPAGE,
        name="Slippage",
        description="Object slips from fingers during manipulation, typically due to insufficient friction or grip force",
        key_indicators=[
            "Decreasing contact count over time",
            "Object position changes while contacts exist",
            "Contacts lost after being established"
        ],
        detection_criteria={
            "contact_trend_threshold": -1.0,  # Contacts decreasing
            "min_contacts_for_slippage": 1,  # At least one contact existed
            "contact_loss_threshold": 0.5,  # Significant contact loss
        }
    ),
    
    FailureMode.UNSTABLE_GRASP: FailureModeDefinition(
        mode=FailureMode.UNSTABLE_GRASP,
        name="Unstable Grasp",
        description="Contacts are established but highly variable, indicating unstable grasp configuration",
        key_indicators=[
            "High variance in contact count",
            "Rapid contact changes",
            "Contacts fluctuate without clear pattern"
        ],
        detection_criteria={
            "contact_variance_threshold": 2.0,  # High variance
            "min_episode_length": 10,  # Need enough steps to assess stability
            "contact_fluctuation_threshold": 3,  # Number of contact changes
        }
    ),
    
    FailureMode.MISALIGNMENT: FailureModeDefinition(
        mode=FailureMode.MISALIGNMENT,
        name="Misalignment",
        description="Fingers contact object but at suboptimal positions, preventing successful grasp",
        key_indicators=[
            "Some contacts exist but insufficient for grasp",
            "Contacts maintained but grasp incomplete",
            "Object not properly positioned relative to hand"
        ],
        detection_criteria={
            "min_contacts": 1,  # At least one contact
            "max_contacts": 2,  # But less than success threshold
            "contact_stability": True,  # Contacts are relatively stable
        }
    ),
    
    FailureMode.TIMEOUT: FailureModeDefinition(
        mode=FailureMode.TIMEOUT,
        name="Timeout",
        description="Episode reached maximum steps without achieving successful grasp",
        key_indicators=[
            "Episode length equals maximum steps",
            "No successful grasp achieved",
            "May have partial progress but incomplete"
        ],
        detection_criteria={
            "episode_length_equals_max": True,
            "success": False,
        }
    ),
    
    FailureMode.OBJECT_DROPPED: FailureModeDefinition(
        mode=FailureMode.OBJECT_DROPPED,
        name="Object Dropped",
        description="Object was dropped or fell during manipulation, losing all contacts",
        key_indicators=[
            "Final contact count is zero",
            "Object may have fallen",
            "Contacts lost completely"
        ],
        detection_criteria={
            "final_contacts": 0,
            "had_contacts_before": True,  # Had contacts at some point
        }
    ),
    
    FailureMode.INSUFFICIENT_CONTACTS: FailureModeDefinition(
        mode=FailureMode.INSUFFICIENT_CONTACTS,
        name="Insufficient Contacts",
        description="Never established sufficient contacts with the object to form a grasp",
        key_indicators=[
            "Contact count always below threshold",
            "Fingers never reached object",
            "No meaningful contact established"
        ],
        detection_criteria={
            "max_contacts_below_threshold": True,
            "never_reached_threshold": True,
        }
    ),
}


class FailureClassifier:
    """
    Classifier for automatically mapping episodes to failure modes.
    
    Uses heuristics based on contact history, episode length,
    and object state to classify failure types.
    """
    
    def __init__(self, success_threshold: int = 3):
        """
        Initialize failure classifier.
        
        Args:
            success_threshold: Minimum number of contacts for successful grasp
        """
        self.success_threshold = success_threshold
        self.definitions = FAILURE_MODE_DEFINITIONS
    
    def classify(
        self,
        episode_data: Dict,
        max_steps: int = 200
    ) -> Tuple[Optional[FailureMode], Dict]:
        """
        Classify failure mode for an episode.
        
        Args:
            episode_data: Dictionary with episode information
            max_steps: Maximum steps per episode
            
        Returns:
            Tuple of (FailureMode or None if successful, confidence dict)
        """
        success = episode_data.get("success", False)
        if success:
            return None, {}
        
        episode_steps = episode_data.get("episode_steps", 0)
        num_contacts = episode_data.get("num_contacts", 0)
        final_contacts = episode_data.get("final_contacts", num_contacts)
        contact_history = episode_data.get("contact_history", [])
        
        confidence = {}
        
        # Extract contact statistics
        if contact_history:
            contact_counts = [
                len([c for c in contacts if c > 0.5]) 
                for contacts in contact_history
            ]
            max_contacts = max(contact_counts) if contact_counts else 0
            contact_variance = np.var(contact_counts) if len(contact_counts) > 1 else 0.0
        else:
            contact_counts = []
            max_contacts = num_contacts
            contact_variance = 0.0
        
        # Priority 1: TIMEOUT (most specific)
        if episode_steps >= max_steps:
            confidence["timeout"] = 1.0
            return FailureMode.TIMEOUT, confidence
        
        # Priority 2: OBJECT_DROPPED (clear indicator)
        if final_contacts == 0 and max_contacts > 0:
            confidence["object_dropped"] = 1.0
            return FailureMode.OBJECT_DROPPED, confidence
        
        # Priority 3: Analyze contact patterns if history available
        if len(contact_counts) > 5:
            # Check for SLIPPAGE (contacts decreasing)
            if len(contact_counts) > 10:
                early_mean = np.mean(contact_counts[:5])
                late_mean = np.mean(contact_counts[-5:])
                trend = late_mean - early_mean
                
                slippage_criteria = self.definitions[FailureMode.SLIPPAGE].detection_criteria
                if trend < slippage_criteria["contact_trend_threshold"] and max_contacts >= slippage_criteria["min_contacts_for_slippage"]:
                    confidence["slippage"] = min(1.0, abs(trend) / 2.0)
                    return FailureMode.SLIPPAGE, confidence
            
            # Check for UNSTABLE_GRASP (high variance)
            unstable_criteria = self.definitions[FailureMode.UNSTABLE_GRASP].detection_criteria
            if contact_variance > unstable_criteria["contact_variance_threshold"]:
                confidence["unstable_grasp"] = min(1.0, contact_variance / 5.0)
                return FailureMode.UNSTABLE_GRASP, confidence
        
        # Priority 4: MISALIGNMENT (some contacts but insufficient)
        misalignment_criteria = self.definitions[FailureMode.MISALIGNMENT].detection_criteria
        if (misalignment_criteria["min_contacts"] <= num_contacts <= misalignment_criteria["max_contacts"]):
            # Check if contacts are relatively stable (not unstable)
            if contact_variance < 1.0:  # Low variance indicates stable but misaligned
                confidence["misalignment"] = 0.8
                return FailureMode.MISALIGNMENT, confidence
        
        # Priority 5: INSUFFICIENT_CONTACTS (default)
        if max_contacts < self.success_threshold:
            confidence["insufficient_contacts"] = 1.0
            return FailureMode.INSUFFICIENT_CONTACTS, confidence
        
        # Fallback
        confidence["insufficient_contacts"] = 0.5
        return FailureMode.INSUFFICIENT_CONTACTS, confidence
    
    def get_failure_mode_info(self, mode: FailureMode) -> FailureModeDefinition:
        """
        Get information about a failure mode.
        
        Args:
            mode: Failure mode
            
        Returns:
            FailureModeDefinition
        """
        return self.definitions[mode]
    
    def classify_batch(
        self,
        episodes: List[Dict],
        max_steps: int = 200
    ) -> Dict[FailureMode, List[Dict]]:
        """
        Classify a batch of episodes.
        
        Args:
            episodes: List of episode data dictionaries
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary mapping failure modes to lists of episodes
        """
        classified = {mode: [] for mode in FailureMode}
        classified[None] = []  # Successful episodes
        
        for episode in episodes:
            mode, confidence = self.classify(episode, max_steps)
            episode["failure_mode"] = mode.value if mode else None
            episode["failure_confidence"] = confidence
            
            if mode:
                classified[mode].append(episode)
            else:
                classified[None].append(episode)
        
        return classified
    
    def get_failure_statistics(
        self,
        episodes: List[Dict],
        max_steps: int = 200
    ) -> Dict:
        """
        Get statistics about failure modes in a set of episodes.
        
        Args:
            episodes: List of episode data dictionaries
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary with failure statistics
        """
        classified = self.classify_batch(episodes, max_steps)
        
        total_episodes = len(episodes)
        successful = len(classified[None])
        failed = total_episodes - successful
        
        failure_counts = {mode.value: len(episodes) for mode, episodes in classified.items() if mode is not None}
        failure_frequencies = {
            mode: count / total_episodes 
            for mode, count in failure_counts.items()
        }
        
        return {
            "total_episodes": total_episodes,
            "successful_episodes": successful,
            "failed_episodes": failed,
            "success_rate": successful / total_episodes if total_episodes > 0 else 0.0,
            "failure_counts": failure_counts,
            "failure_frequencies": failure_frequencies,
            "classified_episodes": classified,
        }
