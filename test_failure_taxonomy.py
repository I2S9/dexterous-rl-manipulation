"""
Test script for failure taxonomy and automatic classification.

This script validates that failure modes are correctly defined
and that automatic classification works properly.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluation import (
    FailureMode,
    FailureClassifier,
    print_failure_taxonomy,
    print_failure_statistics,
    analyze_failure_modes,
)


def test_taxonomy_definitions():
    """Test that all failure modes are properly defined."""
    print("=" * 60)
    print("Test 1: Failure Taxonomy Definitions")
    print("=" * 60)
    
    from evaluation.failure_taxonomy import FAILURE_MODE_DEFINITIONS
    
    # Check all required modes are defined
    required_modes = [
        FailureMode.SLIPPAGE,
        FailureMode.UNSTABLE_GRASP,
        FailureMode.MISALIGNMENT,
        FailureMode.TIMEOUT,
    ]
    
    for mode in required_modes:
        assert mode in FAILURE_MODE_DEFINITIONS, f"Missing definition for {mode}"
        definition = FAILURE_MODE_DEFINITIONS[mode]
        assert definition.name, f"Missing name for {mode}"
        assert definition.description, f"Missing description for {mode}"
        assert definition.key_indicators, f"Missing indicators for {mode}"
        print(f"  [PASS] {mode.value}: {definition.name}")
    
    print("\n[PASS] All failure modes properly defined")
    return True


def test_classification_timeout():
    """Test timeout classification."""
    print("\n" + "=" * 60)
    print("Test 2: Timeout Classification")
    print("=" * 60)
    
    classifier = FailureClassifier()
    
    # Create timeout episode
    timeout_episode = {
        "success": False,
        "episode_steps": 200,  # Max steps
        "num_contacts": 2,
        "final_contacts": 2,
        "contact_history": [],
    }
    
    mode, confidence = classifier.classify(timeout_episode, max_steps=200)
    print(f"  Episode: timeout (200 steps, 2 contacts)")
    print(f"  Classified as: {mode.value if mode else 'None'}")
    print(f"  Confidence: {confidence}")
    
    assert mode == FailureMode.TIMEOUT, f"Expected TIMEOUT, got {mode}"
    print("\n[PASS] Timeout classification correct")
    return True


def test_classification_slippage():
    """Test slippage classification."""
    print("\n" + "=" * 60)
    print("Test 3: Slippage Classification")
    print("=" * 60)
    
    classifier = FailureClassifier()
    
    # Create slippage episode (contacts decreasing)
    contact_history = []
    for i in range(20):
        # Contacts start high and decrease
        contacts = [1.0] * max(0, 4 - i // 5) + [0.0] * (5 - max(0, 4 - i // 5))
        contact_history.append(contacts)
    
    slippage_episode = {
        "success": False,
        "episode_steps": 50,
        "num_contacts": 1,
        "final_contacts": 1,
        "contact_history": contact_history,
    }
    
    mode, confidence = classifier.classify(slippage_episode, max_steps=200)
    print(f"  Episode: slippage (contacts decreasing from 4 to 1)")
    print(f"  Classified as: {mode.value if mode else 'None'}")
    print(f"  Confidence: {confidence}")
    
    assert mode == FailureMode.SLIPPAGE, f"Expected SLIPPAGE, got {mode}"
    print("\n[PASS] Slippage classification correct")
    return True


def test_classification_unstable_grasp():
    """Test unstable grasp classification."""
    print("\n" + "=" * 60)
    print("Test 4: Unstable Grasp Classification")
    print("=" * 60)
    
    classifier = FailureClassifier()
    
    # Create unstable grasp episode (high variance in contacts)
    np.random.seed(42)  # For reproducibility
    contact_history = []
    for i in range(20):
        # Contacts fluctuate randomly with high variance
        num_contacts = np.random.choice([0, 1, 4, 0, 1, 4, 0, 1, 4, 0])  # High variance pattern
        contacts = [1.0] * num_contacts + [0.0] * (5 - num_contacts)
        contact_history.append(contacts)
    
    unstable_episode = {
        "success": False,
        "episode_steps": 50,
        "num_contacts": 2,
        "final_contacts": 2,
        "contact_history": contact_history,
    }
    
    mode, confidence = classifier.classify(unstable_episode, max_steps=200)
    print(f"  Episode: unstable grasp (high variance in contacts)")
    print(f"  Classified as: {mode.value if mode else 'None'}")
    print(f"  Confidence: {confidence}")
    
    # May classify as unstable or slippage depending on variance
    assert mode in [FailureMode.UNSTABLE_GRASP, FailureMode.SLIPPAGE], \
        f"Expected UNSTABLE_GRASP or SLIPPAGE, got {mode}"
    print("\n[PASS] Unstable grasp classification correct")
    return True


def test_classification_misalignment():
    """Test misalignment classification."""
    print("\n" + "=" * 60)
    print("Test 5: Misalignment Classification")
    print("=" * 60)
    
    classifier = FailureClassifier()
    
    # Create misalignment episode (stable but insufficient contacts)
    contact_history = []
    for i in range(20):
        # Stable 2 contacts (below threshold of 3)
        contacts = [1.0, 1.0, 0.0, 0.0, 0.0]
        contact_history.append(contacts)
    
    misalignment_episode = {
        "success": False,
        "episode_steps": 50,
        "num_contacts": 2,
        "final_contacts": 2,
        "contact_history": contact_history,
    }
    
    mode, confidence = classifier.classify(misalignment_episode, max_steps=200)
    print(f"  Episode: misalignment (stable 2 contacts, below threshold)")
    print(f"  Classified as: {mode.value if mode else 'None'}")
    print(f"  Confidence: {confidence}")
    
    assert mode == FailureMode.MISALIGNMENT, f"Expected MISALIGNMENT, got {mode}"
    print("\n[PASS] Misalignment classification correct")
    return True


def test_classification_object_dropped():
    """Test object dropped classification."""
    print("\n" + "=" * 60)
    print("Test 6: Object Dropped Classification")
    print("=" * 60)
    
    classifier = FailureClassifier()
    
    # Create object dropped episode
    contact_history = []
    for i in range(10):
        # Had contacts initially, then lost them
        if i < 5:
            contacts = [1.0, 1.0, 1.0, 0.0, 0.0]
        else:
            contacts = [0.0, 0.0, 0.0, 0.0, 0.0]
        contact_history.append(contacts)
    
    dropped_episode = {
        "success": False,
        "episode_steps": 50,
        "num_contacts": 0,
        "final_contacts": 0,
        "contact_history": contact_history,
    }
    
    mode, confidence = classifier.classify(dropped_episode, max_steps=200)
    print(f"  Episode: object dropped (had contacts, then lost all)")
    print(f"  Classified as: {mode.value if mode else 'None'}")
    print(f"  Confidence: {confidence}")
    
    assert mode == FailureMode.OBJECT_DROPPED, f"Expected OBJECT_DROPPED, got {mode}"
    print("\n[PASS] Object dropped classification correct")
    return True


def test_batch_classification():
    """Test batch classification."""
    print("\n" + "=" * 60)
    print("Test 7: Batch Classification")
    print("=" * 60)
    
    classifier = FailureClassifier()
    
    # Create diverse set of episodes
    episodes = [
        {"success": False, "episode_steps": 200, "num_contacts": 2, "final_contacts": 2, "contact_history": []},  # Timeout
        {"success": False, "episode_steps": 50, "num_contacts": 0, "final_contacts": 0, "contact_history": [[1.0, 1.0, 0.0, 0.0, 0.0]] * 5 + [[0.0] * 5] * 5},  # Dropped
        {"success": False, "episode_steps": 50, "num_contacts": 2, "final_contacts": 2, "contact_history": [[1.0, 1.0, 0.0, 0.0, 0.0]] * 20},  # Misalignment
        {"success": True, "episode_steps": 30, "num_contacts": 4, "final_contacts": 4, "contact_history": []},  # Success
    ]
    
    statistics = classifier.get_failure_statistics(episodes, max_steps=200)
    
    print(f"  Total episodes: {statistics['total_episodes']}")
    print(f"  Successful: {statistics['successful_episodes']}")
    print(f"  Failed: {statistics['failed_episodes']}")
    print(f"  Failure counts: {statistics['failure_counts']}")
    
    assert statistics['total_episodes'] == 4
    assert statistics['successful_episodes'] == 1
    assert statistics['failed_episodes'] == 3
    
    print("\n[PASS] Batch classification correct")
    return True


def test_failure_analysis():
    """Test failure analysis tools."""
    print("\n" + "=" * 60)
    print("Test 8: Failure Analysis Tools")
    print("=" * 60)
    
    # Create test episodes
    episodes = [
        {"success": False, "episode_steps": 200, "num_contacts": 2, "final_contacts": 2, "contact_history": []},
        {"success": False, "episode_steps": 50, "num_contacts": 2, "final_contacts": 2, "contact_history": [[1.0, 1.0, 0.0, 0.0, 0.0]] * 20},
        {"success": True, "episode_steps": 30, "num_contacts": 4, "final_contacts": 4, "contact_history": []},
    ]
    
    analysis = analyze_failure_modes(episodes, max_steps=200)
    
    print_failure_statistics(analysis)
    
    assert "detailed_analysis" in analysis
    print("\n[PASS] Failure analysis tools work correctly")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Failure Taxonomy Tests")
    print("=" * 60)
    
    # Print taxonomy
    print_failure_taxonomy()
    
    tests = [
        ("Taxonomy Definitions", test_taxonomy_definitions),
        ("Timeout Classification", test_classification_timeout),
        ("Slippage Classification", test_classification_slippage),
        ("Unstable Grasp Classification", test_classification_unstable_grasp),
        ("Misalignment Classification", test_classification_misalignment),
        ("Object Dropped Classification", test_classification_object_dropped),
        ("Batch Classification", test_batch_classification),
        ("Failure Analysis", test_failure_analysis),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    main()
