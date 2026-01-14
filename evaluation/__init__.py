"""
Evaluation modules for dexterous manipulation.
"""

from evaluation.heldout_objects import HeldOutObjectSet, ObjectProperties, generate_training_objects
from evaluation.evaluator import Evaluator
from evaluation.metrics import EvaluationMetrics, FailureType, format_metrics_report
from evaluation.robustness_tests import (
    RobustnessTester,
    NoisyObservationWrapper,
    NoisyDynamicsWrapper,
    CombinedNoiseWrapper,
)
from evaluation.failure_taxonomy import (
    FailureMode,
    FailureClassifier,
    FAILURE_MODE_DEFINITIONS,
)
from evaluation.failure_analysis import (
    print_failure_taxonomy,
    print_failure_statistics,
    analyze_failure_modes,
)
from evaluation.failure_logger import FailureLogger, EpisodeRecorder
from evaluation.failure_statistics import (
    load_failure_logs,
    compute_failure_distribution,
    compute_correlations,
    plot_failure_distribution,
    plot_failure_correlations,
    generate_failure_report,
)
from evaluation.seed_variance import (
    SeedVarianceAnalyzer,
    analyze_seed_variance,
    plot_seed_variance,
    print_variance_report,
)

__all__ = [
    "HeldOutObjectSet", 
    "ObjectProperties", 
    "generate_training_objects", 
    "Evaluator",
    "EvaluationMetrics",
    "FailureType",
    "format_metrics_report",
    "RobustnessTester",
    "NoisyObservationWrapper",
    "NoisyDynamicsWrapper",
    "CombinedNoiseWrapper",
    "FailureMode",
    "FailureClassifier",
    "FAILURE_MODE_DEFINITIONS",
    "print_failure_taxonomy",
    "print_failure_statistics",
    "analyze_failure_modes",
    "FailureLogger",
    "EpisodeRecorder",
    "load_failure_logs",
    "compute_failure_distribution",
    "compute_correlations",
    "plot_failure_distribution",
    "plot_failure_correlations",
    "generate_failure_report",
    "SeedVarianceAnalyzer",
    "analyze_seed_variance",
    "plot_seed_variance",
    "print_variance_report",
]
