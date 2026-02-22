"""Threshold calculation strategies for bubble field interpretation.

This module provides threshold calculation strategies split into focused files
matching the TypeScript implementation structure.
"""

from src.processors.detection.threshold.adaptive_threshold import (
    AdaptiveThresholdStrategy,
    create_default_threshold_calculator,
)
from src.processors.detection.threshold.global_threshold import GlobalThresholdStrategy
from src.processors.detection.threshold.local_threshold import LocalThresholdStrategy
from src.processors.detection.threshold.threshold_result import (
    ThresholdConfig,
    ThresholdResult,
)
from src.processors.detection.threshold.threshold_strategy import ThresholdStrategy

__all__ = [
    "ThresholdResult",
    "ThresholdConfig",
    "ThresholdStrategy",
    "GlobalThresholdStrategy",
    "LocalThresholdStrategy",
    "AdaptiveThresholdStrategy",
    "create_default_threshold_calculator",
]
