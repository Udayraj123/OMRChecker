"""Threshold calculation strategies for bubble field interpretation.

This module provides threshold calculation strategies split into focused files
matching the TypeScript implementation structure.
"""

from src.processors.threshold.adaptive_threshold import (
    AdaptiveThresholdStrategy,
    create_default_threshold_calculator,
)
from src.processors.threshold.global_threshold import GlobalThresholdStrategy
from src.processors.threshold.local_threshold import LocalThresholdStrategy
from src.processors.threshold.threshold_result import ThresholdConfig, ThresholdResult
from src.processors.threshold.threshold_strategy import ThresholdStrategy

__all__ = [
    "ThresholdResult",
    "ThresholdConfig",
    "ThresholdStrategy",
    "GlobalThresholdStrategy",
    "LocalThresholdStrategy",
    "AdaptiveThresholdStrategy",
    "create_default_threshold_calculator",
]
