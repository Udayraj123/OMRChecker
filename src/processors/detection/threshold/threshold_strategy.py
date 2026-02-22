"""Abstract base class for threshold calculation strategies."""

from abc import ABC, abstractmethod

from src.processors.detection.threshold.threshold_result import (
    ThresholdConfig,
    ThresholdResult,
)


class ThresholdStrategy(ABC):
    """Abstract base class for threshold calculation strategies."""

    @abstractmethod
    def calculate_threshold(
        self, bubble_mean_values: list[float], config: ThresholdConfig
    ) -> ThresholdResult:
        """Calculate threshold from bubble mean intensity values.

        Args:
            bubble_mean_values: List of bubble mean intensity values
            config: Threshold configuration

        Returns:
            ThresholdResult with threshold and confidence information
        """
