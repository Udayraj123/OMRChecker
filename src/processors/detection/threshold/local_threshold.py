"""Local threshold strategy for bubble detection.

Strategy using field-level statistics.
Based on the existing get_local_threshold logic.
Calculates threshold for individual field, with fallback to global.
"""

import numpy as np

from src.processors.detection.threshold.threshold_result import (
    ThresholdConfig,
    ThresholdResult,
)
from src.processors.detection.threshold.threshold_strategy import ThresholdStrategy


class LocalThresholdStrategy(ThresholdStrategy):
    """Strategy using field-level statistics.

    Based on the existing get_local_threshold logic.
    Calculates threshold for individual field, with fallback to global.
    """

    def __init__(self, global_fallback: float | None = None) -> None:
        """Initialize with optional global fallback threshold.

        Args:
            global_fallback: Global threshold to use when local confidence is low
        """
        self.global_fallback = global_fallback

    def calculate_threshold(
        self, bubble_mean_values: list[float], config: ThresholdConfig
    ) -> ThresholdResult:
        """Calculate local threshold with global fallback."""
        fallback_threshold = self.global_fallback or config.default_threshold

        # Base case: empty or single bubble
        if len(bubble_mean_values) < 2:
            return ThresholdResult(
                threshold_value=fallback_threshold,
                confidence=0.0,
                max_jump=0.0,
                method_used="local_single_bubble_fallback",
                fallback_used=True,
            )

        sorted_values = sorted(bubble_mean_values)

        # Special case: exactly 2 bubbles
        if len(sorted_values) == 2:
            gap = sorted_values[1] - sorted_values[0]
            if gap < config.min_gap_two_bubbles:
                return ThresholdResult(
                    threshold_value=fallback_threshold,
                    confidence=0.3,
                    max_jump=gap,
                    method_used="local_two_bubbles_small_gap_fallback",
                    fallback_used=True,
                )
            return ThresholdResult(
                threshold_value=float(np.mean(sorted_values)),
                confidence=0.7,
                max_jump=gap,
                method_used="local_two_bubbles_mean",
                fallback_used=False,
            )

        # 3+ bubbles: find largest jump
        max_jump = 0.0
        threshold = fallback_threshold

        for i in range(1, len(sorted_values) - 1):
            jump = sorted_values[i + 1] - sorted_values[i - 1]
            if jump > max_jump:
                max_jump = jump
                threshold = sorted_values[i - 1] + jump / 2

        # Check if jump is confident
        confident_jump = config.min_jump + config.min_jump_surplus_for_global_fallback

        if max_jump < confident_jump:
            # Low confidence - use global fallback
            return ThresholdResult(
                threshold_value=fallback_threshold,
                confidence=0.4,
                max_jump=max_jump,
                method_used="local_low_confidence_global_fallback",
                fallback_used=True,
                metadata={"local_threshold": threshold},
            )

        # High confidence
        confidence = min(1.0, max_jump / (confident_jump * 2))

        return ThresholdResult(
            threshold_value=threshold,
            confidence=confidence,
            max_jump=max_jump,
            method_used="local_max_jump",
            fallback_used=False,
            metadata={"num_bubbles": len(bubble_mean_values)},
        )
