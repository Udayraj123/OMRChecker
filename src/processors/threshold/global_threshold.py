"""Global threshold strategy for bubble detection.

Strategy using global file-level statistics.
Based on the existing get_global_threshold logic.
Finds the largest gap in sorted bubble means across all fields.
"""

from src.processors.threshold.threshold_result import ThresholdConfig, ThresholdResult
from src.processors.threshold.threshold_strategy import ThresholdStrategy


class GlobalThresholdStrategy(ThresholdStrategy):
    """Strategy using global file-level statistics.

    Based on the existing get_global_threshold logic.
    Finds the largest gap in sorted bubble means across all fields.
    """

    def calculate_threshold(
        self, bubble_mean_values: list[float], config: ThresholdConfig
    ) -> ThresholdResult:
        """Calculate global threshold by finding largest gap."""
        if len(bubble_mean_values) < 2:
            return ThresholdResult(
                threshold_value=config.default_threshold,
                confidence=0.0,
                max_jump=0.0,
                method_used="global_default",
                fallback_used=True,
            )

        sorted_values = sorted(bubble_mean_values)

        # Find the FIRST LARGE GAP using looseness
        looseness = 1
        ls = (looseness + 1) // 2
        total_bubbles_loose = len(sorted_values) - ls

        max_jump = config.min_jump
        threshold = config.default_threshold

        for i in range(ls, total_bubbles_loose):
            jump = sorted_values[i + ls] - sorted_values[i - ls]
            if jump > max_jump:
                max_jump = jump
                threshold = sorted_values[i - ls] + jump / 2

        # Calculate confidence based on jump size
        # Higher jump = higher confidence
        confidence = min(1.0, max_jump / (config.min_jump * 3))

        return ThresholdResult(
            threshold_value=threshold,
            confidence=confidence,
            max_jump=max_jump,
            method_used="global_max_jump",
            fallback_used=max_jump < config.min_jump,
            metadata={
                "num_bubbles": len(bubble_mean_values),
                "min_value": min(bubble_mean_values),
                "max_value": max(bubble_mean_values),
            },
        )
