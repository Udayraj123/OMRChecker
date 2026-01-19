"""Adaptive threshold strategy for bubble detection.

Adaptive strategy that combines multiple strategies.
Uses weighted average based on confidence scores.
"""

from src.processors.threshold.global_threshold import GlobalThresholdStrategy
from src.processors.threshold.local_threshold import LocalThresholdStrategy
from src.processors.threshold.threshold_result import ThresholdConfig, ThresholdResult
from src.processors.threshold.threshold_strategy import ThresholdStrategy


class AdaptiveThresholdStrategy(ThresholdStrategy):
    """Adaptive strategy that combines multiple strategies.

    Uses weighted average based on confidence scores.
    """

    def __init__(
        self, strategies: list[ThresholdStrategy], weights: list[float] | None = None
    ) -> None:
        """Initialize with strategies and optional weights.

        Args:
            strategies: List of threshold strategies to combine
            weights: Optional weights for each strategy (default: equal weights)
        """
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)

        if len(self.strategies) != len(self.weights):
            msg = "Number of strategies must match number of weights"
            raise ValueError(msg)

    def calculate_threshold(
        self, bubble_mean_values: list[float], config: ThresholdConfig
    ) -> ThresholdResult:
        """Calculate threshold using weighted average of strategies."""
        # Get results from all strategies
        results = [
            strategy.calculate_threshold(bubble_mean_values, config)
            for strategy in self.strategies
        ]

        # Calculate weighted average based on confidence
        total_weight = sum(
            result.confidence * weight
            for result, weight in zip(results, self.weights, strict=False)
        )

        if total_weight == 0:
            # All strategies have zero confidence
            return ThresholdResult(
                threshold_value=config.default_threshold,
                confidence=0.0,
                max_jump=0.0,
                method_used="adaptive_all_zero_confidence",
                fallback_used=True,
            )

        weighted_threshold = (
            sum(
                result.threshold_value * result.confidence * weight
                for result, weight in zip(results, self.weights, strict=False)
            )
            / total_weight
        )

        # Use max confidence and max jump from all strategies
        max_confidence = max(result.confidence for result in results)
        max_jump_value = max(result.max_jump for result in results)

        return ThresholdResult(
            threshold_value=weighted_threshold,
            confidence=max_confidence,
            max_jump=max_jump_value,
            method_used="adaptive_weighted",
            fallback_used=any(result.fallback_used for result in results),
            metadata={
                "strategy_results": [
                    {
                        "method": result.method_used,
                        "threshold": result.threshold_value,
                        "confidence": result.confidence,
                        "weight": weight,
                    }
                    for result, weight in zip(results, self.weights, strict=False)
                ]
            },
        )


def create_default_threshold_calculator(
    global_threshold: float | None = None,
) -> ThresholdStrategy:
    """Factory function to create default threshold calculator.

    Args:
        global_threshold: Optional global threshold for local strategy fallback

    Returns:
        AdaptiveThresholdStrategy combining global and local strategies
    """
    return AdaptiveThresholdStrategy(
        strategies=[
            GlobalThresholdStrategy(),
            LocalThresholdStrategy(global_fallback=global_threshold),
        ],
        weights=[0.4, 0.6],  # Prefer local threshold
    )
