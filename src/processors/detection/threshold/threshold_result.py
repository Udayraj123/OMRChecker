"""Threshold result and configuration dataclasses.

Shared types for threshold calculation strategies.
"""

from dataclasses import dataclass


@dataclass
class ThresholdResult:
    """Result from threshold calculation."""

    threshold_value: float
    confidence: float  # 0.0 to 1.0
    max_jump: float
    method_used: str
    fallback_used: bool = False
    metadata: dict = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ThresholdConfig:
    """Configuration for threshold calculation."""

    min_jump: float = 30.0
    """Minimum jump to consider significant."""

    jump_delta: float = 20.0
    """Delta between jumps for two-jump detection."""

    min_gap_two_bubbles: float = 20.0
    """Minimum gap required when only two bubbles present."""

    min_jump_surplus_for_global_fallback: float = 10.0
    """Extra jump required to avoid global fallback."""

    confident_jump_surplus_for_disparity: float = 15.0
    """Extra jump for high confidence despite disparity."""

    global_threshold_margin: float = 10.0
    """Safety margin for global threshold."""

    outlier_deviation_threshold: float = 5.0
    """Std deviation threshold for outlier detection."""

    default_threshold: float = 127.5
    """Default fallback threshold."""
