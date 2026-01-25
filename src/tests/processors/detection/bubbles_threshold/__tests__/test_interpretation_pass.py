"""Comprehensive tests for BubblesThresholdInterpretationPass.

This test file focuses on the interpretation pass logic
to ensure proper functionality and test coverage matching TypeScript.
"""

from unittest.mock import Mock

import pytest

from src.processors.detection.bubbles_threshold.interpretation import (
    BubblesFieldInterpretation,
)
from src.processors.detection.bubbles_threshold.interpretation_pass import (
    BubblesThresholdInterpretationPass,
)
from src.processors.detection.models.detection_results import (
    BubbleFieldDetectionResult,
    BubbleMeanValue,
)
from src.processors.repositories.detection_repository import DetectionRepository


class MockField:
    """Mock Field for testing."""

    def __init__(self, field_id: str, empty_value: str = ""):
        """Initialize mock field.

        Args:
            field_id: Field identifier
            empty_value: Empty value for the field
        """
        self.id = field_id
        self.field_label = field_id
        self.empty_value = empty_value


class MockBubblesScanBox:
    """Mock BubblesScanBox for testing."""

    def __init__(
        self, x: int, y: int, bubble_value: str = "A", width: int = 10, height: int = 10
    ):
        """Initialize mock bubble.

        Args:
            x: X coordinate
            y: Y coordinate
            bubble_value: Value of the bubble (e.g., "A", "B")
            width: Bubble width
            height: Bubble height
        """
        self.x = x
        self.y = y
        self.bubble_value = bubble_value
        self.bubble_dimensions = (width, height)
        self.label = f"bubble_{x}_{y}"

    def get_shifted_position(self):
        """Get position (mimics real method)."""
        return (self.x, self.y)


class TestBubblesThresholdInterpretationPass:
    """Test BubblesThresholdInterpretationPass class."""

    def create_mock_tuning_config(self):
        """Create a mock tuning config."""
        config = Mock()
        config.thresholding = Mock()
        config.thresholding.MIN_JUMP_STD = 5.0
        config.thresholding.GLOBAL_PAGE_THRESHOLD_STD = 10.0
        config.thresholding.GLOBAL_PAGE_THRESHOLD = 180
        config.thresholding.MIN_JUMP = 10
        config.thresholding.JUMP_DELTA = 20.0
        config.thresholding.MIN_GAP_TWO_BUBBLES = 20.0
        config.thresholding.MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK = 10.0
        config.thresholding.CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY = 15.0
        config.thresholding.GLOBAL_THRESHOLD_MARGIN = 10.0
        return config

    def test_constructor_initializes_repository(self):
        """Test repository initialization."""
        config = self.create_mock_tuning_config()
        repository = DetectionRepository()

        pass_instance = BubblesThresholdInterpretationPass(
            config, repository=repository, field_detection_type="BUBBLES_THRESHOLD"
        )

        assert pass_instance.repository == repository
        assert pass_instance.repository is not None

    def test_get_field_interpretation_returns_bubbles_field_interpretation(self):
        """Test that get_field_interpretation returns BubblesFieldInterpretation instance."""
        config = self.create_mock_tuning_config()
        repository = DetectionRepository()
        pass_instance = BubblesThresholdInterpretationPass(
            config, repository=repository, field_detection_type="BUBBLES_THRESHOLD"
        )

        field = MockField("q1", "")
        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble_mean1 = BubbleMeanValue(100.0, bubble1, (0, 0))
        bubble_mean2 = BubbleMeanValue(200.0, bubble2, (10, 0))
        detection_result = BubbleFieldDetectionResult(
            "q1", "q1", [bubble_mean1, bubble_mean2]
        )

        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = pass_instance.get_field_interpretation(
            field, file_level_detection_aggregates, file_level_aggregates
        )

        assert isinstance(interpretation, BubblesFieldInterpretation)

    def test_initialize_file_level_aggregates(self):
        """Test aggregate initialization with repository."""
        config = self.create_mock_tuning_config()
        repository = DetectionRepository()
        pass_instance = BubblesThresholdInterpretationPass(
            config, repository=repository, field_detection_type="BUBBLES_THRESHOLD"
        )

        # Initialize repository with test data
        repository.initialize_file("/test/file.jpg")
        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble_mean1 = BubbleMeanValue(100.0, bubble1, (0, 0))
        bubble_mean2 = BubbleMeanValue(200.0, bubble2, (10, 0))

        # Save to repository
        repository.save_bubble_field(
            "q1",
            BubbleFieldDetectionResult("q1", "q1", [bubble_mean1, bubble_mean2]),
        )

        # Initialize aggregates
        pass_instance.initialize_file_level_aggregates("/test/file.jpg")

        # Verify aggregates were initialized
        file_agg = pass_instance.get_file_level_aggregates()
        assert file_agg is not None
        assert "file_level_fallback_threshold" in file_agg
        assert file_agg["file_level_fallback_threshold"] is not None
        assert isinstance(file_agg["file_level_fallback_threshold"], (int, float))

    def test_get_outlier_deviation_threshold(self):
        """Test outlier deviation threshold calculation."""
        config = self.create_mock_tuning_config()
        repository = DetectionRepository()
        pass_instance = BubblesThresholdInterpretationPass(
            config, repository=repository, field_detection_type="BUBBLES_THRESHOLD"
        )

        all_outlier_deviations = [5.0, 6.0, 7.0, 8.0]
        threshold = pass_instance.get_outlier_deviation_threshold(
            all_outlier_deviations
        )

        assert isinstance(threshold, (int, float))
        assert threshold > 0

    def test_get_fallback_threshold(self):
        """Test fallback threshold calculation."""
        config = self.create_mock_tuning_config()
        repository = DetectionRepository()
        pass_instance = BubblesThresholdInterpretationPass(
            config, repository=repository, field_detection_type="BUBBLES_THRESHOLD"
        )

        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        field_wise_means_and_refs = [
            BubbleMeanValue(100.0, bubble1, (0, 0)),
            BubbleMeanValue(200.0, bubble2, (10, 0)),
        ]

        file_level_fallback_threshold, global_max_jump = (
            pass_instance.get_fallback_threshold(field_wise_means_and_refs)
        )

        assert isinstance(file_level_fallback_threshold, (int, float))
        assert file_level_fallback_threshold > 0
        assert isinstance(global_max_jump, (int, float))
        assert global_max_jump >= 0

    def test_update_field_level_aggregates_on_processed_field_interpretation(self):
        """Test field aggregate updates."""
        config = self.create_mock_tuning_config()
        repository = DetectionRepository()
        pass_instance = BubblesThresholdInterpretationPass(
            config, repository=repository, field_detection_type="BUBBLES_THRESHOLD"
        )

        # Initialize file-level aggregates first
        repository.initialize_file("/test/file.jpg")
        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble_mean1 = BubbleMeanValue(50.0, bubble1, (0, 0))
        bubble_mean2 = BubbleMeanValue(200.0, bubble2, (10, 0))
        repository.save_bubble_field(
            "q1",
            BubbleFieldDetectionResult("q1", "q1", [bubble_mean1, bubble_mean2]),
        )
        pass_instance.initialize_file_level_aggregates("/test/file.jpg")

        field = MockField("q1", "")

        # Initialize field-level aggregates
        pass_instance.initialize_field_level_aggregates(field)

        detection_result = BubbleFieldDetectionResult(
            "q1", "q1", [bubble_mean1, bubble_mean2]
        )

        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = pass_instance.get_field_interpretation(
            field, file_level_detection_aggregates, file_level_aggregates
        )

        # Update aggregates
        pass_instance.update_field_level_aggregates_on_processed_field_interpretation(
            field, interpretation
        )

        # Verify field-level aggregates were updated
        field_agg = pass_instance.get_field_level_aggregates()
        assert field_agg is not None
        assert "is_multi_marked" in field_agg
        assert "local_threshold_for_field" in field_agg
        assert "bubble_interpretations" in field_agg

    def test_update_file_level_aggregates_on_processed_field_interpretation(self):
        """Test file aggregate updates."""
        config = self.create_mock_tuning_config()
        repository = DetectionRepository()
        pass_instance = BubblesThresholdInterpretationPass(
            config, repository=repository, field_detection_type="BUBBLES_THRESHOLD"
        )

        # Initialize file-level aggregates first
        repository.initialize_file("/test/file.jpg")
        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble_mean1 = BubbleMeanValue(100.0, bubble1, (0, 0))
        bubble_mean2 = BubbleMeanValue(200.0, bubble2, (10, 0))
        repository.save_bubble_field(
            "q1",
            BubbleFieldDetectionResult("q1", "q1", [bubble_mean1, bubble_mean2]),
        )
        pass_instance.initialize_file_level_aggregates("/test/file.jpg")

        field = MockField("q1", "")
        detection_result = BubbleFieldDetectionResult(
            "q1", "q1", [bubble_mean1, bubble_mean2]
        )
        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = pass_instance.get_field_interpretation(
            field, file_level_detection_aggregates, file_level_aggregates
        )

        field_level_aggregates = {}

        # Update file-level aggregates
        pass_instance.update_file_level_aggregates_on_processed_field_interpretation(
            field, interpretation, field_level_aggregates
        )

        # Verify file-level aggregates were updated
        file_agg = pass_instance.get_file_level_aggregates()
        assert file_agg is not None
        assert "all_fields_local_thresholds" in file_agg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
