"""Comprehensive tests for bubbles_threshold interpretation.

This test file focuses on the bubble field interpretation logic
to ensure proper functionality and test coverage.
"""

from unittest.mock import Mock

import pytest

from src.processors.detection.bubbles_threshold.interpretation import (
    BubbleInterpretation,
    BubblesFieldInterpretation,
)
from src.processors.detection.models.detection_results import (
    BubbleFieldDetectionResult,
    BubbleMeanValue,
)


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


class TestBubbleInterpretation:
    """Test BubbleInterpretation class."""

    def test_constructor_with_marked_bubble(self):
        """Test constructor with mean value below threshold."""
        # Create mock bubble with low mean value (marked)
        bubble = MockBubblesScanBox(0, 0, bubble_value="A")
        bubble_mean = BubbleMeanValue(50.0, bubble, (0, 0))
        threshold = 100.0

        interpretation = BubbleInterpretation(bubble_mean, threshold)

        assert interpretation.bubble_mean == bubble_mean
        assert interpretation.threshold == threshold
        assert interpretation.mean_value == 50.0
        assert interpretation.is_attempted is True
        assert interpretation.bubble_value == "A"
        assert interpretation.item_reference == bubble

    def test_constructor_with_unmarked_bubble(self):
        """Test constructor with mean value above threshold."""
        # Create mock bubble with high mean value (unmarked)
        bubble = MockBubblesScanBox(0, 0, bubble_value="B")
        bubble_mean = BubbleMeanValue(200.0, bubble, (0, 0))
        threshold = 100.0

        interpretation = BubbleInterpretation(bubble_mean, threshold)

        assert interpretation.mean_value == 200.0
        assert interpretation.is_attempted is False
        assert interpretation.bubble_value == "B"
        assert interpretation.item_reference == bubble

    def test_constructor_extracts_bubble_value(self):
        """Test that bubble_value is extracted from unit_bubble."""
        bubble = MockBubblesScanBox(0, 0, bubble_value="C")
        bubble_mean = BubbleMeanValue(80.0, bubble, (0, 0))
        threshold = 100.0

        interpretation = BubbleInterpretation(bubble_mean, threshold)

        assert interpretation.bubble_value == "C"

    def test_constructor_without_bubble_value(self):
        """Test constructor when bubble doesn't have bubble_value attribute."""
        bubble = Mock()
        bubble.bubble_value = None
        bubble_mean = BubbleMeanValue(80.0, bubble, (0, 0))
        threshold = 100.0

        interpretation = BubbleInterpretation(bubble_mean, threshold)

        # Should handle missing bubble_value gracefully
        assert interpretation.bubble_value == ""

    def test_get_value_marked(self):
        """Test getValue() returns bubble_value when marked."""
        bubble = MockBubblesScanBox(0, 0, bubble_value="A")
        bubble_mean = BubbleMeanValue(50.0, bubble, (0, 0))
        threshold = 100.0

        interpretation = BubbleInterpretation(bubble_mean, threshold)

        assert interpretation.get_value() == "A"

    def test_get_value_unmarked(self):
        """Test getValue() returns empty string when unmarked."""
        bubble = MockBubblesScanBox(0, 0, bubble_value="B")
        bubble_mean = BubbleMeanValue(200.0, bubble, (0, 0))
        threshold = 100.0

        interpretation = BubbleInterpretation(bubble_mean, threshold)

        assert interpretation.get_value() == ""

    def test_item_reference_set(self):
        """Test that item_reference is set correctly."""
        bubble = MockBubblesScanBox(0, 0, bubble_value="A")
        bubble_mean = BubbleMeanValue(50.0, bubble, (0, 0))
        threshold = 100.0

        interpretation = BubbleInterpretation(bubble_mean, threshold)

        assert interpretation.item_reference == bubble
        assert interpretation.item_reference.bubble_value == "A"


class TestBubblesFieldInterpretation:
    """Test BubblesFieldInterpretation class."""

    def create_mock_tuning_config(self, show_confidence_metrics: bool = False):
        """Create a mock tuning config."""
        config = Mock()
        config.thresholding = Mock()
        config.thresholding.MIN_JUMP = 10
        config.thresholding.JUMP_DELTA = 5
        config.thresholding.MIN_GAP_TWO_BUBBLES = 20
        config.thresholding.MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK = 15
        config.thresholding.CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY = 10
        config.thresholding.GLOBAL_THRESHOLD_MARGIN = 5
        config.thresholding.GLOBAL_PAGE_THRESHOLD = 180
        config.outputs = Mock()
        config.outputs.show_confidence_metrics = show_confidence_metrics
        return config

    def create_mock_field(self, field_id: str = "q1", empty_value: str = ""):
        """Create a mock field."""
        field = MockField(field_id, empty_value)
        return field

    def create_mock_detection_result(
        self, field_id: str, bubble_means: list[BubbleMeanValue]
    ):
        """Create a mock detection result."""
        return BubbleFieldDetectionResult(field_id, field_id, bubble_means)

    def test_get_field_interpretation_string_no_bubbles_marked(self):
        """Test get_field_interpretation_string returns empty value when no bubbles marked."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config()

        # Create mock detection data with unmarked bubbles
        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble_mean1 = BubbleMeanValue(200.0, bubble1, (0, 0))  # Unmarked (high value)
        bubble_mean2 = BubbleMeanValue(210.0, bubble2, (10, 0))  # Unmarked (high value)

        detection_result = self.create_mock_detection_result(
            "q1", [bubble_mean1, bubble_mean2]
        )
        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = BubblesFieldInterpretation(
            config,
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

        result = interpretation.get_field_interpretation_string()
        assert result == ""

    def test_get_field_interpretation_string_single_bubble_marked(self):
        """Test get_field_interpretation_string returns bubble value when one bubble marked."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config()

        # Create mock detection data with one marked bubble
        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble_mean1 = BubbleMeanValue(50.0, bubble1, (0, 0))  # Marked (low value)
        bubble_mean2 = BubbleMeanValue(200.0, bubble2, (10, 0))  # Unmarked (high value)

        detection_result = self.create_mock_detection_result(
            "q1", [bubble_mean1, bubble_mean2]
        )
        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = BubblesFieldInterpretation(
            config,
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

        result = interpretation.get_field_interpretation_string()
        assert result == "A"

    def test_get_field_interpretation_string_multiple_bubbles_marked(self):
        """Test get_field_interpretation_string returns concatenated values when multiple marked."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config()

        # Create mock detection data with multiple marked bubbles
        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble3 = MockBubblesScanBox(20, 0, "C")
        bubble_mean1 = BubbleMeanValue(50.0, bubble1, (0, 0))  # Marked
        bubble_mean2 = BubbleMeanValue(60.0, bubble2, (10, 0))  # Marked
        bubble_mean3 = BubbleMeanValue(200.0, bubble3, (20, 0))  # Unmarked

        detection_result = self.create_mock_detection_result(
            "q1", [bubble_mean1, bubble_mean2, bubble_mean3]
        )
        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = BubblesFieldInterpretation(
            config,
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

        result = interpretation.get_field_interpretation_string()
        assert result == "AB"

    def test_get_field_interpretation_string_all_bubbles_marked(self):
        """Test get_field_interpretation_string returns empty when all bubbles marked (edge case)."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config()

        # Create mock detection data with all bubbles marked
        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble_mean1 = BubbleMeanValue(50.0, bubble1, (0, 0))  # Marked
        bubble_mean2 = BubbleMeanValue(60.0, bubble2, (10, 0))  # Marked

        detection_result = self.create_mock_detection_result(
            "q1", [bubble_mean1, bubble_mean2]
        )
        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = BubblesFieldInterpretation(
            config,
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

        result = interpretation.get_field_interpretation_string()
        # All bubbles marked should return empty (likely scanning issue)
        assert result == ""

    def test_run_interpretation_basic(self):
        """Test full interpretation flow with mock detection result."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config()

        # Create detection result
        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble_mean1 = BubbleMeanValue(50.0, bubble1, (0, 0))
        bubble_mean2 = BubbleMeanValue(200.0, bubble2, (10, 0))
        detection_result = self.create_mock_detection_result(
            "q1", [bubble_mean1, bubble_mean2]
        )

        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = BubblesFieldInterpretation(
            config,
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

        # Verify interpretation ran
        assert len(interpretation.bubble_interpretations) == 2
        assert interpretation.local_threshold_for_field > 0
        assert interpretation.threshold_result is not None

    def test_run_interpretation_extracts_detection_result(self):
        """Test that detection result is extracted correctly."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config()

        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble_mean1 = BubbleMeanValue(50.0, bubble1, (0, 0))
        detection_result = self.create_mock_detection_result("q1", [bubble_mean1])

        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = BubblesFieldInterpretation(
            config,
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

        # Verify detection result was extracted
        assert interpretation.bubble_interpretations[0].bubble_mean == bubble_mean1

    def test_run_interpretation_calculates_threshold(self):
        """Test that threshold is calculated correctly."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config()

        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble_mean1 = BubbleMeanValue(50.0, bubble1, (0, 0))
        bubble_mean2 = BubbleMeanValue(200.0, bubble2, (10, 0))
        detection_result = self.create_mock_detection_result(
            "q1", [bubble_mean1, bubble_mean2]
        )

        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = BubblesFieldInterpretation(
            config,
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

        # Verify threshold was calculated
        assert interpretation.local_threshold_for_field > 0
        assert interpretation.threshold_result is not None
        assert interpretation.threshold_result.threshold_value > 0

    def test_run_interpretation_creates_bubble_interpretations(self):
        """Test that bubble interpretations are created correctly."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config()

        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble_mean1 = BubbleMeanValue(50.0, bubble1, (0, 0))
        bubble_mean2 = BubbleMeanValue(200.0, bubble2, (10, 0))
        detection_result = self.create_mock_detection_result(
            "q1", [bubble_mean1, bubble_mean2]
        )

        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = BubblesFieldInterpretation(
            config,
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

        # Verify bubble interpretations were created
        assert len(interpretation.bubble_interpretations) == 2
        assert all(
            isinstance(interp, BubbleInterpretation)
            for interp in interpretation.bubble_interpretations
        )

    def test_check_multi_marking_single(self):
        """Test is_multi_marked = False for single mark."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config()

        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble_mean1 = BubbleMeanValue(50.0, bubble1, (0, 0))  # Marked
        bubble_mean2 = BubbleMeanValue(200.0, bubble2, (10, 0))  # Unmarked
        detection_result = self.create_mock_detection_result(
            "q1", [bubble_mean1, bubble_mean2]
        )

        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = BubblesFieldInterpretation(
            config,
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

        assert interpretation.is_multi_marked is False

    def test_check_multi_marking_multiple(self):
        """Test is_multi_marked = True for multiple marks."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config()

        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble3 = MockBubblesScanBox(20, 0, "C")
        bubble_mean1 = BubbleMeanValue(50.0, bubble1, (0, 0))  # Marked
        bubble_mean2 = BubbleMeanValue(60.0, bubble2, (10, 0))  # Marked
        bubble_mean3 = BubbleMeanValue(200.0, bubble3, (20, 0))  # Unmarked
        detection_result = self.create_mock_detection_result(
            "q1", [bubble_mean1, bubble_mean2, bubble_mean3]
        )

        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = BubblesFieldInterpretation(
            config,
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

        assert interpretation.is_multi_marked is True

    def test_calculate_confidence_metrics_enabled(self):
        """Test confidence metrics calculation when enabled."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config(show_confidence_metrics=True)

        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble_mean1 = BubbleMeanValue(50.0, bubble1, (0, 0))
        bubble_mean2 = BubbleMeanValue(200.0, bubble2, (10, 0))
        detection_result = self.create_mock_detection_result(
            "q1", [bubble_mean1, bubble_mean2]
        )

        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = BubblesFieldInterpretation(
            config,
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

        # Verify confidence metrics were calculated
        metrics = interpretation.get_field_level_confidence_metrics()
        assert "overall_confidence_score" in metrics
        assert "local_threshold" in metrics
        assert "global_threshold" in metrics

    def test_calculate_confidence_metrics_disabled(self):
        """Test confidence metrics are not calculated when disabled."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config(show_confidence_metrics=False)

        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble_mean1 = BubbleMeanValue(50.0, bubble1, (0, 0))
        bubble_mean2 = BubbleMeanValue(200.0, bubble2, (10, 0))
        detection_result = self.create_mock_detection_result(
            "q1", [bubble_mean1, bubble_mean2]
        )

        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = BubblesFieldInterpretation(
            config,
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

        # Confidence metrics should not be calculated
        metrics = interpretation.get_field_level_confidence_metrics()
        # Should be empty or minimal when disabled
        assert (
            "overall_confidence_score" not in metrics
            or metrics.get("overall_confidence_score") is None
        )

    def test_calculate_overall_confidence_score(self):
        """Test confidence score calculation logic."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config(show_confidence_metrics=True)

        # Create detection result with good separation (high confidence scenario)
        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble2 = MockBubblesScanBox(10, 0, "B")
        bubble_mean1 = BubbleMeanValue(50.0, bubble1, (0, 0))  # Clearly marked
        bubble_mean2 = BubbleMeanValue(200.0, bubble2, (10, 0))  # Clearly unmarked
        detection_result = self.create_mock_detection_result(
            "q1", [bubble_mean1, bubble_mean2]
        )

        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = BubblesFieldInterpretation(
            config,
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

        # Verify confidence score is calculated and in valid range
        metrics = interpretation.get_field_level_confidence_metrics()
        confidence_score = metrics.get("overall_confidence_score")
        assert confidence_score is not None
        assert 0.0 <= confidence_score <= 1.0

    def test_extract_detection_result_from_bubble_fields(self):
        """Test extraction from new typed format (bubble_fields)."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config()

        bubble1 = MockBubblesScanBox(0, 0, "A")
        bubble_mean1 = BubbleMeanValue(50.0, bubble1, (0, 0))
        detection_result = self.create_mock_detection_result("q1", [bubble_mean1])

        file_level_detection_aggregates = {"bubble_fields": {"q1": detection_result}}
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        interpretation = BubblesFieldInterpretation(
            config,
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

        # Verify detection result was extracted from bubble_fields
        assert len(interpretation.bubble_interpretations) == 1
        assert interpretation.bubble_interpretations[0].bubble_mean == bubble_mean1

    def test_extract_detection_result_missing_field(self):
        """Test error raised when field is missing from bubble_fields."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config()

        file_level_detection_aggregates = {"bubble_fields": {}}  # Missing q1
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        # Should raise ValueError with informative message
        with pytest.raises(ValueError, match="No detection result for field 'q1'"):
            BubblesFieldInterpretation(
                config,
                field,
                file_level_detection_aggregates,
                file_level_interpretation_aggregates,
            )

    def test_extract_detection_result_missing_bubble_fields(self):
        """Test error raised when bubble_fields key is missing."""
        field = self.create_mock_field("q1", "")
        config = self.create_mock_tuning_config()

        file_level_detection_aggregates = {}  # Missing bubble_fields key
        file_level_interpretation_aggregates = {"file_level_fallback_threshold": 180}

        # Should raise ValueError with informative message
        with pytest.raises(ValueError, match="No detection result for field 'q1'"):
            BubblesFieldInterpretation(
                config,
                field,
                file_level_detection_aggregates,
                file_level_interpretation_aggregates,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
