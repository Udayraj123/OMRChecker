"""Comprehensive tests for bubbles_threshold detection.

This test file specifically focuses on the bubble field detection logic
to ensure proper porting to TypeScript.
"""

import numpy as np
import pytest

from src.processors.detection.bubbles_threshold.detection import (
    BubblesFieldDetection,
)
from src.processors.detection.models.detection_results import (
    BubbleFieldDetectionResult,
    BubbleMeanValue,
    ScanQuality,
)


class MockBubblesScanBox:
    """Mock BubblesScanBox for testing."""

    def __init__(self, x: int, y: int, width: int = 10, height: int = 10):
        """Initialize mock bubble.

        Args:
            x: X coordinate
            y: Y coordinate
            width: Bubble width
            height: Bubble height
        """
        self.x = x
        self.y = y
        self.bubble_dimensions = (width, height)
        self.label = f"bubble_{x}_{y}"

    def get_shifted_position(self):
        """Get position (mimics real method)."""
        return (self.x, self.y)


class MockField:
    """Mock Field for testing."""

    def __init__(self, field_id: str, scan_boxes: list):
        """Initialize mock field.

        Args:
            field_id: Field identifier
            scan_boxes: List of scan boxes
        """
        self.id = field_id
        self.field_label = field_id
        self.scan_boxes = scan_boxes


class TestBubblesFieldDetection:
    """Test BubblesFieldDetection class."""

    def create_test_image(self, bubble_values: list[float]) -> np.ndarray:
        """Create a test grayscale image with specific bubble values.

        Args:
            bubble_values: List of mean values for each bubble region

        Returns:
            Grayscale image as numpy array
        """
        # Create a simple image with each bubble as a 10x10 square
        image = np.ones((100, 100), dtype=np.uint8) * 255  # White background

        for i, value in enumerate(bubble_values):
            x = (i * 20) + 5
            y = 10
            image[y : y + 10, x : x + 10] = int(value)

        return image

    def test_basic_detection(self):
        """Test basic bubble detection."""
        # Create field with 4 bubbles
        scan_boxes = [
            MockBubblesScanBox(5, 10),
            MockBubblesScanBox(25, 10),
            MockBubblesScanBox(45, 10),
            MockBubblesScanBox(65, 10),
        ]
        field = MockField("q1", scan_boxes)

        # Create image with different intensities
        gray_image = self.create_test_image([100, 150, 200, 250])

        # Run detection
        detection = BubblesFieldDetection(field, gray_image, None)

        # Check result
        assert detection.result is not None
        assert isinstance(detection.result, BubbleFieldDetectionResult)
        assert detection.result.field_id == "q1"
        assert len(detection.result.bubble_means) == 4

    def test_bubble_mean_values(self):
        """Test that bubble mean values are calculated correctly."""
        scan_boxes = [
            MockBubblesScanBox(5, 10),
            MockBubblesScanBox(25, 10),
            MockBubblesScanBox(45, 10),
        ]
        field = MockField("q1", scan_boxes)

        # Create image with known values
        gray_image = self.create_test_image([100, 150, 200])

        # Run detection
        detection = BubblesFieldDetection(field, gray_image, None)

        # Check mean values
        mean_values = detection.result.mean_values
        assert len(mean_values) == 3
        # Values should be close to expected (allow for edge effects)
        assert abs(mean_values[0] - 100) < 10
        assert abs(mean_values[1] - 150) < 10
        assert abs(mean_values[2] - 200) < 10

    def test_scan_quality_excellent(self):
        """Test scan quality assessment - excellent case."""
        scan_boxes = [
            MockBubblesScanBox(5, 10),
            MockBubblesScanBox(25, 10),
            MockBubblesScanBox(45, 10),
            MockBubblesScanBox(65, 10),
        ]
        field = MockField("q1", scan_boxes)

        # Create image with high contrast (marked vs unmarked)
        gray_image = self.create_test_image([100, 105, 200, 205])

        detection = BubblesFieldDetection(field, gray_image, None)

        # Should have excellent quality due to high std deviation
        assert detection.result.scan_quality == ScanQuality.EXCELLENT
        assert detection.result.is_reliable
        assert detection.result.std_deviation > 45

    def test_scan_quality_poor(self):
        """Test scan quality assessment - poor case."""
        scan_boxes = [
            MockBubblesScanBox(5, 10),
            MockBubblesScanBox(25, 10),
            MockBubblesScanBox(45, 10),
        ]
        field = MockField("q1", scan_boxes)

        # Create image with low contrast (all similar)
        gray_image = self.create_test_image([150, 152, 154])

        detection = BubblesFieldDetection(field, gray_image, None)

        # Should have poor quality due to low std deviation
        assert detection.result.scan_quality == ScanQuality.POOR
        assert not detection.result.is_reliable
        assert detection.result.std_deviation < 15

    def test_jumps_calculation(self):
        """Test that jumps between bubble means are calculated correctly."""
        scan_boxes = [
            MockBubblesScanBox(5, 10),
            MockBubblesScanBox(25, 10),
            MockBubblesScanBox(45, 10),
        ]
        field = MockField("q1", scan_boxes)

        # Create image with specific gaps
        gray_image = self.create_test_image([100, 120, 200])

        detection = BubblesFieldDetection(field, gray_image, None)

        jumps = detection.result.jumps
        assert len(jumps) == 2

        # First jump should be around 20 (120-100)
        assert abs(jumps[0][0] - 20) < 5
        # Second jump should be around 80 (200-120)
        assert abs(jumps[1][0] - 80) < 5

    def test_max_jump(self):
        """Test max_jump property."""
        scan_boxes = [
            MockBubblesScanBox(5, 10),
            MockBubblesScanBox(25, 10),
            MockBubblesScanBox(45, 10),
        ]
        field = MockField("q1", scan_boxes)

        # Create image with clear separation
        gray_image = self.create_test_image([100, 105, 200])

        detection = BubblesFieldDetection(field, gray_image, None)

        # Max jump should be around 95 (200-105)
        assert abs(detection.result.max_jump - 95) < 10

    def test_sorted_bubble_means(self):
        """Test that bubbles are sorted correctly by mean value."""
        scan_boxes = [
            MockBubblesScanBox(5, 10),  # Will be 200
            MockBubblesScanBox(25, 10),  # Will be 100
            MockBubblesScanBox(45, 10),  # Will be 150
        ]
        field = MockField("q1", scan_boxes)

        gray_image = self.create_test_image([200, 100, 150])

        detection = BubblesFieldDetection(field, gray_image, None)

        sorted_means = detection.result.sorted_bubble_means

        # Should be sorted: 100, 150, 200
        assert sorted_means[0].mean_value < sorted_means[1].mean_value
        assert sorted_means[1].mean_value < sorted_means[2].mean_value

    def test_result_property(self):
        """Test that result property contains bubble_means."""
        scan_boxes = [MockBubblesScanBox(5, 10), MockBubblesScanBox(25, 10)]
        field = MockField("q1", scan_boxes)

        gray_image = self.create_test_image([100, 200])

        detection = BubblesFieldDetection(field, gray_image, None)

        # Use result property instead of field_bubble_means
        assert hasattr(detection, "result")
        assert detection.result is not None
        assert len(detection.result.bubble_means) == 2
        assert isinstance(detection.result.bubble_means[0], BubbleMeanValue)

    def test_min_max_mean_values(self):
        """Test min and max mean value properties."""
        scan_boxes = [
            MockBubblesScanBox(5, 10),
            MockBubblesScanBox(25, 10),
            MockBubblesScanBox(45, 10),
        ]
        field = MockField("q1", scan_boxes)

        gray_image = self.create_test_image([100, 150, 200])

        detection = BubblesFieldDetection(field, gray_image, None)

        # Min should be around 100, max around 200
        assert abs(detection.result.min_mean - 100) < 10
        assert abs(detection.result.max_mean - 200) < 10

    def test_empty_field(self):
        """Test detection with no bubbles."""
        scan_boxes = []
        field = MockField("q1", scan_boxes)

        gray_image = self.create_test_image([])

        detection = BubblesFieldDetection(field, gray_image, None)

        assert detection.result.std_deviation == 0.0
        assert len(detection.result.bubble_means) == 0
        assert detection.result.min_mean == 0.0

    def test_single_bubble(self):
        """Test detection with single bubble."""
        scan_boxes = [MockBubblesScanBox(5, 10)]
        field = MockField("q1", scan_boxes)

        gray_image = self.create_test_image([150])

        detection = BubblesFieldDetection(field, gray_image, None)

        assert len(detection.result.bubble_means) == 1
        assert detection.result.std_deviation == 0.0
        assert len(detection.result.jumps) == 0

    def test_read_bubble_mean_value_static_method(self):
        """Test the static read_bubble_mean_value method directly."""
        bubble = MockBubblesScanBox(10, 10)

        # Create a uniform image with value 123
        gray_image = np.ones((100, 100), dtype=np.uint8) * 123

        mean_value = BubblesFieldDetection.read_bubble_mean_value(bubble, gray_image)

        assert isinstance(mean_value, BubbleMeanValue)
        assert abs(mean_value.mean_value - 123) < 1  # Should be exactly 123
        assert mean_value.position == (10, 10)
        assert mean_value.unit_bubble == bubble


class TestBubbleMeanValue:
    """Test BubbleMeanValue model."""

    def test_sorting_behavior(self):
        """Test that BubbleMeanValue can be sorted."""
        values = [
            BubbleMeanValue(200, None),
            BubbleMeanValue(50, None),
            BubbleMeanValue(150, None),
            BubbleMeanValue(100, None),
        ]

        sorted_values = sorted(values)

        assert sorted_values[0].mean_value == 50
        assert sorted_values[1].mean_value == 100
        assert sorted_values[2].mean_value == 150
        assert sorted_values[3].mean_value == 200

    def test_string_representation(self):
        """Test string representation."""
        value = BubbleMeanValue(123.456, None)

        assert "123.5" in str(value)

    def test_comparison_operators(self):
        """Test comparison between BubbleMeanValues."""
        low = BubbleMeanValue(100, None)
        high = BubbleMeanValue(200, None)

        assert low < high
        assert not high < low
        assert low < high


class TestIntegrationWithRealScenarios:
    """Integration tests with realistic OMR scenarios."""

    def test_typical_mcq_4_options(self):
        """Test typical MCQ with 4 options (A, B, C, D)."""
        # Simulate 4-option MCQ where option B is marked
        scan_boxes = [
            MockBubblesScanBox(10, 10),  # A - unmarked
            MockBubblesScanBox(30, 10),  # B - marked
            MockBubblesScanBox(50, 10),  # C - unmarked
            MockBubblesScanBox(70, 10),  # D - unmarked
        ]
        field = MockField("q1", scan_boxes)

        # Unmarked bubbles: ~230 (light), Marked bubble: ~50 (dark)
        gray_image = np.ones((100, 100), dtype=np.uint8) * 230
        gray_image[10:20, 30:40] = 50  # Mark option B

        detection = BubblesFieldDetection(field, gray_image, None)

        # Should have excellent quality due to clear difference
        assert detection.result.scan_quality in [
            ScanQuality.EXCELLENT,
            ScanQuality.GOOD,
        ]
        assert detection.result.is_reliable

        # Max jump should be significant (marked vs unmarked)
        assert detection.result.max_jump > 100

    def test_no_answer_marked(self):
        """Test when no answer is marked (all bubbles similar)."""
        scan_boxes = [
            MockBubblesScanBox(10, 10),
            MockBubblesScanBox(30, 10),
            MockBubblesScanBox(50, 10),
            MockBubblesScanBox(70, 10),
        ]
        field = MockField("q1", scan_boxes)

        # All bubbles unmarked - similar values
        gray_image = np.ones((100, 100), dtype=np.uint8) * 230

        detection = BubblesFieldDetection(field, gray_image, None)

        # Should have poor quality due to low variance
        assert detection.result.scan_quality in [
            ScanQuality.POOR,
            ScanQuality.ACCEPTABLE,
        ]
        assert detection.result.std_deviation < 30

    def test_multiple_answers_marked(self):
        """Test when multiple answers are marked (multi-mark scenario)."""
        scan_boxes = [
            MockBubblesScanBox(10, 10),
            MockBubblesScanBox(30, 10),
            MockBubblesScanBox(50, 10),
            MockBubblesScanBox(70, 10),
        ]
        field = MockField("q1", scan_boxes)

        # Mark options A and C
        gray_image = np.ones((100, 100), dtype=np.uint8) * 230
        gray_image[10:20, 10:20] = 60  # Mark A
        gray_image[10:20, 50:60] = 55  # Mark C

        detection = BubblesFieldDetection(field, gray_image, None)

        # Should still be detectable
        assert detection.result.is_reliable
        # Two darkest bubbles should be close in value
        sorted_means = detection.result.sorted_bubble_means
        assert abs(sorted_means[0].mean_value - sorted_means[1].mean_value) < 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
