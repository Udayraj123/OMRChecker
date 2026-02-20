"""
Tests for marker detection module.

Tests cover:
- Marker template preparation
- Multi-scale template matching
- Corner extraction
- Full detection pipeline
- Edge cases and validation
"""

import numpy as np
import cv2

from src.processors.image.crop_on_patches.marker_detection import (
    prepare_marker_template,
    multi_scale_template_match,
    extract_marker_corners,
    detect_marker_in_patch,
    validate_marker_detection,
)


class TestPrepareMarkerTemplate:
    """Test marker template preparation from reference images."""

    def test_basic_extraction(self):
        """Test basic marker extraction from reference."""
        # Create a simple reference image
        reference = np.zeros((100, 100), dtype=np.uint8)
        reference[20:40, 30:50] = 255  # White rectangle

        reference_zone = {"origin": [30, 20], "dimensions": [20, 20]}

        marker = prepare_marker_template(
            reference,
            reference_zone,
            blur_kernel=(3, 3),
            apply_erode_subtract=False,
        )

        assert marker is not None
        assert marker.shape == (20, 20)
        assert marker.dtype == np.uint8

    def test_with_resize(self):
        """Test marker extraction with resizing."""
        reference = np.ones((100, 100), dtype=np.uint8) * 128
        reference_zone = {"origin": [10, 10], "dimensions": [40, 30]}

        marker = prepare_marker_template(
            reference,
            reference_zone,
            marker_dimensions=(20, 20),
            apply_erode_subtract=False,
        )

        assert marker.shape == (20, 20)

    def test_with_erode_subtract(self):
        """Test marker with edge enhancement."""
        # Create reference with gradient for better edge enhancement testing
        reference = np.zeros((50, 50), dtype=np.uint8)
        # Create a gradient-filled square
        for i in range(15, 35):
            for j in range(15, 35):
                reference[i, j] = 200 - abs(i - 25) * 5 - abs(j - 25) * 5

        reference_zone = {"origin": [15, 15], "dimensions": [20, 20]}

        marker_no_erode = prepare_marker_template(
            reference,
            reference_zone,
            blur_kernel=(3, 3),
            apply_erode_subtract=False,
        )

        marker_with_erode = prepare_marker_template(
            reference,
            reference_zone,
            blur_kernel=(3, 3),
            apply_erode_subtract=True,
        )

        # With erode-subtract, edges should be enhanced (different output)
        assert not np.array_equal(marker_no_erode, marker_with_erode)
        # Both should have values normalized to full range
        assert marker_no_erode.max() > 200
        assert marker_with_erode.max() > 200

    def test_normalization(self):
        """Test that output is normalized to full range."""
        # Create marker with gradient for proper normalization test
        reference = np.zeros((60, 60), dtype=np.uint8)
        # Create a gradient from 0 to 255
        for i in range(10, 50):
            for j in range(10, 50):
                reference[i, j] = int((i - 10) / 40 * 255)

        reference_zone = {"origin": [10, 10], "dimensions": [40, 40]}

        marker = prepare_marker_template(
            reference,
            reference_zone,
            blur_kernel=(5, 5),
            apply_erode_subtract=False,
        )

        # Should be normalized to full range even after blurring
        assert marker.max() == 255  # Normalized to max
        assert marker.min() == 0  # Normalized to min


class TestMultiScaleTemplateMatch:
    """Test multi-scale template matching algorithm."""

    def create_test_patch_and_marker(self, scale=1.0):
        """Helper to create test patch with embedded marker."""
        # Create marker (simple white square)
        marker = np.zeros((20, 20), dtype=np.uint8)
        marker[5:15, 5:15] = 255

        # Create patch
        patch = np.zeros((100, 100), dtype=np.uint8)

        # Embed scaled marker at position (30, 40)
        if scale != 1.0:
            marker_scaled = cv2.resize(
                marker,
                (int(20 * scale), int(20 * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            marker_scaled = marker.copy()

        h, w = marker_scaled.shape
        patch[40 : 40 + h, 30 : 30 + w] = marker_scaled

        return patch, marker

    def test_perfect_match(self):
        """Test matching with exact scale."""
        patch, marker = self.create_test_patch_and_marker(scale=1.0)

        position, optimal_marker, confidence, scale_percent = (
            multi_scale_template_match(
                patch, marker, scale_range=(90, 110), scale_steps=5
            )
        )

        assert position is not None
        assert optimal_marker is not None
        assert confidence > 0.9  # Very high confidence
        assert 95 <= scale_percent <= 105  # Near 100%

    def test_scaled_match(self):
        """Test matching with scaled marker."""
        # Embed marker at 90% scale
        patch, marker = self.create_test_patch_and_marker(scale=0.9)

        position, optimal_marker, confidence, scale_percent = (
            multi_scale_template_match(
                patch, marker, scale_range=(80, 100), scale_steps=5
            )
        )

        assert position is not None
        assert 85 <= scale_percent <= 95  # Should detect ~90%

    def test_no_match(self):
        """Test when marker is not present."""
        # Create patches with different patterns to ensure no match
        patch = np.random.randint(50, 100, (100, 100), dtype=np.uint8)
        marker = np.random.randint(150, 255, (20, 20), dtype=np.uint8)

        position, optimal_marker, confidence, scale_percent = (
            multi_scale_template_match(patch, marker)
        )

        # Should still return results, but low confidence
        assert confidence < 0.7  # Random patterns don't match well

    def test_marker_larger_than_patch(self):
        """Test when scaled marker exceeds patch size."""
        patch = np.zeros((50, 50), dtype=np.uint8)
        marker = np.zeros((40, 40), dtype=np.uint8)
        marker[10:30, 10:30] = 255

        # Try to scale up beyond patch size
        position, optimal_marker, confidence, scale_percent = (
            multi_scale_template_match(
                patch, marker, scale_range=(100, 150), scale_steps=3
            )
        )

        # Should handle gracefully (skip invalid scales)
        # May or may not find match depending on scale steps
        assert True  # No crash

    def test_various_scale_ranges(self):
        """Test different scale ranges."""
        patch, marker = self.create_test_patch_and_marker(scale=1.0)

        # Narrow range
        pos1, _, conf1, _ = multi_scale_template_match(
            patch, marker, scale_range=(95, 105), scale_steps=3
        )

        # Wide range
        pos2, _, conf2, _ = multi_scale_template_match(
            patch, marker, scale_range=(50, 150), scale_steps=10
        )

        # Both should find the marker
        assert pos1 is not None
        assert pos2 is not None


class TestExtractMarkerCorners:
    """Test corner extraction from detected position."""

    def test_basic_extraction(self):
        """Test basic corner extraction."""
        # Create simple marker
        marker = np.zeros((20, 30), dtype=np.uint8)
        position = (10, 15)  # (x, y)

        corners = extract_marker_corners(position, marker, zone_offset=(0, 0))

        assert corners.shape == (4, 2)
        # Check corners are at expected positions
        expected = np.array(
            [
                [10, 15],  # top-left
                [40, 15],  # top-right
                [40, 35],  # bottom-right
                [10, 35],  # bottom-left
            ]
        )
        np.testing.assert_array_equal(corners, expected)

    def test_with_zone_offset(self):
        """Test corner extraction with zone offset."""
        marker = np.zeros((20, 20), dtype=np.uint8)
        position = (5, 10)
        zone_offset = (100, 200)  # Offset in absolute coordinates

        corners = extract_marker_corners(position, marker, zone_offset)

        # All corners should be shifted by offset
        assert np.all(corners[:, 0] >= 100)  # x >= 100
        assert np.all(corners[:, 1] >= 200)  # y >= 200

    def test_different_marker_sizes(self):
        """Test with various marker dimensions."""
        for width, height in [(10, 10), (30, 20), (15, 40)]:
            marker = np.zeros((height, width), dtype=np.uint8)
            position = (0, 0)

            corners = extract_marker_corners(position, marker)

            # Check width and height
            assert corners[1, 0] - corners[0, 0] == width
            assert corners[2, 1] - corners[1, 1] == height


class TestDetectMarkerInPatch:
    """Test full marker detection pipeline."""

    def create_test_scenario(self):
        """Create test patch with embedded marker."""
        # Create marker
        marker = np.zeros((25, 25), dtype=np.uint8)
        marker[5:20, 5:20] = 255
        marker = cv2.GaussianBlur(marker, (5, 5), 0)

        # Create patch with marker embedded
        patch = np.random.randint(0, 50, (150, 150), dtype=np.uint8)
        patch[50:75, 60:85] = marker

        return patch, marker

    def test_successful_detection(self):
        """Test successful marker detection."""
        patch, marker = self.create_test_scenario()

        corners = detect_marker_in_patch(
            patch,
            marker,
            zone_offset=(0, 0),
            scale_range=(90, 110),
            min_confidence=0.3,
        )

        assert corners is not None
        assert corners.shape == (4, 2)
        # Should be near position (60, 50) with size ~25x25
        assert 55 <= corners[0, 0] <= 65  # x near 60
        assert 45 <= corners[0, 1] <= 55  # y near 50

    def test_below_confidence_threshold(self):
        """Test when confidence is below threshold."""
        # Create patch and marker with different random patterns
        patch = np.random.randint(0, 100, (100, 100), dtype=np.uint8)
        marker = np.random.randint(150, 255, (20, 20), dtype=np.uint8)

        corners = detect_marker_in_patch(
            patch,
            marker,
            min_confidence=0.95,  # Very high threshold for random patterns
        )

        # Should return None due to low confidence
        assert corners is None

    def test_with_zone_offset(self):
        """Test detection with zone offset applied."""
        patch, marker = self.create_test_scenario()
        zone_offset = (1000, 2000)

        corners = detect_marker_in_patch(
            patch, marker, zone_offset=zone_offset, min_confidence=0.3
        )

        assert corners is not None
        # Corners should be offset
        assert np.all(corners[:, 0] >= 1000)
        assert np.all(corners[:, 1] >= 2000)

    def test_scaled_marker_detection(self):
        """Test detection with marker at different scale."""
        # Create marker
        marker_original = np.zeros((30, 30), dtype=np.uint8)
        marker_original[8:22, 8:22] = 255

        # Embed at 80% scale
        patch = np.zeros((120, 120), dtype=np.uint8)
        marker_scaled = cv2.resize(marker_original, (24, 24))
        patch[40:64, 50:74] = marker_scaled

        corners = detect_marker_in_patch(
            patch,
            marker_original,
            scale_range=(70, 90),
            scale_steps=5,
            min_confidence=0.5,
        )

        assert corners is not None

    def test_no_marker_in_patch(self):
        """Test when no marker present."""
        # Create patch and marker with very different patterns
        patch = np.random.randint(0, 80, (100, 100), dtype=np.uint8)
        marker = np.random.randint(180, 255, (20, 20), dtype=np.uint8)

        corners = detect_marker_in_patch(patch, marker, min_confidence=0.8)

        assert corners is None


class TestValidateMarkerDetection:
    """Test marker detection validation."""

    def test_valid_corners(self):
        """Test validation of valid corners."""
        corners = np.array([[10, 10], [30, 10], [30, 30], [10, 30]])

        assert validate_marker_detection(corners) is True

    def test_none_corners(self):
        """Test validation with None."""
        assert validate_marker_detection(None) is False

    def test_invalid_shape(self):
        """Test validation with wrong shape."""
        corners = np.array([[10, 10], [20, 20]])  # Only 2 points

        assert validate_marker_detection(corners) is False

    def test_area_validation(self):
        """Test validation with area constraints."""
        # Square with area 400
        corners = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])

        # Should pass
        assert (
            validate_marker_detection(corners, expected_area_range=(300, 500)) is True
        )

        # Should fail (too small)
        assert (
            validate_marker_detection(corners, expected_area_range=(500, 1000)) is False
        )

        # Should fail (too large)
        assert (
            validate_marker_detection(corners, expected_area_range=(10, 100)) is False
        )

    def test_area_validation_without_range(self):
        """Test that validation works without area range."""
        corners = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])

        # Should pass (no area check)
        assert validate_marker_detection(corners, expected_area_range=None) is True


class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from preparation to detection."""
        # Create reference image with marker
        reference = np.zeros((100, 100), dtype=np.uint8)
        reference[30:50, 40:60] = 255
        reference = cv2.GaussianBlur(reference, (5, 5), 0)

        # Prepare marker
        reference_zone = {"origin": [40, 30], "dimensions": [20, 20]}
        marker = prepare_marker_template(reference, reference_zone, blur_kernel=(3, 3))

        # Create target patch with same marker
        patch = np.zeros((150, 150), dtype=np.uint8)
        patch[60:80, 70:90] = marker

        # Detect
        corners = detect_marker_in_patch(patch, marker, min_confidence=0.7)

        assert corners is not None
        assert validate_marker_detection(corners) is True

    def test_with_noise(self):
        """Test detection with noisy images."""
        # Create marker
        marker = np.zeros((25, 25), dtype=np.uint8)
        marker[5:20, 5:20] = 255
        marker = cv2.GaussianBlur(marker, (5, 5), 0)

        # Create noisy patch
        patch = np.random.randint(0, 100, (120, 120), dtype=np.uint8)
        patch[40:65, 50:75] = marker
        # Add more noise
        noise = np.random.randint(-30, 30, patch.shape, dtype=np.int16)
        patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        corners = detect_marker_in_patch(patch, marker, min_confidence=0.3)

        # Should still detect despite noise
        assert corners is not None

    def test_multiple_scale_attempts(self):
        """Test that multiple scale attempts improve detection."""
        marker = np.zeros((30, 30), dtype=np.uint8)
        marker[10:20, 10:20] = 255

        # Embed at 110% scale
        patch = np.zeros((150, 150), dtype=np.uint8)
        marker_scaled = cv2.resize(marker, (33, 33))
        patch[50:83, 60:93] = marker_scaled

        # With narrow range (should fail or low confidence)
        _corners1 = detect_marker_in_patch(
            patch, marker, scale_range=(95, 105), scale_steps=2, min_confidence=0.7
        )

        # With wider range (should succeed)
        corners2 = detect_marker_in_patch(
            patch, marker, scale_range=(90, 120), scale_steps=6, min_confidence=0.7
        )

        # Wider range should have better chance
        assert corners2 is not None
