"""
Tests for Dot and Line Detection Module

Comprehensive tests for dot/line preprocessing and detection algorithms.
"""

import cv2
import numpy as np
import pytest

from src.processors.constants import ScannerType
from src.processors.image.crop_on_patches.dot_line_detection import (
    preprocess_dot_zone,
    preprocess_line_zone,
    detect_contours_using_canny,
    extract_patch_corners_and_edges,
    detect_dot_corners,
    detect_line_corners_and_edges,
    validate_blur_kernel,
    create_structuring_element,
)


class TestPreprocessDotZone:
    """Tests for dot zone preprocessing"""

    @pytest.fixture
    def dot_kernel(self):
        """Create dot morphological kernel"""
        return cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    @pytest.fixture
    def zone_with_dot(self):
        """Create zone with a dark dot"""
        zone = np.full((100, 100), 200, dtype=np.uint8)
        cv2.circle(zone, (50, 50), 15, 50, -1)
        return zone

    def test_preprocess_basic(self, zone_with_dot, dot_kernel):
        """Test basic preprocessing"""
        result = preprocess_dot_zone(zone_with_dot, dot_kernel)

        assert result.shape == zone_with_dot.shape
        assert result.dtype == np.uint8
        # Center should be darker (dot present)
        assert result[50, 50] < result[10, 10]

    def test_preprocess_with_blur(self, zone_with_dot, dot_kernel):
        """Test preprocessing with Gaussian blur"""
        result = preprocess_dot_zone(zone_with_dot, dot_kernel, blur_kernel=(5, 5))

        assert result.shape == zone_with_dot.shape
        assert result[50, 50] < 200  # Dot should still be visible

    def test_preprocess_with_threshold(self, zone_with_dot, dot_kernel):
        """Test different threshold values"""
        result_low = preprocess_dot_zone(zone_with_dot, dot_kernel, dot_threshold=100)
        result_high = preprocess_dot_zone(zone_with_dot, dot_kernel, dot_threshold=200)

        # Both should be valid preprocessed images
        assert result_low.shape == zone_with_dot.shape
        assert result_high.shape == zone_with_dot.shape


class TestPreprocessLineZone:
    """Tests for line zone preprocessing"""

    @pytest.fixture
    def line_kernel(self):
        """Create line morphological kernel"""
        return cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))

    @pytest.fixture
    def zone_with_line(self):
        """Create zone with a dark vertical line"""
        zone = np.full((100, 100), 200, dtype=np.uint8)
        cv2.line(zone, (50, 10), (50, 90), 50, 3)
        return zone

    def test_preprocess_basic(self, zone_with_line, line_kernel):
        """Test basic preprocessing"""
        result = preprocess_line_zone(zone_with_line, line_kernel, gamma_low=0.5)

        assert result.shape == zone_with_line.shape
        assert result.dtype == np.uint8
        # Result should be valid preprocessed image
        assert result.mean() < 255

    def test_preprocess_with_gamma(self, zone_with_line, line_kernel):
        """Test gamma adjustment effect"""
        result_low_gamma = preprocess_line_zone(
            zone_with_line, line_kernel, gamma_low=0.3
        )
        result_high_gamma = preprocess_line_zone(
            zone_with_line, line_kernel, gamma_low=0.9
        )

        # Both should be valid preprocessed images
        assert result_low_gamma.shape == zone_with_line.shape
        assert result_high_gamma.shape == zone_with_line.shape

    def test_preprocess_with_blur(self, zone_with_line, line_kernel):
        """Test preprocessing with Gaussian blur"""
        result = preprocess_line_zone(
            zone_with_line, line_kernel, gamma_low=0.5, blur_kernel=(5, 5)
        )

        assert result.shape == zone_with_line.shape


class TestDetectContoursUsingCanny:
    """Tests for Canny-based contour detection"""

    @pytest.fixture
    def zone_with_shape(self):
        """Create zone with a clear rectangle"""
        zone = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(zone, (20, 20), (80, 80), 255, -1)
        return zone

    def test_detect_contours_success(self, zone_with_shape):
        """Test successful contour detection"""
        contours = detect_contours_using_canny(zone_with_shape)

        assert len(contours) > 0
        # Largest contour should be the rectangle
        largest = contours[0]
        assert cv2.contourArea(largest) > 1000

    def test_detect_contours_empty_zone(self):
        """Test detection on empty zone"""
        empty_zone = np.zeros((100, 100), dtype=np.uint8)

        contours = detect_contours_using_canny(empty_zone)

        assert len(contours) == 0

    def test_detect_contours_sorted_by_area(self):
        """Test that contours are sorted by area"""
        zone = np.zeros((100, 100), dtype=np.uint8)
        # Draw two rectangles of different sizes
        cv2.rectangle(zone, (10, 10), (30, 30), 255, -1)  # Small
        cv2.rectangle(zone, (40, 40), (90, 90), 255, -1)  # Large

        contours = detect_contours_using_canny(zone)

        assert len(contours) >= 2
        # First should be larger than second
        assert cv2.contourArea(contours[0]) > cv2.contourArea(contours[1])


class TestExtractPatchCornersAndEdges:
    """Tests for corner and edge extraction"""

    @pytest.fixture
    def rectangle_contour(self):
        """Create a rectangular contour"""
        zone = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(zone, (20, 20), (80, 80), 255, -1)
        contours = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cv2.findContours(zone, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        from src.utils.image import ImageUtils

        return ImageUtils.grab_contours(contours)[0]

    def test_extract_dot_corners(self, rectangle_contour):
        """Test corner extraction for dot scanner"""
        corners, edges = extract_patch_corners_and_edges(
            rectangle_contour, ScannerType.PATCH_DOT
        )

        assert corners is not None
        assert len(corners) == 4
        assert edges is not None
        assert len(edges) == 4

    def test_extract_line_corners(self, rectangle_contour):
        """Test corner extraction for line scanner"""
        corners, edges = extract_patch_corners_and_edges(
            rectangle_contour, ScannerType.PATCH_LINE
        )

        assert corners is not None
        assert len(corners) == 4
        assert edges is not None
        assert len(edges) == 4

    def test_unsupported_scanner_type(self, rectangle_contour):
        """Test that unsupported scanner type raises error"""
        with pytest.raises(ValueError, match="Unsupported scanner type"):
            extract_patch_corners_and_edges(rectangle_contour, "INVALID_TYPE")


class TestDetectDotCorners:
    """Integration tests for dot detection"""

    @pytest.fixture
    def dot_kernel(self):
        """Create dot kernel"""
        return cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    @pytest.fixture
    def zone_with_dot(self):
        """Create zone with clear dot"""
        zone = np.full((100, 100), 200, dtype=np.uint8)
        cv2.rectangle(zone, (40, 40), (60, 60), 50, -1)
        return zone

    def test_detect_dot_success(self, zone_with_dot, dot_kernel):
        """Test successful dot detection"""
        corners = detect_dot_corners(
            zone_with_dot, zone_offset=(0, 0), dot_kernel=dot_kernel, dot_threshold=150
        )

        assert corners is not None
        assert corners.shape == (4, 2)
        # Corners should be roughly around the dot position
        assert 30 <= corners[0][0] <= 70
        assert 30 <= corners[0][1] <= 70

    def test_detect_dot_with_offset(self, zone_with_dot, dot_kernel):
        """Test detection with zone offset"""
        corners = detect_dot_corners(
            zone_with_dot, zone_offset=(100, 100), dot_kernel=dot_kernel
        )

        assert corners is not None
        # All corners should be offset
        assert np.all(corners >= 100)

    def test_detect_dot_returns_none_when_not_found(self, dot_kernel):
        """Test that None is returned when dot not found"""
        empty_zone = np.full((100, 100), 200, dtype=np.uint8)

        corners = detect_dot_corners(
            empty_zone, zone_offset=(0, 0), dot_kernel=dot_kernel
        )

        assert corners is None


class TestDetectLineCornersAndEdges:
    """Integration tests for line detection"""

    @pytest.fixture
    def line_kernel(self):
        """Create line kernel"""
        return cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))

    @pytest.fixture
    def zone_with_line(self):
        """Create zone with clear vertical line"""
        zone = np.full((100, 100), 200, dtype=np.uint8)
        cv2.rectangle(zone, (40, 10), (60, 90), 50, -1)
        return zone

    def test_detect_line_success(self, zone_with_line, line_kernel):
        """Test successful line detection"""
        corners, edges = detect_line_corners_and_edges(
            zone_with_line,
            zone_offset=(0, 0),
            line_kernel=line_kernel,
            gamma_low=0.5,
            line_threshold=180,
        )

        assert corners is not None
        assert corners.shape == (4, 2)
        assert edges is not None
        assert len(edges) == 4

    def test_detect_line_with_offset(self, zone_with_line, line_kernel):
        """Test detection with zone offset"""
        corners, edges = detect_line_corners_and_edges(
            zone_with_line,
            zone_offset=(100, 100),
            line_kernel=line_kernel,
            gamma_low=0.5,
        )

        assert corners is not None
        # All corners should be offset
        assert np.all(corners[:, 0] >= 100)  # X coordinates
        assert np.all(corners[:, 1] >= 100)  # Y coordinates

    def test_detect_line_returns_none_when_not_found(self, line_kernel):
        """Test that None is returned when line not found"""
        # Use completely uniform image (no edges)
        empty_zone = np.full((100, 100), 200, dtype=np.uint8)

        corners, edges = detect_line_corners_and_edges(
            empty_zone,
            zone_offset=(0, 0),
            line_kernel=line_kernel,
            gamma_low=0.5,
            line_threshold=50,  # Very low threshold to avoid finding noise
        )

        # Should return None since no clear line edges
        assert corners is None
        assert edges is None


class TestValidateBlurKernel:
    """Tests for blur kernel validation"""

    def test_valid_kernel(self):
        """Test validation of valid kernel"""
        assert validate_blur_kernel((100, 100), (5, 5))

    def test_kernel_too_large(self):
        """Test rejection of too-large kernel"""
        with pytest.raises(ValueError, match="smaller than provided blur kernel"):
            validate_blur_kernel((10, 10), (15, 15))

    def test_kernel_equal_size(self):
        """Test rejection of equal-size kernel"""
        with pytest.raises(ValueError, match="smaller than provided blur kernel"):
            validate_blur_kernel((10, 10), (10, 10))

    def test_validation_with_label(self):
        """Test validation with zone label"""
        with pytest.raises(ValueError, match="zone 'test_zone'"):
            validate_blur_kernel((10, 10), (15, 15), zone_label="test_zone")


class TestCreateStructuringElement:
    """Tests for structuring element creation"""

    def test_create_rect(self):
        """Test rectangular element creation"""
        element = create_structuring_element("rect", (5, 5))

        assert element.shape == (5, 5)
        assert element.dtype == np.uint8

    def test_create_ellipse(self):
        """Test ellipse element creation"""
        element = create_structuring_element("ellipse", (7, 7))

        assert element.shape == (7, 7)

    def test_create_cross(self):
        """Test cross element creation"""
        element = create_structuring_element("cross", (5, 5))

        assert element.shape == (5, 5)

    def test_invalid_shape(self):
        """Test that invalid shape raises error"""
        with pytest.raises(ValueError, match="Unknown shape"):
            create_structuring_element("invalid", (5, 5))


class TestDotLineDetectionIntegration:
    """End-to-end integration tests"""

    def test_realistic_dot_detection(self):
        """Test detection on realistic dot scenario"""
        # Create realistic page image
        image = np.full((400, 600), 220, dtype=np.uint8)

        # Add noise
        noise = np.random.randint(-10, 10, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add dot marker (filled square with border)
        cv2.rectangle(image, (50, 50), (90, 90), 50, -1)
        cv2.rectangle(image, (55, 55), (85, 85), 70, 2)

        # Detect dot
        dot_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        corners = detect_dot_corners(
            image, zone_offset=(0, 0), dot_kernel=dot_kernel, dot_threshold=150
        )

        assert corners is not None
        assert corners.shape == (4, 2)
        # Corners should be in the correct area
        assert 30 <= corners[0][0] <= 110
        assert 30 <= corners[0][1] <= 110

    def test_realistic_line_detection(self):
        """Test detection on realistic line scenario"""
        # Create realistic page image
        image = np.full((400, 600), 220, dtype=np.uint8)

        # Add noise
        noise = np.random.randint(-10, 10, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add vertical line
        cv2.rectangle(image, (50, 50), (60, 350), 60, -1)

        # Detect line
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
        corners, edges = detect_line_corners_and_edges(
            image,
            zone_offset=(0, 0),
            line_kernel=line_kernel,
            gamma_low=0.5,
            line_threshold=180,
        )

        assert corners is not None
        assert corners.shape == (4, 2)
        assert edges is not None
        assert len(edges) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
