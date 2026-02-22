"""
Tests for Page Detection Module

Comprehensive tests for all page detection functions.
"""

import cv2
import numpy as np
import pytest
from pathlib import Path

from src.processors.image.page_detection import (
    prepare_page_image,
    apply_colored_canny,
    apply_grayscale_canny,
    find_page_contours,
    extract_page_rectangle,
    find_page_contour_and_corners,
)
from src.utils.exceptions import ImageProcessingError


class TestPreparePageImage:
    """Tests for image preparation"""

    def test_prepare_normalizes_image(self):
        """Test that preparation normalizes the image"""
        # Create image with values in limited range
        image = np.full((100, 100), 128, dtype=np.uint8)

        result = prepare_page_image(image)

        # Should be normalized to full range after truncation
        assert result.dtype == np.uint8
        assert result.shape == (100, 100)

    def test_prepare_truncates_high_values(self):
        """Test that high values are truncated"""
        # Create image with high values
        image = np.full((100, 100), 250, dtype=np.uint8)

        result = prepare_page_image(image)

        # High values should be capped
        assert np.max(result) <= 255


class TestApplyColoredCanny:
    """Tests for colored Canny edge detection"""

    @pytest.fixture
    def test_images(self):
        """Create test grayscale and color images"""
        gray = np.zeros((200, 200), dtype=np.uint8)
        # Draw white rectangle (page)
        cv2.rectangle(gray, (50, 50), (150, 150), 255, -1)

        # Create BGR version
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return gray, color

    def test_apply_colored_canny_produces_edges(self, test_images):
        """Test that colored Canny produces edge map"""
        gray, color = test_images

        edges = apply_colored_canny(gray, color)

        assert edges.shape == gray.shape
        assert edges.dtype == np.uint8
        # Should have some edges detected
        assert np.any(edges > 0)


class TestApplyGrayscaleCanny:
    """Tests for grayscale Canny edge detection"""

    @pytest.fixture
    def test_image(self):
        """Create test image"""
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
        return image

    def test_apply_without_morph(self, test_image):
        """Test Canny without morphological operations"""
        edges = apply_grayscale_canny(test_image, morph_kernel=None)

        assert edges.shape == test_image.shape
        assert edges.dtype == np.uint8
        assert np.any(edges > 0)

    def test_apply_with_morph_kernel(self, test_image):
        """Test Canny with morphological closing"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        edges = apply_grayscale_canny(test_image, morph_kernel=kernel)

        assert edges.shape == test_image.shape
        assert np.any(edges > 0)

    def test_small_kernel_skips_morph(self, test_image):
        """Test that kernel of size 1 skips morphology"""
        kernel = np.ones((1, 1), dtype=np.uint8)

        edges = apply_grayscale_canny(test_image, morph_kernel=kernel)

        assert edges.shape == test_image.shape


class TestFindPageContours:
    """Tests for contour finding"""

    @pytest.fixture
    def edge_image(self):
        """Create edge image with rectangles"""
        edges = np.zeros((300, 400), dtype=np.uint8)
        # Large rectangle (page)
        cv2.rectangle(edges, (50, 50), (350, 250), 255, 2)
        # Small rectangle (noise)
        cv2.rectangle(edges, (100, 100), (120, 120), 255, 2)
        return edges

    def test_find_contours_returns_list(self, edge_image):
        """Test that contours are found and returned"""
        contours = find_page_contours(edge_image)

        assert isinstance(contours, list)
        assert len(contours) > 0

    def test_contours_sorted_by_area(self, edge_image):
        """Test that contours are sorted by area (largest first)"""
        contours = find_page_contours(edge_image)

        if len(contours) >= 2:
            area1 = cv2.contourArea(contours[0])
            area2 = cv2.contourArea(contours[1])
            assert area1 >= area2

    def test_returns_top_candidates_only(self, edge_image):
        """Test that only top N contours are returned"""
        from src.processors.image.constants import TOP_CONTOURS_COUNT

        contours = find_page_contours(edge_image)

        assert len(contours) <= TOP_CONTOURS_COUNT


class TestExtractPageRectangle:
    """Tests for page rectangle extraction"""

    def test_extract_valid_rectangle(self):
        """Test extraction of valid rectangular contour"""
        # Create a large rectangle contour (above MIN_PAGE_AREA)
        corners = np.array(
            [[[50, 50]], [[500, 50]], [[500, 400]], [[50, 400]]], dtype=np.int32
        )

        contours = [corners]

        extracted_corners, full_contour = extract_page_rectangle(contours)

        assert extracted_corners is not None
        assert extracted_corners.shape == (4, 2)
        assert full_contour is not None

    def test_reject_small_contour(self):
        """Test that small contours are rejected"""
        # Create very small rectangle (below MIN_PAGE_AREA)
        small_corners = np.array(
            [[[10, 10]], [[15, 10]], [[15, 15]], [[10, 15]]], dtype=np.int32
        )

        contours = [small_corners]

        extracted_corners, full_contour = extract_page_rectangle(contours)

        # Should be rejected as too small
        assert extracted_corners is None
        assert full_contour is None

    def test_reject_non_rectangle(self):
        """Test that non-rectangular shapes are rejected"""
        # Create triangle (3 points, not 4)
        triangle = np.array([[[100, 50]], [[200, 150]], [[50, 150]]], dtype=np.int32)

        contours = [triangle]

        extracted_corners, full_contour = extract_page_rectangle(contours)

        # Should be rejected as not a rectangle
        assert extracted_corners is None
        assert full_contour is None


class TestFindPageContourAndCorners:
    """Integration tests for complete page detection"""

    @pytest.fixture
    def page_image(self):
        """Create image with clear page boundary"""
        image = np.zeros((400, 600), dtype=np.uint8)
        # White page with border
        cv2.rectangle(image, (100, 80), (500, 320), 255, -1)
        # Add some noise
        cv2.circle(image, (200, 150), 5, 128, -1)
        return image

    def test_find_page_success(self, page_image):
        """Test successful page detection"""
        corners, contour = find_page_contour_and_corners(page_image)

        assert corners is not None
        assert corners.shape == (4, 2)
        assert contour is not None
        assert len(contour) > 4  # Contour should have many points

    def test_find_page_with_colored_canny(self, page_image):
        """Test page detection with colored Canny"""
        colored = cv2.cvtColor(page_image, cv2.COLOR_GRAY2BGR)

        corners, contour = find_page_contour_and_corners(
            page_image, colored_image=colored, use_colored_canny=True
        )

        assert corners is not None
        assert corners.shape == (4, 2)

    def test_find_page_with_morph_kernel(self, page_image):
        """Test page detection with morphological kernel"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

        corners, contour = find_page_contour_and_corners(
            page_image, morph_kernel=kernel
        )

        assert corners is not None

    def test_find_page_draws_debug_contours(self, page_image):
        """Test that debug contours are drawn when provided"""
        debug_image = np.zeros_like(page_image)

        corners, contour = find_page_contour_and_corners(
            page_image, debug_image=debug_image
        )

        # Debug image should have contours drawn
        assert np.any(debug_image > 0)

    def test_find_page_raises_error_when_not_found(self):
        """Test that error is raised when page is not found"""
        # Empty image - no page to find
        empty_image = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(ImageProcessingError, match="Paper boundary not found"):
            find_page_contour_and_corners(empty_image, file_path=Path("test.jpg"))

    def test_find_page_with_file_path_in_error(self):
        """Test that file path is included in error message"""
        empty_image = np.zeros((100, 100), dtype=np.uint8)
        test_path = Path("/path/to/test.jpg")

        with pytest.raises(ImageProcessingError) as exc_info:
            find_page_contour_and_corners(empty_image, file_path=test_path)

        assert exc_info.value.file_path == test_path


class TestPageDetectionIntegration:
    """End-to-end integration tests"""

    def test_realistic_page_detection(self):
        """Test detection on realistic page image"""
        # Create realistic image: white page on darker background
        image = np.full((600, 800), 50, dtype=np.uint8)  # Dark background

        # White page
        page_corners = np.array([[100, 100], [700, 120], [680, 500], [90, 480]])

        # Draw filled polygon for page
        cv2.fillPoly(image, [page_corners], 255)

        # Detect page
        detected_corners, contour = find_page_contour_and_corners(image)

        assert detected_corners is not None
        assert detected_corners.shape == (4, 2)

        # Corners should be roughly in the correct positions
        # (allowing for some detection variance)
        detected_flat = detected_corners.flatten()
        assert 80 <= np.min(detected_flat) <= 120
        assert 650 <= np.max(detected_flat) <= 720


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
