"""
Tests for Page Detection Module

Tests the PageDetector class and related functionality for finding
page boundaries in images.
"""

import cv2
import numpy as np
import pytest

from src.processors.image.page_detection import PageDetector, detect_page_corners
from src.exceptions import ImageProcessingError


class TestPageDetector:
    """Test suite for PageDetector class"""

    @pytest.fixture
    def simple_page_image(self):
        """Create a simple test image with a clear page boundary"""
        # Create a white background
        image = np.ones((1000, 800), dtype=np.uint8) * 255

        # Draw a dark rectangular page with some margin
        page_rect = np.array(
            [[100, 100], [700, 100], [700, 900], [100, 900]], dtype=np.int32
        )

        cv2.fillPoly(image, [page_rect], color=50)

        # Add some noise/texture to the page
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)

        return image

    @pytest.fixture
    def rotated_page_image(self):
        """Create a test image with a slightly rotated page"""
        image = np.ones((1000, 800), dtype=np.uint8) * 255

        # Rotated rectangle
        page_rect = np.array(
            [[120, 100], [680, 150], [660, 900], [100, 850]], dtype=np.int32
        )

        cv2.fillPoly(image, [page_rect], color=50)
        return image

    @pytest.fixture
    def no_page_image(self):
        """Create an image with no clear page boundary"""
        # Just random noise
        return np.random.randint(100, 200, (1000, 800), dtype=np.uint8)

    def test_detector_initialization(self):
        """Test PageDetector initializes correctly"""
        detector = PageDetector(morph_kernel=(10, 10), use_colored_canny=False)

        assert detector.morph_kernel.shape == (10, 10)
        assert detector.use_colored_canny is False

    def test_detect_simple_page(self, simple_page_image):
        """Test detection of a simple rectangular page"""
        detector = PageDetector()

        corners, contour = detector.detect_page_boundary(simple_page_image)

        # Should find 4 corners
        assert corners.shape == (4, 2)

        # Contour should be non-empty
        assert len(contour) > 4

        # Corners should be roughly at expected positions (with tolerance)
        # Expected corners around [100,100], [700,100], [700,900], [100,900]
        corner_distances = []
        expected = np.array([[100, 100], [700, 100], [700, 900], [100, 900]])

        for expected_corner in expected:
            min_dist = min(
                np.linalg.norm(corner - expected_corner) for corner in corners
            )
            corner_distances.append(min_dist)

        # All corners should be within 50 pixels of expected
        assert all(dist < 50 for dist in corner_distances)

    def test_detect_rotated_page(self, rotated_page_image):
        """Test detection of a rotated page"""
        detector = PageDetector()

        corners, contour = detector.detect_page_boundary(rotated_page_image)

        # Should still find 4 corners
        assert corners.shape == (4, 2)
        assert len(contour) > 4

    def test_no_page_raises_error(self, no_page_image):
        """Test that missing page raises appropriate error"""
        detector = PageDetector()

        with pytest.raises(ImageProcessingError) as exc_info:
            detector.detect_page_boundary(no_page_image, file_path="test.jpg")

        assert "Paper boundary not found" in str(exc_info.value)
        assert "test.jpg" in str(exc_info.value.context)

    def test_preprocess_image(self, simple_page_image):
        """Test image preprocessing"""
        detector = PageDetector()

        preprocessed = detector._preprocess_image(simple_page_image)

        # Should be normalized (full range 0-255)
        assert preprocessed.min() >= 0
        assert preprocessed.max() <= 255

        # Should be same shape
        assert preprocessed.shape == simple_page_image.shape

    def test_canny_detection(self, simple_page_image):
        """Test Canny edge detection"""
        detector = PageDetector()

        preprocessed = detector._preprocess_image(simple_page_image)
        edges = detector._apply_canny_detection(preprocessed, simple_page_image)

        # Should produce binary edge map
        assert edges.dtype == np.uint8
        assert set(np.unique(edges)).issubset({0, 255})

        # Should detect edges (non-empty)
        assert np.any(edges > 0)

    def test_colored_canny(self, simple_page_image):
        """Test colored Canny with HSV masking"""
        detector = PageDetector(use_colored_canny=True)

        # Create a colored version (BGR)
        colored = cv2.cvtColor(simple_page_image, cv2.COLOR_GRAY2BGR)

        preprocessed = detector._preprocess_image(simple_page_image)
        edges = detector._apply_canny_detection(
            preprocessed, simple_page_image, colored
        )

        assert edges.dtype == np.uint8
        assert np.any(edges > 0)

    def test_morph_kernel_effect(self, simple_page_image):
        """Test that morphological kernel affects detection"""
        detector_small = PageDetector(morph_kernel=(3, 3))
        detector_large = PageDetector(morph_kernel=(15, 15))

        preprocessed = detector_small._preprocess_image(simple_page_image)

        edges_small = detector_small._apply_canny_detection(
            preprocessed, simple_page_image
        )
        edges_large = detector_large._apply_canny_detection(
            preprocessed, simple_page_image
        )

        # Different kernel sizes should produce different results
        assert not np.array_equal(edges_small, edges_large)

    def test_convenience_function(self, simple_page_image):
        """Test the convenience function works"""
        corners, contour = detect_page_corners(
            simple_page_image, morph_kernel=(10, 10), file_path="test.jpg"
        )

        assert corners.shape == (4, 2)
        assert len(contour) > 4

    def test_min_area_filtering(self):
        """Test that small contours are filtered out"""
        # Create image with tiny "page"
        image = np.ones((1000, 800), dtype=np.uint8) * 255

        # Very small rectangle (below MIN_PAGE_AREA)
        tiny_rect = np.array(
            [[400, 400], [410, 400], [410, 410], [400, 410]], dtype=np.int32
        )

        cv2.fillPoly(image, [tiny_rect], color=0)

        detector = PageDetector()

        with pytest.raises(ImageProcessingError):
            detector.detect_page_boundary(image)

    def test_multiple_contours_selects_largest(self):
        """Test that largest valid rectangle is selected"""
        image = np.ones((1000, 800), dtype=np.uint8) * 255

        # Create two rectangles, larger one should be selected
        small_rect = np.array(
            [[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.int32
        )

        large_rect = np.array(
            [[50, 50], [750, 50], [750, 950], [50, 950]], dtype=np.int32
        )

        cv2.fillPoly(image, [small_rect], color=50)
        cv2.fillPoly(image, [large_rect], color=50)

        detector = PageDetector()
        corners, _ = detector.detect_page_boundary(image)

        # Should select the larger rectangle
        # Calculate bounding box area
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]
        width = x_coords.max() - x_coords.min()
        height = y_coords.max() - y_coords.min()

        # Should be close to large rectangle dimensions
        assert width > 600  # Large rect is 700 wide
        assert height > 800  # Large rect is 900 tall


class TestPageDetectionIntegration:
    """Integration tests for page detection"""

    def test_full_pipeline_with_noise(self):
        """Test detection works with noisy images"""
        # Create page with significant noise
        image = np.ones((1000, 800), dtype=np.uint8) * 255

        page_rect = np.array(
            [[100, 100], [700, 100], [700, 900], [100, 900]], dtype=np.int32
        )

        cv2.fillPoly(image, [page_rect], color=50)

        # Add heavy noise
        noise = np.random.randint(-50, 50, image.shape, dtype=np.int8)
        image = cv2.add(image.astype(np.int16), noise.astype(np.int16))
        image = np.clip(image, 0, 255).astype(np.uint8)

        # Should still detect
        corners, contour = detect_page_corners(image)
        assert corners.shape == (4, 2)

    def test_detection_with_shadows(self):
        """Test detection works with gradient shadows"""
        image = np.ones((1000, 800), dtype=np.uint8) * 255

        # Add gradient shadow
        y_gradient = np.linspace(0, 100, 1000).reshape(-1, 1)
        x_gradient = np.linspace(0, 50, 800).reshape(1, -1)
        shadow = (y_gradient + x_gradient).astype(np.uint8)

        image = cv2.subtract(image, shadow)

        # Add page
        page_rect = np.array(
            [[100, 100], [700, 100], [700, 900], [100, 900]], dtype=np.int32
        )

        cv2.fillPoly(image, [page_rect], color=30)

        # Should still detect despite gradient
        corners, contour = detect_page_corners(image)
        assert corners.shape == (4, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
