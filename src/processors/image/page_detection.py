"""
Page Detection Module - Extracted from CropPage

Handles page boundary detection using edge detection and contour analysis.
This module is focused solely on finding the page boundary in an image.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from src.constants import (
    APPROX_POLY_EPSILON_FACTOR,
    CANNY_THRESHOLD_HIGH,
    CANNY_THRESHOLD_LOW,
    MIN_PAGE_AREA,
    PIXEL_VALUE_MAX,
    THRESH_PAGE_TRUNCATE_HIGH,
    THRESH_PAGE_TRUNCATE_SECONDARY,
    TOP_CONTOURS_COUNT,
)
from src.exceptions import ImageProcessingError
from src.utils.constants import hsv_white_high, hsv_white_low
from src.utils.image import ImageUtils
from src.utils.math import MathUtils
from src.utils.logger import logger


class PageDetector:
    """
    Detects page boundaries in images using edge detection.

    This class encapsulates all logic related to finding the rectangular
    boundary of a page/document in an image.
    """

    def __init__(
        self,
        morph_kernel: Tuple[int, int] = (10, 10),
        use_colored_canny: bool = False,
    ):
        """
        Initialize the page detector.

        Args:
            morph_kernel: Kernel size for morphological operations
            use_colored_canny: Whether to use HSV masking for colored images
        """
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
        self.use_colored_canny = use_colored_canny

    def detect_page_boundary(
        self,
        image: np.ndarray,
        colored_image: Optional[np.ndarray] = None,
        file_path: str = "unknown",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect the page boundary in an image.

        Args:
            image: Grayscale image
            colored_image: Optional colored version for HSV masking
            file_path: Path for error reporting

        Returns:
            Tuple of (page_corners, page_contour)
            - page_corners: 4x2 array of corner points
            - page_contour: Full contour of the page boundary

        Raises:
            ImageProcessingError: If no page boundary is found
        """
        # Step 1: Preprocess image
        preprocessed = self._preprocess_image(image)

        # Step 2: Apply Canny edge detection
        canny_edges = self._apply_canny_detection(preprocessed, image, colored_image)

        # Step 3: Find and validate contours
        page_corners, page_contour = self._find_page_contour(canny_edges, file_path)

        if page_contour is None:
            raise ImageProcessingError(
                "Paper boundary not found",
                context={"file_path": str(file_path), "image_shape": image.shape[:2]},
            )

        return page_corners, page_contour

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image before edge detection.

        Applies threshold truncation and normalization.
        """
        # Truncate high values to reduce bright spots
        _, thresholded = cv2.threshold(
            image, THRESH_PAGE_TRUNCATE_HIGH, PIXEL_VALUE_MAX, cv2.THRESH_TRUNC
        )
        normalized = ImageUtils.normalize(thresholded)

        # Secondary truncation
        _, thresholded2 = cv2.threshold(
            normalized,
            THRESH_PAGE_TRUNCATE_SECONDARY,
            PIXEL_VALUE_MAX,
            cv2.THRESH_TRUNC,
        )
        return ImageUtils.normalize(thresholded2)

    def _apply_canny_detection(
        self,
        preprocessed: np.ndarray,
        original: np.ndarray,
        colored_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply Canny edge detection with optional HSV masking.
        """
        if self.use_colored_canny and colored_image is not None:
            return self._apply_colored_canny(preprocessed, colored_image)

        # Standard grayscale Canny
        # Apply morphological closing to complete edges
        closed = preprocessed
        if self.morph_kernel.shape[0] > 1:
            closed = cv2.morphologyEx(preprocessed, cv2.MORPH_CLOSE, self.morph_kernel)

        return cv2.Canny(closed, CANNY_THRESHOLD_HIGH, CANNY_THRESHOLD_LOW)

    def _apply_colored_canny(
        self, image: np.ndarray, colored_image: np.ndarray
    ) -> np.ndarray:
        """
        Apply Canny edge detection with HSV white masking.

        This helps isolate the page from colored backgrounds.
        """
        hsv = cv2.cvtColor(colored_image, cv2.COLOR_BGR2HSV)
        # Mask to select only white-ish regions
        mask = cv2.inRange(hsv, hsv_white_low, hsv_white_high)
        mask_result = cv2.bitwise_and(image, image, mask=mask)

        return cv2.Canny(mask_result, CANNY_THRESHOLD_HIGH, CANNY_THRESHOLD_LOW)

    def _find_page_contour(
        self, canny_edges: np.ndarray, file_path: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find the page contour from Canny edges.

        Returns the largest rectangular contour that likely represents
        the page boundary.
        """
        # Find all contours
        all_contours = ImageUtils.grab_contours(
            cv2.findContours(canny_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        )

        if not all_contours:
            return None, None

        # Apply convex hull to resolve noise
        all_contours = [cv2.convexHull(contour) for contour in all_contours]

        # Sort by area and take top N
        all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)[
            :TOP_CONTOURS_COUNT
        ]

        # Find the first valid rectangular contour
        for contour in all_contours:
            area = cv2.contourArea(contour)

            # Skip small contours
            if area < MIN_PAGE_AREA:
                continue

            # Approximate contour to polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(
                contour,
                epsilon=APPROX_POLY_EPSILON_FACTOR * perimeter,
                closed=True,
            )

            # Check if it's a valid rectangle (4 corners)
            if MathUtils.validate_rect(approx):
                page_corners = np.reshape(approx, (4, -1))
                page_contour = np.vstack(contour).squeeze()

                logger.debug(f"Found page boundary for {file_path}")
                return page_corners, page_contour

        # No valid page found
        logger.warning(
            f"No valid page boundary found in {file_path}. "
            f"Largest contour area: {cv2.contourArea(all_contours[0]) if all_contours else 0}"
        )
        return None, None


def detect_page_corners(
    image: np.ndarray,
    colored_image: Optional[np.ndarray] = None,
    morph_kernel: Tuple[int, int] = (10, 10),
    use_colored_canny: bool = False,
    file_path: str = "unknown",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to detect page corners in an image.

    Args:
        image: Grayscale image
        colored_image: Optional colored version
        morph_kernel: Morphological kernel size
        use_colored_canny: Use HSV masking
        file_path: Path for error reporting

    Returns:
        Tuple of (page_corners, page_contour)
    """
    detector = PageDetector(morph_kernel, use_colored_canny)
    return detector.detect_page_boundary(image, colored_image, file_path)
