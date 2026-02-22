"""
Page Detection Module

Extracted from CropPage to provide focused, testable page boundary detection.

This module handles:
- Edge detection using Canny
- Contour finding and filtering
- Page boundary identification
- Corner extraction
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from pathlib import Path

from src.processors.image.constants import (
    APPROX_POLY_EPSILON_FACTOR,
    CANNY_THRESHOLD_HIGH,
    CANNY_THRESHOLD_LOW,
    CONTOUR_THICKNESS_STANDARD,
    MIN_PAGE_AREA,
    PIXEL_VALUE_MAX,
    THRESH_PAGE_TRUNCATE_HIGH,
    THRESH_PAGE_TRUNCATE_SECONDARY,
    TOP_CONTOURS_COUNT,
)
from src.utils.exceptions import ImageProcessingError
from src.utils.constants import CLR_WHITE, hsv_white_high, hsv_white_low
from src.utils.drawing import DrawingUtils
from src.utils.image import ImageUtils
from src.utils.logger import logger
from src.utils.math import MathUtils


def prepare_page_image(image: np.ndarray) -> np.ndarray:
    """
    Prepare image for page detection.

    Applies truncation and normalization to enhance page boundaries.

    Args:
        image: Grayscale input image

    Returns:
        Preprocessed image ready for edge detection
    """
    # Truncate high values to reduce noise
    _, truncated = cv2.threshold(
        image, THRESH_PAGE_TRUNCATE_HIGH, PIXEL_VALUE_MAX, cv2.THRESH_TRUNC
    )

    # Normalize to full range
    return ImageUtils.normalize(truncated)


def apply_colored_canny(image: np.ndarray, colored_image: np.ndarray) -> np.ndarray:
    """
    Apply Canny edge detection on color-masked image.

    Uses HSV color space to isolate white-ish regions (the page).

    Args:
        image: Grayscale image
        colored_image: Original BGR color image

    Returns:
        Canny edge map
    """
    # Convert to HSV for better color-based masking
    hsv = cv2.cvtColor(colored_image, cv2.COLOR_BGR2HSV)

    # Mask to select only white-ish zones (the page)
    mask = cv2.inRange(hsv, hsv_white_low, hsv_white_high)
    mask_result = cv2.bitwise_and(image, image, mask=mask)

    # Apply Canny edge detection
    canny_edge = cv2.Canny(mask_result, CANNY_THRESHOLD_HIGH, CANNY_THRESHOLD_LOW)

    return canny_edge


def apply_grayscale_canny(
    image: np.ndarray, morph_kernel: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply Canny edge detection on grayscale image.

    Optionally applies morphological closing to complete broken edges.

    Args:
        image: Preprocessed grayscale image
        morph_kernel: Optional morphological kernel for closing operation

    Returns:
        Canny edge map
    """
    # Second truncation threshold for cleaner edges
    _, truncated = cv2.threshold(
        image, THRESH_PAGE_TRUNCATE_SECONDARY, PIXEL_VALUE_MAX, cv2.THRESH_TRUNC
    )

    normalized = ImageUtils.normalize(truncated)

    # Close small holes to complete edges
    if morph_kernel is not None and morph_kernel.shape[0] > 1:
        closed = cv2.morphologyEx(normalized, cv2.MORPH_CLOSE, morph_kernel)
    else:
        closed = normalized

    # Apply Canny edge detection
    canny_edge = cv2.Canny(closed, CANNY_THRESHOLD_HIGH, CANNY_THRESHOLD_LOW)

    return canny_edge


def find_page_contours(canny_edge: np.ndarray) -> list[np.ndarray]:
    """
    Find and filter contours that could be the page boundary.

    Args:
        canny_edge: Canny edge map

    Returns:
        List of candidate contours, sorted by area (largest first)
    """
    # Find all contours
    all_contours = ImageUtils.grab_contours(
        cv2.findContours(canny_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    )

    # Apply convex hull to resolve disordered curves from noise
    all_contours = [cv2.convexHull(contour) for contour in all_contours]

    # Sort by area and take top candidates
    sorted_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)
    return sorted_contours[:TOP_CONTOURS_COUNT]


def extract_page_rectangle(
    contours: list[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract the page rectangle from candidate contours.

    Finds the first contour that:
    1. Has area >= MIN_PAGE_AREA
    2. Can be approximated as a 4-sided polygon
    3. Forms a valid rectangle

    Args:
        contours: List of candidate contours

    Returns:
        Tuple of (corners, full_contour) or (None, None) if not found
        - corners: 4x2 array of corner points
        - full_contour: Full page contour for edge mapping
    """
    for contour in contours:
        # Skip if too small
        if cv2.contourArea(contour) < MIN_PAGE_AREA:
            continue

        # Approximate contour to polygon
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = APPROX_POLY_EPSILON_FACTOR * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        # Check if it's a valid rectangle (4 corners)
        if MathUtils.validate_rect(approx):
            corners = np.reshape(approx, (4, -1))
            full_contour = np.vstack(contour).squeeze()
            return corners, full_contour

    return None, None


def find_page_contour_and_corners(
    image: np.ndarray,
    colored_image: Optional[np.ndarray] = None,
    use_colored_canny: bool = False,
    morph_kernel: Optional[np.ndarray] = None,
    file_path: Optional[Path] = None,
    debug_image: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find page boundary and extract corners.

    Main entry point for page detection. Combines all steps:
    1. Image preparation
    2. Edge detection (colored or grayscale)
    3. Contour finding
    4. Rectangle extraction

    Args:
        image: Grayscale input image
        colored_image: Optional color image for colored Canny
        use_colored_canny: Whether to use color-based edge detection
        morph_kernel: Optional morphological kernel
        file_path: Optional file path for error messages
        debug_image: Optional image to draw debug contours on

    Returns:
        Tuple of (corners, page_contour)
        - corners: 4x2 array of page corners
        - page_contour: Full contour of the page boundary

    Raises:
        ImageProcessingError: If page boundary cannot be found
    """
    # Step 1: Prepare image
    prepared_image = prepare_page_image(image)

    # Step 2: Edge detection
    if use_colored_canny and colored_image is not None:
        canny_edge = apply_colored_canny(prepared_image, colored_image)
    else:
        canny_edge = apply_grayscale_canny(prepared_image, morph_kernel)

    # Step 3: Find contours
    contours = find_page_contours(canny_edge)

    # Step 4: Extract rectangle
    corners, page_contour = extract_page_rectangle(contours)

    # Step 5: Draw debug visualization if requested
    if corners is not None and debug_image is not None:
        approx = np.reshape(corners, (4, 1, 2))
        DrawingUtils.draw_contour(
            canny_edge,
            approx,
            color=CLR_WHITE,
            thickness=CONTOUR_THICKNESS_STANDARD,
        )
        DrawingUtils.draw_contour(
            debug_image,
            approx,
            color=CLR_WHITE,
            thickness=CONTOUR_THICKNESS_STANDARD,
        )

    # Step 6: Error if not found
    if page_contour is None:
        file_str = str(file_path) if file_path else "unknown"
        logger.error(f"Paper boundary not found for: '{file_str}'")
        logger.warning(
            "Have you accidentally included CropPage preprocessor?\n"
            f"If no, increase processing dimensions from config. Current image size: {image.shape[:2]}"
        )
        raise ImageProcessingError(
            "Paper boundary not found",
            file_path=file_path,
            reason=f"No valid rectangle found in {len(contours)} candidates",
        )

    return corners, page_contour
