"""
Dot and Line Detection Module

Extracted from CropOnDotLines to provide focused, testable detection algorithms.

This module handles:
- Dot detection using morphological operations
- Line detection using morphological operations
- Edge detection using Canny
- Contour extraction and processing
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from src.processors.constants import EDGE_TYPES_IN_ORDER, ScannerType
from src.utils.image import ImageUtils
from src.utils.math import MathUtils


def preprocess_dot_zone(
    zone: np.ndarray,
    dot_kernel: np.ndarray,
    dot_threshold: int = 150,
    blur_kernel: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Preprocess image zone for dot detection.

    Steps:
    1. Optional Gaussian blur
    2. White padding to prevent edge artifacts
    3. Morphological opening (erode then dilate)
    4. Thresholding and normalization

    Args:
        zone: Image zone to process
        dot_kernel: Morphological structuring element for dots
        dot_threshold: Threshold value for dot detection (darker = lower)
        blur_kernel: Optional Gaussian blur kernel size

    Returns:
        Preprocessed zone ready for contour detection
    """
    # Optional blur
    if blur_kernel is not None:
        zone = cv2.GaussianBlur(zone, blur_kernel, 0)

    # Add white padding to avoid dilations sticking to edges
    kernel_height, kernel_width = dot_kernel.shape[:2]
    white_padded_zone, pad_range = ImageUtils.pad_image_from_center(
        zone, kernel_width * 2, kernel_height * 2, 255
    )

    # Morphological opening: removes small noise while preserving dot shapes
    morphed_zone = cv2.morphologyEx(
        white_padded_zone, cv2.MORPH_OPEN, dot_kernel, iterations=3
    )

    # Threshold and normalize
    _, white_thresholded = cv2.threshold(
        morphed_zone, dot_threshold, 255, cv2.THRESH_TRUNC
    )
    white_normalised = ImageUtils.normalize(white_thresholded)

    # Remove white padding
    white_normalised = white_normalised[
        pad_range[0] : pad_range[1], pad_range[2] : pad_range[3]
    ]

    return white_normalised


def preprocess_line_zone(
    zone: np.ndarray,
    line_kernel: np.ndarray,
    gamma_low: float,
    line_threshold: int = 180,
    blur_kernel: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Preprocess image zone for line detection.

    Steps:
    1. Optional Gaussian blur
    2. Gamma adjustment to darken lines
    3. Thresholding and normalization
    4. White padding for morphology
    5. Morphological opening

    Args:
        zone: Image zone to process
        line_kernel: Morphological structuring element for lines
        gamma_low: Gamma value for darkening (<1.0)
        line_threshold: Threshold value for line detection
        blur_kernel: Optional Gaussian blur kernel size

    Returns:
        Preprocessed zone ready for contour detection
    """
    # Optional blur
    if blur_kernel is not None:
        zone = cv2.GaussianBlur(zone, blur_kernel, 0)

    # Darken the image to make lines more prominent
    darker_image = ImageUtils.adjust_gamma(zone, gamma_low)

    # Threshold and normalize
    _, thresholded = cv2.threshold(darker_image, line_threshold, 255, cv2.THRESH_TRUNC)
    normalised = ImageUtils.normalize(thresholded)

    # Add white padding for morphology
    kernel_height, kernel_width = line_kernel.shape[:2]
    white, pad_range = ImageUtils.pad_image_from_center(
        normalised, kernel_width * 2, kernel_height * 2, 255
    )

    # Threshold-normalize again after padding
    _, white_thresholded = cv2.threshold(white, line_threshold, 255, cv2.THRESH_TRUNC)
    white_normalised = ImageUtils.normalize(white_thresholded)

    # Morphological opening: removes small noise while preserving line shapes
    line_morphed = cv2.morphologyEx(
        white_normalised,
        cv2.MORPH_OPEN,
        line_kernel,
        iterations=3,
    )

    # Remove white padding
    line_morphed = line_morphed[
        pad_range[0] : pad_range[1], pad_range[2] : pad_range[3]
    ]

    return line_morphed


def detect_contours_using_canny(
    zone: np.ndarray,
    canny_low: int = 55,
    canny_high: int = 185,
) -> list:
    """
    Detect contours in zone using Canny edge detection.

    Args:
        zone: Preprocessed image zone
        canny_low: Low threshold for Canny
        canny_high: High threshold for Canny

    Returns:
        List of contours (sorted by area, largest first)
    """
    # Apply Canny edge detection
    canny_edges = cv2.Canny(zone, canny_high, canny_low)

    # Find contours
    all_contours = ImageUtils.grab_contours(
        cv2.findContours(canny_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    )

    if len(all_contours) == 0:
        return []

    # Sort by area (largest first)
    sorted_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)

    return sorted_contours


def extract_patch_corners_and_edges(
    contour: np.ndarray,
    scanner_type: str,
) -> Tuple[np.ndarray, Dict[str, list]]:
    """
    Extract corner points and edge contours from detected contour.

    Args:
        contour: Detected contour (largest from zone)
        scanner_type: Type of scanner (PATCH_DOT or PATCH_LINE)

    Returns:
        Tuple of (ordered_corners, edge_contours_map)
        - ordered_corners: 4x2 array of corner points
        - edge_contours_map: Dictionary mapping edge types to contour segments
    """
    # Convert to 2D points and get convex hull
    bounding_contour = np.vstack(contour).squeeze()
    bounding_hull = cv2.convexHull(bounding_contour)

    if scanner_type == ScannerType.PATCH_DOT:
        # Use axis-aligned bounding rectangle for dots
        x, y, w, h = cv2.boundingRect(bounding_hull)
        patch_corners = MathUtils.get_rectangle_points(x, y, w, h)
    elif scanner_type == ScannerType.PATCH_LINE:
        # Use rotated rectangle for lines (handles slight rotations)
        rotated_rect = cv2.minAreaRect(bounding_hull)
        rotated_rect_points = cv2.boxPoints(rotated_rect)
        patch_corners = np.intp(rotated_rect_points)
    else:
        raise ValueError(f"Unsupported scanner type: {scanner_type}")

    # Split contour into edges based on corners
    (
        ordered_patch_corners,
        edge_contours_map,
    ) = ImageUtils.split_patch_contour_on_corners(patch_corners, bounding_contour)

    return ordered_patch_corners, edge_contours_map


def detect_dot_corners(
    zone: np.ndarray,
    zone_offset: Tuple[int, int],
    dot_kernel: np.ndarray,
    dot_threshold: int = 150,
    blur_kernel: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """
    Detect dot corners in image zone.

    Main entry point for dot detection. Combines:
    1. Preprocessing
    2. Contour detection
    3. Corner extraction

    Args:
        zone: Image zone to search in
        zone_offset: Offset for absolute coordinates
        dot_kernel: Morphological kernel for dot detection
        dot_threshold: Threshold value for dots
        blur_kernel: Optional Gaussian blur kernel

    Returns:
        4x2 array of corner points in absolute coordinates, or None if not found
    """
    # Preprocess zone
    preprocessed = preprocess_dot_zone(
        zone,
        dot_kernel,
        dot_threshold=dot_threshold,
        blur_kernel=blur_kernel,
    )

    # Detect contours
    contours = detect_contours_using_canny(preprocessed)

    if len(contours) == 0:
        return None

    # Extract corners from largest contour
    largest_contour = contours[0]
    corners, _ = extract_patch_corners_and_edges(
        largest_contour,
        ScannerType.PATCH_DOT,
    )

    if corners is None:
        return None

    # Convert to absolute coordinates
    absolute_corners = MathUtils.shift_points_from_origin(zone_offset, corners)

    return np.array(absolute_corners, dtype=np.float32)


def detect_line_corners_and_edges(
    zone: np.ndarray,
    zone_offset: Tuple[int, int],
    line_kernel: np.ndarray,
    gamma_low: float,
    line_threshold: int = 180,
    blur_kernel: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, list]]]:
    """
    Detect line corners and edge contours in image zone.

    Main entry point for line detection. Combines:
    1. Preprocessing
    2. Contour detection
    3. Corner and edge extraction

    Args:
        zone: Image zone to search in
        zone_offset: Offset for absolute coordinates
        line_kernel: Morphological kernel for line detection
        gamma_low: Gamma value for darkening
        line_threshold: Threshold value for lines
        blur_kernel: Optional Gaussian blur kernel

    Returns:
        Tuple of (corners, edge_contours_map) or (None, None) if not found
        - corners: 4x2 array of corner points in absolute coordinates
        - edge_contours_map: Dictionary mapping edge types to absolute contour segments
    """
    # Preprocess zone
    preprocessed = preprocess_line_zone(
        zone,
        line_kernel,
        gamma_low,
        line_threshold=line_threshold,
        blur_kernel=blur_kernel,
    )

    # Detect contours
    contours = detect_contours_using_canny(preprocessed)

    if len(contours) == 0:
        return None, None

    # Extract corners and edges from largest contour
    largest_contour = contours[0]
    corners, edge_contours_map = extract_patch_corners_and_edges(
        largest_contour,
        ScannerType.PATCH_LINE,
    )

    if corners is None or edge_contours_map is None:
        return None, None

    # Convert to absolute coordinates
    absolute_corners = MathUtils.shift_points_from_origin(zone_offset, corners)

    shifted_edge_contours_map = {
        edge_type: MathUtils.shift_points_from_origin(
            zone_offset, edge_contours_map[edge_type]
        )
        for edge_type in EDGE_TYPES_IN_ORDER
    }

    return (
        np.array(absolute_corners, dtype=np.float32),
        shifted_edge_contours_map,
    )


def validate_blur_kernel(
    zone_shape: Tuple[int, int],
    blur_kernel: Tuple[int, int],
    zone_label: str = "",
) -> bool:
    """
    Validate that blur kernel is smaller than zone.

    Args:
        zone_shape: (height, width) of zone
        blur_kernel: (height, width) of blur kernel
        zone_label: Optional label for error messages

    Returns:
        True if valid, False otherwise
    """
    zone_h, zone_w = zone_shape
    blur_h, blur_w = blur_kernel

    if not (zone_h > blur_h and zone_w > blur_w):
        label_str = f" '{zone_label}'" if zone_label else ""
        raise ValueError(
            f"The zone{label_str} is smaller than provided blur kernel: "
            f"{zone_shape} < {blur_kernel}"
        )

    return True


def create_structuring_element(
    shape: str,
    size: Tuple[int, int],
) -> np.ndarray:
    """
    Create morphological structuring element.

    Args:
        shape: 'rect', 'ellipse', or 'cross'
        size: (width, height) of element

    Returns:
        Structuring element as numpy array
    """
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }

    if shape not in shape_map:
        raise ValueError(f"Unknown shape: {shape}. Use {list(shape_map.keys())}")

    return cv2.getStructuringElement(shape_map[shape], size)
