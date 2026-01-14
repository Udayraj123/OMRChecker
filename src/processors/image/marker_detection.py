"""
Marker Detection Module

Extracted from CropOnCustomMarkers to provide focused, testable
marker template matching algorithms.

This module handles:
- Marker template preparation
- Multi-scale template matching
- Best match selection
- Marker corner extraction
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from src.utils.image import ImageUtils
from src.utils.math import MathUtils
from src.utils.logger import logger


def prepare_marker_template(
    reference_image: np.ndarray,
    reference_zone: dict,
    marker_dimensions: Optional[Tuple[int, int]] = None,
    blur_kernel: Tuple[int, int] = (5, 5),
    apply_erode_subtract: bool = True,
) -> np.ndarray:
    """
    Extract and prepare marker template from reference image.

    Applies preprocessing to enhance marker features:
    1. Extract region of interest
    2. Resize if dimensions specified
    3. Gaussian blur to reduce noise
    4. Normalize to full range
    5. Optional erode-subtract to enhance edges

    Args:
        reference_image: Source image containing the marker
        reference_zone: Dict with 'origin' and 'dimensions' keys
        marker_dimensions: Optional (width, height) to resize marker
        blur_kernel: Gaussian blur kernel size
        apply_erode_subtract: Whether to enhance edges with erosion

    Returns:
        Preprocessed marker template ready for matching
    """
    origin, dimensions = reference_zone["origin"], reference_zone["dimensions"]
    x, y = origin
    w, h = dimensions

    # Extract marker region
    marker = reference_image[y : y + h, x : x + w]

    # Resize if dimensions specified
    if marker_dimensions is not None:
        marker = ImageUtils.resize_to_dimensions(marker_dimensions, marker)

    # Blur to reduce noise
    marker = cv2.GaussianBlur(marker, blur_kernel, 0)

    # Normalize to full range
    marker = cv2.normalize(
        marker, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Optional edge enhancement
    if apply_erode_subtract:
        marker = marker - cv2.erode(marker, kernel=np.ones((5, 5)), iterations=5)
        # Re-normalize after edge enhancement
        marker = cv2.normalize(
            marker, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

    return marker


def multi_scale_template_match(
    patch: np.ndarray,
    marker: np.ndarray,
    scale_range: Tuple[int, int] = (85, 115),
    scale_steps: int = 5,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, Optional[int]]:
    """
    Perform multi-scale template matching to find best match.

    Tests marker at different scales within scale_range to account for
    size variations due to scanning/printing differences.

    Args:
        patch: Image patch to search in
        marker: Template marker to find
        scale_range: (min_percent, max_percent) for scaling
        scale_steps: Number of scale increments to test

    Returns:
        Tuple of (position, optimal_marker, confidence, optimal_scale_percent)
        - position: (x, y) of top-left corner, or None if not found
        - optimal_marker: Rescaled marker that gave best match
        - confidence: Match confidence (0.0 to 1.0)
        - optimal_scale_percent: Scale that gave best match
    """
    descent_per_step = (scale_range[1] - scale_range[0]) // scale_steps
    marker_height, marker_width = marker.shape[:2]
    patch_height, patch_width = patch.shape[:2]

    best_position = None
    best_marker = None
    best_confidence = 0.0
    best_scale_percent = None

    # Test different scales
    for scale_percent in np.arange(
        scale_range[1], scale_range[0], -1 * descent_per_step
    ):
        scale = float(scale_percent / 100)
        if scale <= 0.0:
            continue

        # Rescale marker
        scaled_marker = ImageUtils.resize_single(
            marker,
            u_width=int(marker_width * scale),
            u_height=int(marker_height * scale),
        )

        # Skip if rescaled marker is larger than patch
        scaled_height, scaled_width = scaled_marker.shape[:2]
        if scaled_height > patch_height or scaled_width > patch_width:
            continue

        # Template matching
        match_result = cv2.matchTemplate(patch, scaled_marker, cv2.TM_CCOEFF_NORMED)

        max_confidence = match_result.max()

        # Update best match if this is better
        if max_confidence > best_confidence:
            best_scale_percent = scale_percent
            best_marker = scaled_marker
            best_confidence = max_confidence

            # Get position of best match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
            best_position = max_loc  # (x, y)

    logger.debug(
        f"Best match: scale={best_scale_percent}%, confidence={best_confidence:.2%}"
    )

    return best_position, best_marker, best_confidence, best_scale_percent


def extract_marker_corners(
    position: Tuple[int, int],
    marker: np.ndarray,
    zone_offset: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """
    Extract corner points of detected marker.

    Args:
        position: (x, y) position of marker's top-left corner in patch
        marker: The marker template (for dimensions)
        zone_offset: Offset to convert from patch coordinates to absolute

    Returns:
        4x2 array of corner points in absolute coordinates
    """
    h, w = marker.shape[:2]
    x, y = position

    # Get rectangle corners in patch coordinates
    corners = MathUtils.get_rectangle_points(x, y, w, h)

    # Shift to absolute coordinates
    absolute_corners = MathUtils.shift_points_from_origin(zone_offset, corners)

    # Ensure we return a numpy array
    return np.array(absolute_corners, dtype=np.float32)


def detect_marker_in_patch(
    patch: np.ndarray,
    marker: np.ndarray,
    zone_offset: Tuple[int, int] = (0, 0),
    scale_range: Tuple[int, int] = (85, 115),
    scale_steps: int = 5,
    min_confidence: float = 0.3,
) -> Optional[np.ndarray]:
    """
    Detect marker in a patch using multi-scale template matching.

    Main entry point for marker detection. Combines:
    1. Multi-scale template matching
    2. Confidence threshold check
    3. Corner extraction

    Args:
        patch: Image patch to search in
        marker: Template marker to find
        zone_offset: Offset for absolute coordinate conversion
        scale_range: (min_percent, max_percent) for scaling
        scale_steps: Number of scale increments to test
        min_confidence: Minimum confidence threshold (0.0 to 1.0)

    Returns:
        4x2 array of corner points in absolute coordinates, or None if not found
    """
    # Perform multi-scale matching
    position, optimal_marker, confidence, scale_percent = multi_scale_template_match(
        patch, marker, scale_range, scale_steps
    )

    # Check if we found a valid match
    if position is None or optimal_marker is None:
        logger.warning("No marker match found in patch")
        return None

    if confidence < min_confidence:
        logger.warning(
            f"Marker match confidence {confidence:.2%} below threshold {min_confidence:.2%}"
        )
        return None

    # Extract corners
    corners = extract_marker_corners(position, optimal_marker, zone_offset)

    logger.debug(
        f"Marker detected at {position} with confidence {confidence:.2%} "
        f"(scale={scale_percent}%)"
    )

    return corners


def validate_marker_detection(
    corners: Optional[np.ndarray],
    expected_area_range: Optional[Tuple[float, float]] = None,
) -> bool:
    """
    Validate detected marker corners.

    Args:
        corners: 4x2 array of corner points
        expected_area_range: Optional (min_area, max_area) for validation

    Returns:
        True if corners are valid, False otherwise
    """
    if corners is None:
        return False

    if corners.shape != (4, 2):
        logger.warning(f"Invalid corner shape: {corners.shape}, expected (4, 2)")
        return False

    # Optional area validation
    if expected_area_range is not None:
        # Calculate area using cross product
        # Assumes corners are ordered
        area = cv2.contourArea(corners)
        min_area, max_area = expected_area_range

        if not (min_area <= area <= max_area):
            logger.warning(
                f"Marker area {area} outside expected range [{min_area}, {max_area}]"
            )
            return False

    return True
