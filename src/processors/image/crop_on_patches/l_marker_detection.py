"""
L-Marker Detection Module

Detects L-shaped corner markers used for OMR sheet alignment.

Each L-marker is an L-shaped printed symbol at a page corner. The inner
right-angle vertex (the concave turning point) is used as the control
point for warping.

This module handles:
- Morphological preprocessing to clean noise
- Canny + contour detection to isolate the L shape
- Convexity defect analysis to extract the inner corner point
"""

from typing import Optional
import cv2
import numpy as np

from src.utils.logger import logger


def preprocess_for_l_marker(
    patch: np.ndarray,
    morph_kernel_size: tuple = (5, 5),
    morph_iterations: int = 2,
) -> np.ndarray:
    """
    Preprocess a patch to isolate L-marker shape.

    Steps:
    1. Gaussian blur to reduce noise
    2. Otsu threshold to binarise
    3. morphologyEx OPEN (remove small noise blobs)
    4. morphologyEx CLOSE (fill gaps inside L body)

    Args:
        patch: Grayscale image patch
        morph_kernel_size: Structuring element size for morphology
        morph_iterations: Number of morphological iterations

    Returns:
        Binary image (uint8, 0 or 255) with L shape isolated
    """
    # Gaussian blur
    blurred = cv2.GaussianBlur(patch, (5, 5), 0)

    # Otsu threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological open: remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
    opened = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel, iterations=morph_iterations
    )

    # Morphological close: fill gaps in L body
    closed = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations
    )

    return closed


def detect_l_contours(
    binary: np.ndarray,
    min_area: float = 500.0,
    max_area: float = 50000.0,
) -> list:
    """
    Find contours in a binary image that could be L-shaped markers.

    Uses Canny edge detection on the binary image, then findContours,
    then filters by area.

    Args:
        binary: Binary (uint8) image from preprocess_for_l_marker
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider

    Returns:
        List of contours sorted by area descending (largest first)
    """
    # Canny on binary
    edges = cv2.Canny(binary, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area
    valid = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

    # Sort largest first
    valid.sort(key=cv2.contourArea, reverse=True)

    return valid


def extract_l_inner_corner(contour: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract the inner right-angle corner of an L-shaped contour.

    Strategy:
    1. Primary: convexityDefects — the defect with maximum depth is the
       concave inner corner of the L.
    2. Fallback: approxPolyDP — find the vertex that lies inside the
       convex hull (concave vertex).

    Args:
        contour: Single contour array (N, 1, 2)

    Returns:
        (x, y) point as shape (2,) float32 array, or None if no valid
        inner corner found.
    """
    if len(contour) < 4:
        return None

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return None

    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        defects = None

    if defects is not None and len(defects) > 0:
        # Find defect with maximum depth
        max_depth = 0
        best_point = None
        for defect in defects:
            start_idx, end_idx, far_idx, depth = defect[0]
            if depth > max_depth:
                max_depth = depth
                best_point = contour[far_idx][0]

        if best_point is not None and max_depth > 100:  # minimum depth threshold
            logger.debug(f"L inner corner via convexity defect, depth={max_depth}")
            return np.float32(best_point)

    # Fallback: approxPolyDP — find the concave vertex
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    hull = cv2.convexHull(approx, returnPoints=True)
    hull_pts = set(map(tuple, hull.reshape(-1, 2)))

    for pt in approx.reshape(-1, 2):
        if tuple(pt) not in hull_pts:
            logger.debug("L inner corner via approxPolyDP fallback")
            return np.float32(pt)

    logger.warning("Could not extract L inner corner from contour")
    return None


def detect_l_marker_in_patch(
    patch: np.ndarray,
    zone_offset: tuple = (0, 0),
    tuning_options: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """
    Detect an L-shaped marker in a patch and return its inner corner
    in absolute image coordinates.

    Orchestrates: preprocess -> detect_l_contours -> pick largest ->
    extract_l_inner_corner -> shift by zone_offset.

    Args:
        patch: Grayscale image patch
        zone_offset: (x, y) offset to convert patch coords to image coords
        tuning_options: Optional dict with keys:
            morph_kernel_size (list[int, int])
            morph_iterations (int)
            min_marker_area (float)
            max_marker_area (float)

    Returns:
        (x, y) point as shape (2,) float32 in absolute image coordinates,
        or None if no L marker found.
    """
    if tuning_options is None:
        tuning_options = {}

    morph_kernel_size = tuple(tuning_options.get("morph_kernel_size", [5, 5]))
    morph_iterations = int(tuning_options.get("morph_iterations", 2))
    min_area = float(tuning_options.get("min_marker_area", 500.0))
    max_area = float(tuning_options.get("max_marker_area", 50000.0))

    # Preprocess
    binary = preprocess_for_l_marker(patch, morph_kernel_size, morph_iterations)

    # Find contours
    contours = detect_l_contours(binary, min_area, max_area)

    if not contours:
        logger.warning("No L-marker contours found in patch")
        return None

    # Take the largest contour (most likely to be the L marker)
    best_contour = contours[0]

    # Extract inner corner
    inner_corner = extract_l_inner_corner(best_contour)

    if inner_corner is None:
        logger.warning("Found contour but could not extract L inner corner")
        return None

    # Shift to absolute image coordinates
    ox, oy = zone_offset
    absolute_point = np.float32([inner_corner[0] + ox, inner_corner[1] + oy])

    logger.debug(
        f"L marker detected at patch {inner_corner}, absolute {absolute_point}"
    )

    return absolute_point
