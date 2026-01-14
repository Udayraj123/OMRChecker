"""
Utilities for patch-based scanning and point detection.

This module provides reusable utilities for:
- Selecting points from rectangles (corners, centers)
- Computing and drawing scan zones
- Managing edge contours from zone points
- Scan zone validation
"""

import cv2
import numpy as np

from src.processors.constants import (
    EDGE_TYPES_IN_ORDER,
    TARGET_ENDPOINTS_FOR_EDGES,
    EdgeType,
)
from src.utils.constants import CLR_DARK_GREEN, CLR_NEAR_BLACK
from src.utils.drawing import DrawingUtils
from src.utils.shapes import ShapeUtils


def select_point_from_rectangle(rectangle, points_selector):
    """
    Select a specific point from a rectangle based on selector type.

    Args:
        rectangle: Array of 4 corner points [tl, tr, br, bl]
        points_selector: Selector type (e.g., "SELECT_CENTER", "SELECT_TOP_LEFT")

    Returns:
        Selected point as [x, y] or None if selector is invalid
    """
    tl, tr, br, bl = rectangle
    if points_selector == "SELECT_TOP_LEFT":
        return tl
    if points_selector == "SELECT_TOP_RIGHT":
        return tr
    if points_selector == "SELECT_BOTTOM_RIGHT":
        return br
    if points_selector == "SELECT_BOTTOM_LEFT":
        return bl
    if points_selector == "SELECT_CENTER":
        return [
            (tl[0] + br[0]) // 2,
            (tl[1] + br[1]) // 2,
        ]
    return None


def compute_scan_zone(image, zone_description):
    """
    Extract image zone and compute zone boundaries.

    Args:
        image: Source image to extract zone from
        zone_description: Dictionary with zone origin, dimensions, and margins

    Returns:
        Tuple of (zone_image, zone_start_point, zone_end_point)
    """
    zone, scan_zone_rectangle = ShapeUtils.extract_image_from_zone_description(
        image, zone_description
    )

    zone_start = scan_zone_rectangle[0]
    zone_end = scan_zone_rectangle[2]
    return zone, np.array(zone_start), np.array(zone_end)


def get_edge_contours_map_from_zone_points(zone_preset_points):
    """
    Build edge contours map from zone points.

    Args:
        zone_preset_points: Dictionary mapping zone preset names to point arrays

    Returns:
        Dictionary mapping EdgeType to list of contour points
    """
    edge_contours_map = {
        EdgeType.TOP: [],
        EdgeType.RIGHT: [],
        EdgeType.BOTTOM: [],
        EdgeType.LEFT: [],
    }

    for edge_type in EDGE_TYPES_IN_ORDER:
        for zone_preset, contour_point_index in TARGET_ENDPOINTS_FOR_EDGES[edge_type]:
            if zone_preset in zone_preset_points:
                zone_points = zone_preset_points[zone_preset]
                if contour_point_index == "ALL":
                    edge_contours_map[edge_type] += zone_points
                else:
                    edge_contours_map[edge_type].append(
                        zone_points[contour_point_index]
                    )
    return edge_contours_map


def draw_zone_contours_and_anchor_shifts(
    debug_image, zone_control_points, zone_destination_points
):
    """
    Draw detected contours and alignment arrows for debugging.

    Args:
        debug_image: Image to draw on
        zone_control_points: List of detected control points
        zone_destination_points: List of target destination points
    """
    if len(zone_control_points) > 1:
        two_points = 2
        if len(zone_control_points) == two_points:
            # Draw line if it's just two points
            DrawingUtils.draw_contour(debug_image, zone_control_points)
        else:
            # Draw convex hull of the found control points
            DrawingUtils.draw_contour(
                debug_image,
                cv2.convexHull(np.intp(zone_control_points)),
            )

    # Helper for alignment
    DrawingUtils.draw_arrows(
        debug_image,
        zone_control_points,
        zone_destination_points,
        tip_length=0.4,
    )
    for control_point in zone_control_points:
        # Show current detections too
        DrawingUtils.draw_box(
            debug_image,
            control_point,
            # TODO: change this based on image shape
            [20, 20],
            color=CLR_DARK_GREEN,
            border=1,
            centered=True,
        )


def draw_scan_zone(debug_image, zone_description):
    """
    Draw scan zone boundaries on debug image.

    Draws two rectangles:
    - Outer rectangle (green): includes margins
    - Inner rectangle (black): actual scan zone without margins

    Args:
        debug_image: Image to draw on
        zone_description: Dictionary with zone origin, dimensions, and margins
    """
    scan_zone_rectangle = ShapeUtils.compute_scan_zone_rectangle(
        zone_description, include_margins=True
    )
    scan_zone_rectangle_without_margins = ShapeUtils.compute_scan_zone_rectangle(
        zone_description, include_margins=False
    )
    zone_start = scan_zone_rectangle[0]
    zone_end = scan_zone_rectangle[2]
    zone_start_without_margins = scan_zone_rectangle_without_margins[0]
    zone_end_without_margins = scan_zone_rectangle_without_margins[2]

    DrawingUtils.draw_box_diagonal(
        debug_image,
        zone_start,
        zone_end,
        color=CLR_DARK_GREEN,
        border=2,
    )
    DrawingUtils.draw_box_diagonal(
        debug_image,
        zone_start_without_margins,
        zone_end_without_margins,
        color=CLR_NEAR_BLACK,
        border=1,
    )
