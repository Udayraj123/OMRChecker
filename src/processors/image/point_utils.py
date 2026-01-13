"""
Point Parsing Utilities for Image Warping

Extracted from WarpOnPointsCommon to handle parsing and validation
of control and destination points from various formats.
"""

from typing import Tuple, List, Union, Optional
import numpy as np

from src.utils.logger import logger


class PointParser:
    """
    Parse and validate point specifications for warping operations.

    Handles different formats:
    - Direct arrays: [[x1,y1], [x2,y2], ...]
    - String references: "template.dimensions", "page_dimensions"
    - Tuples of arrays: (control_points, destination_points)
    """

    @staticmethod
    def parse_points(
        points_spec: Union[List, Tuple, str, np.ndarray],
        template_dimensions: Optional[Tuple[int, int]] = None,
        page_dimensions: Optional[Tuple[int, int]] = None,
        context: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse point specification into control and destination arrays.

        Args:
            points_spec: Point specification in various formats
            template_dimensions: Template (width, height) for reference
            page_dimensions: Page (width, height) for reference
            context: Additional context dict for resolving references

        Returns:
            Tuple of (control_points, destination_points) as numpy arrays

        Raises:
            ValueError: If points_spec format is invalid
        """
        if isinstance(points_spec, str):
            return PointParser._parse_string_reference(
                points_spec, template_dimensions, page_dimensions, context
            )

        if isinstance(points_spec, tuple) and len(points_spec) == 2:
            # Assume (control, destination) tuple
            control, dest = points_spec
            return (
                PointParser._ensure_numpy_array(control),
                PointParser._ensure_numpy_array(dest),
            )

        if isinstance(points_spec, (list, np.ndarray)):
            # Single array - use as both control and destination
            points = PointParser._ensure_numpy_array(points_spec)
            return points, points.copy()

        raise ValueError(
            f"Invalid points specification type: {type(points_spec)}. "
            f"Expected list, tuple, string, or numpy array."
        )

    @staticmethod
    def _parse_string_reference(
        reference: str,
        template_dimensions: Optional[Tuple[int, int]],
        page_dimensions: Optional[Tuple[int, int]],
        context: Optional[dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse string reference to points.

        Supported references:
        - "template.dimensions" -> corners of template
        - "page_dimensions" -> corners of page
        """
        if reference == "template.dimensions":
            if template_dimensions is None:
                raise ValueError(
                    "template.dimensions reference requires template_dimensions"
                )
            return PointParser._create_corner_points(template_dimensions)

        if reference == "page_dimensions":
            if page_dimensions is None:
                raise ValueError("page_dimensions reference requires page_dimensions")
            return PointParser._create_corner_points(page_dimensions)

        # Try to resolve from context
        if context and reference in context:
            value = context[reference]
            return PointParser.parse_points(
                value, template_dimensions, page_dimensions, context
            )

        raise ValueError(f"Unknown point reference: {reference}")

    @staticmethod
    def _create_corner_points(
        dimensions: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create corner points for a rectangle.

        Args:
            dimensions: (width, height)

        Returns:
            Tuple of (corners, corners) for control and destination
        """
        w, h = dimensions
        corners = np.array(
            [
                [0, 0],  # top-left
                [w - 1, 0],  # top-right
                [w - 1, h - 1],  # bottom-right
                [0, h - 1],  # bottom-left
            ],
            dtype=np.float32,
        )

        return corners, corners.copy()

    @staticmethod
    def _ensure_numpy_array(points: Union[List, np.ndarray]) -> np.ndarray:
        """Convert points to numpy array with proper dtype"""
        if isinstance(points, np.ndarray):
            return points.astype(np.float32)
        return np.array(points, dtype=np.float32)

    @staticmethod
    def validate_points(
        control_points: np.ndarray,
        destination_points: np.ndarray,
        min_points: int = 4,
    ) -> None:
        """
        Validate that point arrays are properly formed.

        Args:
            control_points: Source points
            destination_points: Target points
            min_points: Minimum number of required points

        Raises:
            ValueError: If validation fails
        """
        # Check shapes
        if control_points.ndim != 2 or control_points.shape[1] != 2:
            raise ValueError(
                f"control_points must be Nx2 array, got shape {control_points.shape}"
            )

        if destination_points.ndim != 2 or destination_points.shape[1] != 2:
            raise ValueError(
                f"destination_points must be Nx2 array, "
                f"got shape {destination_points.shape}"
            )

        # Check count match
        if len(control_points) != len(destination_points):
            raise ValueError(
                f"Mismatch: {len(control_points)} control points vs "
                f"{len(destination_points)} destination points"
            )

        # Check minimum
        if len(control_points) < min_points:
            raise ValueError(
                f"At least {min_points} points required, got {len(control_points)}"
            )

        logger.debug(f"Validated {len(control_points)} point pairs")


class WarpedDimensionsCalculator:
    """
    Calculate appropriate dimensions for warped images.

    Determines output image size based on destination points and
    optional constraints.
    """

    @staticmethod
    def calculate_from_points(
        destination_points: np.ndarray,
        padding: int = 0,
        max_dimension: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Calculate warped dimensions from destination points.

        Args:
            destination_points: Target points array
            padding: Extra padding to add
            max_dimension: Maximum width or height

        Returns:
            (width, height) tuple
        """
        # Find bounding box
        min_x = np.min(destination_points[:, 0])
        max_x = np.max(destination_points[:, 0])
        min_y = np.min(destination_points[:, 1])
        max_y = np.max(destination_points[:, 1])

        width = int(np.ceil(max_x - min_x)) + 1 + 2 * padding
        height = int(np.ceil(max_y - min_y)) + 1 + 2 * padding

        # Apply max dimension constraint if provided
        if max_dimension is not None:
            if width > max_dimension or height > max_dimension:
                scale = max_dimension / max(width, height)
                width = int(width * scale)
                height = int(height * scale)
                logger.debug(f"Scaled dimensions to fit max_dimension={max_dimension}")

        logger.debug(f"Calculated warped dimensions: {width}x{height}")
        return width, height

    @staticmethod
    def calculate_from_dimensions(
        dimensions: Tuple[int, int],
        scale: float = 1.0,
    ) -> Tuple[int, int]:
        """
        Calculate warped dimensions from explicit dimensions.

        Args:
            dimensions: (width, height)
            scale: Scaling factor

        Returns:
            (scaled_width, scaled_height)
        """
        w, h = dimensions
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)

        logger.debug(f"Dimensions {w}x{h} scaled by {scale} -> {scaled_w}x{scaled_h}")
        return scaled_w, scaled_h


def order_four_points(points: np.ndarray) -> np.ndarray:
    """
    Order 4 points in consistent order: TL, TR, BR, BL.

    Args:
        points: 4x2 array of points (in any order)

    Returns:
        4x2 array ordered as [top-left, top-right, bottom-right, bottom-left]

    Raises:
        ValueError: If not exactly 4 points
    """
    if len(points) != 4:
        raise ValueError(
            f"order_four_points requires exactly 4 points, got {len(points)}"
        )

    # Sort by y-coordinate to get top 2 and bottom 2
    sorted_by_y = points[np.argsort(points[:, 1])]

    top_points = sorted_by_y[:2]
    bottom_points = sorted_by_y[2:]

    # Sort each pair by x-coordinate
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]

    ordered = np.array(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        ],
        dtype=np.float32,
    )

    return ordered


def compute_point_distances(
    points1: np.ndarray,
    points2: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean distances between corresponding points.

    Args:
        points1: Nx2 array
        points2: Nx2 array

    Returns:
        N-length array of distances
    """
    if len(points1) != len(points2):
        raise ValueError("Point arrays must have same length")

    differences = points2 - points1
    distances = np.sqrt(np.sum(differences**2, axis=1))

    return distances


def compute_bounding_box(points: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute axis-aligned bounding box for points.

    Args:
        points: Nx2 array of points

    Returns:
        (min_x, min_y, max_x, max_y) tuple
    """
    min_x = int(np.floor(np.min(points[:, 0])))
    max_x = int(np.ceil(np.max(points[:, 0])))
    min_y = int(np.floor(np.min(points[:, 1])))
    max_y = int(np.ceil(np.max(points[:, 1])))

    return min_x, min_y, max_x, max_y
