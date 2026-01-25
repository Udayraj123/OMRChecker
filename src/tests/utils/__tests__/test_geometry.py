"""Tests for geometry utility functions."""

import math

import pytest

from src.utils.geometry import bbox_center, euclidean_distance, vector_magnitude


class TestGeometryUtils:
    """Test suite for geometry utility functions."""

    def test_euclidean_distance_2d(self) -> None:
        """Test Euclidean distance calculation in 2D."""
        point1 = [0, 0]
        point2 = [3, 4]
        distance = euclidean_distance(point1, point2)
        assert distance == 5.0

    def test_euclidean_distance_same_point(self) -> None:
        """Test distance between identical points is zero."""
        point = [10, 20]
        distance = euclidean_distance(point, point)
        assert distance == 0.0

    def test_euclidean_distance_negative_coords(self) -> None:
        """Test distance with negative coordinates."""
        point1 = [-3, -4]
        point2 = [0, 0]
        distance = euclidean_distance(point1, point2)
        assert distance == 5.0

    def test_euclidean_distance_float_coords(self) -> None:
        """Test distance with floating point coordinates."""
        point1 = [1.5, 2.5]
        point2 = [4.5, 6.5]
        distance = euclidean_distance(point1, point2)
        assert distance == pytest.approx(5.0)

    def test_vector_magnitude_zero(self) -> None:
        """Test magnitude of zero vector."""
        vector = [0, 0]
        magnitude = vector_magnitude(vector)
        assert magnitude == 0.0

    def test_vector_magnitude_unit_vector(self) -> None:
        """Test magnitude of unit vector."""
        vector = [1, 0]
        magnitude = vector_magnitude(vector)
        assert magnitude == 1.0

    def test_vector_magnitude_2d(self) -> None:
        """Test vector magnitude in 2D."""
        vector = [3, 4]
        magnitude = vector_magnitude(vector)
        assert magnitude == 5.0

    def test_vector_magnitude_negative(self) -> None:
        """Test magnitude with negative components."""
        vector = [-3, -4]
        magnitude = vector_magnitude(vector)
        assert magnitude == 5.0

    def test_vector_magnitude_3d(self) -> None:
        """Test vector magnitude in 3D."""
        vector = [1, 2, 2]
        magnitude = vector_magnitude(vector)
        assert magnitude == 3.0

    def test_bbox_center_unit_square(self) -> None:
        """Test center calculation for unit square at origin."""
        origin = [0, 0]
        dimensions = [2, 2]
        center = bbox_center(origin, dimensions)
        assert center == [1.0, 1.0]

    def test_bbox_center_offset_box(self) -> None:
        """Test center calculation for offset bounding box."""
        origin = [10, 20]
        dimensions = [30, 40]
        center = bbox_center(origin, dimensions)
        assert center == [25.0, 40.0]

    def test_bbox_center_float_coords(self) -> None:
        """Test center with floating point coordinates."""
        origin = [1.5, 2.5]
        dimensions = [3.0, 4.0]
        center = bbox_center(origin, dimensions)
        assert center == [3.0, 4.5]

    def test_bbox_center_zero_dimensions(self) -> None:
        """Test center calculation for zero-size box (point)."""
        origin = [10, 20]
        dimensions = [0, 0]
        center = bbox_center(origin, dimensions)
        assert center == [10.0, 20.0]

    def test_bbox_center_large_box(self) -> None:
        """Test center calculation for large bounding box."""
        origin = [100, 200]
        dimensions = [800, 600]
        center = bbox_center(origin, dimensions)
        assert center == [500.0, 500.0]

    def test_geometry_consistency(self) -> None:
        """Test that geometry functions are consistent with each other."""
        # Two boxes with known centers
        origin1 = [0, 0]
        dimensions1 = [10, 10]
        center1 = bbox_center(origin1, dimensions1)

        origin2 = [20, 0]
        dimensions2 = [10, 10]
        center2 = bbox_center(origin2, dimensions2)

        # Distance between centers should equal distance between origins + half widths
        distance = euclidean_distance(center1, center2)
        expected_distance = 20.0  # Centers are at (5, 5) and (25, 5)
        assert distance == pytest.approx(expected_distance)

    def test_pythagorean_theorem(self) -> None:
        """Verify Pythagorean theorem holds using geometry utils."""
        # Right triangle with legs 3 and 4, hypotenuse should be 5
        point1 = [0, 0]
        point2 = [3, 0]
        point3 = [0, 4]

        # Calculate all three sides
        side_a = euclidean_distance(point1, point2)
        side_b = euclidean_distance(point1, point3)
        hypotenuse = euclidean_distance(point2, point3)

        # Verify Pythagorean theorem: a² + b² = c²
        assert math.isclose(side_a**2 + side_b**2, hypotenuse**2)
        assert hypotenuse == 5.0
