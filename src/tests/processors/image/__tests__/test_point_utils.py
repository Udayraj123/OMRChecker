"""
Tests for Point Parsing and Manipulation Utilities
"""

import numpy as np
import pytest

from src.processors.image.point_utils import (
    PointParser,
    WarpedDimensionsCalculator,
    order_four_points,
    compute_point_distances,
    compute_bounding_box,
)


class TestPointParser:
    """Tests for PointParser"""

    def test_parse_simple_array(self):
        """Test parsing simple point array"""
        points_list = [[10, 20], [30, 40], [50, 60], [70, 80]]

        control, dest = PointParser.parse_points(points_list)

        assert isinstance(control, np.ndarray)
        assert isinstance(dest, np.ndarray)
        assert control.shape == (4, 2)
        assert dest.shape == (4, 2)
        assert np.array_equal(control, dest)

    def test_parse_numpy_array(self):
        """Test parsing numpy array"""
        points_array = np.array([[10, 20], [30, 40]], dtype=np.float32)

        control, dest = PointParser.parse_points(points_array)

        assert np.array_equal(control, points_array)
        assert np.array_equal(dest, points_array)

    def test_parse_tuple_of_arrays(self):
        """Test parsing tuple of (control, destination)"""
        control_in = [[0, 0], [100, 0], [100, 100], [0, 100]]
        dest_in = [[10, 10], [90, 10], [90, 90], [10, 90]]

        control, dest = PointParser.parse_points((control_in, dest_in))

        assert control.shape == (4, 2)
        assert dest.shape == (4, 2)
        assert not np.array_equal(control, dest)

    def test_parse_template_dimensions_reference(self):
        """Test parsing 'template.dimensions' reference"""
        template_dims = (800, 1200)

        control, dest = PointParser.parse_points(
            "template.dimensions", template_dimensions=template_dims
        )

        # Should create corner points
        expected = np.array(
            [[0, 0], [799, 0], [799, 1199], [0, 1199]], dtype=np.float32
        )

        assert np.array_equal(control, expected)
        assert np.array_equal(dest, expected)

    def test_parse_page_dimensions_reference(self):
        """Test parsing 'page_dimensions' reference"""
        page_dims = (600, 800)

        control, dest = PointParser.parse_points(
            "page_dimensions", page_dimensions=page_dims
        )

        expected = np.array([[0, 0], [599, 0], [599, 799], [0, 799]], dtype=np.float32)

        assert np.array_equal(control, expected)

    def test_parse_context_reference(self):
        """Test parsing from context dict"""
        context = {"my_points": [[100, 100], [200, 100], [200, 200], [100, 200]]}

        control, dest = PointParser.parse_points("my_points", context=context)

        assert control.shape == (4, 2)

    def test_missing_template_dimensions_raises_error(self):
        """Test that missing template_dimensions raises error"""
        with pytest.raises(ValueError, match="requires template_dimensions"):
            PointParser.parse_points("template.dimensions")

    def test_missing_page_dimensions_raises_error(self):
        """Test that missing page_dimensions raises error"""
        with pytest.raises(ValueError, match="requires page_dimensions"):
            PointParser.parse_points("page_dimensions")

    def test_unknown_reference_raises_error(self):
        """Test that unknown string reference raises error"""
        with pytest.raises(ValueError, match="Unknown point reference"):
            PointParser.parse_points("unknown_reference")

    def test_invalid_type_raises_error(self):
        """Test that invalid type raises error"""
        with pytest.raises(ValueError, match="Invalid points specification type"):
            PointParser.parse_points(42)  # Invalid type

    def test_validate_points_valid(self):
        """Test validation of valid points"""
        control = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        dest = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)

        # Should not raise
        PointParser.validate_points(control, dest, min_points=4)

    def test_validate_points_wrong_shape(self):
        """Test validation rejects wrong shape"""
        control = np.array([0, 1, 2, 3])  # 1D array
        dest = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

        with pytest.raises(ValueError, match="must be Nx2 array"):
            PointParser.validate_points(control, dest)

    def test_validate_points_mismatch_count(self):
        """Test validation rejects count mismatch"""
        control = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
        dest = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)

        with pytest.raises(ValueError, match="Mismatch"):
            PointParser.validate_points(control, dest)

    def test_validate_points_too_few(self):
        """Test validation rejects too few points"""
        control = np.array([[0, 0], [1, 1]], dtype=np.float32)
        dest = np.array([[0, 0], [1, 1]], dtype=np.float32)

        with pytest.raises(ValueError, match="At least 4 points required"):
            PointParser.validate_points(control, dest, min_points=4)


class TestWarpedDimensionsCalculator:
    """Tests for WarpedDimensionsCalculator"""

    def test_calculate_from_points_simple(self):
        """Test calculating dimensions from points"""
        points = np.array([[0, 0], [100, 0], [100, 200], [0, 200]], dtype=np.float32)

        width, height = WarpedDimensionsCalculator.calculate_from_points(points)

        assert width == 101  # 0 to 100 inclusive + 1
        assert height == 201  # 0 to 200 inclusive + 1

    def test_calculate_from_points_with_padding(self):
        """Test calculating dimensions with padding"""
        points = np.array([[0, 0], [100, 100]], dtype=np.float32)

        width, height = WarpedDimensionsCalculator.calculate_from_points(
            points, padding=10
        )

        assert width == 101 + 20  # +10 on each side
        assert height == 101 + 20

    def test_calculate_from_points_with_max_dimension(self):
        """Test calculating dimensions with max constraint"""
        points = np.array(
            [[0, 0], [2000, 0], [2000, 3000], [0, 3000]], dtype=np.float32
        )

        width, height = WarpedDimensionsCalculator.calculate_from_points(
            points, max_dimension=1000
        )

        # Should be scaled down to fit within 1000
        assert max(width, height) <= 1000
        # Aspect ratio should be preserved
        assert abs(width / height - 2001 / 3001) < 0.01

    def test_calculate_from_dimensions(self):
        """Test calculating from explicit dimensions"""
        dims = (800, 1200)

        width, height = WarpedDimensionsCalculator.calculate_from_dimensions(
            dims, scale=1.0
        )

        assert width == 800
        assert height == 1200

    def test_calculate_from_dimensions_with_scale(self):
        """Test calculating with scaling"""
        dims = (800, 1200)

        width, height = WarpedDimensionsCalculator.calculate_from_dimensions(
            dims, scale=0.5
        )

        assert width == 400
        assert height == 600


class TestOrderFourPoints:
    """Tests for order_four_points function"""

    def test_order_already_ordered(self):
        """Test ordering already ordered points"""
        points = np.array(
            [
                [0, 0],  # TL
                [100, 0],  # TR
                [100, 100],  # BR
                [0, 100],  # BL
            ],
            dtype=np.float32,
        )

        ordered = order_four_points(points)

        assert np.array_equal(ordered, points)

    def test_order_random_order(self):
        """Test ordering randomly ordered points"""
        # Bottom-right, top-left, bottom-left, top-right
        points = np.array(
            [
                [100, 100],  # BR
                [0, 0],  # TL
                [0, 100],  # BL
                [100, 0],  # TR
            ],
            dtype=np.float32,
        )

        ordered = order_four_points(points)

        expected = np.array(
            [
                [0, 0],  # TL
                [100, 0],  # TR
                [100, 100],  # BR
                [0, 100],  # BL
            ],
            dtype=np.float32,
        )

        assert np.array_equal(ordered, expected)

    def test_order_tilted_rectangle(self):
        """Test ordering tilted rectangle"""
        points = np.array(
            [
                [150, 50],  # TR (higher x, lower y)
                [50, 100],  # TL (lower x, lower y)
                [100, 200],  # BL (lower x, higher y)
                [200, 150],  # BR (higher x, higher y)
            ],
            dtype=np.float32,
        )

        ordered = order_four_points(points)

        # Top two should have lower y values
        assert ordered[0][1] < 150
        assert ordered[1][1] < 150
        # Left points should have lower x values
        assert ordered[0][0] < ordered[1][0]
        assert ordered[3][0] < ordered[2][0]

    def test_order_requires_four_points(self):
        """Test that exactly 4 points are required"""
        points = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)

        with pytest.raises(ValueError, match="exactly 4 points"):
            order_four_points(points)


class TestComputePointDistances:
    """Tests for compute_point_distances function"""

    def test_zero_distance(self):
        """Test distance between identical points"""
        points1 = np.array([[0, 0], [10, 10]], dtype=np.float32)
        points2 = np.array([[0, 0], [10, 10]], dtype=np.float32)

        distances = compute_point_distances(points1, points2)

        assert len(distances) == 2
        assert np.allclose(distances, [0, 0])

    def test_horizontal_distance(self):
        """Test horizontal distance"""
        points1 = np.array([[0, 0]], dtype=np.float32)
        points2 = np.array([[3, 0]], dtype=np.float32)

        distances = compute_point_distances(points1, points2)

        assert np.allclose(distances, [3.0])

    def test_diagonal_distance(self):
        """Test diagonal distance (Pythagoras)"""
        points1 = np.array([[0, 0]], dtype=np.float32)
        points2 = np.array([[3, 4]], dtype=np.float32)

        distances = compute_point_distances(points1, points2)

        assert np.allclose(distances, [5.0])  # 3-4-5 triangle

    def test_multiple_distances(self):
        """Test multiple point pairs"""
        points1 = np.array([[0, 0], [10, 10], [20, 20]], dtype=np.float32)

        points2 = np.array([[0, 0], [13, 14], [23, 24]], dtype=np.float32)

        distances = compute_point_distances(points1, points2)

        assert len(distances) == 3
        assert np.allclose(distances[0], 0.0)
        assert np.allclose(distances[1], 5.0)  # 3-4-5 triangle
        assert np.allclose(distances[2], 5.0)  # 3-4-5 triangle

    def test_mismatched_length_raises_error(self):
        """Test that mismatched arrays raise error"""
        points1 = np.array([[0, 0], [1, 1]], dtype=np.float32)
        points2 = np.array([[0, 0]], dtype=np.float32)

        with pytest.raises(ValueError, match="same length"):
            compute_point_distances(points1, points2)


class TestComputeBoundingBox:
    """Tests for compute_bounding_box function"""

    def test_simple_rectangle(self):
        """Test bounding box of rectangle"""
        points = np.array(
            [[10, 20], [100, 20], [100, 200], [10, 200]], dtype=np.float32
        )

        min_x, min_y, max_x, max_y = compute_bounding_box(points)

        assert min_x == 10
        assert min_y == 20
        assert max_x == 100
        assert max_y == 200

    def test_single_point(self):
        """Test bounding box of single point"""
        points = np.array([[50, 75]], dtype=np.float32)

        min_x, min_y, max_x, max_y = compute_bounding_box(points)

        assert min_x == 50
        assert min_y == 75
        assert max_x == 50
        assert max_y == 75

    def test_scattered_points(self):
        """Test bounding box of scattered points"""
        points = np.array([[30, 50], [10, 80], [90, 20], [50, 100]], dtype=np.float32)

        min_x, min_y, max_x, max_y = compute_bounding_box(points)

        assert min_x == 10
        assert min_y == 20
        assert max_x == 90
        assert max_y == 100

    def test_negative_coordinates(self):
        """Test bounding box with negative coordinates"""
        points = np.array([[-10, -20], [10, 20]], dtype=np.float32)

        min_x, min_y, max_x, max_y = compute_bounding_box(points)

        assert min_x == -10
        assert min_y == -20
        assert max_x == 10
        assert max_y == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
