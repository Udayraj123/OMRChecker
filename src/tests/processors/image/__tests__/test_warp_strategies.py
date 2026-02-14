"""
Tests for Image Warping Strategies

Comprehensive tests for all warping transformation strategies.
"""

import cv2
import numpy as np
import pytest

from src.processors.image.warp_strategies import (
    PerspectiveTransformStrategy,
    HomographyStrategy,
    GridDataRemapStrategy,
    DocRefineRectifyStrategy,
    WarpStrategyFactory,
)


class TestPerspectiveTransformStrategy:
    """Tests for perspective transformation"""

    @pytest.fixture
    def test_image(self):
        """Create a test image with clear features"""
        img = np.zeros((400, 400), dtype=np.uint8)
        # Draw a white square
        cv2.rectangle(img, (50, 50), (350, 350), 255, -1)
        # Add corners for tracking
        cv2.circle(img, (50, 50), 10, 0, -1)
        cv2.circle(img, (350, 50), 10, 0, -1)
        cv2.circle(img, (350, 350), 10, 0, -1)
        cv2.circle(img, (50, 350), 10, 0, -1)
        return img

    @pytest.fixture
    def control_points(self):
        """4 corners of a tilted rectangle"""
        return np.array(
            [
                [100, 150],  # top-left
                [300, 100],  # top-right
                [320, 300],  # bottom-right
                [80, 350],  # bottom-left
            ],
            dtype=np.float32,
        )

    @pytest.fixture
    def destination_points(self):
        """Target rectangle (axis-aligned)"""
        return np.array(
            [
                [50, 50],
                [350, 50],
                [350, 350],
                [50, 350],
            ],
            dtype=np.float32,
        )

    def test_initialization(self):
        """Test strategy initializes correctly"""
        strategy = PerspectiveTransformStrategy()
        assert strategy.get_name() == "PerspectiveTransform"
        assert strategy.interpolation_flag == cv2.INTER_LINEAR

        strategy = PerspectiveTransformStrategy(cv2.INTER_CUBIC)
        assert strategy.interpolation_flag == cv2.INTER_CUBIC

    def test_warp_simple_image(self, test_image, control_points, destination_points):
        """Test basic perspective warp"""
        strategy = PerspectiveTransformStrategy()

        warped, warped_colored, _ = strategy.warp_image(
            test_image, None, control_points, destination_points, (400, 400)
        )

        # Warped image should have correct shape
        assert warped.shape == (400, 400)

        # Should be different from original
        assert not np.array_equal(warped, test_image)

        # Colored should be None
        assert warped_colored is None

    def test_warp_with_colored_image(
        self, test_image, control_points, destination_points
    ):
        """Test warping both grayscale and colored"""
        strategy = PerspectiveTransformStrategy()
        colored = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)

        warped_gray, warped_colored, _ = strategy.warp_image(
            test_image, colored, control_points, destination_points, (400, 400)
        )

        assert warped_gray.shape == (400, 400)
        assert warped_colored.shape == (400, 400, 3)

    def test_requires_4_points(self, test_image):
        """Test that exactly 4 points are required"""
        strategy = PerspectiveTransformStrategy()

        # Test with 3 points
        with pytest.raises(ValueError, match="exactly 4 control points"):
            strategy.warp_image(
                test_image,
                None,
                np.array([[0, 0], [100, 0], [100, 100]]),
                np.array([[0, 0], [100, 0], [100, 100]]),
                (400, 400),
            )

        # Test with 5 points
        with pytest.raises(ValueError, match="exactly 4 control points"):
            strategy.warp_image(
                test_image,
                None,
                np.array([[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]]),
                np.array([[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]]),
                (400, 400),
            )

    def test_identity_transform(self, test_image):
        """Test that identity transform preserves image"""
        strategy = PerspectiveTransformStrategy()

        # Same control and destination points
        points = np.array([[0, 0], [399, 0], [399, 399], [0, 399]], dtype=np.float32)

        warped, _, _ = strategy.warp_image(test_image, None, points, points, (400, 400))

        # Should be nearly identical (allowing for interpolation artifacts)
        difference = np.abs(warped.astype(int) - test_image.astype(int))
        assert np.mean(difference) < 1.0  # Less than 1 pixel average difference


class TestHomographyStrategy:
    """Tests for homography transformation"""

    @pytest.fixture
    def test_image(self):
        """Create test image"""
        img = np.zeros((400, 400), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 300), 255, -1)
        return img

    def test_initialization(self):
        """Test homography strategy initialization"""
        strategy = HomographyStrategy()
        assert strategy.get_name() == "Homography"
        assert not strategy.use_ransac

        strategy = HomographyStrategy(use_ransac=True, ransac_threshold=5.0)
        assert strategy.use_ransac
        assert strategy.ransac_threshold == 5.0

    def test_warp_with_4_points(self, test_image):
        """Test homography with 4 points (minimum)"""
        strategy = HomographyStrategy()

        control = np.array(
            [[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.float32
        )

        dest = np.array([[50, 50], [350, 50], [350, 350], [50, 350]], dtype=np.float32)

        warped, _, _ = strategy.warp_image(test_image, None, control, dest, (400, 400))

        assert warped.shape == (400, 400)

    def test_warp_with_many_points(self, test_image):
        """Test homography with more than 4 points"""
        strategy = HomographyStrategy()

        # 8 points around a rectangle
        control = np.array(
            [
                [100, 100],
                [200, 100],
                [300, 100],
                [300, 200],
                [300, 300],
                [200, 300],
                [100, 300],
                [100, 200],
            ],
            dtype=np.float32,
        )

        dest = np.array(
            [
                [50, 50],
                [200, 50],
                [350, 50],
                [350, 200],
                [350, 350],
                [200, 350],
                [50, 350],
                [50, 200],
            ],
            dtype=np.float32,
        )

        warped, _, _ = strategy.warp_image(test_image, None, control, dest, (400, 400))

        assert warped.shape == (400, 400)

    def test_requires_at_least_4_points(self, test_image):
        """Test that at least 4 points are required"""
        strategy = HomographyStrategy()

        with pytest.raises(ValueError, match="at least 4 control points"):
            strategy.warp_image(
                test_image,
                None,
                np.array([[0, 0], [100, 0], [100, 100]]),
                np.array([[0, 0], [100, 0], [100, 100]]),
                (400, 400),
            )

    def test_ransac_mode(self, test_image):
        """Test RANSAC mode for robust estimation"""
        strategy = HomographyStrategy(use_ransac=True, ransac_threshold=3.0)

        # Add some outliers to test robustness
        control = np.array(
            [
                [100, 100],
                [300, 100],
                [300, 300],
                [100, 300],
                [200, 200],  # Outlier
            ],
            dtype=np.float32,
        )

        dest = np.array(
            [
                [50, 50],
                [350, 50],
                [350, 350],
                [50, 350],
                [500, 500],  # Outlier destination
            ],
            dtype=np.float32,
        )

        # Should still work with RANSAC
        warped, _, _ = strategy.warp_image(test_image, None, control, dest, (400, 400))

        assert warped.shape == (400, 400)


class TestGridDataRemapStrategy:
    """Tests for griddata interpolation"""

    @pytest.fixture
    def test_image(self):
        """Create test image with gradient"""
        x = np.linspace(0, 255, 400)
        y = np.linspace(0, 255, 400)
        xv, yv = np.meshgrid(x, y)
        img = ((xv + yv) / 2).astype(np.uint8)
        return img

    def test_initialization(self):
        """Test griddata strategy initialization"""
        strategy = GridDataRemapStrategy()
        assert strategy.get_name() == "GridDataRemap"
        assert strategy.interpolation_method == "cubic"

        strategy = GridDataRemapStrategy("linear")
        assert strategy.interpolation_method == "linear"

    def test_warp_with_sparse_points(self, test_image):
        """Test warping with sparse control points"""
        strategy = GridDataRemapStrategy()

        # Sparse points (corners + center)
        control = np.array(
            [
                [0, 0],
                [399, 0],
                [399, 399],
                [0, 399],
                [200, 200],
            ],
            dtype=np.float32,
        )

        dest = np.array(
            [
                [0, 0],
                [399, 0],
                [399, 399],
                [0, 399],
                [200, 200],
            ],
            dtype=np.float32,
        )

        warped, _, _ = strategy.warp_image(test_image, None, control, dest, (400, 400))

        assert warped.shape == (400, 400)

    def test_different_interpolation_methods(self, test_image):
        """Test different interpolation methods"""
        control = np.array(
            [
                [0, 0],
                [399, 0],
                [399, 399],
                [0, 399],
            ],
            dtype=np.float32,
        )

        dest = np.array(
            [
                [50, 50],
                [349, 50],
                [349, 349],
                [50, 349],
            ],
            dtype=np.float32,
        )

        for method in ["linear", "nearest", "cubic"]:
            strategy = GridDataRemapStrategy(method)
            warped, _, _ = strategy.warp_image(
                test_image, None, control, dest, (400, 400)
            )
            assert warped.shape == (400, 400)


class TestWarpStrategyFactory:
    """Tests for the strategy factory"""

    def test_create_perspective_transform(self):
        """Test creating perspective transform strategy"""
        strategy = WarpStrategyFactory.create("PERSPECTIVE_TRANSFORM")
        assert isinstance(strategy, PerspectiveTransformStrategy)

    def test_create_homography(self):
        """Test creating homography strategy"""
        strategy = WarpStrategyFactory.create("HOMOGRAPHY")
        assert isinstance(strategy, HomographyStrategy)

    def test_create_griddata_remap(self):
        """Test creating griddata strategy"""
        strategy = WarpStrategyFactory.create("REMAP_GRIDDATA")
        assert isinstance(strategy, GridDataRemapStrategy)

    def test_create_doc_refine(self):
        """Test creating doc-refine strategy"""
        strategy = WarpStrategyFactory.create("DOC_REFINE")
        assert isinstance(strategy, DocRefineRectifyStrategy)

    def test_unknown_method_raises_error(self):
        """Test that unknown method raises error"""
        with pytest.raises(ValueError, match="Unknown warp method"):
            WarpStrategyFactory.create("INVALID_METHOD")

    def test_get_available_methods(self):
        """Test getting available methods"""
        methods = WarpStrategyFactory.get_available_methods()

        assert "PERSPECTIVE_TRANSFORM" in methods
        assert "HOMOGRAPHY" in methods
        assert "REMAP_GRIDDATA" in methods
        assert "DOC_REFINE" in methods

    def test_create_with_config(self):
        """Test creating strategy with configuration"""
        strategy = WarpStrategyFactory.create(
            "PERSPECTIVE_TRANSFORM", interpolation_flag=cv2.INTER_CUBIC
        )
        assert strategy.interpolation_flag == cv2.INTER_CUBIC

        strategy = WarpStrategyFactory.create(
            "HOMOGRAPHY", use_ransac=True, ransac_threshold=5.0
        )
        assert strategy.use_ransac
        assert strategy.ransac_threshold == 5.0


class TestWarpStrategyIntegration:
    """Integration tests across strategies"""

    @pytest.fixture
    def checkerboard(self):
        """Create a checkerboard pattern for visual verification"""
        img = np.zeros((400, 400), dtype=np.uint8)
        square_size = 50
        for i in range(0, 400, square_size):
            for j in range(0, 400, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    img[j : j + square_size, i : i + square_size] = 255
        return img

    def test_all_strategies_produce_output(self, checkerboard):
        """Test that all strategies can warp an image"""
        control = np.array(
            [
                [100, 100],
                [300, 100],
                [300, 300],
                [100, 300],
            ],
            dtype=np.float32,
        )

        dest = np.array(
            [
                [50, 50],
                [350, 50],
                [350, 350],
                [50, 350],
            ],
            dtype=np.float32,
        )

        strategies = [
            "PERSPECTIVE_TRANSFORM",
            "HOMOGRAPHY",
            "REMAP_GRIDDATA",
        ]

        for method_name in strategies:
            strategy = WarpStrategyFactory.create(method_name)
            warped, _, _ = strategy.warp_image(
                checkerboard, None, control, dest, (400, 400)
            )

            # All should produce valid output
            assert warped.shape == (400, 400)
            assert warped.dtype == np.uint8
            # Should not be all zeros
            assert np.any(warped > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
