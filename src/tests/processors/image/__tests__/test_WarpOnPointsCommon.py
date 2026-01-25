"""
Tests for Refactored WarpOnPointsCommon

Tests the orchestration layer of the warp processing pipeline.
"""

import cv2
import numpy as np
import pytest
from unittest.mock import Mock

from src.processors.constants import WarpMethod
from src.processors.image.WarpOnPointsCommon import WarpOnPointsCommon


class ConcreteWarpProcessor(WarpOnPointsCommon):
    """Concrete implementation for testing"""

    def __init__(
        self,
        options=None,
        relative_dir=".",
        save_image_ops=None,
        default_processing_image_shape=(1000, 1000),
    ):
        # Mock save_image_ops if not provided
        if save_image_ops is None:
            save_image_ops = Mock()
            save_image_ops.tuning_config = Mock()
            save_image_ops.tuning_config.outputs = Mock()
            save_image_ops.tuning_config.outputs.show_image_level = 0
            save_image_ops.tuning_config.outputs.colored_outputs_enabled = False

        if options is None:
            options = {}

        self._test_control_points = None
        self._test_destination_points = None
        self._test_edge_contours_map = None

        super().__init__(
            options, relative_dir, save_image_ops, default_processing_image_shape
        )

    def validate_and_remap_options_schema(self, options):
        """Simple passthrough validation"""
        return {
            "enableCropping": options.get("enableCropping", False),
            "tuningOptions": options.get("tuningOptions", {}),
        }

    def prepare_image_before_extraction(self, image):
        """Simple passthrough"""
        return image

    def extract_control_destination_points(self, image, colored_image, file_path):
        """Return pre-configured test points"""
        return (
            self._test_control_points,
            self._test_destination_points,
            self._test_edge_contours_map,
        )

    def set_test_points(self, control, destination, edge_map=None):
        """Helper to set test data"""
        self._test_control_points = control
        self._test_destination_points = destination
        self._test_edge_contours_map = edge_map


class TestWarpOnPointsCommonInitialization:
    """Tests for initialization and configuration"""

    def test_default_initialization(self):
        """Test processor initializes with defaults"""
        processor = ConcreteWarpProcessor()

        assert processor.enable_cropping is False
        assert (
            processor.warp_method == WarpMethod.HOMOGRAPHY
        )  # Default when not cropping
        assert processor.warp_strategy is not None

    def test_initialization_with_cropping(self):
        """Test initialization with cropping enabled"""
        options = {"enableCropping": True}
        processor = ConcreteWarpProcessor(options=options)

        assert processor.enable_cropping is True
        assert processor.warp_method == WarpMethod.PERSPECTIVE_TRANSFORM

    def test_custom_warp_method(self):
        """Test custom warp method in tuning options"""
        options = {
            "enableCropping": False,
            "tuningOptions": {"warpMethod": WarpMethod.REMAP_GRIDDATA},
        }
        processor = ConcreteWarpProcessor(options=options)

        assert processor.warp_method == WarpMethod.REMAP_GRIDDATA

    def test_custom_interpolation_flag(self):
        """Test custom interpolation flag"""

        options = {"tuningOptions": {"warpMethodFlag": "INTER_CUBIC"}}
        processor = ConcreteWarpProcessor(options=options)

        assert processor.warp_method_flag == cv2.INTER_CUBIC


class TestWarpOnPointsCommonPointParsing:
    """Tests for point parsing and preparation"""

    @pytest.fixture
    def test_image(self):
        """Create a test image"""
        return np.zeros((400, 600), dtype=np.uint8)

    def test_parse_simple_points_no_cropping(self, test_image):
        """Test parsing points without cropping"""
        processor = ConcreteWarpProcessor()

        control = [[100, 100], [500, 100], [500, 300], [100, 300]]
        destination = [[0, 0], [400, 0], [400, 200], [0, 200]]

        parsed_ctrl, parsed_dest, dims = processor._parse_and_prepare_points(
            test_image, control, destination
        )

        assert parsed_ctrl.shape == (4, 2)
        assert parsed_dest.shape == (4, 2)
        assert dims == (600, 400)  # Original image dimensions

    def test_parse_points_with_cropping(self, test_image):
        """Test parsing points with cropping enabled"""
        options = {"enableCropping": True}
        processor = ConcreteWarpProcessor(options=options)

        control = [[100, 100], [500, 100], [500, 300], [100, 300]]
        destination = [[50, 50], [450, 50], [450, 250], [50, 250]]

        parsed_ctrl, parsed_dest, dims = processor._parse_and_prepare_points(
            test_image, control, destination
        )

        # Dimensions should be based on destination bounding box
        assert dims == (400, 200)  # 450-50, 250-50

        # Destination points should be shifted to origin
        assert parsed_dest[0][0] == 0  # Was 50, now 0
        assert parsed_dest[0][1] == 0  # Was 50, now 0

    def test_deduplicate_points(self, test_image):
        """Test that duplicate points are removed"""
        processor = ConcreteWarpProcessor()

        # Includes duplicates
        control = [[100, 100], [200, 200], [100, 100], [300, 300]]
        destination = [[0, 0], [100, 100], [0, 0], [200, 200]]

        parsed_ctrl, parsed_dest, _ = processor._parse_and_prepare_points(
            test_image, control, destination
        )

        # Should have removed one duplicate
        assert len(parsed_ctrl) == 3
        assert len(parsed_dest) == 3


class TestWarpOnPointsCommonWarpingStrategies:
    """Tests for warping strategy application"""

    @pytest.fixture
    def test_image(self):
        """Create a test image"""
        img = np.zeros((400, 400), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 300), 255, -1)
        return img

    @pytest.fixture
    def simple_points(self):
        """Simple 4-point transformation"""
        control = np.array(
            [[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.float32
        )

        destination = np.array(
            [[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32
        )

        return control, destination

    def test_perspective_transform_strategy(self, test_image, simple_points):
        """Test perspective transform application"""
        options = {
            "enableCropping": True,
            "tuningOptions": {"warpMethod": WarpMethod.PERSPECTIVE_TRANSFORM},
        }
        processor = ConcreteWarpProcessor(options=options)

        control, destination = simple_points

        warped, warped_colored = processor._apply_warp_strategy(
            test_image, None, control, destination, (200, 200), None
        )

        assert warped.shape == (200, 200)
        assert warped_colored is None

    def test_homography_strategy(self, test_image, simple_points):
        """Test homography application"""
        options = {"tuningOptions": {"warpMethod": WarpMethod.HOMOGRAPHY}}
        processor = ConcreteWarpProcessor(options=options)

        control, destination = simple_points

        warped, _ = processor._apply_warp_strategy(
            test_image, None, control, destination, (200, 200), None
        )

        assert warped.shape == (200, 200)

    def test_griddata_strategy(self, test_image):
        """Test griddata remap strategy"""
        options = {"tuningOptions": {"warpMethod": WarpMethod.REMAP_GRIDDATA}}
        processor = ConcreteWarpProcessor(options=options)

        # Can use more than 4 points with griddata
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

        destination = np.array(
            [
                [0, 0],
                [100, 0],
                [200, 0],
                [200, 100],
                [200, 200],
                [100, 200],
                [0, 200],
                [0, 100],
            ],
            dtype=np.float32,
        )

        warped, _ = processor._apply_warp_strategy(
            test_image, None, control, destination, (200, 200), None
        )

        assert warped.shape == (200, 200)

    def test_perspective_transform_requires_4_points(self, test_image):
        """Test that perspective transform validates point count"""
        options = {"tuningOptions": {"warpMethod": WarpMethod.PERSPECTIVE_TRANSFORM}}
        processor = ConcreteWarpProcessor(options=options)

        # Only 3 points (should fail)
        control = np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float32)
        destination = np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float32)

        from src.exceptions import TemplateValidationError

        with pytest.raises(TemplateValidationError, match="Expected 4 control points"):
            processor._apply_warp_strategy(
                test_image, None, control, destination, (200, 200), None
            )

    def test_colored_output_when_enabled(self, test_image):
        """Test colored output is generated when enabled"""
        # Create mock with colored outputs enabled
        save_image_ops = Mock()
        save_image_ops.tuning_config = Mock()
        save_image_ops.tuning_config.outputs = Mock()
        save_image_ops.tuning_config.outputs.show_image_level = 0
        save_image_ops.tuning_config.outputs.colored_outputs_enabled = True

        processor = ConcreteWarpProcessor(save_image_ops=save_image_ops)

        colored = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)

        control = np.array(
            [[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.float32
        )
        destination = np.array(
            [[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32
        )

        warped, warped_colored = processor._apply_warp_strategy(
            test_image, colored, control, destination, (200, 200), None
        )

        assert warped.shape == (200, 200)
        assert warped_colored is not None
        assert warped_colored.shape == (200, 200, 3)


class TestWarpOnPointsCommonFullPipeline:
    """Integration tests for the full pipeline"""

    @pytest.fixture
    def test_image(self):
        """Create test image"""
        img = np.zeros((400, 600), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (500, 300), 255, -1)
        return img

    def test_full_apply_filter_pipeline(self, test_image):
        """Test the complete apply_filter pipeline"""
        processor = ConcreteWarpProcessor()

        # Set test points
        control = [[100, 100], [500, 100], [500, 300], [100, 300]]
        destination = [[0, 0], [400, 0], [400, 200], [0, 200]]
        processor.set_test_points(control, destination)

        # Mock the append_save_image method
        processor.append_save_image = Mock()

        # Run the pipeline
        warped, warped_colored, template = processor.apply_filter(
            test_image, test_image, None, "test.jpg"
        )

        # Check outputs
        assert warped is not None
        assert warped.shape[0] == 400
        assert warped.shape[1] == 600

        # Check that debug images were saved
        assert processor.append_save_image.call_count > 0

    def test_pipeline_with_cropping(self, test_image):
        """Test pipeline with cropping enabled"""
        options = {"enableCropping": True}
        processor = ConcreteWarpProcessor(options=options)

        # Set test points
        control = [[100, 100], [500, 100], [500, 300], [100, 300]]
        destination = [[50, 50], [450, 50], [450, 250], [50, 250]]
        processor.set_test_points(control, destination)

        processor.append_save_image = Mock()

        warped, _, _ = processor.apply_filter(test_image, test_image, None, "test.jpg")

        # With cropping, dimensions should match destination bounding box
        assert warped is not None


class TestWarpOnPointsCommonAbstractMethods:
    """Tests for abstract method enforcement"""

    def test_validate_and_remap_not_implemented(self):
        """Test that abstract methods must be implemented"""

        # Create a class that doesn't implement required methods
        class IncompleteProcessor(WarpOnPointsCommon):
            pass

        save_image_ops = Mock()
        save_image_ops.tuning_config = Mock()
        save_image_ops.tuning_config.outputs = Mock()
        save_image_ops.tuning_config.outputs.show_image_level = 0
        save_image_ops.tuning_config.outputs.colored_outputs_enabled = False

        # Should raise during initialization
        with pytest.raises(NotImplementedError):
            IncompleteProcessor({}, ".", save_image_ops, (1000, 1000))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
