"""
WarpOnPointsCommon - Refactored Version

This is the refactored base class for processors that warp images based on control points.
The original 404-line monolithic class has been broken down into:

1. warp_strategies.py - Different warping transformation methods (Strategy pattern)
2. point_utils.py - Point parsing, validation, and manipulation utilities

This class now focuses on:
- Orchestration of the warping pipeline
- Configuration management
- Debug visualization and image saving
- Template-specific abstract methods for subclasses
"""

from pathlib import Path
from typing import Any, ClassVar, Optional, Tuple
import cv2
import numpy as np

from src.exceptions import ImageProcessingError, TemplateValidationError
from src.processors.constants import WarpMethod, WarpMethodFlags
from src.processors.image.base import ImageTemplatePreprocessor
from src.processors.image.warp_strategies import WarpStrategyFactory, WarpStrategy
from src.utils.drawing import DrawingUtils
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils
from src.utils.parsing import OVERRIDE_MERGER


class WarpOnPointsCommon(ImageTemplatePreprocessor):
    """
    Base class for image processors that apply warping transformations.

    This class provides a template method pattern for warping images:
    1. Prepare image for extraction (subclass-specific)
    2. Extract control and destination points (subclass-specific)
    3. Parse and validate points
    4. Apply warping strategy
    5. Save debug images and visualizations

    Subclasses must implement:
    - validate_and_remap_options_schema()
    - prepare_image_before_extraction()
    - extract_control_destination_points()
    """

    __is_internal_preprocessor__: ClassVar = True

    # Map configuration flags to OpenCV constants
    warp_method_flags_map: ClassVar = {
        WarpMethodFlags.INTER_LINEAR: cv2.INTER_LINEAR,
        WarpMethodFlags.INTER_CUBIC: cv2.INTER_CUBIC,
        WarpMethodFlags.INTER_NEAREST: cv2.INTER_NEAREST,
    }

    def __init__(
        self, options, relative_dir, save_image_ops, default_processing_image_shape
    ) -> None:
        """
        Initialize the warp processor.

        Args:
            options: Processor configuration options
            relative_dir: Base directory for relative paths
            save_image_ops: Image saving configuration
            default_processing_image_shape: Default image dimensions
        """
        # Store tuning config before parent initialization
        self.tuning_config = save_image_ops.tuning_config

        # Validate and parse options (subclass-specific)
        parsed_options = self.validate_and_remap_options_schema(options)

        # Merge tuning options
        parsed_options = OVERRIDE_MERGER.merge(
            {
                "tuningOptions": options.get("tuningOptions", {}),
            },
            parsed_options,
        )

        # Initialize parent
        super().__init__(
            parsed_options,
            relative_dir,
            save_image_ops,
            default_processing_image_shape,
        )

        # Extract configuration
        options = self.options
        tuning_options = self.tuning_options

        # Cropping configuration
        self.enable_cropping = options.get("enableCropping", False)

        # Determine warp method (default depends on cropping)
        self.warp_method = tuning_options.get(
            "warpMethod",
            (
                WarpMethod.PERSPECTIVE_TRANSFORM
                if self.enable_cropping
                else WarpMethod.HOMOGRAPHY
            ),
        )

        # Get interpolation flag
        self.warp_method_flag = self.warp_method_flags_map.get(
            tuning_options.get("warpMethodFlag", "INTER_LINEAR")
        )

        # Create the appropriate warp strategy
        self.warp_strategy = self._create_warp_strategy()

        # Debug visualization storage
        self.debug_image = None
        self.debug_hstack = []
        self.debug_vstack = []

    def _create_warp_strategy(self) -> WarpStrategy:
        """
        Create the appropriate warp strategy based on configuration.

        Returns:
            Configured WarpStrategy instance
        """
        strategy_config = {"interpolation_flag": self.warp_method_flag}

        # Add method-specific config
        if self.warp_method == WarpMethod.HOMOGRAPHY:
            # Could add RANSAC configuration here if needed
            strategy_config["use_ransac"] = False
        elif self.warp_method == WarpMethod.REMAP_GRIDDATA:
            strategy_config["interpolation_method"] = "cubic"

        return WarpStrategyFactory.create(self.warp_method, **strategy_config)

    # =========================================================================
    # Abstract methods for subclasses
    # =========================================================================

    def validate_and_remap_options_schema(self, _options) -> dict:
        """
        Validate and transform processor-specific options.

        Subclasses must implement this to define their schema.
        """
        msg = "Subclass must implement validate_and_remap_options_schema"
        raise NotImplementedError(msg)

    def prepare_image_before_extraction(self, _image):
        """
        Prepare the image before extracting control points.

        Subclasses can apply preprocessing (blur, threshold, etc).
        """
        msg = "Subclass must implement prepare_image_before_extraction"
        raise NotImplementedError(msg)

    def extract_control_destination_points(
        self, _image, _colored_image, _file_path
    ) -> Tuple[Any, Any, Any]:
        """
        Extract control and destination points from the image.

        Returns:
            Tuple of (control_points, destination_points, edge_contours_map)

            - control_points: Points in the source image
            - destination_points: Corresponding points in the target space
            - edge_contours_map: Optional edge map for doc-refine method
        """
        msg = "Subclass must implement extract_control_destination_points"
        raise NotImplementedError(msg)

    def exclude_files(self) -> list[Path]:
        """Return list of files to exclude from processing"""
        return []

    # =========================================================================
    # Main processing pipeline
    # =========================================================================

    def apply_filter(self, image, colored_image, _template, file_path):
        """
        Apply the warping transformation to the image.

        This is the main entry point called by the processing pipeline.

        Args:
            image: Grayscale input image
            colored_image: Colored version of input
            _template: Template configuration (unused in base class)
            file_path: Path to the image file (for logging/debugging)

        Returns:
            Tuple of (warped_image, warped_colored_image, _template)
        """
        config = self.tuning_config

        # Initialize debug state
        self.debug_image = image.copy()
        self.debug_hstack = []
        self.debug_vstack = []

        # Step 1: Prepare image (subclass-specific)
        prepared_image = self.prepare_image_before_extraction(image)

        # Step 2: Extract control/destination points (subclass-specific)
        (
            control_points,
            destination_points,
            edge_contours_map,
        ) = self.extract_control_destination_points(
            prepared_image, colored_image, file_path
        )

        # Step 3: Parse and validate points
        (
            parsed_control_points,
            parsed_destination_points,
            warped_dimensions,
        ) = self._parse_and_prepare_points(image, control_points, destination_points)

        logger.debug(
            f"Cropping Enabled: {self.enable_cropping}\n"
            f"Control points: {len(parsed_control_points)}\n"
            f"Warped dimensions: {warped_dimensions}"
        )

        # Step 4: Apply warping using strategy
        warped_image, warped_colored_image = self._apply_warp_strategy(
            image,
            colored_image,
            parsed_control_points,
            parsed_destination_points,
            warped_dimensions,
            edge_contours_map,
        )

        # Step 5: Debug visualization and image saving
        self._save_debug_visualizations(
            config,
            file_path,
            image,
            warped_image,
            warped_colored_image,
            parsed_control_points,
            parsed_destination_points,
        )

        return warped_image, warped_colored_image, _template

    # =========================================================================
    # Point parsing and preparation
    # =========================================================================

    def _parse_and_prepare_points(
        self,
        image: np.ndarray,
        control_points: Any,
        destination_points: Any,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Parse, deduplicate, and prepare control/destination points.

        Args:
            image: Input image (for dimension reference)
            control_points: Raw control points from extraction
            destination_points: Raw destination points from extraction

        Returns:
            Tuple of (parsed_control_points, parsed_destination_points, warped_dimensions)
        """
        # Deduplicate points
        parsed_control = []
        parsed_dest = []
        seen_control = set()

        for ctrl_pt, dest_pt in zip(control_points, destination_points, strict=False):
            ctrl_tuple = tuple(ctrl_pt)
            if ctrl_tuple not in seen_control:
                seen_control.add(ctrl_tuple)
                parsed_control.append(ctrl_pt)
                parsed_dest.append(dest_pt)

        # Convert to numpy arrays
        parsed_control_points = np.float32(parsed_control)
        parsed_destination_points = np.float32(parsed_dest)

        # Determine warped dimensions
        h, w = image.shape[:2]
        warped_dimensions = (w, h)

        if self.enable_cropping:
            # Get bounding box of destination points
            destination_box, rectangle_dimensions = (
                MathUtils.get_bounding_box_of_points(parsed_destination_points)
            )
            warped_dimensions = rectangle_dimensions

            # Shift destination points to origin for cropping
            from_origin = -1 * destination_box[0]
            parsed_destination_points = MathUtils.shift_points_from_origin(
                from_origin, parsed_destination_points
            )

        return parsed_control_points, parsed_destination_points, warped_dimensions

    # =========================================================================
    # Warping strategy application
    # =========================================================================

    def _apply_warp_strategy(
        self,
        image: np.ndarray,
        colored_image: Optional[np.ndarray],
        control_points: np.ndarray,
        destination_points: np.ndarray,
        warped_dimensions: Tuple[int, int],
        edge_contours_map: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply the configured warp strategy.

        Args:
            image: Grayscale input
            colored_image: Optional colored input
            control_points: Validated control points
            destination_points: Validated destination points
            warped_dimensions: Output dimensions
            edge_contours_map: Optional edge map for doc-refine

        Returns:
            Tuple of (warped_gray, warped_colored)
        """
        config = self.tuning_config

        # Special handling for perspective transform (needs 4 ordered points)
        if self.warp_method == WarpMethod.PERSPECTIVE_TRANSFORM:
            if len(control_points) != 4:
                msg = (
                    f"Expected 4 control points for perspective transform, "
                    f"found {len(control_points)}. "
                    f"Use tuningOptions['warpMethod'] for different methods."
                )
                raise TemplateValidationError(
                    msg,
                    context={
                        "control_points_count": len(control_points),
                        "expected_count": 4,
                    },
                )

            # Order the 4 points consistently
            control_points, _ = MathUtils.order_four_points(
                control_points, dtype="float32"
            )

            # Recalculate destination points from ordered control points
            destination_points, warped_dimensions = (
                ImageUtils.get_cropped_warped_rectangle_points(control_points)
            )

        # Prepare kwargs for strategy
        strategy_kwargs = {}
        if self.warp_method == WarpMethod.DOC_REFINE:
            if edge_contours_map is None:
                msg = "DOC_REFINE method requires edge_contours_map"
                raise ImageProcessingError(msg)
            strategy_kwargs["edge_contours_map"] = edge_contours_map

        # Handle colored image
        colored_input = (
            colored_image if config.outputs.colored_outputs_enabled else None
        )

        # Apply the warp
        warped_image, warped_colored = self.warp_strategy.warp_image(
            image,
            colored_input,
            control_points,
            destination_points,
            warped_dimensions,
            **strategy_kwargs,
        )

        return warped_image, warped_colored

    # =========================================================================
    # Debug visualization and image saving
    # =========================================================================

    def _save_debug_visualizations(
        self,
        config,
        file_path,
        original_image: np.ndarray,
        warped_image: np.ndarray,
        warped_colored_image: Optional[np.ndarray],
        control_points: np.ndarray,
        destination_points: np.ndarray,
    ):
        """
        Save debug images and show interactive visualizations.

        Args:
            config: Tuning configuration
            file_path: Path for display titles
            original_image: Original input image
            warped_image: Warped output
            warped_colored_image: Optional warped colored output
            control_points: Control points used
            destination_points: Destination points used
        """
        title_prefix = "Warped Image"

        # High-detail visualizations
        if config.outputs.show_image_level >= 4:
            if self.enable_cropping:
                title_prefix = "Cropped Image"
                # Draw convex hull of control points
                DrawingUtils.draw_contour(
                    self.debug_image, cv2.convexHull(control_points)
                )

            if config.outputs.show_image_level >= 5:
                InteractionUtils.show("Anchor Points", self.debug_image, pause=False)

            # Draw match lines between control and destination
            matched_lines = DrawingUtils.draw_matches(
                original_image,
                control_points,
                warped_image,
                destination_points,
            )

            InteractionUtils.show(
                f"{title_prefix} with Match Lines: {file_path}",
                matched_lines,
                pause=True,
                config=config,
            )

        # Save warped image
        self.append_save_image(
            f"Warped Image(no resize): {self}",
            range(4, 7),
            warped_image,
            warped_colored_image,
        )

        # Save anchor points visualization
        # CropPage gets different level range
        if str(self) == "CropPage":
            self.append_save_image(
                f"Anchor Points: {self}", range(6, 7), self.debug_image
            )
        else:
            self.append_save_image(
                f"Anchor Points: {self}", range(3, 7), self.debug_image
            )

        # Show warped preview
        if config.outputs.show_image_level >= 5:
            InteractionUtils.show(
                f"{title_prefix} Preview of Warp: {file_path}",
                warped_image,
                pause=True,
            )
