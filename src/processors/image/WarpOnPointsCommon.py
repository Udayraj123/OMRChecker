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
            options: Processor configuration options (raw, will be validated)
            relative_dir: Base directory for relative paths
            save_image_ops: Image saving configuration
            default_processing_image_shape: Default image dimensions
        """
        # Store tuning config before validation (needed by some subclasses)
        self.tuning_config = save_image_ops.tuning_config

        # Validate and parse options (polymorphic - calls subclass implementation)
        parsed_options = self.validate_and_remap_options_schema(options)

        # Merge tuning options
        merged_options = self.merge_tuning_options(parsed_options, options)

        # Initialize parent with merged options
        super().__init__(
            merged_options,
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
        # Method-specific configurations
        method_configs = {
            WarpMethod.PERSPECTIVE_TRANSFORM: {
                "interpolation_flag": self.warp_method_flag
            },
            WarpMethod.HOMOGRAPHY: {
                "interpolation_flag": self.warp_method_flag,
                "use_ransac": False,
            },
            WarpMethod.REMAP_GRIDDATA: {"interpolation_method": "cubic"},
            WarpMethod.DOC_REFINE: {},
        }

        # Get config for this method, default to interpolation_flag only
        strategy_config = method_configs.get(
            self.warp_method, {"interpolation_flag": self.warp_method_flag}
        )

        return WarpStrategyFactory.create(self.warp_method, **strategy_config)

    # =========================================================================
    # Abstract methods for subclasses
    # =========================================================================

    def validate_and_remap_options_schema(self, _options) -> dict:
        """
        Validate and transform processor-specific options.

        Subclasses must override this to define their schema.

        This method is called by the parent's __init__ with polymorphic dispatch,
        so subclasses just need to override it - no need to call it explicitly.

        Args:
            _options: Raw options to validate and transform

        Returns:
            Validated and transformed options dict
        """
        msg = "Subclass must implement validate_and_remap_options_schema"
        raise NotImplementedError(msg)

    def merge_tuning_options(
        self, parsed_options: dict, original_options: dict
    ) -> dict:
        """
        Merge tuning options from original options into parsed options.

        This ensures tuningOptions from the original config aren't lost during validation.
        Can be overridden by subclasses if custom merge logic is needed.

        Args:
            parsed_options: Options returned from validate_and_remap_options_schema
            original_options: Original raw options dict

        Returns:
            Merged options dict
        """
        return OVERRIDE_MERGER.merge(
            {
                "tuningOptions": original_options.get("tuningOptions", {}),
            },
            parsed_options,
        )

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
        # Deduplicate points using dict to preserve order
        unique_pairs = {
            tuple(ctrl): dest
            for ctrl, dest in zip(control_points, destination_points, strict=False)
        }

        parsed_control_points = np.float32(list(unique_pairs.keys()))
        parsed_destination_points = np.float32(list(unique_pairs.values()))

        # Calculate warped dimensions
        h, w = image.shape[:2]
        warped_dimensions = self._calculate_warped_dimensions(
            (w, h), parsed_destination_points
        )

        return parsed_control_points, parsed_destination_points, warped_dimensions

    def _calculate_warped_dimensions(
        self, default_dims: Tuple[int, int], destination_points: np.ndarray
    ) -> Tuple[int, int]:
        """Calculate warped dimensions based on cropping settings."""
        if not self.enable_cropping:
            return default_dims

        destination_box, rectangle_dimensions = MathUtils.get_bounding_box_of_points(
            destination_points
        )

        # Shift points to origin for cropping (modifies destination_points in-place)
        from_origin = -1 * destination_box[0]
        destination_points[:] = MathUtils.shift_points_from_origin(
            from_origin, destination_points
        )

        return rectangle_dimensions

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
        # Prepare points for perspective transform
        control_points, destination_points, warped_dimensions = (
            self._prepare_points_for_strategy(
                control_points, destination_points, warped_dimensions
            )
        )

        # Build strategy kwargs
        strategy_kwargs = self._build_strategy_kwargs(edge_contours_map)

        # Select colored input based on config
        colored_input = self._get_colored_input(colored_image)

        # Apply the warp
        return self.warp_strategy.warp_image(
            image,
            colored_input,
            control_points,
            destination_points,
            warped_dimensions,
            **strategy_kwargs,
        )

    def _prepare_points_for_strategy(
        self,
        control_points: np.ndarray,
        destination_points: np.ndarray,
        warped_dimensions: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Prepare points specifically for perspective transform if needed."""
        if self.warp_method != WarpMethod.PERSPECTIVE_TRANSFORM:
            return control_points, destination_points, warped_dimensions

        if len(control_points) != 4:
            from pathlib import Path

            raise TemplateValidationError(
                Path("template"),
                reason=(
                    f"Expected 4 control points for perspective transform, "
                    f"found {len(control_points)}. "
                    f"Use tuningOptions['warpMethod'] for different methods."
                ),
            )

        # Order the 4 points consistently
        ordered_control, _ = MathUtils.order_four_points(
            control_points, dtype="float32"
        )

        # Recalculate destination points from ordered control points
        new_destination, new_dimensions = (
            ImageUtils.get_cropped_warped_rectangle_points(ordered_control)
        )

        return ordered_control, new_destination, new_dimensions

    def _build_strategy_kwargs(self, edge_contours_map: Optional[np.ndarray]) -> dict:
        """Build kwargs dict for strategy based on warp method."""
        if self.warp_method != WarpMethod.DOC_REFINE:
            return {}

        if edge_contours_map is None:
            msg = "DOC_REFINE method requires edge_contours_map"
            raise ImageProcessingError(msg)

        return {"edge_contours_map": edge_contours_map}

    def _get_colored_input(
        self, colored_image: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Return colored image only if colored outputs are enabled."""
        return (
            colored_image
            if self.tuning_config.outputs.colored_outputs_enabled
            else None
        )

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
        # Show high-detail visualizations if configured
        if config.outputs.show_image_level >= 4:
            self._show_high_detail_visualizations(
                config,
                file_path,
                original_image,
                warped_image,
                control_points,
                destination_points,
            )

        # Save images at appropriate levels
        self._save_debug_images(warped_image, warped_colored_image)

        # Show final preview if configured
        if config.outputs.show_image_level >= 5:
            title = f"{'Cropped' if self.enable_cropping else 'Warped'} Image Preview"
            InteractionUtils.show(f"{title}: {file_path}", warped_image, pause=True)

    def _show_high_detail_visualizations(
        self,
        config,
        file_path,
        original_image,
        warped_image,
        control_points,
        destination_points,
    ):
        """Show detailed debug visualizations."""
        title_prefix = "Cropped Image" if self.enable_cropping else "Warped Image"

        if self.enable_cropping:
            DrawingUtils.draw_contour(self.debug_image, cv2.convexHull(control_points))

        if config.outputs.show_image_level >= 5:
            InteractionUtils.show("Anchor Points", self.debug_image, pause=False)

        # Draw and show match lines
        matched_lines = DrawingUtils.draw_matches(
            original_image, control_points, warped_image, destination_points
        )
        InteractionUtils.show(
            f"{title_prefix} with Match Lines: {file_path}",
            matched_lines,
            pause=True,
            config=config,
        )

    def _save_debug_images(self, warped_image, warped_colored_image):
        """Save warped and debug images."""
        # Save warped image
        self.append_save_image(
            f"Warped Image(no resize): {self}",
            range(4, 7),
            warped_image,
            warped_colored_image,
        )

        # Save anchor points (different level ranges for CropPage)
        level_range = range(6, 7) if str(self) == "CropPage" else range(3, 7)
        self.append_save_image(f"Anchor Points: {self}", level_range, self.debug_image)
