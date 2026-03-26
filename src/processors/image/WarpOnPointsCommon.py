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

from src.utils.exceptions import ImageProcessingError
from src.processors.constants import WarpMethod, WarpMethodFlags
from src.processors.image.base import ImageTemplatePreprocessor
from src.processors.image.warp_strategies import WarpStrategyFactory, WarpStrategy
from src.utils.drawing import DrawingUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
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
        # Validate and parse options (polymorphic - calls subclass implementation)
        parsed_options = self.validate_and_remap_options_schema(options)

        # Merge tuning options
        merged_options = self.merge_tuning_options(parsed_options, options)

        # Initialize parent with merged options (parent will set self.tuning_config)
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
        self.cropping_enabled = options.get("cropping_enabled", False)
        self.cropping_use_bounding_box = options.get("cropping_use_bounding_box", True)
        self.cropping_use_approx_poly = options.get("cropping_use_approx_poly", True)

        # Determine warp method (default depends on cropping)
        self.warp_method = tuning_options.get(
            "warp_method",
            (
                WarpMethod.PERSPECTIVE_TRANSFORM
                if self.cropping_enabled
                else WarpMethod.HOMOGRAPHY
            ),
        )

        # Get interpolation flag
        self.warp_method_flag = self.warp_method_flags_map.get(
            tuning_options.get("warp_method_flag", "INTER_LINEAR")
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
                "interpolation_flag": self.warp_method_flag,
                "use_bounding_box": self.cropping_use_bounding_box,
                "use_approx_poly": self.cropping_use_approx_poly,
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
                "tuning_options": original_options.get("tuning_options", {}),
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

        # Step 1: Prepare image (subclass-specific)
        prepared_image = self.prepare_image_before_extraction(image)

        # Initialize debug state with prepared image to ensure consistent dimensions
        self.debug_image = prepared_image.copy()
        self.debug_hstack = []
        self.debug_vstack = []

        # Step 2: Extract control/destination points (subclass-specific)
        (
            control_points,
            destination_points,
            edge_contours_map,
        ) = self.extract_control_destination_points(
            prepared_image, colored_image, file_path
        )

        # Step 3: Prepare points via strategy (dedup + shape + dims)
        h, w = prepared_image.shape[:2]
        (
            parsed_control_points,
            parsed_destination_points,
            warped_dimensions,
        ) = self.warp_strategy.prepare_points(
            control_points, destination_points, (w, h), self.cropping_enabled
        )

        logger.debug(
            f"Cropping Enabled: {self.cropping_enabled}\n"
            f"Control points: {parsed_control_points}\n"
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
        # Build strategy kwargs
        strategy_kwargs = self._build_strategy_kwargs(edge_contours_map)

        # Select colored input based on config
        colored_input = self._get_colored_input(colored_image)

        # Apply the warp (strategy warps image, colored_input, and debug_image in one go)
        warped_image, warped_colored_image, warped_debug_image = (
            self.warp_strategy.warp_image(
                image,
                colored_input,
                control_points,
                destination_points,
                warped_dimensions,
                debug_image=self.debug_image,
                **strategy_kwargs,
            )
        )
        if warped_debug_image is not None:
            self.debug_image = warped_debug_image

        return warped_image, warped_colored_image

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
            title = f"{'Cropped' if self.cropping_enabled else 'Warped'} Image Preview"
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
        title_prefix = "Cropped Image" if self.cropping_enabled else "Warped Image"

        if self.cropping_enabled:
            # debug_image is warped, so draw contour in destination space
            DrawingUtils.draw_contour(
                self.debug_image, cv2.convexHull(destination_points)
            )

        if config.outputs.show_image_level >= 5:
            InteractionUtils.show("Anchor Points", self.debug_image, pause=False)

        # Draw and show match lines
        # TODO: debug why resizing the images is making matches not match with destination points.
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
