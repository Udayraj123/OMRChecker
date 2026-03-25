from typing import ClassVar

import cv2

from src.processors.constants import EDGE_TYPES_IN_ORDER, WarpMethod
from src.processors.image.WarpOnPointsCommon import WarpOnPointsCommon
from src.processors.image.page_detection import find_page_contour_and_corners
from src.utils.image import ImageUtils
from src.utils.logger import logger
from src.utils.math import MathUtils


class CropPage(WarpOnPointsCommon):
    """
    Preprocessor for detecting and cropping the page boundary.

    Uses edge detection and contour analysis to find the page rectangle,
    then crops and warps the image to align the page.
    """

    __is_internal_preprocessor__: ClassVar = False
    defaults: ClassVar = {
        "morph_kernel": (10, 10),
        "use_colored_canny": False,
    }

    def get_class_name(self) -> str:
        return "CropPage"

    def validate_and_remap_options_schema(self, options):
        tuning_options = options.get("tuning_options", {})

        return {
            # Local defaults
            "morph_kernel": options.get(
                "morph_kernel", CropPage.defaults["morph_kernel"]
            ),
            "use_colored_canny": options.get(
                "use_colored_canny", CropPage.defaults["use_colored_canny"]
            ),
            "max_points_per_edge": options.get("max_points_per_edge", None),
            "cropping_enabled": True,
            "tuning_options": {
                "warp_method": tuning_options.get(
                    "warp_method", WarpMethod.PERSPECTIVE_TRANSFORM
                ),
                "normalize_config": [],
                "canny_config": [],
            },
        }

    def __init__(self, options, *args, **kwargs) -> None:
        # Parent's __init__ will call validate_and_remap_options_schema via polymorphism
        super().__init__(options, *args, **kwargs)
        options = self.options
        self.use_colored_canny = options["use_colored_canny"]

        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, tuple(options["morph_kernel"])
        )

    def __str__(self) -> str:
        return "CropPage"

    def prepare_image_before_extraction(self, image):
        """Normalize image before page detection."""
        return ImageUtils.normalize(image)

    def extract_control_destination_points(self, image, colored_image, file_path):
        """
        Extract page corners and generate control/destination points.

        Uses the extracted page_detection module for cleaner separation.
        """
        config = self.tuning_config
        options = self.options

        # Check colored Canny configuration
        if self.use_colored_canny and not config.outputs.colored_outputs_enabled:
            logger.warning(
                "Cannot process colored image for CropPage. "
                "useColoredCanny is true but colored_outputs_enabled is false."
            )

        # Use extracted page detection module
        sheet, page_contour = find_page_contour_and_corners(
            image,
            colored_image=colored_image
            if config.outputs.colored_outputs_enabled
            else None,
            use_colored_canny=self.use_colored_canny,
            morph_kernel=self.morph_kernel,
            file_path=file_path,
            debug_image=self.debug_image,
        )

        # Split contour into edges
        (
            ordered_page_corners,
            edge_contours_map,
        ) = ImageUtils.split_patch_contour_on_corners(sheet, page_contour)

        logger.debug(f"Found page corners: \t {ordered_page_corners}")

        # Calculate destination corners
        (
            destination_page_corners,
            _,
        ) = ImageUtils.get_cropped_warped_rectangle_points(ordered_page_corners)

        # For DOC_REFINE and PERSPECTIVE_TRANSFORM, just return corners
        if self.warp_method in {
            WarpMethod.DOC_REFINE,
            WarpMethod.PERSPECTIVE_TRANSFORM,
        }:
            return ordered_page_corners, destination_page_corners, edge_contours_map

        # For other methods (HOMOGRAPHY, REMAP_GRIDDATA), generate edge points
        max_points_per_edge = options.get("max_points_per_edge", None)

        control_points, destination_points = [], []
        for edge_type in EDGE_TYPES_IN_ORDER:
            destination_line = MathUtils.select_edge_from_rectangle(
                destination_page_corners, edge_type
            )
            # Extrapolate destination_line to get approximate destination points
            (
                edge_control_points,
                edge_destination_points,
            ) = ImageUtils.get_control_destination_points_from_contour(
                edge_contours_map[edge_type], destination_line, max_points_per_edge
            )
            control_points += edge_control_points
            destination_points += edge_destination_points
        logger.debug(
            f"control_points: {control_points}, destination_points: {destination_points}"
        )
        return control_points, destination_points, edge_contours_map
