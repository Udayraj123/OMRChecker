"""
CropPage - Refactored Page Cropping Preprocessor

Simplified and modular version that delegates page detection to a separate module.
"""

from typing import ClassVar
from pathlib import Path

import numpy as np

from src.processors.constants import WarpMethod
from src.processors.image.WarpOnPointsCommon import WarpOnPointsCommon
from src.processors.image.page_detection import PageDetector
from src.utils.image import ImageUtils
from src.utils.logger import logger


class CropPage(WarpOnPointsCommon):
    """
    Automatic page detection and cropping preprocessor.

    This class handles the high-level workflow of detecting a page boundary
    and applying perspective correction. The actual detection logic is
    delegated to PageDetector for better separation of concerns.
    """

    __is_internal_preprocessor__: ClassVar = False

    defaults: ClassVar = {
        "morphKernel": (10, 10),
        "useColoredCanny": False,
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        options = self.options

        # Initialize page detector with configured options
        self.page_detector = PageDetector(
            morph_kernel=tuple(options["morphKernel"]),
            use_colored_canny=options["useColoredCanny"],
        )

    def get_class_name(self) -> str:
        return "CropPage"

    def __str__(self) -> str:
        return "CropPage"

    def validate_and_remap_options_schema(self, options):
        """
        Validate and remap options to internal schema.

        This method transforms user-provided options into the internal
        format expected by the processor.
        """
        tuning_options = options.get("tuningOptions", {})

        return {
            # Apply defaults from class-level defaults
            "morphKernel": options.get("morphKernel", self.defaults["morphKernel"]),
            "useColoredCanny": options.get(
                "useColoredCanny", self.defaults["useColoredCanny"]
            ),
            "enableCropping": True,  # CropPage always enables cropping
            "tuningOptions": {
                "warpMethod": tuning_options.get(
                    "warpMethod", WarpMethod.PERSPECTIVE_TRANSFORM
                ),
                # TODO: Support additional warp configuration
                "normalizeConfig": tuning_options.get("normalizeConfig", []),
                "cannyConfig": tuning_options.get("cannyConfig", []),
            },
        }

    def prepare_image_before_extraction(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for page detection.

        Normalizes the image to enhance edge detection performance.
        """
        return ImageUtils.normalize(image)

    def extract_control_destination_points(
        self,
        image: np.ndarray,
        colored_image: np.ndarray,
        file_path: str,
    ) -> tuple:
        """
        Extract control and destination points for warping.

        This method:
        1. Detects the page boundary using PageDetector
        2. Splits the contour into edge segments
        3. Computes destination points for perspective correction

        Returns:
            Tuple of (control_points, destination_points, edge_contours_map)
        """
        # Use the dedicated page detector
        sheet, page_contour = self.page_detector.detect_page_boundary(
            image, colored_image, file_path
        )

        # Split the contour into ordered corners and edge segments
        (
            ordered_page_corners,
            edge_contours_map,
        ) = ImageUtils.split_patch_contour_on_corners(sheet, page_contour)

        logger.debug(f"Found page corners: {ordered_page_corners}")

        # Compute ideal destination rectangle
        (
            destination_page_corners,
            _,
        ) = ImageUtils.get_cropped_warped_rectangle_points(ordered_page_corners)

        # For perspective transform, we only need the 4 corners
        if self.warp_method in {
            WarpMethod.DOC_REFINE,
            WarpMethod.PERSPECTIVE_TRANSFORM,
        }:
            return ordered_page_corners, destination_page_corners, edge_contours_map

        # For other warp methods (e.g., HOMOGRAPHY, REMAP), extract more points
        return self._extract_dense_control_points(
            ordered_page_corners,
            destination_page_corners,
            edge_contours_map,
        )

    def _extract_dense_control_points(
        self,
        ordered_corners: np.ndarray,
        destination_corners: np.ndarray,
        edge_contours_map: dict,
    ) -> tuple:
        """
        Extract dense control points from edge contours.

        Used for advanced warping methods that benefit from more control points.
        """
        from src.processors.constants import EDGE_TYPES_IN_ORDER
        from src.utils.math import MathUtils

        options = self.options
        max_points_per_edge = options.get("maxPointsPerEdge", None)

        control_points, destination_points = [], []

        for edge_type in EDGE_TYPES_IN_ORDER:
            destination_line = MathUtils.select_edge_from_rectangle(
                destination_corners, edge_type
            )

            # Extrapolate destination points along the edge
            (
                edge_control_points,
                edge_destination_points,
            ) = ImageUtils.get_control_destination_points_from_contour(
                edge_contours_map[edge_type],
                destination_line,
                max_points_per_edge,
            )

            control_points += edge_control_points
            destination_points += edge_destination_points

        return control_points, destination_points, edge_contours_map

    def exclude_files(self) -> list[Path]:
        """
        Return list of files to exclude from processing.

        CropPage doesn't use any external reference files.
        """
        return []


# Convenience function for standalone usage
def crop_page_from_image(
    image: np.ndarray,
    colored_image: np.ndarray = None,
    morph_kernel: tuple = (10, 10),
    use_colored_canny: bool = False,
    warp_method: str = WarpMethod.PERSPECTIVE_TRANSFORM,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standalone function to detect and crop a page from an image.

    Args:
        image: Grayscale input image
        colored_image: Optional colored version
        morph_kernel: Morphological kernel size
        use_colored_canny: Use HSV masking for colored images
        warp_method: Warping method to use

    Returns:
        Tuple of (warped_gray_image, warped_colored_image)
    """
    # This is a simplified interface - for production use,
    # instantiate CropPage properly with all required parameters
    from src.processors.image.page_detection import detect_page_corners

    corners, _ = detect_page_corners(
        image,
        colored_image,
        morph_kernel,
        use_colored_canny,
    )

    # Get destination rectangle
    (
        destination_corners,
        warped_dimensions,
    ) = ImageUtils.get_cropped_warped_rectangle_points(corners)

    # Apply perspective transform
    import cv2

    transform_matrix = cv2.getPerspectiveTransform(
        corners.astype(np.float32),
        destination_corners.astype(np.float32),
    )

    warped_gray = cv2.warpPerspective(image, transform_matrix, warped_dimensions)

    warped_colored = None
    if colored_image is not None:
        warped_colored = cv2.warpPerspective(
            colored_image, transform_matrix, warped_dimensions
        )

    return warped_gray, warped_colored
