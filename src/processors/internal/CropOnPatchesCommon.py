import cv2
import numpy as np

from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger


class CropOnPatchesCommon(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = True

    # Common code used by both types of croppers
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        options = self.options

        # Default to select centers for roi
        self.default_points_selector = self.default_points_selector_map[
            options.get("pointsSelector", "CENTERS")
        ]

    def exclude_files(self):
        return []

    def __str__(self):
        return f"CropOnMarkers[\"{self.options['type']}\"]"

    def prepare_image(self, image):
        return image

    def apply_filter(self, image, colored_image, _template, file_path):
        config = self.tuning_config

        self.debug_image = image.copy()
        self.debug_hstack = []
        self.debug_vstack = []

        image = self.prepare_image(image)

        four_corners = self.find_four_corners(image, file_path)

        # Crop the image
        warped_image = ImageUtils.four_point_transform(image, four_corners)

        if config.outputs.show_colored_outputs:
            colored_image = ImageUtils.four_point_transform(colored_image, four_corners)

        # TODO: Save intuitive meta data
        # self.append_save_image(1,warped_image)

        if config.outputs.show_image_level >= 4:
            hstack = ImageUtils.get_padded_hstack([self.debug_image, warped_image])
            InteractionUtils.show(
                f"warped_image: {file_path}", hstack, 1, 1, config=config
            )

        return warped_image, colored_image, _template

    def find_four_corners(self, image, file_path):
        options = self.options
        config = self.tuning_config

        # TODO: support for defaulting to using quadrants?
        patch_areas = self.patch_areas_for_type[options["type"]]
        corners = []
        for patch_type in patch_areas["DOTS"]:
            corners.append(self.select_point_from_dot(patch_type, image, file_path))

        for patch_type in patch_areas["LINES"]:
            corners += self.select_points_from_line(patch_type, image)

            if config.outputs.show_image_level >= 5:
                if len(self.debug_vstack) > 0:
                    InteractionUtils.show(
                        f"Line Patches: {patch_type}",
                        ImageUtils.get_vstack_image_grid(self.debug_vstack),
                        0,
                        0,
                        config=config,
                    )
                self.debug_vstack = []

        if config.outputs.show_image_level >= 4:
            logger.info(f"corners={corners}")
            corners = ImageUtils.order_points(corners)
            cv2.drawContours(
                self.debug_image, [np.intp(corners)], -1, (200, 200, 200), 2
            )

            for corner in corners:
                cv2.rectangle(
                    self.debug_image,
                    tuple(corner),
                    (corner[0] + 2, corner[1] + 2),
                    (20, 255, 20),
                    2,
                )

        return corners

    default_points_selector_map = {
        "CENTERS": {
            "topLeftDot": "DOT_CENTER",
            "topRightDot": "DOT_CENTER",
            "bottomRightDot": "DOT_CENTER",
            "bottomLeftDot": "DOT_CENTER",
            "leftLine": "LINE_OUTER_EDGE",
            "rightLine": "LINE_OUTER_EDGE",
        },
        "INNER_WIDTHS": {
            "topLeftDot": "DOT_TOP_RIGHT",
            "topRightDot": "DOT_TOP_LEFT",
            "bottomRightDot": "DOT_BOTTOM_LEFT",
            "bottomLeftDot": "DOT_BOTTOM_RIGHT",
            "leftLine": "LINE_INNER_EDGE",
            "rightLine": "LINE_INNER_EDGE",
        },
        "INNER_HEIGHTS": {
            "topLeftDot": "DOT_BOTTOM_LEFT",
            "topRightDot": "DOT_BOTTOM_RIGHT",
            "bottomRightDot": "DOT_TOP_RIGHT",
            "bottomLeftDot": "DOT_TOP_LEFT",
            "leftLine": "LINE_OUTER_EDGE",
            "rightLine": "LINE_OUTER_EDGE",
        },
        "INNER_CORNERS": {
            "topLeftDot": "DOT_BOTTOM_RIGHT",
            "topRightDot": "DOT_BOTTOM_LEFT",
            "bottomRightDot": "DOT_TOP_LEFT",
            "bottomLeftDot": "DOT_TOP_RIGHT",
            "leftLine": "LINE_INNER_EDGE",
            "rightLine": "LINE_INNER_EDGE",
        },
        "OUTER_CORNERS": {
            "topLeftDot": "DOT_TOP_LEFT",
            "topRightDot": "DOT_TOP_RIGHT",
            "bottomRightDot": "DOT_BOTTOM_RIGHT",
            "bottomLeftDot": "DOT_BOTTOM_LEFT",
            "leftLine": "LINE_OUTER_EDGE",
            "rightLine": "LINE_OUTER_EDGE",
        },
    }
    all_dots = ["topLeftDot", "topRightDot", "bottomRightDot", "bottomLeftDot"]

    def select_point_from_dot_rect(self, patch_type, dot_rect):
        options = self.options
        points_selector = self.default_points_selector[patch_type]
        if patch_type in options:
            points_selector = options[patch_type].get("pointsSelector", points_selector)

        tl, tr, br, bl = dot_rect
        if points_selector == "DOT_TOP_LEFT":
            return tl
        if points_selector == "DOT_TOP_RIGHT":
            return tr
        if points_selector == "DOT_BOTTOM_RIGHT":
            return br
        if points_selector == "DOT_BOTTOM_LEFT":
            return bl
        if points_selector == "DOT_CENTER":
            return [
                (tl[0] + br[0]) // 2,
                (tl[1] + br[1]) // 2,
            ]
        return None

    def compute_scan_area(self, image, area_description):
        # parse arguments
        h, w = image.shape[:2]
        origin, dimensions, margins = map(
            area_description.get, ["origin", "dimensions", "margins"]
        )

        # compute area and clip to image dimensions
        area_start = [
            max(0, origin[0] - margins["horizontal"]),
            max(0, origin[1] - margins["vertical"]),
        ]
        area_end = [
            min(w, origin[0] + margins["horizontal"] + dimensions[0]),
            min(h, origin[1] + margins["vertical"] + dimensions[1]),
        ]

        if (
            area_start[0] == 0
            or area_start[1] == 0
            or area_end[0] == w
            or area_end[1] == h
        ):
            logger.warning(
                f"Scan area clipped to image boundary for patch item with origin: {origin}"
            )

        # Extract image area
        area = image[area_start[1] : area_end[1], area_start[0] : area_end[0]]

        config = self.tuning_config
        if config.outputs.show_image_level >= 1:
            h, w = area.shape[:2]
            cv2.rectangle(
                self.debug_image,
                tuple(area_start),
                (area_start[0] + w, area_start[1] + h),
                (20, 255, 20),
                2,
            )

        return area, np.array(area_start)
