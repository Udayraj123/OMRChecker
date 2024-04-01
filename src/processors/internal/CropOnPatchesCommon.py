import cv2
import numpy as np

from src.processors.internal.CropOnIndexPointsCommon import CropOnIndexPointsCommon
from src.utils.constants import CLR_DARK_GREEN
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils


class CropOnPatchesCommon(CropOnIndexPointsCommon):
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

    def find_corners_and_edges(self, image, file_path):
        options = self.options
        config = self.tuning_config

        patch_selectors = self.patch_types_for_layout[options["type"]]

        (
            control_points,
            destination_points,
        ) = (
            [],
            [],
        )
        corner_points = []
        for patch_type in patch_selectors["DOTS"]:
            dot_point, destination_point = self.find_and_select_point_from_dot(
                image, patch_type, file_path
            )
            control_points.append(dot_point)
            destination_points.append(destination_point)

            corner_points.append(dot_point)

        for patch_type in patch_selectors["LINES"]:
            (
                edge_points,
                line_control_points,
                line_destination_points,
            ) = self.find_and_select_points_from_line(patch_type, image)

            control_points += line_control_points
            destination_points += line_destination_points
            corner_points += edge_points

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

        ordered_corner_points = MathUtils.order_four_points(
            corner_points, dtype="float32"
        )

        if config.outputs.show_image_level >= 4:
            logger.info(f"corner_points={ordered_corner_points}")
            cv2.drawContours(
                self.debug_image,
                [np.intp(ordered_corner_points)],
                -1,
                (200, 200, 200),
                2,
            )

            for corner in ordered_corner_points:
                ImageUtils.draw_box(
                    self.debug_image,
                    corner,
                    [2, 2],
                    color=CLR_DARK_GREEN,
                    border=2,
                )

        # return ordered_corner_points, edge_contours_map
        return ordered_corner_points, control_points, destination_points

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

    def find_and_select_point_from_dot(self, image, patch_type, file_path):
        options = self.options
        logger.info(f"options={options}")

        # Note: dot_description is computed at runtime(e.g. for CropOnMarkers with default quadrants)
        dot_rect, dot_description = self.find_dot_corners_from_options(
            image, patch_type, file_path
        )

        points_selector = self.default_points_selector[patch_type]
        points_selector = dot_description.get("pointsSelector", points_selector)

        dot_point = self.select_point_from_rectangle(dot_rect, points_selector)

        destination_rect = self.compute_scan_area_destination_rect(dot_description)
        destination_point = self.select_point_from_rectangle(
            destination_rect, points_selector
        )

        return dot_point, destination_point

    @staticmethod
    def select_point_from_rectangle(rectangle, points_selector):
        tl, tr, br, bl = rectangle
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

    def compute_scan_area_destination_rect(self, area_description):
        x, y = area_description["origin"]
        w, h = area_description["dimensions"]
        return MathUtils.get_rectangle_points(x, y, w, h)

    def compute_scan_area_util(self, image, area_description):
        logger.info(f"area_description={area_description}")
        # parse arguments
        h, w = image.shape[:2]
        origin, dimensions, margins = map(
            area_description.get, ["origin", "dimensions", "margins"]
        )

        # compute area and clip to image dimensions
        area_start = [
            max(0, int(origin[0] - margins["horizontal"])),
            max(0, int(origin[1] - margins["vertical"])),
        ]
        area_end = [
            min(w, int(origin[0] + margins["horizontal"] + dimensions[0])),
            min(h, int(origin[1] + margins["vertical"] + dimensions[1])),
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
            ImageUtils.draw_box_diagonal(
                self.debug_image,
                area_start,
                area_end,
                color=CLR_DARK_GREEN,
                border=2,
            )
        return area, np.array(area_start)
