import cv2
import numpy as np

from src.processors.constants import AreaTemplate, ScannerType
from src.processors.internal.WarpOnPointsCommon import WarpOnPointsCommon
from src.utils.constants import CLR_DARK_GREEN
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils


class CropOnPatchesCommon(WarpOnPointsCommon):
    __is_internal_preprocessor__ = True

    default_points_selector_map = {
        "CENTERS": {
            AreaTemplate.topLeftDot: "DOT_CENTER",
            AreaTemplate.topRightDot: "DOT_CENTER",
            AreaTemplate.bottomRightDot: "DOT_CENTER",
            AreaTemplate.bottomLeftDot: "DOT_CENTER",
            AreaTemplate.topLeftMarker: "DOT_CENTER",
            AreaTemplate.topRightMarker: "DOT_CENTER",
            AreaTemplate.bottomRightMarker: "DOT_CENTER",
            AreaTemplate.bottomLeftMarker: "DOT_CENTER",
            AreaTemplate.leftLine: "LINE_OUTER_EDGE",
            AreaTemplate.rightLine: "LINE_OUTER_EDGE",
        },
        "INNER_WIDTHS": {
            AreaTemplate.topLeftDot: "DOT_TOP_RIGHT",
            AreaTemplate.topRightDot: "DOT_TOP_LEFT",
            AreaTemplate.bottomRightDot: "DOT_BOTTOM_LEFT",
            AreaTemplate.bottomLeftDot: "DOT_BOTTOM_RIGHT",
            AreaTemplate.topLeftMarker: "DOT_TOP_RIGHT",
            AreaTemplate.topRightMarker: "DOT_TOP_LEFT",
            AreaTemplate.bottomRightMarker: "DOT_BOTTOM_LEFT",
            AreaTemplate.bottomLeftMarker: "DOT_BOTTOM_RIGHT",
            AreaTemplate.leftLine: "LINE_INNER_EDGE",
            AreaTemplate.rightLine: "LINE_INNER_EDGE",
        },
        "INNER_HEIGHTS": {
            AreaTemplate.topLeftDot: "DOT_BOTTOM_LEFT",
            AreaTemplate.topRightDot: "DOT_BOTTOM_RIGHT",
            AreaTemplate.bottomRightDot: "DOT_TOP_RIGHT",
            AreaTemplate.bottomLeftDot: "DOT_TOP_LEFT",
            AreaTemplate.topLeftMarker: "DOT_BOTTOM_LEFT",
            AreaTemplate.topRightMarker: "DOT_BOTTOM_RIGHT",
            AreaTemplate.bottomRightMarker: "DOT_TOP_RIGHT",
            AreaTemplate.bottomLeftMarker: "DOT_TOP_LEFT",
            AreaTemplate.leftLine: "LINE_OUTER_EDGE",
            AreaTemplate.rightLine: "LINE_OUTER_EDGE",
        },
        "INNER_CORNERS": {
            AreaTemplate.topLeftDot: "DOT_BOTTOM_RIGHT",
            AreaTemplate.topRightDot: "DOT_BOTTOM_LEFT",
            AreaTemplate.bottomRightDot: "DOT_TOP_LEFT",
            AreaTemplate.bottomLeftDot: "DOT_TOP_RIGHT",
            AreaTemplate.topLeftMarker: "DOT_BOTTOM_RIGHT",
            AreaTemplate.topRightMarker: "DOT_BOTTOM_LEFT",
            AreaTemplate.bottomRightMarker: "DOT_TOP_LEFT",
            AreaTemplate.bottomLeftMarker: "DOT_TOP_RIGHT",
            AreaTemplate.leftLine: "LINE_INNER_EDGE",
            AreaTemplate.rightLine: "LINE_INNER_EDGE",
        },
        "OUTER_CORNERS": {
            AreaTemplate.topLeftDot: "DOT_TOP_LEFT",
            AreaTemplate.topRightDot: "DOT_TOP_RIGHT",
            AreaTemplate.bottomRightDot: "DOT_BOTTOM_RIGHT",
            AreaTemplate.bottomLeftDot: "DOT_BOTTOM_LEFT",
            AreaTemplate.topLeftMarker: "DOT_TOP_LEFT",
            AreaTemplate.topRightMarker: "DOT_TOP_RIGHT",
            AreaTemplate.bottomRightMarker: "DOT_BOTTOM_RIGHT",
            AreaTemplate.bottomLeftMarker: "DOT_BOTTOM_LEFT",
            AreaTemplate.leftLine: "LINE_OUTER_EDGE",
            AreaTemplate.rightLine: "LINE_OUTER_EDGE",
        },
    }

    # Common code used by both types of croppers
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        options = self.options

        # Default to select centers for roi
        self.default_points_selector = self.default_points_selector_map[
            options.get("defaultSelector", "CENTERS")
        ]

    def exclude_files(self):
        return []

    def __str__(self):
        return f"CropOnMarkers[\"{self.options['type']}\"]"

    def prepare_image(self, image):
        return image

    def extract_control_destination_points(self, image, file_path):
        config = self.tuning_config

        (
            control_points,
            destination_points,
        ) = (
            [],
            [],
        )
        for scan_area in self.scan_areas:
            area_description = self.get_runtime_area_description_with_defaults(
                image, scan_area
            )
            (
                area_control_points,
                area_destination_points,
            ) = self.extract_points_from_scan_area(image, area_description, file_path)
            control_points += area_control_points
            destination_points += area_destination_points

            if config.outputs.show_image_level >= 4:
                area_label = area_description["label"]
                logger.info(f"{area_label}: area_control_points={area_control_points}")
                cv2.drawContours(
                    self.debug_image,
                    [np.intp(area_control_points)],
                    -1,
                    (200, 200, 200),
                    2,
                )

                for corner in area_control_points:
                    ImageUtils.draw_box(
                        self.debug_image,
                        corner,
                        [2, 2],
                        color=CLR_DARK_GREEN,
                        border=2,
                    )

            if config.outputs.show_image_level >= 5:
                if len(self.debug_vstack) > 0:
                    area_label, scanner_type = (
                        area_description["label"],
                        area_description["scannerType"],
                    )
                    InteractionUtils.show(
                        f"{area_label} Patches: {scanner_type}",
                        ImageUtils.get_vstack_image_grid(self.debug_vstack),
                        0,
                        0,
                        config=config,
                    )
                self.debug_vstack = []

        return control_points, destination_points

    def get_runtime_area_description_with_defaults(self, image, scan_area):
        return scan_area["areaDescription"]

    def extract_points_from_scan_area(self, image, area_description, file_path):
        scanner_type = area_description["scannerType"]

        # Note: area_description is computed at runtime(e.g. for CropOnCustomMarkers with default quadrants)
        if (
            scanner_type == ScannerType.PATCH_DOT
            or scanner_type == ScannerType.TEMPLATE_MATCH
        ):
            dot_point, destination_point = self.find_and_select_point_from_dot(
                image, area_description, file_path
            )
            area_control_points, area_destination_points = [dot_point], [
                destination_point
            ]
        elif scanner_type == ScannerType.PATCH_LINE:
            (
                line_control_points,
                line_destination_points,
            ) = self.find_and_select_points_from_line(
                image, area_description, file_path
            )

            area_control_points, area_destination_points = (
                line_control_points,
                line_destination_points,
            )
        # TODO: support DASHED_LINE here later

        return area_control_points, area_destination_points

    def find_and_select_point_from_dot(self, image, area_description, file_path):
        patch_type = area_description["scannerType"]

        dot_rect = self.find_dot_corners_from_options(
            image, area_description, file_path
        )

        default_points_selector = self.default_points_selector[patch_type]
        points_selector = area_description.get("selector", default_points_selector)

        dot_point = self.select_point_from_rectangle(dot_rect, points_selector)

        destination_rect = self.compute_scan_area_destination_rect(area_description)
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
