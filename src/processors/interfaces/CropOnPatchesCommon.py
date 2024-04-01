import cv2
import numpy as np

from src.processors.constants import EDGE_TYPES_IN_ORDER
from src.processors.interfaces.CropOnIndexPoints import CropOnIndexPoints
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils


class CropOnPatchesCommon(CropOnIndexPoints):
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

        selectors = self.patch_types_for_layout[options["type"]]
        patch_selectors, edge_selectors = (
            selectors["patch_selectors"],
            selectors["edge_selectors"],
        )

        points_and_edges, corner_points = {}, []
        for patch_type in patch_selectors["DOTS"]:
            dot_point = self.select_point_from_dot(patch_type, image, file_path)

            points_and_edges[patch_type] = dot_point
            corner_points.append(dot_point)

        for patch_type in patch_selectors["LINES"]:
            edge_points, edge_contour = self.select_points_and_edges_from_line(
                patch_type, image
            )
            points_and_edges[patch_type] = [edge_points, edge_contour]
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

        # First element of each contour should necessarily start & end with a corner point
        edge_contours_map = {}
        for edge_type in EDGE_TYPES_IN_ORDER:
            edge_contours_map[edge_type] = []
            logger.info(f"{edge_type}: {edge_selectors[edge_type]}")
            for selector in edge_selectors[edge_type]:
                logger.info(f"selector={selector}, points_and_edges={points_and_edges}")
                patch_type, selection_type = (
                    selector["patch_type"],
                    selector["selection_type"],
                )
                if selection_type == "DOT_PICK_POINT":
                    dot_point = points_and_edges[patch_type]
                    edge_contours_map[edge_type].append(dot_point)
                else:
                    edge_points, edge_contour = points_and_edges[patch_type]
                    if selection_type == "LINE_PICK_FIRST_POINT":
                        edge_contours_map[edge_type].append(edge_points[0])
                    if selection_type == "LINE_PICK_LAST_POINT":
                        edge_contours_map[edge_type].append(edge_points[-1])
                    if selection_type == "LINE_PICK_CONTOUR":
                        edge_contours_map[edge_type] += edge_contour

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
                cv2.rectangle(
                    self.debug_image,
                    tuple(corner),
                    (corner[0] + 2, corner[1] + 2),
                    (20, 255, 20),
                    2,
                )

        return ordered_corner_points, edge_contours_map

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
