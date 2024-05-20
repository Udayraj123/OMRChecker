from copy import deepcopy

import cv2
import numpy as np

from src.processors.constants import (
    EDGE_TYPES_IN_ORDER,
    TARGET_ENDPOINTS_FOR_EDGES,
    EdgeType,
    ScannerType,
    WarpMethod,
)
from src.processors.internal.WarpOnPointsCommon import WarpOnPointsCommon
from src.utils.constants import CLR_DARK_GREEN
from src.utils.drawing import DrawingUtils
from src.utils.logger import logger
from src.utils.math import MathUtils
from src.utils.parsing import OVERRIDE_MERGER


class CropOnPatchesCommon(WarpOnPointsCommon):
    __is_internal_preprocessor__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parse_and_apply_scan_area_templates_and_defaults()
        self.validate_scan_areas()
        self.validate_points_layouts()

        options = self.options
        # Default to select centers for roi
        self.default_points_selector = self.default_points_selector_map[
            options.get("defaultSelector", "CENTERS")
        ]

    def exclude_files(self):
        return []

    def __str__(self):
        return f"CropOnMarkers[\"{self.options['pointsLayout']}\"]"

    def prepare_image(self, image):
        return image

    def parse_and_apply_scan_area_templates_and_defaults(self):
        options = self.options
        scan_areas = options["scanAreas"]
        scan_areas_with_defaults = []
        for scan_area in scan_areas:
            area_template, area_description, custom_options = (
                scan_area["areaTemplate"],
                scan_area.get("areaDescription", {}),
                scan_area.get("customOptions", {}),
            )
            area_description["label"] = area_description.get("label", area_template)
            scan_areas_with_defaults += [
                {
                    "areaTemplate": area_template,
                    "areaDescription": OVERRIDE_MERGER.merge(
                        deepcopy(self.default_scan_area_descriptions[area_template]),
                        area_description,
                    ),
                    "customOptions": custom_options,
                }
            ]
        self.scan_areas = scan_areas_with_defaults
        # logger.debug(self.scan_areas)

    def validate_scan_areas(self):
        seen_labels = set()
        repeat_labels = set()
        for scan_area in self.scan_areas:
            area_label = scan_area["areaDescription"]["label"]
            if area_label in seen_labels:
                repeat_labels.add(area_label)
            seen_labels.add(area_label)
        if len(repeat_labels) > 0:
            raise Exception(f"Found repeated labels in scanAreas: {repeat_labels}")

    # TODO: check if this needs to move into child for working properly (accessing self attributes declared in child in parent's constructor)
    def validate_points_layouts(self):
        options = self.options
        points_layout = options["pointsLayout"]
        if (
            points_layout not in self.scan_area_templates_for_layout
            and points_layout != "CUSTOM"
        ):
            raise Exception(
                f"Invalid pointsLayout provided: {points_layout} for {self}"
            )

        expected_templates = set(self.scan_area_templates_for_layout[points_layout])
        provided_templates = set(
            [scan_area["areaTemplate"] for scan_area in self.scan_areas]
        )
        not_provided_area_templates = expected_templates.difference(provided_templates)

        if len(not_provided_area_templates) > 0:
            logger.error(f"not_provided_area_templates={not_provided_area_templates}")
            raise Exception(
                f"Missing a few scanAreaTemplates for the pointsLayout {points_layout}"
            )

    def extract_control_destination_points(self, image, _colored_image, file_path):
        config = self.tuning_config

        (
            control_points,
            destination_points,
        ) = (
            [],
            [],
        )

        # TODO: use shapely and corner points to split easily?

        area_template_points = {}
        page_corners, destination_page_corners = [], []
        for scan_area in self.scan_areas:
            area_template = scan_area["areaTemplate"]
            area_description = self.get_runtime_area_description_with_defaults(
                image, scan_area
            )
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
                area_template_points[area_template] = area_control_points

                page_corners.append(dot_point)
                destination_page_corners.append(destination_point)

            elif scanner_type == ScannerType.PATCH_LINE:
                (
                    area_control_points,
                    area_destination_points,
                    selected_contour,
                ) = self.find_and_select_points_from_line(
                    image, area_template, area_description, file_path
                )
                area_template_points[area_template] = selected_contour
                page_corners += [area_control_points[0], area_control_points[-1]]
                destination_page_corners += [
                    area_destination_points[0],
                    area_destination_points[-1],
                ]
            # TODO: support DASHED_LINE here later
            control_points += area_control_points
            destination_points += area_destination_points

            if config.outputs.show_image_level >= 4:
                self.draw_area_contours_and_anchor_shifts(
                    area_control_points, area_destination_points
                )

        # Fill edge contours
        edge_contours_map = self.get_edge_contours_map_from_area_points(
            area_template_points
        )

        if self.warp_method in [
            WarpMethod.PERSPECTIVE_TRANSFORM,
        ]:
            ordered_page_corners, ordered_indices = MathUtils.order_four_points(
                page_corners, dtype="float32"
            )
            destination_page_corners = [
                destination_page_corners[i] for i in ordered_indices
            ]
            return ordered_page_corners, destination_page_corners, edge_contours_map

        # TODO: sort edge_contours_map manually?
        return control_points, destination_points, edge_contours_map

    def get_runtime_area_description_with_defaults(self, image, scan_area):
        return scan_area["areaDescription"]

    def get_edge_contours_map_from_area_points(self, area_template_points):
        edge_contours_map = {
            EdgeType.TOP: [],
            EdgeType.RIGHT: [],
            EdgeType.BOTTOM: [],
            EdgeType.LEFT: [],
        }

        for edge_type in EDGE_TYPES_IN_ORDER:
            for area_template, contour_point_index in TARGET_ENDPOINTS_FOR_EDGES[
                edge_type
            ]:
                if area_template in area_template_points:
                    area_points = area_template_points[area_template]
                    if contour_point_index == "ALL":
                        edge_contours_map[edge_type] += area_points
                    else:
                        edge_contours_map[edge_type].append(
                            area_points[contour_point_index]
                        )
        return edge_contours_map

    def draw_area_contours_and_anchor_shifts(
        self, area_control_points, area_destination_points
    ):
        if len(area_control_points) > 1:
            if len(area_control_points) == 2:
                # Draw line if it's just two points
                DrawingUtils.draw_contour(self.debug_image, area_control_points)
            else:
                # Draw convex hull of the found control points
                DrawingUtils.draw_contour(
                    self.debug_image,
                    cv2.convexHull(np.intp(area_control_points)),
                )

        # Helper for alignment
        DrawingUtils.draw_arrows(
            self.debug_image,
            area_control_points,
            area_destination_points,
            tip_length=0.4,
        )
        for control_point in area_control_points:
            # Show current detections too
            DrawingUtils.draw_box(
                self.debug_image,
                control_point,
                # TODO: change this based on image shape
                [20, 20],
                color=CLR_DARK_GREEN,
                border=1,
                centered=True,
            )

    def find_and_select_point_from_dot(self, image, area_description, file_path):
        area_label = area_description["label"]
        points_selector = area_description.get(
            "selector", self.default_points_selector.get(area_label, None)
        )

        dot_rect = self.find_dot_corners_from_options(
            image, area_description, file_path
        )

        dot_point = self.select_point_from_rectangle(dot_rect, points_selector)

        if dot_point is None:
            raise Exception(f"No dot found for area {area_label}")

        destination_rect = self.compute_scan_area_destination_rect(area_description)
        destination_point = self.select_point_from_rectangle(
            destination_rect, points_selector
        )

        return dot_point, destination_point

    @staticmethod
    def select_point_from_rectangle(rectangle, points_selector):
        tl, tr, br, bl = rectangle
        if points_selector == "SELECT_TOP_LEFT":
            return tl
        if points_selector == "SELECT_TOP_RIGHT":
            return tr
        if points_selector == "SELECT_BOTTOM_RIGHT":
            return br
        if points_selector == "SELECT_BOTTOM_LEFT":
            return bl
        if points_selector == "SELECT_CENTER":
            return [
                (tl[0] + br[0]) // 2,
                (tl[1] + br[1]) // 2,
            ]
        return None

    @staticmethod
    def compute_scan_area_destination_rect(area_description):
        x, y = area_description["origin"]
        w, h = area_description["dimensions"]
        return np.intp(MathUtils.get_rectangle_points(x, y, w, h))

    def compute_scan_area_util(self, image, area_description):
        area_label = area_description["label"]
        # parse arguments
        h, w = image.shape[:2]
        origin, dimensions, margins = map(
            area_description.get, ["origin", "dimensions", "margins"]
        )
        # TODO: check bug in margins for scan area

        # compute area and clip to image dimensions
        area_start = [
            int(origin[0] - margins["left"]),
            int(origin[1] - margins["top"]),
        ]
        area_end = [
            int(origin[0] + margins["right"] + dimensions[0]),
            int(origin[1] + margins["bottom"] + dimensions[1]),
        ]

        if area_start[0] < 0 or area_start[1] < 0 or area_end[0] > w or area_end[1] > h:
            logger.warning(
                f"Clipping label {area_label} with scan rectangle: {[area_start, area_end]} to image boundary {[w, h]}."
            )

            area_start = [max(0, area_start[0]), max(0, area_start[1])]
            area_end = [min(w, area_end[0]), min(h, area_end[1])]

        # Extract image area
        area = image[area_start[1] : area_end[1], area_start[0] : area_end[0]]

        config = self.tuning_config
        if config.outputs.show_image_level >= 1:
            DrawingUtils.draw_box_diagonal(
                self.debug_image,
                area_start,
                area_end,
                color=CLR_DARK_GREEN,
                border=2,
            )
        return area, np.array(area_start)
