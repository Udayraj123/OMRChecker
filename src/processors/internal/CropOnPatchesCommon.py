from copy import deepcopy

import cv2
import numpy as np

from src.processors.constants import (
    EDGE_TYPES_IN_ORDER,
    TARGET_ENDPOINTS_FOR_EDGES,
    EdgeType,
    ScannerType,
    SelectorType,
    WarpMethod,
)
from src.processors.internal.WarpOnPointsCommon import WarpOnPointsCommon
from src.utils.constants import CLR_DARK_GREEN
from src.utils.drawing import DrawingUtils
from src.utils.logger import logger
from src.utils.math import MathUtils
from src.utils.parsing import OVERRIDE_MERGER
from src.utils.shapes import ShapeUtils


class CropOnPatchesCommon(WarpOnPointsCommon):
    __is_internal_preprocessor__ = True
    default_scan_zone_descriptions = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parse_and_apply_scan_zone_presets_and_defaults()
        self.validate_scan_zones()
        self.validate_points_layouts()

        options = self.options
        # Default to select centers for roi
        self.default_points_selector = self.default_points_selector_map[
            options.get("defaultSelector")
        ]

    def exclude_files(self):
        return []

    def __str__(self):
        return f"CropOnMarkers[\"{self.options['pointsLayout']}\"]"

    def prepare_image(self, image):
        return image

    def parse_and_apply_scan_zone_presets_and_defaults(self):
        options = self.options
        scan_zones = options["scanZones"]
        scan_zones_with_defaults = []
        for scan_zone in scan_zones:
            zone_preset, zone_description, custom_options = (
                scan_zone["zonePreset"],
                scan_zone.get("zoneDescription", {}),
                scan_zone.get("customOptions", {}),
            )
            zone_description["label"] = zone_description.get("label", zone_preset)
            scan_zones_with_defaults += [
                {
                    "zonePreset": zone_preset,
                    "zoneDescription": OVERRIDE_MERGER.merge(
                        deepcopy(self.default_scan_zone_descriptions[zone_preset]),
                        zone_description,
                    ),
                    "customOptions": custom_options,
                }
            ]
        self.scan_zones = scan_zones_with_defaults
        # logger.debug(self.scan_zones)

    def validate_scan_zones(self):
        seen_labels = set()
        repeat_labels = set()
        for scan_zone in self.scan_zones:
            zone_label = scan_zone["zoneDescription"]["label"]
            if zone_label in seen_labels:
                repeat_labels.add(zone_label)
            seen_labels.add(zone_label)
        if len(repeat_labels) > 0:
            raise Exception(f"Found repeated labels in scanZones: {repeat_labels}")

    # TODO: check if this needs to move into child for working properly (accessing self attributes declared in child in parent's constructor)
    def validate_points_layouts(self):
        options = self.options
        points_layout = options["pointsLayout"]
        if (
            points_layout not in self.scan_zone_presets_for_layout
            and points_layout != "CUSTOM"
        ):
            raise Exception(
                f"Invalid pointsLayout provided: {points_layout} for {self}"
            )

        expected_templates = set(self.scan_zone_presets_for_layout[points_layout])
        provided_templates = set(
            [scan_zone["zonePreset"] for scan_zone in self.scan_zones]
        )
        not_provided_zone_presets = expected_templates.difference(provided_templates)

        if len(not_provided_zone_presets) > 0:
            logger.error(f"not_provided_zone_presets={not_provided_zone_presets}")
            raise Exception(
                f"Missing a few zonePresets for the pointsLayout {points_layout}"
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

        zone_preset_points = {}
        page_corners, destination_page_corners = [], []
        for scan_zone in self.scan_zones:
            zone_preset = scan_zone["zonePreset"]
            zone_description = self.get_runtime_zone_description_with_defaults(
                image, scan_zone
            )
            scanner_type = zone_description["scannerType"]
            # Note: zone_description is computed at runtime(e.g. for CropOnCustomMarkers with default quadrants)
            if (
                scanner_type == ScannerType.PATCH_DOT
                or scanner_type == ScannerType.TEMPLATE_MATCH
            ):
                dot_point, destination_point = self.find_and_select_point_from_dot(
                    image, zone_description, file_path
                )
                zone_control_points, zone_destination_points = [dot_point], [
                    destination_point
                ]
                zone_preset_points[zone_preset] = zone_control_points

                page_corners.append(dot_point)
                destination_page_corners.append(destination_point)

            elif scanner_type == ScannerType.PATCH_LINE:
                (
                    zone_control_points,
                    zone_destination_points,
                    selected_contour,
                ) = self.find_and_select_points_from_line(
                    image, zone_preset, zone_description, file_path
                )
                zone_preset_points[zone_preset] = selected_contour
                page_corners += [zone_control_points[0], zone_control_points[-1]]
                destination_page_corners += [
                    zone_destination_points[0],
                    zone_destination_points[-1],
                ]
            # TODO: support DASHED_LINE here later
            control_points += zone_control_points
            destination_points += zone_destination_points

            if config.outputs.show_image_level >= 4:
                self.draw_zone_contours_and_anchor_shifts(
                    zone_control_points, zone_destination_points
                )

        # Fill edge contours
        edge_contours_map = self.get_edge_contours_map_from_zone_points(
            zone_preset_points
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

    def get_runtime_zone_description_with_defaults(self, image, scan_zone):
        return scan_zone["zoneDescription"]

    def get_edge_contours_map_from_zone_points(self, zone_preset_points):
        edge_contours_map = {
            EdgeType.TOP: [],
            EdgeType.RIGHT: [],
            EdgeType.BOTTOM: [],
            EdgeType.LEFT: [],
        }

        for edge_type in EDGE_TYPES_IN_ORDER:
            for zone_preset, contour_point_index in TARGET_ENDPOINTS_FOR_EDGES[
                edge_type
            ]:
                if zone_preset in zone_preset_points:
                    zone_points = zone_preset_points[zone_preset]
                    if contour_point_index == "ALL":
                        edge_contours_map[edge_type] += zone_points
                    else:
                        edge_contours_map[edge_type].append(
                            zone_points[contour_point_index]
                        )
        return edge_contours_map

    def draw_zone_contours_and_anchor_shifts(
        self, zone_control_points, zone_destination_points
    ):
        if len(zone_control_points) > 1:
            if len(zone_control_points) == 2:
                # Draw line if it's just two points
                DrawingUtils.draw_contour(self.debug_image, zone_control_points)
            else:
                # Draw convex hull of the found control points
                DrawingUtils.draw_contour(
                    self.debug_image,
                    cv2.convexHull(np.intp(zone_control_points)),
                )

        # Helper for alignment
        DrawingUtils.draw_arrows(
            self.debug_image,
            zone_control_points,
            zone_destination_points,
            tip_length=0.4,
        )
        for control_point in zone_control_points:
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

    def find_and_select_point_from_dot(self, image, zone_description, file_path):
        zone_label = zone_description["label"]
        points_selector = zone_description.get(
            "selector",
            self.default_points_selector.get(zone_label, SelectorType.SELECT_CENTER),
        )
        dot_rect = self.find_dot_corners_from_options(
            image, zone_description, file_path
        )

        dot_point = self.select_point_from_rectangle(dot_rect, points_selector)

        if dot_point is None:
            raise Exception(f"No dot found for zone {zone_label}")

        destination_rect = np.intp(
            ShapeUtils.compute_scan_zone_rectangle(
                zone_description, include_margins=False
            )
        )
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

    def compute_scan_zone_util(self, image, zone_description):
        zone, scan_zone_rectangle = ShapeUtils.extract_image_from_zone_description(
            image, zone_description
        )

        zone_start = scan_zone_rectangle[0]
        zone_end = scan_zone_rectangle[2]

        config = self.tuning_config
        if config.outputs.show_image_level >= 1:
            DrawingUtils.draw_box_diagonal(
                self.debug_image,
                zone_start,
                zone_end,
                color=CLR_DARK_GREEN,
                border=2,
            )
        return zone, np.array(zone_start)
