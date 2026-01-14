from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from src.exceptions import ImageProcessingError, TemplateValidationError
from src.processors.constants import (
    ScannerType,
    SelectorType,
    WarpMethod,
)
from src.processors.image.patch_utils import (
    compute_scan_zone,
    draw_scan_zone,
    draw_zone_contours_and_anchor_shifts,
    get_edge_contours_map_from_zone_points,
    select_point_from_rectangle,
)
from src.processors.image.WarpOnPointsCommon import WarpOnPointsCommon
from src.utils.constants import OUTPUT_MODES
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils
from src.utils.parsing import OVERRIDE_MERGER
from src.utils.shapes import ShapeUtils


class CropOnPatchesCommon(WarpOnPointsCommon):
    __is_internal_preprocessor__ = True
    default_scan_zone_descriptions: ClassVar = {}
    # Expected to be overridden by child
    default_points_selector_map: ClassVar = {}
    scan_zone_presets_for_layout: ClassVar = {}

    def find_and_select_points_from_line(
        self, _image, _zone_preset, _zone_description, _file_path
    ) -> tuple[Any, Any, Any]:
        msg = "Subclass must implement find_line_corners_from_options"
        raise NotImplementedError(msg)

    def find_dot_corners_from_options(
        self, _image, _zone_description, _file_path
    ) -> tuple[Any, Any]:
        msg = "Subclass must implement find_dot_corners_from_options"
        raise NotImplementedError(msg)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parse_and_apply_scan_zone_presets_and_defaults()
        self.validate_scan_zones()
        self.validate_points_layouts()

        options = self.options
        # Default to select centers for roi
        self.default_points_selector = self.default_points_selector_map[
            options.get("defaultSelector")
        ]

    def exclude_files(self) -> list[Path]:
        return []

    def __str__(self) -> str:
        return f'CropOnMarkers["{self.options["pointsLayout"]}"]'

    def prepare_image_before_extraction(self, image):
        return image

    def parse_and_apply_scan_zone_presets_and_defaults(self) -> None:
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

    def validate_scan_zones(self) -> None:
        seen_labels = set()
        repeat_labels = set()
        for scan_zone in self.scan_zones:
            zone_label = scan_zone["zoneDescription"]["label"]
            if zone_label in seen_labels:
                repeat_labels.add(zone_label)
            seen_labels.add(zone_label)
        if len(repeat_labels) > 0:
            msg = f"Found repeated labels in scanZones: {repeat_labels}"
            raise TemplateValidationError(
                msg,
                context={"repeat_labels": list(repeat_labels)},
            )

    # TODO: check if this needs to move into child for working properly (accessing self attributes declared in child in parent's constructor)
    def validate_points_layouts(self) -> None:
        options = self.options
        points_layout = options["pointsLayout"]
        if (
            points_layout not in self.scan_zone_presets_for_layout
            and points_layout != "CUSTOM"
        ):
            msg = f"Invalid pointsLayout provided: {points_layout} for {self}"
            raise TemplateValidationError(
                msg,
                context={"points_layout": points_layout},
            )

        expected_templates = set(self.scan_zone_presets_for_layout[points_layout])
        provided_templates = {scan_zone["zonePreset"] for scan_zone in self.scan_zones}
        not_provided_zone_presets = expected_templates.difference(provided_templates)

        if len(not_provided_zone_presets) > 0:
            logger.error(f"not_provided_zone_presets={not_provided_zone_presets}")
            msg = f"Missing a few zonePresets for the pointsLayout {points_layout}"
            raise TemplateValidationError(
                msg,
                context={
                    "points_layout": points_layout,
                    "not_provided_zone_presets": list(not_provided_zone_presets),
                },
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

        for scan_zone in self.scan_zones:
            # Note: zone_description is computed at runtime(e.g. for CropOnCustomMarkers with default quadrants)
            zone_description = self.get_runtime_zone_description_with_defaults(
                image, scan_zone
            )
            # Inject runtime zone description
            scan_zone["runtimeZoneDescription"] = zone_description

            self.draw_scan_zone(zone_description)

        if (
            config.outputs.show_image_level >= 4
            or config.outputs.output_mode == OUTPUT_MODES.SET_LAYOUT
        ):
            # TODO: move this before detection part to avoid seeing errors
            InteractionUtils.show(
                f"Control Zones in the debug image: {file_path}",
                self.debug_image,
                pause=1,
            )

        # TODO: use shapely and corner points to split easily?
        zone_preset_points = {}
        page_corners, destination_page_corners = [], []
        for scan_zone in self.scan_zones:
            zone_preset = scan_zone["zonePreset"]
            zone_description = scan_zone["runtimeZoneDescription"]
            scanner_type = zone_description["scannerType"]

            if scanner_type in {ScannerType.PATCH_DOT, ScannerType.TEMPLATE_MATCH}:
                dot_point, destination_point = self.find_and_select_point_from_dot(
                    image, zone_description, file_path
                )
                zone_control_points, zone_destination_points = (
                    [dot_point],
                    [destination_point],
                )
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
                draw_zone_contours_and_anchor_shifts(
                    self.debug_image, zone_control_points, zone_destination_points
                )

        if len(self.debug_hstack) > 0 and config.outputs.show_image_level >= 5:
            InteractionUtils.show(
                f"Zones debug stack of the image: {file_path}",
                ImageUtils.get_padded_hstack(self.debug_hstack),
                pause=1,
            )

        # Fill edge contours
        edge_contours_map = get_edge_contours_map_from_zone_points(zone_preset_points)

        if self.warp_method == WarpMethod.PERSPECTIVE_TRANSFORM:
            ordered_page_corners, ordered_indices = MathUtils.order_four_points(
                page_corners, dtype="float32"
            )
            destination_page_corners = [
                destination_page_corners[i] for i in ordered_indices
            ]
            return ordered_page_corners, destination_page_corners, edge_contours_map

        # TODO: sort edge_contours_map manually?
        return control_points, destination_points, edge_contours_map

    def get_runtime_zone_description_with_defaults(self, _image, scan_zone):
        return scan_zone["zoneDescription"]

    def find_and_select_point_from_dot(self, image, zone_description, file_path):
        zone_label = zone_description["label"]
        points_selector = zone_description.get(
            "selector",
            self.default_points_selector.get(zone_label, SelectorType.SELECT_CENTER),
        )
        dot_rect = self.find_dot_corners_from_options(
            image, zone_description, file_path
        )

        dot_point = select_point_from_rectangle(dot_rect, points_selector)

        if dot_point is None:
            msg = f"No dot found for zone {zone_label}"
            raise ImageProcessingError(
                msg,
                context={"zone_label": zone_label},
            )

        destination_rect = np.intp(
            ShapeUtils.compute_scan_zone_rectangle(
                zone_description, include_margins=False
            )
        )
        destination_point = select_point_from_rectangle(
            destination_rect, points_selector
        )

        return dot_point, destination_point

    def compute_scan_zone_util(self, image, zone_description):
        """
        Extract image zone and compute zone boundaries.

        Delegates to patch_utils.compute_scan_zone for the core logic.
        Kept for backward compatibility with existing code.
        """
        return compute_scan_zone(image, zone_description)

    def draw_scan_zone(self, zone_description):
        """
        Draw scan zone boundaries on debug image.

        Delegates to patch_utils.draw_scan_zone for the core logic.
        """
        config = self.tuning_config
        if config.outputs.show_image_level >= 1:
            draw_scan_zone(self.debug_image, zone_description)
