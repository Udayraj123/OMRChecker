from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from src.utils.exceptions import (
    ImageProcessingError,
    TemplateConfigurationError,
)
from src.processors.constants import (
    ScannerType,
    SelectorType,
)
from src.processors.image.crop_on_patches.patch_utils import (
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
from src.utils.parsing import OVERRIDE_MERGER
from src.utils.shapes import ShapeUtils


class CropOnPatchesCommon(WarpOnPointsCommon):
    __is_internal_preprocessor__ = True
    default_scan_zone_descriptions: ClassVar = {}
    # Expected to be overridden by child
    default_points_selector_map: ClassVar = {}
    scan_zone_presets_for_layout: ClassVar = {}
    dot_like_scanner_types: ClassVar[frozenset] = frozenset(
        {ScannerType.PATCH_DOT, ScannerType.TEMPLATE_MATCH}
    )

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
            options.get("default_selector")
        ]

    def _build_base_parsed_options(
        self,
        options: dict,
        layout_type: str,
        *,
        cropping_enabled: bool = True,
        default_warp_method: str,
    ) -> dict:
        """
        Build the common base structure for validate_and_remap_options_schema.
        Subclasses call this then add their scan_zones and other fields.
        """
        tuning_options = options.get("tuning_options", {})
        return {
            "default_selector": options.get("default_selector", "CENTERS"),
            "points_layout": layout_type,
            "cropping_enabled": cropping_enabled,
            "cropping_use_bounding_box": options.get("cropping_use_bounding_box", True),
            "cropping_use_approx_poly": options.get("cropping_use_approx_poly", False),
            "tuning_options": {
                "warp_method": tuning_options.get("warp_method", default_warp_method),
            },
        }

    def exclude_files(self) -> list[Path]:
        return []

    def __str__(self) -> str:
        return f'CropOnMarkers["{self.options["points_layout"]}"]'

    def prepare_image_before_extraction(self, image):
        return image

    def parse_and_apply_scan_zone_presets_and_defaults(self) -> None:
        options = self.options
        scan_zones = options["scan_zones"]
        scan_zones_with_defaults = []
        for scan_zone in scan_zones:
            zone_preset, zone_description, custom_options = (
                scan_zone["zone_preset"],
                scan_zone.get("zone_description", {}),
                scan_zone.get("custom_options", {}),
            )
            zone_description["label"] = zone_description.get("label", zone_preset)
            scan_zones_with_defaults += [
                {
                    "zone_preset": zone_preset,
                    "zone_description": OVERRIDE_MERGER.merge(
                        deepcopy(self.default_scan_zone_descriptions[zone_preset]),
                        zone_description,
                    ),
                    "custom_options": custom_options,
                }
            ]
        self.scan_zones = scan_zones_with_defaults
        # logger.debug(self.scan_zones)

    def validate_scan_zones(self) -> None:
        seen_labels = set()
        repeat_labels = set()
        for scan_zone in self.scan_zones:
            zone_label = scan_zone["zone_description"]["label"]
            if zone_label in seen_labels:
                repeat_labels.add(zone_label)
            seen_labels.add(zone_label)
        if len(repeat_labels) > 0:
            msg = f"Found repeated labels in scanZones: {repeat_labels}"
            raise TemplateConfigurationError(
                msg,
                repeat_labels=list(repeat_labels),
            )

    # TODO: check if this needs to move into child for working properly (accessing self attributes declared in child in parent's constructor)
    def validate_points_layouts(self) -> None:
        options = self.options
        points_layout = options["points_layout"]
        if (
            points_layout not in self.scan_zone_presets_for_layout
            and points_layout != "CUSTOM"
        ):
            msg = f"Invalid pointsLayout provided: {points_layout} for {self}"
            raise TemplateConfigurationError(
                msg,
                points_layout=points_layout,
            )

        expected_templates = set(self.scan_zone_presets_for_layout[points_layout])
        provided_templates = {scan_zone["zone_preset"] for scan_zone in self.scan_zones}
        not_provided_zone_presets = expected_templates.difference(provided_templates)

        if len(not_provided_zone_presets) > 0:
            logger.error(f"not_provided_zone_presets={not_provided_zone_presets}")
            msg = f"Missing a few zonePresets for the pointsLayout {points_layout}"
            raise TemplateConfigurationError(
                msg,
                points_layout=points_layout,
                not_provided_zone_presets=list(not_provided_zone_presets),
            )

    def extract_control_destination_points(self, image, _colored_image, file_path):
        config = self.tuning_config

        control_points, destination_points = [], []

        # Inject runtime zone description
        for scan_zone in self.scan_zones:
            # Note: zone_description is computed at runtime(e.g. for CropOnCustomMarkers with default quadrants)
            zone_description = self.get_runtime_zone_description_with_defaults(
                image, scan_zone
            )
            scan_zone["runtime_zone_description"] = zone_description

            if config.outputs.show_image_level >= 1:
                draw_scan_zone(self.debug_image, zone_description)

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
        # TODO: have a better control over ordering in the scan zones for getting a contiguous contour
        for scan_zone in self.scan_zones:
            zone_preset = scan_zone["zone_preset"]
            zone_description = scan_zone["runtime_zone_description"]
            scanner_type = zone_description["scanner_type"]

            if scanner_type in self.dot_like_scanner_types:
                dot_point, destination_point = self.find_and_select_point_from_dot(
                    image, zone_description, file_path
                )
                zone_control_points, zone_destination_points = (
                    [dot_point],
                    [destination_point],
                )
                zone_preset_points[zone_preset] = zone_control_points

            elif scanner_type == ScannerType.PATCH_LINE:
                (
                    zone_control_points,
                    zone_destination_points,
                    selected_contour,
                ) = self.find_and_select_points_from_line(
                    image, zone_preset, zone_description, file_path
                )
                zone_preset_points[zone_preset] = selected_contour
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
        # TODO: sort edge_contours_map manually?
        edge_contours_map = get_edge_contours_map_from_zone_points(zone_preset_points)
        return control_points, destination_points, edge_contours_map

    def get_runtime_zone_description_with_defaults(self, _image, scan_zone):
        return scan_zone["zone_description"]

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
