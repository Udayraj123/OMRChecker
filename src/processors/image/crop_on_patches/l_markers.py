from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from src.processors.constants import (
    L_MARKER_ZONE_TYPES_IN_ORDER,
    ScannerType,
    SelectorType,
    WarpMethod,
    ZonePreset,
)
from src.processors.image.crop_on_patches.common import CropOnPatchesCommon
from src.processors.image.crop_on_patches.l_marker_detection import (
    detect_l_marker_in_patch,
)
from src.processors.image.crop_on_patches.patch_utils import compute_scan_zone
from src.utils.drawing import DrawingUtils
from src.utils.exceptions import ImageProcessingError
from src.utils.json_conversion import camel_to_snake
from src.utils.logger import logger
from src.utils.parsing import OVERRIDE_MERGER


class CropOnLMarkers(CropOnPatchesCommon):
    """
    Crop OMR sheets using L-shaped corner markers.

    Detects the inner right-angle corner of each L-marker using
    morphology + Canny + convexity defects, then warps the page
    to a rectangular bounding box of all detected corners.
    """

    __is_internal_preprocessor__: ClassVar[bool] = True

    # Register PATCH_L_MARKER as a dot-like scanner (single point per zone)
    dot_like_scanner_types: ClassVar[frozenset] = frozenset(
        {ScannerType.PATCH_L_MARKER}
    )

    scan_zone_presets_for_layout: ClassVar[dict[str, list[Any]]] = {
        "L_MARKERS": L_MARKER_ZONE_TYPES_IN_ORDER,
        "CUSTOM": [],
    }

    default_scan_zone_descriptions: ClassVar[dict[str, Any]] = {
        **{
            zone_preset: {
                "scanner_type": ScannerType.PATCH_L_MARKER,
                "max_points": 1,
            }
            for zone_preset in L_MARKER_ZONE_TYPES_IN_ORDER
        },
        "CUSTOM": {},
    }

    # Maps each L-marker zone to the inner corner selector
    default_points_selector_map: ClassVar = {
        "CENTERS": {
            ZonePreset.topLeftLMarker: SelectorType.L_INNER_CORNER,
            ZonePreset.topRightLMarker: SelectorType.L_INNER_CORNER,
            ZonePreset.bottomRightLMarker: SelectorType.L_INNER_CORNER,
            ZonePreset.bottomLeftLMarker: SelectorType.L_INNER_CORNER,
        },
    }

    def __init__(self, options, *args, **kwargs) -> None:
        super().__init__(options, *args, **kwargs)
        tuning_options = self.tuning_options
        self.morph_kernel_size = tuple(tuning_options.get("morph_kernel_size", [5, 5]))
        self.morph_iterations = int(tuning_options.get("morph_iterations", 2))
        self.min_marker_area = float(tuning_options.get("min_marker_area", 500.0))
        self.max_marker_area = float(tuning_options.get("max_marker_area", 50000.0))

    def validate_and_remap_options_schema(self, options):
        layout_type = options["type"]
        parsed_options = self._build_base_parsed_options(
            options,
            layout_type,
            enable_cropping=True,
            default_warp_method=WarpMethod.HOMOGRAPHY,
        )

        # Build scan zones from zone preset keys in options
        parsed_scan_zones = []
        for zone_preset in CropOnLMarkers.scan_zone_presets_for_layout.get(
            layout_type, []
        ):
            local_description = options.get(camel_to_snake(zone_preset), {})
            parsed_scan_zones.append(
                {
                    "zone_preset": zone_preset,
                    "zone_description": local_description,
                    "custom_options": {},
                }
            )
        parsed_options["scan_zones"] = parsed_scan_zones
        return parsed_options

    def prepare_image_before_extraction(self, image):
        # Per-patch morphology is done inside detect_l_marker_in_patch.
        # No whole-image preprocessing needed.
        return image

    def get_runtime_zone_description_with_defaults(self, image, scan_zone):
        zone_preset = scan_zone["zone_preset"]
        zone_description = scan_zone["zone_description"]

        if zone_preset not in self.scan_zone_presets_for_layout.get("L_MARKERS", []):
            return zone_description

        origin = zone_description.get("origin")
        dimensions = zone_description.get("dimensions")

        if origin is None or dimensions is None:
            # Auto-assign quadrant zone based on the image dimensions
            h, w = image.shape[:2]
            half_h, half_w = h // 2, w // 2

            search_h = 80  # default search region size when not specified
            search_w = 80

            if zone_preset == ZonePreset.topLeftLMarker:
                zone_start, zone_end = [1, 1], [half_w, half_h]
            elif zone_preset == ZonePreset.topRightLMarker:
                zone_start, zone_end = [half_w, 1], [w, half_h]
            elif zone_preset == ZonePreset.bottomRightLMarker:
                zone_start, zone_end = [half_w, half_h], [w, h]
            elif zone_preset == ZonePreset.bottomLeftLMarker:
                zone_start, zone_end = [1, half_h], [half_w, h]
            else:
                return zone_description

            computed_origin = [
                (zone_start[0] + zone_end[0] - search_w) // 2,
                (zone_start[1] + zone_end[1] - search_h) // 2,
            ]
            margin_horizontal = (zone_end[0] - zone_start[0] - search_w) / 2 - 1
            margin_vertical = (zone_end[1] - zone_start[1] - search_h) / 2 - 1

            computed = {
                "origin": computed_origin,
                "dimensions": [search_w, search_h],
                "margins": {
                    "top": margin_vertical,
                    "right": margin_horizontal,
                    "bottom": margin_vertical,
                    "left": margin_horizontal,
                },
                "scanner_type": ScannerType.PATCH_L_MARKER,
            }
            zone_description = OVERRIDE_MERGER.merge(computed, zone_description)

        return zone_description

    def find_dot_corners_from_options(self, image, zone_description, file_path):
        config = self.tuning_config
        zone_label = zone_description["label"]

        patch, zone_start, _ = compute_scan_zone(image, zone_description)

        tuning = {
            "morph_kernel_size": list(self.morph_kernel_size),
            "morph_iterations": self.morph_iterations,
            "min_marker_area": self.min_marker_area,
            "max_marker_area": self.max_marker_area,
        }

        corner = detect_l_marker_in_patch(
            patch, zone_offset=zone_start, tuning_options=tuning
        )

        if corner is None:
            msg = f"No L-marker found in patch {zone_label}"
            raise ImageProcessingError(
                msg,
                context={
                    "zone_label": zone_label,
                    "origin": zone_description.get("origin"),
                },
            )

        logger.debug(f"L-marker corner for {zone_label}: {corner}")

        if config.outputs.show_image_level >= 1:
            DrawingUtils.draw_contour(
                self.debug_image, np.intp(corner).reshape(1, 1, 2)
            )

        # Return as a degenerate 4-point rectangle where all corners are the
        # detected inner corner. select_point_from_rectangle unpacks 4 points
        # (tl, tr, br, bl); with L_INNER_CORNER selector it returns tl = corner.
        corner_pt = corner.tolist() if hasattr(corner, "tolist") else list(corner)
        return [corner_pt, corner_pt, corner_pt, corner_pt]

    def exclude_files(self) -> list[Path]:
        return []

    def find_and_select_points_from_line(self, *args, **kwargs):
        msg = "CropOnLMarkers does not support PATCH_LINE scanner type"
        raise NotImplementedError(msg)
