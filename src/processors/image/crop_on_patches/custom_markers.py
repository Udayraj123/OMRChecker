from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np

from src.processors.image.crop_on_patches.patch_utils import compute_scan_zone
from src.utils.exceptions import (
    ImageProcessingError,
    ImageReadError,
    TemplateValidationError,
)
from src.processors.constants import (
    MARKER_ZONE_TYPES_IN_ORDER,
    ScannerType,
    WarpMethod,
    ZonePreset,
)
from src.processors.image.crop_on_patches.common import CropOnPatchesCommon
from src.processors.image.crop_on_patches.marker_detection import (
    prepare_marker_template,
    detect_marker_in_patch,
)

from src.utils.drawing import DrawingUtils
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.json_conversion import camel_to_snake
from src.utils.parsing import OVERRIDE_MERGER


# TODO: add support for showing patch zone centers during setLayout option?!
class CropOnCustomMarkers(CropOnPatchesCommon):
    __is_internal_preprocessor__: ClassVar[bool] = True
    scan_zone_presets_for_layout: ClassVar[dict[str, list[Any]]] = {
        "FOUR_MARKERS": MARKER_ZONE_TYPES_IN_ORDER,
    }
    default_scan_zone_descriptions: ClassVar[dict[str, Any]] = {
        **{
            zone_preset: {
                "scanner_type": ScannerType.TEMPLATE_MATCH,
                # "selector": "SELECT_CENTER",
                "max_points": 2,  # for cropping
                # Note: all 4 margins are a required property for a patch zone
            }
            for zone_preset in MARKER_ZONE_TYPES_IN_ORDER
        },
        "CUSTOM": {},
    }

    default_points_selector_map: ClassVar = {
        "CENTERS": {
            ZonePreset.topLeftMarker: "SELECT_CENTER",
            ZonePreset.topRightMarker: "SELECT_CENTER",
            ZonePreset.bottomRightMarker: "SELECT_CENTER",
            ZonePreset.bottomLeftMarker: "SELECT_CENTER",
        },
        "INNER_WIDTHS": {
            ZonePreset.topLeftMarker: "SELECT_TOP_RIGHT",
            ZonePreset.topRightMarker: "SELECT_TOP_LEFT",
            ZonePreset.bottomRightMarker: "SELECT_BOTTOM_LEFT",
            ZonePreset.bottomLeftMarker: "SELECT_BOTTOM_RIGHT",
        },
        "INNER_HEIGHTS": {
            ZonePreset.topLeftMarker: "SELECT_BOTTOM_LEFT",
            ZonePreset.topRightMarker: "SELECT_BOTTOM_RIGHT",
            ZonePreset.bottomRightMarker: "SELECT_TOP_RIGHT",
            ZonePreset.bottomLeftMarker: "SELECT_TOP_LEFT",
        },
        "INNER_CORNERS": {
            ZonePreset.topLeftMarker: "SELECT_BOTTOM_RIGHT",
            ZonePreset.topRightMarker: "SELECT_BOTTOM_LEFT",
            ZonePreset.bottomRightMarker: "SELECT_TOP_LEFT",
            ZonePreset.bottomLeftMarker: "SELECT_TOP_RIGHT",
        },
        "OUTER_CORNERS": {
            ZonePreset.topLeftMarker: "SELECT_TOP_LEFT",
            ZonePreset.topRightMarker: "SELECT_TOP_RIGHT",
            ZonePreset.bottomRightMarker: "SELECT_BOTTOM_RIGHT",
            ZonePreset.bottomLeftMarker: "SELECT_BOTTOM_LEFT",
        },
    }

    def __init__(self, options, *args, **kwargs) -> None:
        # Parent's __init__ will call validate_and_remap_options_schema via polymorphism
        super().__init__(options, *args, **kwargs)
        tuning_options = self.tuning_options
        self.threshold_circles = []

        # TODO: dedicated marker scanZone override support needed for these?
        self.min_matching_threshold = tuning_options.get("min_matching_threshold", 0.3)
        self.marker_rescale_range = tuple(
            tuning_options.get("marker_rescale_range", (85, 115))
        )
        self.marker_rescale_steps = int(tuning_options.get("marker_rescale_steps", 5))
        self.apply_erode_subtract = tuning_options.get("apply_erode_subtract", True)

        self.init_resized_markers()

    def validate_and_remap_options_schema(self, options):
        reference_image_path, layout_type = options["reference_image"], options["type"]
        tuning_options = options.get("tuning_options", {})
        # Note: options["tuning_options"] is accessible in self.tuning_options at Processor level
        parsed_options = {
            "default_selector": options.get("default_selector", "CENTERS"),
            "points_layout": layout_type,
            "enable_cropping": True,
            "tuning_options": {
                "warp_method": tuning_options.get(
                    "warp_method", WarpMethod.PERSPECTIVE_TRANSFORM
                )
            },
        }

        # TODO: add default values for provided scanZones?
        # Allow non-marker scanZones here too?
        default_dimensions = options.get("marker_dimensions", None)
        # inject scanZones (Note: override merge with defaults will happen in parent class)
        parsed_scan_zones = []
        for zone_preset in CropOnCustomMarkers.scan_zone_presets_for_layout[
            layout_type
        ]:
            local_description = options.get(camel_to_snake(zone_preset), {})
            # .pop() will delete the custom_options key from the description if it exists
            local_custom_options = local_description.pop("custom_options", {})
            parsed_scan_zones.append(
                {
                    "zone_preset": zone_preset,
                    "zone_description": {
                        # Default box dimensions to markerDimensions
                        "dimensions": default_dimensions,
                        **local_description,
                    },
                    "custom_options": {
                        "reference_image": reference_image_path,
                        "marker_dimensions": default_dimensions,
                        **local_custom_options,
                    },
                }
            )
        parsed_options["scan_zones"] = parsed_scan_zones
        return parsed_options

    def validate_scan_zones(self) -> None:
        super().validate_scan_zones()
        # Additional marker related validations
        for scan_zone in self.scan_zones:
            zone_preset, zone_description, custom_options = map(
                scan_zone.get, ["zone_preset", "zone_description", "custom_options"]
            )
            zone_label = zone_description["label"]
            if zone_preset in self.scan_zone_presets_for_layout["FOUR_MARKERS"]:
                if "reference_image" not in custom_options:
                    msg = f"referenceImage not provided for custom marker zone {zone_label}"
                    raise TemplateValidationError(
                        Path("unknown"),
                        errors=[msg],
                    )
                reference_image_path = self.get_relative_path(
                    custom_options["reference_image"]
                )

                if not reference_image_path.exists():
                    msg = f"Marker reference image not found for {zone_label} at path provided: {reference_image_path}"
                    raise ImageReadError(
                        msg,
                        context={
                            "zone_label": zone_label,
                            "reference_image_path": str(reference_image_path),
                        },
                    )

    def init_resized_markers(self) -> None:
        self.loaded_reference_images = {}
        self.marker_for_zone_label = {}
        for scan_zone in self.scan_zones:
            zone_description, custom_options = map(
                scan_zone.get, ["zone_description", "custom_options"]
            )
            zone_label, scanner_type = (
                zone_description["label"],
                zone_description["scanner_type"],
            )

            if scanner_type != ScannerType.TEMPLATE_MATCH:
                continue
            reference_image_path = self.get_relative_path(
                custom_options["reference_image"]
            )
            if reference_image_path in self.loaded_reference_images:
                reference_image = self.loaded_reference_images[reference_image_path]
            else:
                # TODO: add colored support later based on image_type passed at parent level
                reference_image = ImageUtils.load_image(
                    reference_image_path, cv2.IMREAD_GRAYSCALE
                )
                self.loaded_reference_images[reference_image_path] = reference_image

            extracted_marker = self.extract_marker_from_reference(
                reference_image, custom_options
            )

            self.marker_for_zone_label[zone_label] = extracted_marker

    def extract_marker_from_reference(self, reference_image, custom_options):
        """
        Extract and prepare marker template from reference image.

        Delegates to marker_detection.prepare_marker_template for the core logic.
        """
        options = self.options
        marker_dimensions = custom_options.get(
            "marker_dimensions", options.get("marker_dimensions", None)
        )
        blur_kernel = custom_options.get("marker_blur_kernel", (5, 5))
        # TODO: expose referenceZone support in schema with a better name (to extract an zone out of the reference image to use as a marker)
        reference_zone = custom_options.get(
            "reference_zone", self.get_default_scan_zone_for_image(reference_image)
        )
        return prepare_marker_template(
            reference_image,
            reference_zone,
            marker_dimensions=marker_dimensions,
            blur_kernel=blur_kernel,
            apply_erode_subtract=self.apply_erode_subtract,
        )

    @staticmethod
    def get_default_scan_zone_for_image(image):
        h, w = image.shape[:2]
        return {
            "origin": [1, 1],
            "dimensions": [w - 1, h - 1],
        }

    def get_runtime_zone_description_with_defaults(self, image, scan_zone):
        zone_preset, zone_description = (
            scan_zone["zone_preset"],
            scan_zone["zone_description"],
        )
        # Note: currently user input would be restricted to only markers at once (no combination of markers and dots)
        # TODO: >> handle a instance of this class from parent using scannerType for applicable ones!
        # Check for zone_preset
        if zone_preset not in self.scan_zone_presets_for_layout["FOUR_MARKERS"]:
            return zone_description

        zone_label, origin, dimensions = map(
            zone_description.get, ["label", "origin", "dimensions"]
        )
        if origin is None or dimensions is None:
            image_shape = image.shape[:2]
            marker_shape = self.marker_for_zone_label[zone_label].shape[:2]
            quadrant_description = self.get_quadrant_zone_description(
                zone_preset, image_shape, marker_shape
            )
            zone_description = OVERRIDE_MERGER.merge(
                quadrant_description, zone_description
            )

        # Note: this runtime template is supposedly always valid
        return zone_description

    def find_dot_corners_from_options(self, image, zone_description, file_path):
        config = self.tuning_config

        # Note we expect fill_runtime_defaults_from_zone_presets to be called from parent

        absolute_corners = self.find_marker_corners_in_patch(
            zone_description, image, file_path
        )

        if config.outputs.show_image_level >= 1:
            DrawingUtils.draw_contour(self.debug_image, absolute_corners)

        return absolute_corners

    def get_quadrant_zone_description(self, patch_type, image_shape, marker_shape):
        h, w = image_shape
        half_height, half_width = h // 2, w // 2
        marker_h, marker_w = marker_shape

        if patch_type == ZonePreset.topLeftMarker:
            zone_start, zone_end = [1, 1], [half_width, half_height]
        elif patch_type == ZonePreset.topRightMarker:
            zone_start, zone_end = [half_width, 1], [w, half_height]
        elif patch_type == ZonePreset.bottomRightMarker:
            zone_start, zone_end = [half_width, half_height], [w, h]
        elif patch_type == ZonePreset.bottomLeftMarker:
            zone_start, zone_end = [1, half_height], [half_width, h]
        else:
            msg = f"Unexpected quadrant patch_type {patch_type}"
            raise TemplateValidationError(
                msg,
                context={"patch_type": patch_type},
            )

        origin = [
            (zone_start[0] + zone_end[0] - marker_w) // 2,
            (zone_start[1] + zone_end[1] - marker_h) // 2,
        ]

        margin_horizontal = (zone_end[0] - zone_start[0] - marker_w) / 2 - 1
        margin_vertical = (zone_end[1] - zone_start[1] - marker_h) / 2 - 1

        return {
            "origin": origin,
            "dimensions": [marker_w, marker_h],
            "margins": {
                "top": margin_vertical,
                "right": margin_horizontal,
                "bottom": margin_vertical,
                "left": margin_horizontal,
            },
            # "selector": "SELECT_CENTER",
            "scanner_type": "TEMPLATE_MARKER",
        }

    def find_marker_corners_in_patch(self, zone_description, image, file_path):
        """
        Find marker corners in a patch zone.

        Delegates to marker_detection.detect_marker_in_patch for detection,
        with optional debug visualization.
        """
        config = self.tuning_config
        zone_label = zone_description["label"]

        patch_zone, zone_start, _zone_end = compute_scan_zone(image, zone_description)

        marker = self.marker_for_zone_label[zone_label]

        # Detect marker using the new module
        corners = detect_marker_in_patch(
            patch_zone,
            marker,
            zone_offset=zone_start,
            scale_range=self.marker_rescale_range,
            scale_steps=self.marker_rescale_steps,
            min_confidence=self.min_matching_threshold,
        )

        # Handle visualization for debugging
        if config.outputs.show_image_level >= 1:
            self._visualize_marker_detection(
                patch_zone,
                marker,
                corners,
                zone_label,
                zone_start,
                file_path,
            )

        if corners is None:
            msg = f"Error: No marker found in patch {zone_label}"
            raise ImageProcessingError(msg)

        return corners

    def _visualize_marker_detection(
        self,
        patch_zone,
        marker,
        corners,
        zone_label,
        zone_start,
        _file_path,
    ):
        """
        Visualize marker detection for debugging.

        Shows the patch, marker template, and match results in debug output.
        """
        config = self.tuning_config

        # Add to debug stack for later display
        self.debug_hstack += [patch_zone / 255]

        if corners is not None:
            # We found a match - show the marker
            self.debug_hstack += [
                marker / 255,
                # Note: We can't get the match_result from detect_marker_in_patch
                # so we skip showing it in the simplified version
            ]

        self.debug_vstack.append(self.debug_hstack)
        self.debug_hstack = []

        # Show detailed view if requested or on error
        if config.outputs.show_image_level >= 5 or (
            corners is None and config.outputs.show_image_level >= 1
        ):
            is_not_matching = corners is None
            hstack = ImageUtils.get_padded_hstack(
                [
                    self.debug_image / 255,
                    patch_zone / 255,
                    marker / 255,
                ]
            )
            title = (
                f"Template Marker Matching(Error): {zone_label}"
                if is_not_matching
                else f"Template Marker Matching: {zone_label}"
            )
            pause = config.outputs.show_image_level <= 4 or is_not_matching
            InteractionUtils.show(
                title,
                hstack,
                pause=pause,
            )

    def exclude_files(self) -> list[Path]:
        return [Path(key) for key in self.loaded_reference_images]

    def prepare_image_before_extraction(self, image):
        # TODO: remove apply_erode_subtract?
        return ImageUtils.normalize(
            image
            if self.apply_erode_subtract
            else (image - cv2.erode(image, kernel=np.ones((5, 5)), iterations=5))
        )
