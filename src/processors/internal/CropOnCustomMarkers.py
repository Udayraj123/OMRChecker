import os

import cv2
import numpy as np

from src.processors.constants import (
    MARKER_AREA_TYPES_IN_ORDER,
    AreaTemplate,
    ScannerType,
    WarpMethod,
)
from src.processors.internal.CropOnPatchesCommon import CropOnPatchesCommon
from src.utils.drawing import DrawingUtils
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils
from src.utils.parsing import OVERRIDE_MERGER


# TODO: add support for showing patch area centers during setLayout option?!
class CropOnCustomMarkers(CropOnPatchesCommon):
    __is_internal_preprocessor__ = True
    scan_area_templates_for_layout = {
        "FOUR_MARKERS": MARKER_AREA_TYPES_IN_ORDER,
    }
    default_scan_area_descriptions = {
        **{
            area_template: {
                "scannerType": ScannerType.TEMPLATE_MATCH,
                "selector": "SELECT_CENTER",
                "maxPoints": 2,  # for cropping
                # Note: all 4 margins are a required property for a patch area
            }
            for area_template in MARKER_AREA_TYPES_IN_ORDER
        },
        "CUSTOM": {},
    }

    default_points_selector_map = {
        "CENTERS": {
            AreaTemplate.topLeftMarker: "SELECT_CENTER",
            AreaTemplate.topRightMarker: "SELECT_CENTER",
            AreaTemplate.bottomRightMarker: "SELECT_CENTER",
            AreaTemplate.bottomLeftMarker: "SELECT_CENTER",
        },
        "INNER_WIDTHS": {
            AreaTemplate.topLeftMarker: "SELECT_TOP_RIGHT",
            AreaTemplate.topRightMarker: "SELECT_TOP_LEFT",
            AreaTemplate.bottomRightMarker: "SELECT_BOTTOM_LEFT",
            AreaTemplate.bottomLeftMarker: "SELECT_BOTTOM_RIGHT",
        },
        "INNER_HEIGHTS": {
            AreaTemplate.topLeftMarker: "SELECT_BOTTOM_LEFT",
            AreaTemplate.topRightMarker: "SELECT_BOTTOM_RIGHT",
            AreaTemplate.bottomRightMarker: "SELECT_TOP_RIGHT",
            AreaTemplate.bottomLeftMarker: "SELECT_TOP_LEFT",
        },
        "INNER_CORNERS": {
            AreaTemplate.topLeftMarker: "SELECT_BOTTOM_RIGHT",
            AreaTemplate.topRightMarker: "SELECT_BOTTOM_LEFT",
            AreaTemplate.bottomRightMarker: "SELECT_TOP_LEFT",
            AreaTemplate.bottomLeftMarker: "SELECT_TOP_RIGHT",
        },
        "OUTER_CORNERS": {
            AreaTemplate.topLeftMarker: "SELECT_TOP_LEFT",
            AreaTemplate.topRightMarker: "SELECT_TOP_RIGHT",
            AreaTemplate.bottomRightMarker: "SELECT_BOTTOM_RIGHT",
            AreaTemplate.bottomLeftMarker: "SELECT_BOTTOM_LEFT",
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tuning_options = self.tuning_options
        self.threshold_circles = []

        # TODO: dedicated marker scanArea override support needed for these?
        self.min_matching_threshold = tuning_options.get("min_matching_threshold", 0.3)
        self.marker_rescale_range = tuple(
            tuning_options.get("marker_rescale_range", (85, 115))
        )
        self.marker_rescale_steps = int(tuning_options.get("marker_rescale_steps", 5))
        self.apply_erode_subtract = tuning_options.get("apply_erode_subtract", True)

        self.init_resized_markers()

    def validate_and_remap_options_schema(self, options):
        reference_image_path, layout_type = options["referenceImage"], options["type"]
        tuning_options = options.get("tuningOptions", {})
        # Note: options["tuningOptions"] is accessible in self.tuning_options at Processor level
        parsed_options = {
            "pointsLayout": layout_type,
            "enableCropping": True,
            "tuningOptions": {
                "warpMethod": tuning_options.get(
                    "warpMethod", WarpMethod.PERSPECTIVE_TRANSFORM
                )
            },
        }

        # TODO: add default values for provided scanAreas?
        # Allow non-marker scanAreas here too?
        defaultDimensions = options.get("markerDimensions", None)
        # inject scanAreas (Note: override merge with defaults will happen in parent class)
        parsed_scan_areas = []
        for area_template in self.scan_area_templates_for_layout[layout_type]:
            local_description = options.get(area_template, {})
            # .pop() will delete the customOptions key from the description if it exists
            local_custom_options = local_description.pop("customOptions", {})
            parsed_scan_areas.append(
                {
                    "areaTemplate": area_template,
                    "areaDescription": {
                        # Default box dimensions to markerDimensions
                        "dimensions": defaultDimensions,
                        **local_description,
                    },
                    "customOptions": {
                        "referenceImage": reference_image_path,
                        "markerDimensions": defaultDimensions,
                        **local_custom_options,
                    },
                }
            )
        parsed_options["scanAreas"] = parsed_scan_areas
        return parsed_options

    def validate_scan_areas(self):
        super().validate_scan_areas()
        # Additional marker related validations
        for scan_area in self.scan_areas:
            area_template, area_description, custom_options = map(
                scan_area.get, ["areaTemplate", "areaDescription", "customOptions"]
            )
            area_label = area_description["label"]
            if area_template in self.scan_area_templates_for_layout["FOUR_MARKERS"]:
                if "referenceImage" not in custom_options:
                    raise Exception(
                        f"referenceImage not provided for custom marker area {area_label}"
                    )
                reference_image_path = self.get_relative_path(
                    custom_options["referenceImage"]
                )

                if not os.path.exists(reference_image_path):
                    raise Exception(
                        f"Marker reference image not found for {area_label} at path provided: {reference_image_path}"
                    )

    def init_resized_markers(self):
        self.loaded_reference_images = {}
        self.marker_for_area_label = {}
        for scan_area in self.scan_areas:
            area_description, custom_options = map(
                scan_area.get, ["areaDescription", "customOptions"]
            )
            area_label, scanner_type = (
                area_description["label"],
                area_description["scannerType"],
            )

            if scanner_type != ScannerType.TEMPLATE_MATCH:
                continue
            reference_image_path = self.get_relative_path(
                custom_options["referenceImage"]
            )
            if reference_image_path in self.loaded_reference_images:
                reference_image = self.loaded_reference_images[reference_image_path]
            else:
                # TODO: add colored support later based on image_type passed at parent level
                reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
                self.loaded_reference_images[reference_image_path] = reference_image

            # TODO: expose referenceArea support in schema with a better name (to extract an area out of the reference image to use as a marker)
            reference_area = custom_options.get(
                "referenceArea", self.get_default_scan_area_for_image(reference_image)
            )
            # logger.debug("area_label=", area_label, custom_options)

            extracted_marker = self.extract_marker_from_reference(
                reference_image, reference_area, custom_options
            )

            self.marker_for_area_label[area_label] = extracted_marker

    def extract_marker_from_reference(
        self, reference_image, reference_area, custom_options
    ):
        options = self.options
        origin, dimensions = reference_area["origin"], reference_area["dimensions"]
        x, y = origin
        w, h = dimensions
        marker = reference_image[y : y + h, x : x + w]

        marker_dimensions = custom_options.get(
            "markerDimensions", options.get("markerDimensions", None)
        )
        if marker_dimensions is not None:
            marker = ImageUtils.resize_to_dimensions(marker, marker_dimensions)

        blur_kernel = custom_options.get("markerBlurKernel", (5, 5))
        marker = cv2.GaussianBlur(marker, blur_kernel, 0)

        marker = cv2.normalize(
            marker, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )

        if self.apply_erode_subtract:
            marker -= cv2.erode(marker, kernel=np.ones((5, 5)), iterations=5)

        return marker

    @staticmethod
    def get_default_scan_area_for_image(image):
        h, w = image.shape[:2]
        return {
            "origin": [1, 1],
            "dimensions": [w - 1, h - 1],
        }

    def get_runtime_area_description_with_defaults(self, image, scan_area):
        area_template, area_description = (
            scan_area["areaTemplate"],
            scan_area["areaDescription"],
        )
        logger.debug(scan_area)
        # Note: currently user input would be restricted to only markers at once (no combination of markers and dots)
        # TODO: >> handle a instance of this class from parent using scannerType for applicable ones!
        # Check for area_template
        if area_template not in self.scan_area_templates_for_layout["FOUR_MARKERS"]:
            return area_description

        area_label, origin, dimensions = map(
            area_description.get, ["label", "origin", "dimensions"]
        )
        if origin is None or dimensions is None:
            image_shape = image.shape[:2]
            marker_shape = self.marker_for_area_label[area_label].shape[:2]
            quadrant_description = self.get_quadrant_area_description(
                area_template, image_shape, marker_shape
            )
            area_description = OVERRIDE_MERGER.merge(
                quadrant_description, area_description
            )
        # Check for area_description["scannerType"]

        # Note: this runtime template is supposedly always valid
        return area_description

    def find_dot_corners_from_options(self, image, area_description, file_path):
        config = self.tuning_config

        # Note we expect fill_runtime_defaults_from_area_templates to be called from parent

        absolute_corners = self.find_marker_corners_in_patch(
            area_description, image, file_path
        )

        if config.outputs.show_image_level >= 1:
            DrawingUtils.draw_contour(self.debug_image, absolute_corners)

        return absolute_corners

    def get_quadrant_area_description(self, patch_type, image_shape, marker_shape):
        h, w = image_shape
        half_height, half_width = h // 2, w // 2
        marker_h, marker_w = marker_shape

        if patch_type == AreaTemplate.topLeftMarker:
            area_start, area_end = [1, 1], [half_width, half_height]
        elif patch_type == AreaTemplate.topRightMarker:
            area_start, area_end = [half_width, 1], [w, half_height]
        elif patch_type == AreaTemplate.bottomRightMarker:
            area_start, area_end = [half_width, half_height], [w, h]
        elif patch_type == AreaTemplate.bottomLeftMarker:
            area_start, area_end = [1, half_height], [half_width, h]
        else:
            raise Exception(f"Unexpected quadrant patch_type {patch_type}")

        origin = [
            (area_start[0] + area_end[0] - marker_w) // 2,
            (area_start[1] + area_end[1] - marker_h) // 2,
        ]

        margin_horizontal = (area_end[0] - area_start[0] - marker_w) / 2 - 1
        margin_vertical = (area_end[1] - area_start[1] - marker_h) / 2 - 1

        return {
            "origin": origin,
            "dimensions": [marker_w, marker_h],
            "margins": {
                "top": margin_vertical,
                "right": margin_horizontal,
                "bottom": margin_vertical,
                "left": margin_horizontal,
            },
            "selector": "SELECT_CENTER",
            "scannerType": "TEMPLATE_MARKER",
        }

    def find_marker_corners_in_patch(self, area_description, image, file_path):
        area_label = area_description["label"]

        patch_area, area_start = self.compute_scan_area_util(image, area_description)
        # Note: now best match is being computed separately inside each patch_area
        (marker_position, optimal_marker) = self.get_best_match(area_label, patch_area)

        if marker_position is None or optimal_marker is None:
            return None

        h, w = optimal_marker.shape[:2]
        x, y = marker_position
        ordered_patch_corners = MathUtils.get_rectangle_points(x, y, w, h)

        absolute_corners = MathUtils.shift_points_from_origin(
            area_start, ordered_patch_corners
        )
        return absolute_corners

    # Resizing the marker within scaleRange at rate of descent_per_step to
    # find the best match.
    def get_best_match(self, area_label, patch_area):
        config = self.tuning_config
        descent_per_step = (
            self.marker_rescale_range[1] - self.marker_rescale_range[0]
        ) // self.marker_rescale_steps
        marker = self.marker_for_area_label[area_label]
        marker_height, marker_width = marker.shape[:2]
        marker_position, optimal_match_result, optimal_scale, optimal_marker = (
            None,
            None,
            None,
            None,
        )
        optimal_match_max = 0

        for r0 in np.arange(
            self.marker_rescale_range[1],
            self.marker_rescale_range[0],
            -1 * descent_per_step,
        ):
            scale = float(r0 * 1 / 100)
            if scale <= 0.0:
                continue
            rescaled_marker = ImageUtils.resize_util(
                marker,
                u_width=int(marker_width * scale),
                u_height=int(marker_height * scale),
            )

            # res is the black image with white dots
            match_result = cv2.matchTemplate(
                patch_area, rescaled_marker, cv2.TM_CCOEFF_NORMED
            )

            match_max = match_result.max()
            if optimal_match_max < match_max:
                # print('Scale: '+str(scale)+', Circle Match: '+str(round(match_max*100,2))+'%')
                (
                    optimal_scale,
                    optimal_marker,
                    optimal_match_max,
                    optimal_match_result,
                ) = (
                    scale,
                    rescaled_marker,
                    match_max,
                    match_result,
                )

        if optimal_scale is None:
            logger.warning(
                f"No matchings for {area_label} for given scaleRange:",
                self.marker_rescale_range,
            )
        if config.outputs.show_image_level >= 1:
            self.debug_hstack += [
                patch_area / 255,
            ]
            if optimal_marker is not None:
                # Note: We need images of dtype float for displaying optimal_match_result.
                self.debug_hstack += [
                    optimal_marker / 255,
                    optimal_match_result,
                ]
            self.debug_vstack.append(self.debug_hstack)
            self.debug_hstack = []

        is_not_matching = optimal_match_max < self.min_matching_threshold
        if is_not_matching:
            logger.error(
                f"Error: No marker found in patch {area_label}, (match_max={optimal_match_max:.2f} < {self.min_matching_threshold:.2f}) for {area_label}! Recheck tuningOptions or output of previous preProcessor."
            )
            logger.debug(
                f"Sizes: optimal_marker:{None if optimal_marker is None else optimal_marker.shape[:2]}, patch_area: {patch_area.shape[:2]}"
            )

        else:
            y, x = np.argwhere(optimal_match_result == optimal_match_max)[0]
            marker_position = [x, y]

            logger.info(
                f"{area_label}:\toptimal_match_max={str(round(optimal_match_max, 2))}\t optimal_scale={optimal_scale}\t"
            )

        if config.outputs.show_image_level >= 5 or (
            is_not_matching and config.outputs.show_image_level >= 1
        ):
            hstack = ImageUtils.get_padded_hstack(
                [
                    self.debug_image / 255,
                    patch_area / 255,
                    (rescaled_marker if optimal_marker is None else optimal_marker)
                    / 255,
                    (
                        match_result
                        if optimal_match_result is None
                        else optimal_match_result
                    ),
                ]
            )
            title = (
                f"Template Marker Matching(Error): {area_label} ({optimal_match_max:.2f}/{self.min_matching_threshold:.2f})"
                if is_not_matching
                else "Template Marker Matching"
            )
            InteractionUtils.show(
                title,
                hstack,
                pause=is_not_matching,
            )
        if is_not_matching:
            raise Exception(f"Error: No marker found in patch {area_label}")

        return marker_position, optimal_marker

    def exclude_files(self):
        return self.loaded_reference_images.keys()

    def prepare_image(self, image):
        # TODO: remove apply_erode_subtract?
        return ImageUtils.normalize(
            image
            if self.apply_erode_subtract
            else (image - cv2.erode(image, kernel=np.ones((5, 5)), iterations=5))
        )
