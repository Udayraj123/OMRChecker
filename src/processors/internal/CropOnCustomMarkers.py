import os

import cv2
import numpy as np

from src.processors.constants import MARKER_AREA_TYPES_IN_ORDER, AreaTemplate
from src.processors.internal.CropOnPatchesCommon import CropOnPatchesCommon
from src.utils.constants import CLR_LIGHT_GRAY
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils
from src.utils.parsing import OVERRIDE_MERGER


# TODO: add support for showing patch areas during setLayout option?!
class CropOnCustomMarkers(CropOnPatchesCommon):
    __is_internal_preprocessor__ = True
    scan_area_templates_for_layout = {
        "CUSTOM_MARKER": MARKER_AREA_TYPES_IN_ORDER,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tuning_options = self.tuning_options
        self.threshold_circles = []

        # TODO: dedicated marker scanArea config needed for these
        self.min_matching_threshold = tuning_options.get("min_matching_threshold", 0.3)
        self.marker_rescale_range = tuple(
            tuning_options.get("marker_rescale_range", (85, 115))
        )
        self.marker_rescale_steps = int(tuning_options.get("marker_rescale_steps", 5))
        self.apply_erode_subtract = tuning_options.get("apply_erode_subtract", True)

        self.init_resized_markers()

    def validate_scan_areas(self):
        super().validate_scan_areas()
        # Additional marker related validations
        for scan_area in self.scan_areas:
            area_template, area_description, custom_options = map(
                scan_area.get, ["areaTemplate", "areaDescription", "customOptions"]
            )
            area_label = area_description["label"]
            if area_template in self.scan_area_templates_for_layout["CUSTOM_MARKER"]:
                if "referenceImage" not in custom_options:
                    raise Exception(
                        f"referenceImage not provided for custom marker area {area_label}"
                    )
                reference_image_path = custom_options["referenceImage"]

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

            if scanner_type != "TEMPLATE_MATCH":
                continue
            reference_image_path = custom_options["referenceImage"]
            if reference_image_path in self.loaded_reference_images:
                reference_image = self.loaded_reference_images[reference_image_path]
            else:
                full_path = os.path.join(self.relative_dir, reference_image_path)
                # TODO: add colored support later based on image_type passed at parent level
                reference_image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                self.loaded_reference_images[reference_image_path]
            reference_area = custom_options.get(
                "referenceArea", self.get_default_scan_area_for_image(reference_image)
            )

            extracted_marker = self.extract_marker_from_reference(
                reference_image, reference_area, custom_options
            )

            self.marker_for_area_label[area_label] = extracted_marker

    def extract_marker_from_reference(
        self, reference_image, reference_area, custom_options
    ):
        origin, dimensions = reference_area["origin"], reference_area["dimensions"]
        x, y = origin
        w, h = dimensions
        marker = reference_image[x : x + w, y : y + h]

        marker_dimensions = custom_options.get("markerDimensions", None)
        if marker_dimensions is not None:
            w, h = marker_dimensions
            marker = ImageUtils.resize_util(marker, w, h)

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
            "origin": [0, 0],
            "dimensions": [w, h],
        }

    def get_runtime_area_description_with_defaults(self, image, scan_area):
        area_template, area_description = (
            scan_area["areaTemplate"],
            scan_area["areaDescription"],
        )

        # Note: currently user input would be restricted to only markers at once (no combination of markers and dots)
        # TODO: >> handle a instance of this class from parent using scannerType for applicable ones!
        # Check for area_template
        if area_template not in self.scan_area_templates_for_layout["CUSTOM_MARKER"]:
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
            cv2.drawContours(
                self.debug_image, [np.intp(absolute_corners)], -1, CLR_LIGHT_GRAY, 2
            )

        return absolute_corners

    def get_quadrant_area_description(self, patch_type, image_shape, marker_shape):
        h, w = image_shape
        half_height, half_width = h // 2, w // 2
        marker_h, marker_w = marker_shape

        if patch_type == AreaTemplate.topLeftDot:
            area_start, area_end = [0, 0], [half_width, half_height]
        elif patch_type == AreaTemplate.topRightDot:
            area_start, area_end = [half_width, 0], [w, half_height]
        elif patch_type == AreaTemplate.bottomRightDot:
            area_start, area_end = [half_width, half_height], [w, h]
        elif patch_type == AreaTemplate.bottomLeftDot:
            area_start, area_end = [0, half_height], [half_width, h]
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
                "horizontal": margin_horizontal,
                "vertical": margin_vertical,
            },
            "selector": "DOT_CENTER",
            "scannerType": "TEMPLATE_MARKER",
        }

    def find_marker_corners_in_patch(self, area_description, image, file_path):
        config = self.tuning_config

        area_label = area_description["label"]

        area, area_start = self.compute_scan_area_util(image, area_description)
        # Note: now best match is being computed separately inside each patch area
        (
            optimal_res,
            optimal_marker,
            optimal_scale,
            optimal_match_max,
        ) = self.get_best_match(area_label, area)

        if optimal_scale is None:
            if config.outputs.show_image_level >= 1:
                # TODO check if debug_image is drawn-over
                InteractionUtils.show("Quads", self.debug_image, config=config)
            return None

        if config.outputs.show_image_level >= 1:
            # Note: We need images of dtype float for displaying optimal_res.
            self.debug_hstack += [area / 255, optimal_marker / 255, optimal_res]
            self.debug_vstack.append(self.debug_hstack)
            self.debug_hstack = []

        logger.info(
            f"{area_label}:\toptimal_match_max={str(round(optimal_match_max, 2))}\t optimal_scale={optimal_scale}\t"
        )

        if optimal_match_max < self.min_matching_threshold:
            logger.error(
                f"{file_path}\nError: No marker found in patch {area_label}, match_max={optimal_match_max}",
            )

            if config.outputs.show_image_level >= 1:
                hstack = ImageUtils.get_padded_hstack(
                    [self.debug_image / 255, optimal_res]
                )
                InteractionUtils.show(
                    f"No Markers res_{area_label} ({str(optimal_match_max)})",
                    hstack,
                    1,
                    config=config,
                )

        h, w = optimal_marker.shape[:2]
        y, x = np.argwhere(optimal_res == optimal_match_max)[0]
        ordered_patch_corners = MathUtils.get_rectangle_points(x, y, w, h)

        absolute_corners = MathUtils.shift_origin_for_points(
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
        _h, _w = marker.shape[:2]
        optimal_res, optimal_scale, optimal_marker = None, None, None
        optimal_match_max = 0
        for r0 in np.arange(
            self.marker_rescale_range[1],
            self.marker_rescale_range[0],
            -1 * descent_per_step,
        ):  # reverse order
            s = float(r0 * 1 / 100)
            if s == 0.0:
                continue
            rescaled_marker = ImageUtils.resize_util_h(marker, u_height=int(_h * s))

            # res is the black image with white dots
            res = cv2.matchTemplate(patch_area, rescaled_marker, cv2.TM_CCOEFF_NORMED)

            match_max = res.max()
            if optimal_match_max < match_max:
                # print('Scale: '+str(s)+', Circle Match: '+str(round(match_max*100,2))+'%')
                optimal_marker, optimal_scale, optimal_match_max, optimal_res = (
                    rescaled_marker,
                    s,
                    match_max,
                    res,
                )

        if optimal_scale is None:
            logger.warning(
                f"No matchings for {area_label} for given scaleRange:",
                self.marker_rescale_range,
            )
        is_low_matching = optimal_match_max < self.min_matching_threshold
        if is_low_matching or config.outputs.show_image_level >= 5:
            if is_low_matching:
                logger.warning(
                    f"Template matching too low for {area_label}! Recheck tuningOptions or output of previous preProcessor."
                )
                logger.info(
                    f"Sizes: marker:{marker.shape[:2]}, patch_area: {patch_area.shape[:2]}"
                )
            if config.outputs.show_image_level >= 1:
                hstack = ImageUtils.get_padded_hstack(
                    [rescaled_marker / 255, self.debug_image / 255, res]
                )
                InteractionUtils.show(f"Marker matching: {area_label}", hstack)

        return optimal_res, optimal_marker, optimal_scale, optimal_match_max

    def exclude_files(self):
        return [self.marker_path]

    def prepare_image(self, image):
        # TODO: remove apply_erode_subtract?
        return ImageUtils.normalize(
            image
            if self.apply_erode_subtract
            else (image - cv2.erode(image, kernel=np.ones((5, 5)), iterations=5))
        )
