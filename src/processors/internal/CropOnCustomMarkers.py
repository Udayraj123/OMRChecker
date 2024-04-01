import os

import cv2
import numpy as np

from src.processors.constants import DOTS_IN_ORDER
from src.processors.internal.CropOnPatchesCommon import CropOnPatchesCommon
from src.utils.constants import CLR_LIGHT_GRAY
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils


# TODO: add support for showing patch areas during setLayout option?!
class CropOnCustomMarkers(CropOnPatchesCommon):
    __is_internal_preprocessor__ = True
    patch_types_for_layout = {
        # Note: this will match with the FOUR_DOTS configuration
        "CUSTOM_MARKER": {
            "patch_selectors": {"DOTS": DOTS_IN_ORDER, "LINES": []},
            # "edge_selectors": {
            #     EdgeType.TOP: [
            #         {"patch_type": "topLeftDot", "selection_type": "DOT_PICK_POINT"},
            #         {"patch_type": "topRightDot", "selection_type": "DOT_PICK_POINT"},
            #     ],
            #     EdgeType.RIGHT: [
            #         {"patch_type": "topRightDot", "selection_type": "DOT_PICK_POINT"},
            #         {
            #             "patch_type": "bottomRightDot",
            #             "selection_type": "DOT_PICK_POINT",
            #         },
            #     ],
            #     EdgeType.BOTTOM: [
            #         {
            #             "patch_type": "bottomRightDot",
            #             "selection_type": "DOT_PICK_POINT",
            #         },
            #         {"patch_type": "bottomLeftDot", "selection_type": "DOT_PICK_POINT"},
            #     ],
            #     EdgeType.LEFT: [
            #         {"patch_type": "bottomLeftDot", "selection_type": "DOT_PICK_POINT"},
            #         {"patch_type": "topLeftDot", "selection_type": "DOT_PICK_POINT"},
            #     ],
            # },
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        options = self.options
        tuning_options = self.tuning_options
        self.threshold_circles = []
        self.marker_path = os.path.join(
            self.relative_dir, options.get("relativePath", "omr_marker.jpg")
        )

        self.min_matching_threshold = tuning_options.get("min_matching_threshold", 0.3)
        self.max_matching_variation = tuning_options.get("max_matching_variation", 0.41)
        self.marker_rescale_range = tuple(
            tuning_options.get("marker_rescale_range", (85, 115))
        )
        self.marker_rescale_steps = int(tuning_options.get("marker_rescale_steps", 5))
        self.apply_erode_subtract = tuning_options.get("apply_erode_subtract", True)
        self.init_resized_markers()

    def init_resized_markers(self):
        options = self.options
        if not os.path.exists(self.marker_path):
            logger.error(
                "Marker not found at path provided in template:",
                self.marker_path,
            )
            exit(31)

        original_marker = cv2.imread(self.marker_path, cv2.IMREAD_GRAYSCALE)
        self.marker_for_patch_type = {}
        for patch_type in DOTS_IN_ORDER:
            dot_options = options.get(patch_type, None)
            expected_dimensions = (
                dot_options["dimensions"] if dot_options else options["dimensions"]
            )
            marker = ImageUtils.resize_util(
                original_marker,
                expected_dimensions[0],
                expected_dimensions[1],
            )

            marker = cv2.GaussianBlur(marker, (5, 5), 0)
            marker = cv2.normalize(
                marker, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            if self.apply_erode_subtract:
                marker -= cv2.erode(marker, kernel=np.ones((5, 5)), iterations=5)

            self.marker_for_patch_type[patch_type] = marker

    def find_dot_corners_from_options(self, image, patch_type, file_path):
        config = self.tuning_config

        absolute_corners, dot_description = self.find_marker_corners_in_patch(
            patch_type, image, file_path
        )

        logger.info(
            f"absolute_corners={absolute_corners}, dot_description={dot_description}"
        )

        if config.outputs.show_image_level >= 1:
            cv2.drawContours(
                self.debug_image, [np.intp(absolute_corners)], -1, CLR_LIGHT_GRAY, 2
            )

        return absolute_corners, dot_description

    def find_marker_corners_in_patch(self, patch_type, image, file_path):
        config = self.tuning_config
        options = self.options
        image_shape = image.shape[:2]
        marker_shape = self.marker_for_patch_type[patch_type].shape[:2]
        dot_description = (
            options[patch_type]
            if patch_type in options
            else self.get_quadrant_area_description(
                patch_type, image_shape, marker_shape
            )
        )

        area, area_start = self.compute_scan_area_util(image, dot_description)
        # Note: now best match is being computed separately inside each patch area
        optimal_marker, optimal_scale, all_max_t = self.get_best_match(patch_type, area)

        if optimal_scale is None:
            if config.outputs.show_image_level >= 1:
                # TODO check if debug_image is drawn-over
                InteractionUtils.show("Quads", self.debug_image, config=config)
            return None

        res = cv2.matchTemplate(area, optimal_marker, cv2.TM_CCOEFF_NORMED)
        match_max = res.max()

        if config.outputs.show_image_level >= 1:
            # Note: We need images of dtype float for displaying res.
            self.debug_hstack += [area / 255, optimal_marker / 255, res]
            self.debug_vstack.append(self.debug_hstack)
            self.debug_hstack = []

        logger.info(
            f"{patch_type}:\tmatch_max={str(round(match_max, 2))}\t optimal_scale={optimal_scale}\t"
        )

        # TODO: reduce size of this if block
        if (
            match_max < self.min_matching_threshold
            or abs(all_max_t - match_max) >= self.max_matching_variation
        ):
            logger.error(
                file_path,
                f"\nError: No marker found in patch {patch_type}, match_max={match_max}",
            )

            if config.outputs.show_image_level >= 1:
                InteractionUtils.show(
                    f"No Markers: {file_path}",
                    self.debug_image,
                    0,
                    config=config,
                )
                InteractionUtils.show(
                    f"No Markers res_{patch_type} ({str(match_max)})",
                    res,
                    1,
                    config=config,
                )

        h, w = optimal_marker.shape[:2]
        y, x = np.argwhere(res == match_max)[0]
        ordered_patch_corners = MathUtils.get_rectangle_points(x, y, w, h)

        absolute_corners = MathUtils.shift_origin_for_points(
            area_start, ordered_patch_corners
        )
        return absolute_corners, dot_description

    def get_quadrant_area_description(self, patch_type, image_shape, marker_shape):
        h, w = image_shape
        half_height, half_width = h // 2, w // 2
        marker_h, marker_w = marker_shape

        if patch_type == "topLeftDot":
            area_start, area_end = [0, 0], [half_width, half_height]
        elif patch_type == "topRightDot":
            area_start, area_end = [half_width, 0], [w, half_height]
        elif patch_type == "bottomRightDot":
            area_start, area_end = [half_width, half_height], [w, h]
        elif patch_type == "bottomLeftDot":
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
            "pointsSelector": "DOT_CENTER",
        }

    # Resizing the marker within scaleRange at rate of descent_per_step to
    # find the best match.
    def get_best_match(self, patch_type, patch_area):
        config = self.tuning_config
        descent_per_step = (
            self.marker_rescale_range[1] - self.marker_rescale_range[0]
        ) // self.marker_rescale_steps
        marker = self.marker_for_patch_type[patch_type]
        _h, _w = marker.shape[:2]
        res, optimal_scale, optimal_marker = None, None, None
        all_max_t = 0
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
            if all_max_t < match_max:
                # print('Scale: '+str(s)+', Circle Match: '+str(round(match_max*100,2))+'%')
                optimal_marker, optimal_scale, all_max_t = rescaled_marker, s, match_max

        if optimal_scale is None:
            logger.warning(
                f"No matchings for {patch_type} for given scaleRange:",
                self.marker_rescale_range,
            )
        is_low_matching = all_max_t < self.min_matching_threshold
        if is_low_matching or config.outputs.show_image_level >= 5:
            if is_low_matching:
                logger.warning(
                    f"Template matching too low for {patch_type}! Recheck tuningOptions or output of previous preProcessor."
                )
                logger.info(
                    f"Sizes: marker:{marker.shape[:2]}, patch_area: {patch_area.shape[:2]}"
                )
            if config.outputs.show_image_level >= 1:
                hstack = ImageUtils.get_padded_hstack(
                    [rescaled_marker / 255, self.debug_image / 255, res]
                )
                InteractionUtils.show(f"Marker matching: {patch_type}", hstack)

        return optimal_marker, optimal_scale, all_max_t

    def exclude_files(self):
        return [self.marker_path]

    def prepare_image(self, image):
        # TODO: remove apply_erode_subtract?
        return ImageUtils.normalize(
            image
            if self.apply_erode_subtract
            else (image - cv2.erode(image, kernel=np.ones((5, 5)), iterations=5))
        )
