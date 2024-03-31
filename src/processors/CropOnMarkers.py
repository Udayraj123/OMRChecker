import os

import cv2
import numpy as np

from src.processors.constants import DOTS_IN_ORDER, EdgeType
from src.processors.interfaces.CropOnPatchesCommon import CropOnPatchesCommon
from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger


# TODO: add support for showing patch areas during setLayout option?!
class CropOnCustomMarkers(CropOnPatchesCommon):
    patch_types_for_layout = {
        # Note: this will match with the FOUR_DOTS configuration
        "CUSTOM_MARKER": {
            "patch_selectors": {"DOTS": DOTS_IN_ORDER, "LINES": []},
            "edge_selectors": {
                EdgeType.TOP: [
                    {"patch_type": "topLeftDot", "selection_type": "DOT_PICK_POINT"},
                    {"patch_type": "topRightDot", "selection_type": "DOT_PICK_POINT"},
                ],
                EdgeType.RIGHT: [
                    {"patch_type": "topRightDot", "selection_type": "DOT_PICK_POINT"},
                    {
                        "patch_type": "bottomRightDot",
                        "selection_type": "DOT_PICK_POINT",
                    },
                ],
                EdgeType.BOTTOM: [
                    {
                        "patch_type": "bottomRightDot",
                        "selection_type": "DOT_PICK_POINT",
                    },
                    {"patch_type": "bottomLeftDot", "selection_type": "DOT_PICK_POINT"},
                ],
                EdgeType.LEFT: [
                    {"patch_type": "bottomLeftDot", "selection_type": "DOT_PICK_POINT"},
                    {"patch_type": "topLeftDot", "selection_type": "DOT_PICK_POINT"},
                ],
            },
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

    def exclude_files(self):
        return [self.marker_path]

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
            expected_dimensions = (
                options[patch_type]["dimensions"]
                if patch_type in options
                else options["dimensions"]
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

    def compute_marker_quadrants(self, image, patch_type):
        h, w = image.shape[:2]
        half_height, half_width = h // 2, w // 2
        if patch_type == "topLeftDot":
            area_start, area_end = [0, 0], [half_width, half_height]
        if patch_type == "topRightDot":
            area_start, area_end = [half_width, 0], [w, half_height]
        if patch_type == "bottomRightDot":
            area_start, area_end = [half_width, half_height], [w, h]
        if patch_type == "bottomLeftDot":
            area_start, area_end = [0, half_height], [half_width, h]

        area = image[area_start[1] : area_end[1], area_start[0] : area_end[0]]

        # TODO: tiny util to draw areas
        config = self.tuning_config
        if config.outputs.show_image_level >= 4:
            h, w = area.shape[:2]
            cv2.rectangle(
                self.debug_image,
                tuple(area_start),
                (area_start[0] + w, area_start[1] + h),
                (20, 255, 20),
                2,
            )
        return area, area_start

    def compute_marker_scan_area(self, patch_type, dot_options, image):
        if dot_options is not None:
            return self.compute_scan_area(image, dot_options)
        else:
            return self.compute_marker_quadrants(image, patch_type)

    def find_marker_corners_in_patch(
        self, patch_type, dot_options, image_eroded_sub, file_path
    ):
        config = self.tuning_config

        area, area_start = self.compute_marker_scan_area(
            patch_type, dot_options, image_eroded_sub
        )
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
        patch_corners = ImageUtils.get_rectangle_points(x, y, w, h)

        # TODO: reuse bottom code
        absolute_corners = list(
            map(lambda point: list(np.add(area_start, point)), patch_corners)
        )
        return absolute_corners

    def select_point_from_dot(self, patch_type, image, file_path):
        options = self.options
        config = self.tuning_config
        dot_options = options.get(patch_type, None)
        absolute_corners = self.find_marker_corners_in_patch(
            patch_type, dot_options, image, file_path
        )

        if config.outputs.show_image_level >= 1:
            cv2.drawContours(
                self.debug_image, [np.intp(absolute_corners)], -1, (200, 200, 200), 2
            )

        return self.select_point_from_dot_rect(patch_type, absolute_corners)

    def prepare_image(self, image):
        # TODO: remove apply_erode_subtract?
        return ImageUtils.normalize(
            image
            if self.apply_erode_subtract
            else (image - cv2.erode(image, kernel=np.ones((5, 5)), iterations=5))
        )

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


class CropOnDotLines(CropOnPatchesCommon):
    patch_types_for_layout = {
        "ONE_LINE_TWO_DOTS": {
            "patch_selectors": {
                "LINES": ["leftLine"],
                "DOTS": ["topRightDot", "bottomRightDot"],
            },
            "edge_selectors": {
                EdgeType.TOP: [
                    # Note: points in leftLine are also in clockwise order
                    {
                        "patch_type": "leftLine",
                        "selection_type": "LINE_PICK_LAST_POINT",
                    },
                    {"patch_type": "topRightDot", "selection_type": "DOT_PICK_POINT"},
                ],
                EdgeType.RIGHT: [
                    {"patch_type": "topRightDot", "selection_type": "DOT_PICK_POINT"},
                    {
                        "patch_type": "bottomRightDot",
                        "selection_type": "DOT_PICK_POINT",
                    },
                ],
                EdgeType.BOTTOM: [
                    {
                        "patch_type": "bottomRightDot",
                        "selection_type": "DOT_PICK_POINT",
                    },
                    {
                        "patch_type": "leftLine",
                        "selection_type": "LINE_PICK_FIRST_POINT",
                    },
                ],
                EdgeType.LEFT: [
                    # TODO: add a config for selecting only first and last (if contour method is not enabled)
                    {"patch_type": "leftLine", "selection_type": "LINE_PICK_CONTOUR"},
                ],
            },
        },
        "TWO_DOTS_ONE_LINE": {
            "patch_selectors": {
                "LINES": ["rightLine"],
                "DOTS": ["topLeftDot", "bottomLeftDot"],
            },
            "edge_selectors": {
                EdgeType.TOP: [
                    {"patch_type": "topLeftDot", "selection_type": "DOT_PICK_POINT"},
                    {
                        "patch_type": "rightLine",
                        "selection_type": "LINE_PICK_FIRST_POINT",
                    },
                ],
                EdgeType.RIGHT: [
                    # TODO: add a config for selecting only first and last (if contour method is not enabled)
                    {"patch_type": "rightLine", "selection_type": "LINE_PICK_CONTOUR"},
                ],
                EdgeType.BOTTOM: [
                    {
                        "patch_type": "rightLine",
                        "selection_type": "LINE_PICK_LAST_POINT",
                    },
                    {"patch_type": "bottomLeftDot", "selection_type": "DOT_PICK_POINT"},
                ],
                EdgeType.LEFT: [
                    {"patch_type": "bottomLeftDot", "selection_type": "DOT_PICK_POINT"},
                    {"patch_type": "topLeftDot", "selection_type": "DOT_PICK_POINT"},
                ],
            },
        },
        "TWO_LINES": {
            "patch_selectors": {"LINES": ["leftLine", "rightLine"], "DOTS": []},
            "edge_selectors": {
                EdgeType.TOP: [
                    {
                        "patch_type": "leftLine",
                        "selection_type": "LINE_PICK_LAST_POINT",
                    },
                    {
                        "patch_type": "rightLine",
                        "selection_type": "LINE_PICK_FIRST_POINT",
                    },
                ],
                EdgeType.RIGHT: [
                    {"patch_type": "rightLine", "selection_type": "LINE_PICK_CONTOUR"},
                ],
                EdgeType.BOTTOM: [
                    {
                        "patch_type": "rightLine",
                        "selection_type": "LINE_PICK_LAST_POINT",
                    },
                    {
                        "patch_type": "leftLine",
                        "selection_type": "LINE_PICK_FIRST_POINT",
                    },
                ],
                EdgeType.LEFT: [
                    {"patch_type": "leftLine", "selection_type": "LINE_PICK_CONTOUR"},
                ],
            },
        },
        "FOUR_DOTS": {
            "patch_selectors": {
                "LINES": [],
                "DOTS": DOTS_IN_ORDER,
            },
            "edge_selectors": {
                EdgeType.TOP: [
                    {"patch_type": "topLeftDot", "selection_type": "DOT_PICK_POINT"},
                    {"patch_type": "topRightDot", "selection_type": "DOT_PICK_POINT"},
                ],
                EdgeType.RIGHT: [
                    {"patch_type": "topRightDot", "selection_type": "DOT_PICK_POINT"},
                    {
                        "patch_type": "bottomRightDot",
                        "selection_type": "DOT_PICK_POINT",
                    },
                ],
                EdgeType.BOTTOM: [
                    {
                        "patch_type": "bottomRightDot",
                        "selection_type": "DOT_PICK_POINT",
                    },
                    {"patch_type": "bottomLeftDot", "selection_type": "DOT_PICK_POINT"},
                ],
                EdgeType.LEFT: [
                    {"patch_type": "bottomLeftDot", "selection_type": "DOT_PICK_POINT"},
                    {"patch_type": "topLeftDot", "selection_type": "DOT_PICK_POINT"},
                ],
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tuning_options = self.tuning_options
        self.line_kernel_morph = cv2.getStructuringElement(
            cv2.MORPH_RECT, tuple(tuning_options.get("lineKernel", [2, 10]))
        )
        self.dot_kernel_morph = self.dot_kernel_morph = cv2.getStructuringElement(
            cv2.MORPH_RECT, tuple(tuning_options.get("dotKernel", [5, 5]))
        )

    def select_points_and_edges_from_line(self, patch_type, image):
        options = self.options
        points_selector = options[patch_type].get(
            "pointsSelector", self.default_points_selector[patch_type]
        )
        # TODO: extract the line_contours points based on pointsSelector
        ordered_patch_corners, line_edge_contours = self.find_line_edges_from_options(
            image, options[patch_type], patch_type
        )
        tl, tr, br, bl = ordered_patch_corners

        # TODO: move over the horizontal line support as well from M3!
        if patch_type == "leftLine":
            if points_selector == "LINE_INNER_EDGE":
                return [tr, br], line_edge_contours[EdgeType.RIGHT]
            if points_selector == "LINE_OUTER_EDGE":
                return [tl, bl], line_edge_contours[EdgeType.LEFT]
        if patch_type == "rightLine":
            if points_selector == "LINE_INNER_EDGE":
                return [tl, bl], line_edge_contours[EdgeType.LEFT]
            if points_selector == "LINE_OUTER_EDGE":
                return [tr, br], line_edge_contours[EdgeType.RIGHT]

    def select_point_from_dot(self, patch_type, image, file_path):
        options = self.options
        dot_rect = self.find_dot_corners_from_options(image, options[patch_type])
        return self.select_point_from_dot_rect(patch_type, dot_rect)

    def find_line_edges_from_options(self, image, line_options, patch_type):
        config = self.tuning_config
        area, area_start = self.compute_scan_area(image, line_options)

        # Make boxes darker (less gamma)
        morph = ImageUtils.adjust_gamma(area, config.thresholding.GAMMA_LOW)
        _, morph = cv2.threshold(morph, 200, 255, cv2.THRESH_TRUNC)
        morph = ImageUtils.normalize(morph)

        # add white padding
        kernel_height, kernel_width = self.line_kernel_morph.shape[:2]
        white, box = ImageUtils.pad_image_from_center(
            morph, kernel_width, kernel_height, 255
        )

        # Threshold-Normalize after white padding
        _, morph = cv2.threshold(white, 180, 255, cv2.THRESH_TRUNC)
        morph = ImageUtils.normalize(morph)

        if config.outputs.show_image_level >= 5:
            self.debug_hstack += [morph]

        # Open : erode then dilate
        morph_v = cv2.morphologyEx(
            morph, cv2.MORPH_OPEN, self.line_kernel_morph, iterations=3
        )

        # remove white padding
        morph_v = morph_v[box[0] : box[1], box[2] : box[3]]

        if config.outputs.show_image_level >= 5:
            self.debug_hstack += [morph, morph_v]
            InteractionUtils.show(
                f"morph_opened_{patch_type}", morph_v, 0, 1, config=config
            )

        # Note: points are returned in the order of order_four_points: (tl, tr, br, bl)
        ordered_patch_corners, edge_contours_map = self.find_largest_patch_area(
            area_start, morph_v, patch_type="line"
        )

        if ordered_patch_corners is None:
            raise Exception(
                f"No line match found at origin: {line_options['origin']} with dimensions: { line_options['dimensions']}"
            )
        return ordered_patch_corners, edge_contours_map

    def find_dot_corners_from_options(self, image, dot_options):
        config = self.tuning_config
        area, area_start = self.compute_scan_area(image, dot_options)

        # simple thresholding, maybe small morphology (extract self.options)

        # TODO: nope, first make it like a patch_area then get contour

        # Open : erode then dilate
        morph_c = cv2.morphologyEx(
            area, cv2.MORPH_OPEN, self.dot_kernel_morph, iterations=3
        )

        _, thresholded = cv2.threshold(morph_c, 200, 255, cv2.THRESH_TRUNC)

        if config.outputs.show_image_level >= 5:
            self.debug_hstack.append(morph_c)

        corners, _ = self.find_largest_patch_area(
            area_start, thresholded, patch_type="dot"
        )
        if corners is None:
            hstack = ImageUtils.get_padded_hstack([self.debug_image, area, thresholded])
            InteractionUtils.show(f"No patch/dot found:", hstack, pause=1)
            raise Exception(
                f"No patch/dot found at origin: {dot_options['origin']} with dimensions: { dot_options['dimensions']}"
            )

        return corners

    def find_largest_patch_area(self, area_start, area, patch_type):
        config = self.tuning_config
        edge = cv2.Canny(area, 185, 55)

        if config.outputs.show_image_level >= 5:
            self.debug_hstack.append(edge.copy())

        # Should mostly return a single contour in the area
        all_contours = ImageUtils.grab_contours(
            cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        )

        # convexHull to resolve disordered curves due to noise
        all_contours = [cv2.convexHull(c) for c in all_contours]

        if len(all_contours) == 0:
            return None, None
        ordered_patch_corners, edge_contours_map = None, None

        bounding_contour = sorted(all_contours, key=cv2.contourArea, reverse=True)[0]

        if patch_type == "dot":
            # Bounding rectangle will not be rotated
            x, y, w, h = cv2.boundingRect(bounding_contour)
            patch_corners = ImageUtils.get_rectangle_points(x, y, w, h)
            (
                ordered_patch_corners,
                edge_contours_map,
            ) = ImageUtils.split_patch_contour_on_corners(
                patch_corners, bounding_contour
            )
        elif patch_type == "line":
            # Rotated rectangle can correct slight rotations better
            rotated_rect = cv2.minAreaRect(bounding_contour)
            # TODO: less confidence if angle = rotated_rect[2] is too skew
            rotated_rect_points = cv2.boxPoints(rotated_rect)
            patch_corners = np.intp(rotated_rect_points)
            (
                ordered_patch_corners,
                edge_contours_map,
            ) = ImageUtils.split_patch_contour_on_corners(
                patch_corners, bounding_contour
            )

        # TODO: Give a warning if given dimensions differ from matched block size
        cv2.drawContours(edge, [ordered_patch_corners], -1, (200, 200, 200), 2)

        if config.outputs.show_image_level >= 5:
            self.debug_hstack += [area, edge]
            self.debug_vstack.append(self.debug_hstack)
            self.debug_hstack = []

        # TODO: >> add shifts to each contour point as well
        shifted_edge_contours = edge_contours_map
        # absolute_corners = []
        # for point in patch_corners:
        #     absolute_corners.append(list(np.add(area_start, point)))
        return ordered_patch_corners, shifted_edge_contours


class CropOnMarkers(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.options["type"] == "CUSTOM_MARKER":
            self.instance = CropOnCustomMarkers(*args, **kwargs)
        else:
            self.instance = CropOnDotLines(*args, **kwargs)

    def exclude_files(self):
        return self.instance.exclude_files()

    def __str__(self):
        return self.instance.__str__()

    def apply_filter(self, *args, **kwargs):
        return self.instance.apply_filter(*args, **kwargs)
