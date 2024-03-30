import os

import cv2
import numpy as np

from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger


# Internal Processor for separation of code
class CropOnPatchesCommon(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = True

    # Common code used by both types of croppers
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        options = self.options

        # Default to select centers for roi
        self.default_points_selector = self.default_points_selector_map[
            options.get("pointsSelector", "CENTERS")
        ]

    def exclude_files(self):
        return []

    def __str__(self):
        return f"CropOnMarkers[\"{self.options['type']}\"]"

    default_points_selector_map = {
        "CENTERS": {
            "topLeftDot": "DOT_CENTER",
            "topRightDot": "DOT_CENTER",
            "bottomRightDot": "DOT_CENTER",
            "bottomLeftDot": "DOT_CENTER",
            "leftLine": "LINE_OUTER_EDGE",
            "rightLine": "LINE_OUTER_EDGE",
        },
        "INNER_WIDTHS": {
            "topLeftDot": "DOT_TOP_RIGHT",
            "topRightDot": "DOT_TOP_LEFT",
            "bottomRightDot": "DOT_BOTTOM_LEFT",
            "bottomLeftDot": "DOT_BOTTOM_RIGHT",
            "leftLine": "LINE_INNER_EDGE",
            "rightLine": "LINE_INNER_EDGE",
        },
        "INNER_HEIGHTS": {
            "topLeftDot": "DOT_BOTTOM_LEFT",
            "topRightDot": "DOT_BOTTOM_RIGHT",
            "bottomRightDot": "DOT_TOP_RIGHT",
            "bottomLeftDot": "DOT_TOP_LEFT",
            "leftLine": "LINE_OUTER_EDGE",
            "rightLine": "LINE_OUTER_EDGE",
        },
        "INNER_CORNERS": {
            "topLeftDot": "DOT_BOTTOM_RIGHT",
            "topRightDot": "DOT_BOTTOM_LEFT",
            "bottomRightDot": "DOT_TOP_LEFT",
            "bottomLeftDot": "DOT_TOP_RIGHT",
            "leftLine": "LINE_INNER_EDGE",
            "rightLine": "LINE_INNER_EDGE",
        },
        "OUTER_CORNERS": {
            "topLeftDot": "DOT_TOP_LEFT",
            "topRightDot": "DOT_TOP_RIGHT",
            "bottomRightDot": "DOT_BOTTOM_RIGHT",
            "bottomLeftDot": "DOT_BOTTOM_LEFT",
            "leftLine": "LINE_OUTER_EDGE",
            "rightLine": "LINE_OUTER_EDGE",
        },
    }
    all_dots = ["topLeftDot", "topRightDot", "bottomRightDot", "bottomLeftDot"]

    def select_point_from_dot_rect(self, patch_type, dot_rect):
        options = self.options
        points_selector = self.default_points_selector[patch_type]
        if patch_type in options:
            points_selector = options[patch_type].get("pointsSelector", points_selector)

        tl, tr, br, bl = dot_rect
        if points_selector == "DOT_TOP_LEFT":
            return tl
        if points_selector == "DOT_TOP_RIGHT":
            return tr
        if points_selector == "DOT_BOTTOM_RIGHT":
            return br
        if points_selector == "DOT_BOTTOM_LEFT":
            return bl
        if points_selector == "DOT_CENTER":
            return [
                (tl[0] + br[0]) // 2,
                (tl[1] + br[1]) // 2,
            ]
        return None

    def compute_scan_area(self, image, area_description):
        # parse arguments
        h, w = image.shape[:2]
        origin, dimensions, margins = map(
            area_description.get, ["origin", "dimensions", "margins"]
        )

        # compute area and clip to image dimensions
        area_start = [
            max(0, origin[0] - margins["horizontal"]),
            max(0, origin[1] - margins["vertical"]),
        ]
        area_end = [
            min(w, origin[0] + margins["horizontal"] + dimensions[0]),
            min(h, origin[1] + margins["vertical"] + dimensions[1]),
        ]

        if (
            area_start[0] == 0
            or area_start[1] == 0
            or area_end[0] == w
            or area_end[1] == h
        ):
            logger.warning(
                f"Scan area clipped to image boundary for patch item with origin: {origin}"
            )

        # Extract image area
        area = image[area_start[1] : area_end[1], area_start[0] : area_end[0]]

        config = self.tuning_config
        if config.outputs.show_image_level >= 1:
            h, w = area.shape[:2]
            cv2.rectangle(
                self.debug_image,
                tuple(area_start),
                (area_start[0] + w, area_start[1] + h),
                (20, 255, 20),
                2,
            )

        return area, np.array(area_start)

    def prepare_image(self, image):
        return image

    def apply_filter(self, image, colored_image, _template, file_path):
        config = self.tuning_config

        self.debug_image = image.copy()
        self.debug_hstack = []
        self.debug_vstack = []

        image = self.prepare_image(image)

        four_corners = self.find_four_corners(image, file_path)

        # Crop the image
        warped_image = ImageUtils.four_point_transform(image, four_corners)

        if config.outputs.show_colored_outputs:
            colored_image = ImageUtils.four_point_transform(colored_image, four_corners)

        # TODO: Save intuitive meta data
        # appendSaveImg(1,warped_image)

        if config.outputs.show_image_level >= 4:
            hstack = ImageUtils.get_padded_hstack([self.debug_image, warped_image])
            InteractionUtils.show(
                f"warped_image: {file_path}", hstack, 1, 1, config=config
            )

        return warped_image, colored_image, _template

    def find_four_corners(self, image, file_path):
        options = self.options
        config = self.tuning_config

        # TODO: support for defaulting to using quadrants?
        patch_areas = self.patch_areas_for_type[options["type"]]
        corners = []
        for patch_type in patch_areas["DOTS"]:
            corners.append(self.select_point_from_dot(patch_type, image, file_path))

        for patch_type in patch_areas["LINES"]:
            corners += self.select_points_from_line(patch_type, image)

            if config.outputs.show_image_level >= 5:
                if len(self.debug_vstack) > 0:
                    InteractionUtils.show(
                        f"Line Patches: {patch_type}",
                        ImageUtils.get_vstack_image_grid(self.debug_vstack),
                        0,
                        0,
                        config=config,
                    )
                self.debug_vstack = []

        if config.outputs.show_image_level >= 4:
            logger.info(f"corners={corners}")
            corners = ImageUtils.order_points(corners)
            cv2.drawContours(
                self.debug_image, [np.intp(corners)], -1, (200, 200, 200), 2
            )

            for corner in corners:
                cv2.rectangle(
                    self.debug_image,
                    tuple(corner),
                    (corner[0] + 2, corner[1] + 2),
                    (20, 255, 20),
                    2,
                )

        return corners


# TODO: add support for showing patch areas during setLayout option?!
class CropOnCustomMarkers(CropOnPatchesCommon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_areas_for_type = {
            "CUSTOM_MARKER": {"DOTS": self.all_dots, "LINES": []}
        }

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
        for patch_type in self.all_dots:
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
        optimal_marker, best_scale, all_max_t = self.get_best_match(patch_type, area)

        if best_scale is None:
            if config.outputs.show_image_level >= 1:
                # TODO check if debug_image is drawn-over
                InteractionUtils.show("Quads", self.debug_image, config=config)
            return None

        res = cv2.matchTemplate(area, optimal_marker, cv2.TM_CCOEFF_NORMED)
        max_t = res.max()

        if config.outputs.show_image_level >= 1:
            # Note: We need images of dtype float for displaying res.
            self.debug_hstack += [area / 255, optimal_marker / 255, res]
            self.debug_vstack.append(self.debug_hstack)
            self.debug_hstack = []

        logger.info(
            f"{patch_type}:\tmax_t={str(round(max_t, 2))}\t best_scale={best_scale}\t"
        )

        # TODO: reduce size of this if block
        if (
            max_t < self.min_matching_threshold
            or abs(all_max_t - max_t) >= self.max_matching_variation
        ):
            logger.error(
                file_path,
                f"\nError: No marker found in patch {patch_type}, max_t={max_t}",
            )

            if config.outputs.show_image_level >= 1:
                InteractionUtils.show(
                    f"No Markers: {file_path}",
                    self.debug_image,
                    0,
                    config=config,
                )
                InteractionUtils.show(
                    f"No Markers res_{patch_type} ({str(max_t)})",
                    res,
                    1,
                    config=config,
                )

        h, w = optimal_marker.shape[:2]
        y, x = np.argwhere(res == max_t)[0]
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
        return ImageUtils.normalize_util(
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
        res, best_scale, optimal_marker = None, None, None
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

            max_t = res.max()
            if all_max_t < max_t:
                # print('Scale: '+str(s)+', Circle Match: '+str(round(max_t*100,2))+'%')
                optimal_marker, best_scale, all_max_t = rescaled_marker, s, max_t

        if best_scale is None:
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

        return optimal_marker, best_scale, all_max_t


class CropOnDotLines(CropOnPatchesCommon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tuning_options = self.tuning_options
        self.patch_areas_for_type = self.get_patch_areas_for_type()
        self.line_kernel_morph = cv2.getStructuringElement(
            cv2.MORPH_RECT, tuple(tuning_options.get("lineKernel", [2, 10]))
        )
        self.dot_kernel_morph = self.dot_kernel_morph = cv2.getStructuringElement(
            cv2.MORPH_RECT, tuple(tuning_options.get("dotKernel", [5, 5]))
        )

    def select_points_from_line(self, patch_type, image):
        options = self.options
        points_selector = options[patch_type].get(
            "pointsSelector", self.default_points_selector[patch_type]
        )
        tl, tr, br, bl = self.find_line_corners_from_options(
            image, options[patch_type], patch_type
        )
        if patch_type == "leftLine":
            if points_selector == "LINE_INNER_EDGE":
                return [tr, br]
            if points_selector == "LINE_OUTER_EDGE":
                return [tl, bl]
        if patch_type == "rightLine":
            if points_selector == "LINE_INNER_EDGE":
                return [tl, bl]
            if points_selector == "LINE_OUTER_EDGE":
                return [tr, br]

    def select_point_from_dot(self, patch_type, image, file_path):
        options = self.options
        dot_rect = self.find_dot_corners_from_options(image, options[patch_type])
        return self.select_point_from_dot_rect(patch_type, dot_rect)

    def get_patch_areas_for_type(self):
        return {
            "ONE_LINE_TWO_DOTS": {
                "LINES": ["leftLine"],
                "DOTS": ["topRightDot", "bottomRightDot"],
            },
            "TWO_DOTS_ONE_LINE": {
                "LINES": ["rightLine"],
                "DOTS": ["topLeftDot", "bottomLeftDot"],
            },
            "TWO_LINES": {"LINES": ["leftLine", "rightLine"], "DOTS": []},
            "FOUR_DOTS": {
                "LINES": [],
                "DOTS": self.all_dots,
            },
        }

    def find_line_corners_from_options(self, image, line_options, patch_type):
        config = self.tuning_config
        area, area_start = self.compute_scan_area(image, line_options)

        # Make boxes darker (less gamma)
        morph = ImageUtils.adjust_gamma(area, config.thresholding.GAMMA_LOW)
        _, morph = cv2.threshold(morph, 200, 255, cv2.THRESH_TRUNC)
        morph = ImageUtils.normalize_util(morph)

        # add white padding
        kernel_height, kernel_width = self.line_kernel_morph.shape[:2]
        white, box = ImageUtils.pad_image_from_center(
            morph, kernel_width, kernel_height, 255
        )

        # Threshold-Normalize after white padding
        _, morph = cv2.threshold(white, 180, 255, cv2.THRESH_TRUNC)
        morph = ImageUtils.normalize_util(morph)

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

        # Note: points are returned in the order of order_points: (tl, tr, br, bl)
        corners = self.find_largest_patch_area_corners(
            area_start, morph_v, patch_type="line"
        )
        if corners is None:
            raise Exception(
                f"No line match found at origin: {line_options['origin']} with dimensions: { line_options['dimensions']}"
            )
        return corners

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

        corners = self.find_largest_patch_area_corners(
            area_start, thresholded, patch_type="dot"
        )
        if corners is None:
            hstack = ImageUtils.get_padded_hstack([self.debug_image, area, thresholded])
            InteractionUtils.show(f"No patch/dot found:", hstack, pause=1)
            raise Exception(
                f"No patch/dot found at origin: {dot_options['origin']} with dimensions: { dot_options['dimensions']}"
            )

        return corners

    def find_largest_patch_area_corners(self, area_start, area, patch_type):
        config = self.tuning_config

        edge = cv2.Canny(area, 185, 55)

        if config.outputs.show_image_level >= 5:
            self.debug_hstack.append(edge.copy())

        # Should mostly return a single contour in the area
        cnts = ImageUtils.grab_contours(
            cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        )

        # convexHull to resolve disordered curves due to noise
        cnts = [cv2.convexHull(c) for c in cnts]

        if len(cnts) == 0:
            return None

        bounding_cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        if patch_type == "dot":
            # Bounding rectangle will not be rotated
            x, y, w, h = cv2.boundingRect(bounding_cnt)
            patch_corners = ImageUtils.get_rectangle_points(x, y, w, h)
        elif patch_type == "line":
            # Rotated rectangle can correct slight rotations better
            rotated_rect = cv2.minAreaRect(bounding_cnt)
            # TODO: less confidence if angle = rotated_rect[2] is too skew
            rotated_rect_points = cv2.boxPoints(rotated_rect)
            patch_corners = np.intp(rotated_rect_points)
            patch_corners = ImageUtils.order_points(patch_corners)

        # TODO: Give a warning if given dimensions differ from matched block size
        cv2.drawContours(edge, [patch_corners], -1, (200, 200, 200), 2)

        if config.outputs.show_image_level >= 5:
            self.debug_hstack += [area, edge]
            self.debug_vstack.append(self.debug_hstack)
            self.debug_hstack = []

        absolute_corners = []
        for point in patch_corners:
            absolute_corners.append(list(np.add(area_start, point)))
        return absolute_corners


class CropOnMarkers(ImageTemplatePreprocessor):
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
