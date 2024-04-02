import cv2
import numpy as np

from src.processors.constants import (
    DOT_AREA_TYPES_IN_ORDER,
    EDGE_TYPES_IN_ORDER,
    AreaTemplate,
    EdgeType,
    ScannerType,
)
from src.processors.internal.CropOnPatchesCommon import CropOnPatchesCommon
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.math import MathUtils


class CropOnDotLines(CropOnPatchesCommon):
    __is_internal_preprocessor__ = True

    scan_area_templates_for_layout = {
        "ONE_LINE_TWO_DOTS": [
            AreaTemplate.topRightDot,
            AreaTemplate.bottomRightDot,
            AreaTemplate.leftLine,
        ],
        "TWO_DOTS_ONE_LINE": [
            AreaTemplate.rightLine,
            AreaTemplate.topLeftDot,
            AreaTemplate.bottomLeftDot,
        ],
        "TWO_LINES": [
            AreaTemplate.leftLine,
            AreaTemplate.rightLine,
        ],
        "FOUR_DOTS": DOT_AREA_TYPES_IN_ORDER,
    }

    default_points_selector_map = {
        "CENTERS": {
            AreaTemplate.topLeftDot: "SELECT_CENTER",
            AreaTemplate.topRightDot: "SELECT_CENTER",
            AreaTemplate.bottomRightDot: "SELECT_CENTER",
            AreaTemplate.bottomLeftDot: "SELECT_CENTER",
            AreaTemplate.leftLine: "LINE_OUTER_EDGE",
            AreaTemplate.rightLine: "LINE_OUTER_EDGE",
        },
        "INNER_WIDTHS": {
            AreaTemplate.topLeftDot: "SELECT_TOP_RIGHT",
            AreaTemplate.topRightDot: "SELECT_TOP_LEFT",
            AreaTemplate.bottomRightDot: "SELECT_BOTTOM_LEFT",
            AreaTemplate.bottomLeftDot: "SELECT_BOTTOM_RIGHT",
            AreaTemplate.leftLine: "LINE_INNER_EDGE",
            AreaTemplate.rightLine: "LINE_INNER_EDGE",
        },
        "INNER_HEIGHTS": {
            AreaTemplate.topLeftDot: "SELECT_BOTTOM_LEFT",
            AreaTemplate.topRightDot: "SELECT_BOTTOM_RIGHT",
            AreaTemplate.bottomRightDot: "SELECT_TOP_RIGHT",
            AreaTemplate.bottomLeftDot: "SELECT_TOP_LEFT",
            AreaTemplate.leftLine: "LINE_OUTER_EDGE",
            AreaTemplate.rightLine: "LINE_OUTER_EDGE",
        },
        "INNER_CORNERS": {
            AreaTemplate.topLeftDot: "SELECT_BOTTOM_RIGHT",
            AreaTemplate.topRightDot: "SELECT_BOTTOM_LEFT",
            AreaTemplate.bottomRightDot: "SELECT_TOP_LEFT",
            AreaTemplate.bottomLeftDot: "SELECT_TOP_RIGHT",
            AreaTemplate.leftLine: "LINE_INNER_EDGE",
            AreaTemplate.rightLine: "LINE_INNER_EDGE",
        },
        "OUTER_CORNERS": {
            AreaTemplate.topLeftDot: "SELECT_TOP_LEFT",
            AreaTemplate.topRightDot: "SELECT_TOP_RIGHT",
            AreaTemplate.bottomRightDot: "SELECT_BOTTOM_RIGHT",
            AreaTemplate.bottomLeftDot: "SELECT_BOTTOM_LEFT",
            AreaTemplate.leftLine: "LINE_OUTER_EDGE",
            AreaTemplate.rightLine: "LINE_OUTER_EDGE",
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

    def validate_and_remap_options_schema(self, options):
        # TODO: implement
        return options

    def find_and_select_points_from_line(self, image, area_description, _file_path):
        area_label = area_description["label"]
        points_selector = area_description.get(
            "selector", self.default_points_selector[area_label]
        )

        ordered_patch_corners, line_edge_contours = self.find_line_edges_from_options(
            image, area_description, area_label
        )

        # Extract the line_contours points based on defaultSelector
        edge_contour, edge_line = self.select_contour_and_edge_from_patch_area(
            ordered_patch_corners, area_label, points_selector, line_edge_contours
        )

        # Extrapolate the edge_line and get approximate destination points
        max_points = area_description.get("maxPoints", None)
        (
            control_points,
            destination_points,
        ) = ImageUtils.get_control_destination_points_from_contour(
            edge_contour, edge_line, max_points
        )
        return control_points, destination_points

    @staticmethod
    def select_contour_and_edge_from_patch_area(
        rectangle, area_label, points_selector, line_edge_contours
    ):
        # TODO: copy over the horizontal line support as well from M3!
        if area_label == AreaTemplate.leftLine:
            if points_selector == "LINE_INNER_EDGE":
                edge_type = EdgeType.RIGHT
            elif points_selector == "LINE_OUTER_EDGE":
                edge_type = EdgeType.LEFT
        if area_label == AreaTemplate.rightLine:
            if points_selector == "LINE_INNER_EDGE":
                edge_type = EdgeType.LEFT
            elif points_selector == "LINE_OUTER_EDGE":
                edge_type = EdgeType.RIGHT

        edge_contour = line_edge_contours[edge_type]
        edge_line = MathUtils.select_edge_from_rectangle(rectangle, edge_type)
        return edge_contour, edge_line

    def find_line_edges_from_options(self, image, area_description, area_label):
        config = self.tuning_config
        area, area_start = self.compute_scan_area_util(image, area_description)

        # Make boxes darker (less gamma)
        morph = ImageUtils.adjust_gamma(area, config.thresholding.GAMMA_LOW)
        _, morph = cv2.threshold(morph, 200, 255, cv2.THRESH_TRUNC)
        morph = ImageUtils.normalize(morph)

        # add white padding
        kernel_height, kernel_width = self.line_kernel_morph.shape[:2]
        white, pad_range = ImageUtils.pad_image_from_center(
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
        morph_v = morph_v[pad_range[0] : pad_range[1], pad_range[2] : pad_range[3]]

        if config.outputs.show_image_level >= 5:
            self.debug_hstack += [morph, morph_v]
            InteractionUtils.show(
                f"morph_opened_{area_label}", morph_v, 0, 1, config=config
            )

        # Note: points are returned in the order of order_four_points: (tl, tr, br, bl)
        (
            ordered_patch_corners,
            edge_contours_map,
        ) = self.find_largest_patch_area_and_contours_map(
            area_start, morph_v, scanner_type=ScannerType.PATCH_LINE
        )

        if ordered_patch_corners is None:
            raise Exception(
                f"No line match found at origin: {area_description['origin']} with dimensions: { area_description['dimensions']}"
            )
        return ordered_patch_corners, edge_contours_map

    def find_dot_corners_from_options(self, image, area_description, _file_path):
        config = self.tuning_config

        area, area_start = self.compute_scan_area_util(image, area_description)

        # TODO: simple colored thresholding to clear out noise?

        # Open : erode then dilate
        morph_c = cv2.morphologyEx(
            area, cv2.MORPH_OPEN, self.dot_kernel_morph, iterations=3
        )

        _, thresholded = cv2.threshold(morph_c, 200, 255, cv2.THRESH_TRUNC)

        if config.outputs.show_image_level >= 5:
            self.debug_hstack.append(morph_c)

        corners, _ = self.find_largest_patch_area_and_contours_map(
            area_start, thresholded, scanner_type=ScannerType.PATCH_DOT
        )
        if corners is None:
            hstack = ImageUtils.get_padded_hstack([self.debug_image, area, thresholded])
            InteractionUtils.show(f"No patch/dot found:", hstack, pause=1)
            raise Exception(
                f"No patch/dot found at origin: {area_description['origin']} with dimensions: { area_description['dimensions']}"
            )

        return corners

    def find_largest_patch_area_and_contours_map(self, area_start, area, scanner_type):
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

        if scanner_type == ScannerType.PATCH_DOT:
            # Bounding rectangle will not be rotated
            x, y, w, h = cv2.boundingRect(bounding_contour)
            patch_corners = MathUtils.get_rectangle_points(x, y, w, h)
            (
                ordered_patch_corners,
                edge_contours_map,
            ) = ImageUtils.split_patch_contour_on_corners(
                patch_corners, bounding_contour
            )
        if scanner_type == ScannerType.PATCH_LINE:
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

        # TODO: less confidence if given dimensions differ from matched block size (also give a warning)
        ImageUtils.draw_contour(edge, ordered_patch_corners)
        if config.outputs.show_image_level >= 5:
            self.debug_hstack += [area, edge]
            self.debug_vstack.append(self.debug_hstack)
            self.debug_hstack = []

        absolute_corners = MathUtils.shift_origin_for_points(
            area_start, ordered_patch_corners
        )

        shifted_edge_contours_map = {
            edge_type: MathUtils.shift_origin_for_points(
                area_start, edge_contours_map[edge_type]
            )
            for edge_type in EDGE_TYPES_IN_ORDER
        }

        return absolute_corners, shifted_edge_contours_map
