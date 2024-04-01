import cv2
import numpy as np

from src.processors.constants import DOTS_IN_ORDER, EDGE_TYPES_IN_ORDER, EdgeType
from src.processors.internal.CropOnPatchesCommon import CropOnPatchesCommon
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.math import MathUtils


class CropOnDotLines(CropOnPatchesCommon):
    __is_internal_preprocessor__ = True

    patch_types_for_layout = {
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
            "DOTS": DOTS_IN_ORDER,
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

    def find_and_select_points_from_line(self, patch_type, image):
        options = self.options
        line_options = options[patch_type]

        points_selector = line_options.get(
            "pointsSelector", self.default_points_selector[patch_type]
        )

        ordered_patch_corners, line_edge_contours = self.find_line_edges_from_options(
            image, line_options, patch_type
        )

        # Extract the line_contours points based on pointsSelector
        edge_contour, edge_line = self.select_contour_and_edge_from_patch_area(
            ordered_patch_corners, patch_type, points_selector, line_edge_contours
        )

        # Extrapolate the edge_line and get approximate destination points
        max_points = line_options.get("maxPoints", None)
        (
            control_points,
            destination_points,
        ) = ImageUtils.get_control_destination_points_from_contour(
            edge_contour, edge_line, max_points
        )
        return control_points, destination_points

    @staticmethod
    def select_contour_and_edge_from_patch_area(
        rectangle, patch_type, points_selector, line_edge_contours
    ):
        # TODO: copy over the horizontal line support as well from M3!
        if patch_type == "leftLine":
            if points_selector == "LINE_INNER_EDGE":
                edge_type = EdgeType.RIGHT
            elif points_selector == "LINE_OUTER_EDGE":
                edge_type = EdgeType.LEFT
        if patch_type == "rightLine":
            if points_selector == "LINE_INNER_EDGE":
                edge_type = EdgeType.LEFT
            elif points_selector == "LINE_OUTER_EDGE":
                edge_type = EdgeType.RIGHT

        edge_contour = line_edge_contours[edge_type]
        edge_line = MathUtils.select_edge_from_rectangle(rectangle, edge_type)
        return edge_contour, edge_line

    def find_line_edges_from_options(self, image, line_options, patch_type):
        config = self.tuning_config
        area, area_start = self.compute_scan_area_util(image, line_options)

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
        (
            ordered_patch_corners,
            edge_contours_map,
        ) = self.find_largest_patch_area_and_contours_map(
            area_start, morph_v, patch_type="line"
        )

        if ordered_patch_corners is None:
            raise Exception(
                f"No line match found at origin: {line_options['origin']} with dimensions: { line_options['dimensions']}"
            )
        return ordered_patch_corners, edge_contours_map

    def find_dot_corners_from_options(self, image, patch_type, _file_path):
        config = self.tuning_config
        options = self.options
        dot_options = options.get(patch_type, None)

        area, area_start = self.compute_scan_area_util(image, dot_options)

        # simple thresholding, maybe small morphology (extract self.options)

        # TODO: nope, first make it like a patch_area then get contour

        # Open : erode then dilate
        morph_c = cv2.morphologyEx(
            area, cv2.MORPH_OPEN, self.dot_kernel_morph, iterations=3
        )

        _, thresholded = cv2.threshold(morph_c, 200, 255, cv2.THRESH_TRUNC)

        if config.outputs.show_image_level >= 5:
            self.debug_hstack.append(morph_c)

        corners, _ = self.find_largest_patch_area_and_contours_map(
            area_start, thresholded, patch_type="dot"
        )
        if corners is None:
            hstack = ImageUtils.get_padded_hstack([self.debug_image, area, thresholded])
            InteractionUtils.show(f"No patch/dot found:", hstack, pause=1)
            raise Exception(
                f"No patch/dot found at origin: {dot_options['origin']} with dimensions: { dot_options['dimensions']}"
            )

        return corners, dot_options

    def find_largest_patch_area_and_contours_map(self, area_start, area, patch_type):
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
            patch_corners = MathUtils.get_rectangle_points(x, y, w, h)
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
