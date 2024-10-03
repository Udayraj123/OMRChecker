import math

import matplotlib.colors as colors
import numpy as np

from src.processors.constants import EdgeType
from src.utils.logger import logger


class MathUtils:
    """A Static-only Class to hold common math utilities & wrappers for easy integration with OpenCV and OMRChecker"""

    @staticmethod
    def distance(point1, point2):
        return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

    @staticmethod
    def shift_points_from_origin(new_origin, list_of_points):
        return list(
            map(
                lambda point: MathUtils.add_points(new_origin, point),
                list_of_points,
            )
        )

    @staticmethod
    def add_points(point, new_origin):
        return [
            new_origin[0] + point[0],
            new_origin[1] + point[1],
        ]

    @staticmethod
    def subtract_points(point, new_origin):
        return [
            point[0] - new_origin[0],
            point[1] - new_origin[1],
        ]

    @staticmethod
    def shift_points_to_origin(new_origin, list_of_points):
        return list(
            map(
                lambda point: MathUtils.subtract_points(point, new_origin),
                list_of_points,
            )
        )

    @staticmethod
    def get_point_on_line_by_ratio(edge_line, length_ratio):
        start, end = edge_line
        return [
            start[0] + (end[0] - start[0]) * length_ratio,
            start[1] + (end[1] - start[1]) * length_ratio,
        ]

    @staticmethod
    def order_four_points(points, dtype="int"):
        points = np.array(points, dtype=dtype)

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        sum = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        ordered_indices = [
            np.argmin(sum),
            np.argmin(diff),
            np.argmax(sum),
            np.argmax(diff),
        ]
        rect = points[ordered_indices]
        # returns the ordered coordinates (tl, tr, br, bl)
        return rect, ordered_indices

    @staticmethod
    def get_tuple_points(points):
        return [(int(point[0]), int(point[1])) for point in points]

    @staticmethod
    def get_bounding_box_of_points(points):
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        # returns the ordered coordinates (tl, tr, br, bl)
        bounding_box = np.array(
            [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        )
        box_dimensions = (int(max_x - min_x), int(max_y - min_y))
        return bounding_box, box_dimensions

    @staticmethod
    def validate_rect(approx):
        return len(approx) == 4 and MathUtils.check_max_cosine(approx.reshape(4, 2))

    @staticmethod
    def get_rectangle_points_from_box(origin, dimensions):
        x, y = origin
        w, h = dimensions
        # order same as order_four_points: (tl, tr, br, bl)
        return MathUtils.get_rectangle_points(x, y, w, h)

    @staticmethod
    def get_rectangle_points(x, y, w, h):
        # order same as order_four_points: (tl, tr, br, bl)
        return [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ]

    @staticmethod
    def select_edge_from_rectangle(rectangle, edge_type):
        tl, tr, br, bl = rectangle
        if edge_type == EdgeType.TOP:
            return [tl, tr]
        if edge_type == EdgeType.RIGHT:
            return [tr, br]
        if edge_type == EdgeType.BOTTOM:
            return [br, bl]
        if edge_type == EdgeType.LEFT:
            return [bl, tl]
        return [tl, tr]

    @staticmethod
    def rectangle_contains(point, rect):
        # Note: this function accepts a rectangle in a 4-tuple
        rect_start, rect_end = rect[0:2], rect[2:4]
        return not (
            point[0] < rect_start[0]
            or point[1] < rect_start[1]
            or point[0] > rect_end[0]
            or point[1] > rect_end[1]
        )

    @staticmethod
    def check_max_cosine(approx):
        # assumes 4 points present
        max_cosine = 0
        min_cosine = 1.5
        for i in range(2, 5):
            cosine = abs(MathUtils.angle(approx[i % 4], approx[i - 2], approx[i - 1]))
            max_cosine = max(cosine, max_cosine)
            min_cosine = min(cosine, min_cosine)

        if max_cosine >= 0.35:
            logger.warning("Quadrilateral is not a rectangle.")
            return False
        return True

    @staticmethod
    def angle(p_1, p_2, p_0):
        dx1 = float(p_1[0] - p_0[0])
        dy1 = float(p_1[1] - p_0[1])
        dx2 = float(p_2[0] - p_0[0])
        dy2 = float(p_2[1] - p_0[1])
        return (dx1 * dx2 + dy1 * dy2) / np.sqrt(
            (dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10
        )

    @staticmethod
    def check_collinear_points(point1, point2, point3):
        (
            [x1, y1],
            [x2, y2],
            [x3, y3],
        ) = (
            point1,
            point2,
            point3,
        )
        return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)

    @staticmethod
    def to_bgr(any_color):
        r, g, b = tuple(int(x * 255) for x in colors.to_rgb(any_color))
        return (b, g, r)

    @staticmethod
    def chunks(list, chunk_size):
        chunk_size = max(1, chunk_size)
        return (list[i : i + chunk_size] for i in range(0, len(list), chunk_size))
