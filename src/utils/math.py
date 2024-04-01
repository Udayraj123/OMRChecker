import math

import numpy as np

from src.utils.logger import logger


class MathUtils:
    """A Static-only Class to hold common math utilities & wrappers for easy integration with OpenCV and OMRChecker"""

    # TODO: move into math utils
    @staticmethod
    def distance(point1, point2):
        return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

    @staticmethod
    def get_point_on_line_by_ratio(edge_line, length_ratio):
        start, end = edge_line
        return [
            start[0] + (end[0] - start[0]) * length_ratio,
            start[1] + (end[1] - start[1]) * length_ratio,
        ]

    @staticmethod
    def order_four_points(points, dtype="int"):
        points = np.array(points)
        rect = np.zeros((4, 2), dtype=dtype)

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]

        # return the ordered coordinates (tl, tr, br, bl)
        return rect

    @staticmethod
    def validate_rect(approx):
        return len(approx) == 4 and MathUtils.check_max_cosine(approx.reshape(4, 2))

    @staticmethod
    def get_rectangle_points(x, y, w, h):
        # order same as order_four_points: (tl, tr, br, bl)
        return np.intp(
            [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ]
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
