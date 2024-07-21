import cv2
import numpy as np

from src.utils.constants import CLR_BLACK, CLR_DARK_GRAY, CLR_GRAY, CLR_GREEN, TEXT_SIZE
from src.utils.image import ImageUtils
from src.utils.math import MathUtils


class DrawingUtils:
    @staticmethod
    def draw_matches(image, from_points, warped_image, to_points):
        horizontal_stack = ImageUtils.get_padded_hstack([image, warped_image])
        h, w = image.shape[:2]
        from_points = MathUtils.get_tuple_points(from_points)
        to_points = MathUtils.get_tuple_points(to_points)
        for from_point, to_point in zip(from_points, to_points):
            horizontal_stack = cv2.line(
                horizontal_stack,
                from_point,
                (w + to_point[0], to_point[1]),
                color=CLR_GREEN,
                thickness=3,
            )
        return horizontal_stack

    @staticmethod
    def draw_box_diagonal(
        image,
        position,
        position_diagonal,
        color=CLR_DARK_GRAY,
        border=3,
    ):
        cv2.rectangle(
            image,
            position,
            position_diagonal,
            color,
            border,
        )

    @staticmethod
    def draw_contour(
        image,
        contour,
        color=CLR_GREEN,
        thickness=2,
    ):
        assert None not in contour, "Invalid contour provided"
        cv2.drawContours(
            image,
            [np.intp(contour)],
            contourIdx=-1,
            color=color,
            thickness=thickness,
        )

    @staticmethod
    def draw_box(
        image,
        position,
        box_dimensions,
        color=None,
        style="BOX_HOLLOW",
        thickness_factor=1 / 12,
        border=3,
        centered=False,
    ):
        assert position is not None
        x, y = position
        box_w, box_h = box_dimensions

        position = (
            int(x + box_w * thickness_factor),
            int(y + box_h * thickness_factor),
        )
        position_diagonal = (
            int(x + box_w - box_w * thickness_factor),
            int(y + box_h - box_h * thickness_factor),
        )

        if centered:
            centered_position = [
                (3 * position[0] - position_diagonal[0]) // 2,
                (3 * position[1] - position_diagonal[1]) // 2,
            ]
            centered_diagonal = [
                (position[0] + position_diagonal[0]) // 2,
                (position[1] + position_diagonal[1]) // 2,
            ]
            position = centered_position
            position_diagonal = centered_diagonal

        if style == "BOX_HOLLOW":
            if color is None:
                color = CLR_GRAY
        elif style == "BOX_FILLED":
            if color is None:
                color = CLR_DARK_GRAY
            border = -1

        DrawingUtils.draw_box_diagonal(
            image,
            position,
            position_diagonal,
            color,
            border,
        )
        return position, position_diagonal

    @staticmethod
    def draw_arrows(
        image,
        start_points,
        end_points,
        color=CLR_GREEN,
        thickness=2,
        line_type=cv2.LINE_AA,
        tip_length=0.1,
    ):
        start_points = MathUtils.get_tuple_points(start_points)
        end_points = MathUtils.get_tuple_points(end_points)
        for start_point, end_point in zip(start_points, end_points):
            image = cv2.arrowedLine(
                image,
                start_point,
                end_point,
                color,
                thickness,
                line_type,
                tipLength=tip_length,
            )

        return image

    @staticmethod
    def draw_text(
        image,
        text_value,
        position,
        text_size=TEXT_SIZE,
        thickness=2,
        centered=False,
        color=CLR_BLACK,
        # available LineTypes: FILLED, LINE_4, LINE_8, LINE_AA
        line_type=cv2.LINE_AA,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
    ):
        if centered:
            assert not callable(
                position
            ), f"centered={centered} but position={position}"
            text_position = position
            position = lambda size_x, size_y: (
                text_position[0] - size_x // 2,
                text_position[1] + size_y // 2,
            )

        if callable(position):
            size_x, size_y = cv2.getTextSize(
                text_value,
                font_face,
                text_size,
                thickness,
            )[0]
            position = position(size_x, size_y)

        position = (int(position[0]), int(position[1]))
        cv2.putText(
            image,
            text_value,
            position,
            font_face,
            text_size,
            color,
            thickness,
            lineType=line_type,
        )

    @staticmethod
    def draw_symbol(image, symbol, position, position_diagonal, color=CLR_BLACK):
        center_position = lambda size_x, size_y: (
            (position[0] + position_diagonal[0] - size_x) // 2,
            (position[1] + position_diagonal[1] + size_y) // 2,
        )

        DrawingUtils.draw_text(image, symbol, center_position, color=color)

    @staticmethod
    def draw_line(image, start, end, color=CLR_BLACK, thickness=3):
        cv2.line(image, start, end, color, thickness)

    @staticmethod
    def draw_polygon(image, points, color=CLR_BLACK, thickness=1, closed=True):
        n = len(points)
        for i in range(n):
            if not closed and i == n - 1:
                continue
            DrawingUtils.draw_line(
                image, points[i % n], points[(i + 1) % n], color, thickness
            )

    @staticmethod
    def draw_group(
        image,
        start,
        bubble_dimensions,
        box_edge,
        color,
        thickness=3,
        thickness_factor=7 / 10,
    ):
        start_x, start_y = start
        box_w, box_h = bubble_dimensions
        if box_edge == "TOP":
            end_position = (start_x + int(box_w * thickness_factor), start_y)
            start = (start_x + int(box_w * (1 - thickness_factor)), start_y)
            DrawingUtils.draw_line(image, start, end_position, color, thickness)
        elif box_edge == "RIGHT":
            start = (start_x + box_w, start_y)
            end_position = (start_x, int(start_y + box_h * thickness_factor))
            start = (start_x, int(start_y + box_h * (1 - thickness_factor)))
            DrawingUtils.draw_line(image, start, end_position, color, thickness)
        elif box_edge == "BOTTOM":
            start = (start_x, start_y + box_h)
            end_position = (int(start_x + box_w * thickness_factor), start_y)
            start = (int(start_x + box_w * (1 - thickness_factor)), start_y)
            DrawingUtils.draw_line(image, start, end_position, color, thickness)
        elif box_edge == "LEFT":
            end_position = (start_x, int(start_y + box_h * thickness_factor))
            start = (start_x, int(start_y + box_h * (1 - thickness_factor)))
            DrawingUtils.draw_line(image, start, end_position, color, thickness)
