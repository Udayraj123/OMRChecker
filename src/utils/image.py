import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.processors.constants import EdgeType
from src.utils.constants import CLR_BLACK, CLR_DARK_GRAY, CLR_GRAY, CLR_WHITE, TEXT_SIZE
from src.utils.logger import logger
from src.utils.math import MathUtils

plt.rcParams["figure.figsize"] = (10.0, 8.0)
CLAHE_HELPER = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))


class ImageUtils:
    """A Static-only Class to hold common image processing utilities & wrappers over OpenCV functions"""

    @staticmethod
    def read_image_util(file_path, tuning_config):
        if tuning_config.outputs.show_colored_outputs:
            colored_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            gray_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            colored_image = None
        return gray_image, colored_image

    @staticmethod
    def save_img(path, final_marked):
        logger.info(f"Saving Image to '{path}'")
        cv2.imwrite(path, final_marked)

    @staticmethod
    def resize_util(img, u_width, u_height=None):
        h, w = img.shape[:2]
        if u_height is None:
            u_height = int(h * u_width / w)

        if u_height == h and u_width == w:
            # No need to resize
            return img
        return cv2.resize(img, (int(u_width), int(u_height)))

    @staticmethod
    def resize_util_h(img, u_height, u_width=None):
        if u_width is None:
            h, w = img.shape[:2]
            u_width = int(w * u_height / h)
        return cv2.resize(img, (int(u_width), int(u_height)))

    @staticmethod
    def grab_contours(cnts):
        # source: imutils package

        # if the length the contours tuple returned by cv2.findContours
        # is '2' then we are using either OpenCV v2.4, v4-beta, or
        # v4-official
        if len(cnts) == 2:
            cnts = cnts[0]

        # if the length of the contours tuple is '3' then we are using
        # either OpenCV v3, v4-pre, or v4-alpha
        elif len(cnts) == 3:
            cnts = cnts[1]

        # otherwise OpenCV has changed their cv2.findContours return
        # signature yet again and I have no idea WTH is going on
        else:
            raise Exception(
                (
                    "Contours tuple must have length 2 or 3, "
                    "otherwise OpenCV changed their cv2.findContours return "
                    "signature yet again. Refer to OpenCV's documentation "
                    "in that case"
                )
            )

        # return the actual contours array
        return cnts

    @staticmethod
    def normalize(img, alpha=0, beta=255):
        return cv2.normalize(img, alpha, beta, norm_type=cv2.NORM_MINMAX)

    @staticmethod
    def auto_canny(image, sigma=0.93):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged

    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    @staticmethod
    def get_control_destination_points_from_contour(
        edge_contour, edge_line, max_points=None
    ):
        logger.info(f"edge_contour={edge_contour}")
        logger.info(f"edge_line={edge_line}")
        total_points = len(edge_contour)
        if max_points is None:
            max_points = total_points
        start, end = edge_line

        contour_length = 0
        for i in range(1, total_points):
            contour_length += MathUtils.distance(edge_contour[i], edge_contour[i - 1])

        # TODO: replace with this if the assertion passes on uncommenting
        # contour_length = cv2.arcLength(edge_contour)

        average_min_gap = (contour_length / (max_points - 1)) - 1

        # Initialize with first point mapping
        control_points, destination_points = [edge_contour[0]], [start]
        current_arc_length = 0
        current_arc_gap = 0
        previous_point = None
        for i in range(1, total_points):
            boundary_point, previous_point = edge_contour[i], edge_contour[i - 1]
            edge_length = MathUtils.distance(previous_point, boundary_point)
            current_arc_gap += edge_length
            if current_arc_gap > average_min_gap:
                current_arc_gap = 0
                current_arc_length += edge_length
                length_ratio = current_arc_length / contour_length
                destination_point = MathUtils.get_point_on_line_by_ratio(
                    edge_line, length_ratio
                )
                control_points.append(boundary_point)
                destination_points.append(destination_point)

        assert current_arc_length == contour_length
        assert MathUtils.distance(destination_points[-1], end) < 1.0

        return control_points, destination_points

    @staticmethod
    def get_cropped_rectangle_destination_points(ordered_corner_points):
        (tl, tr, br, bl) = ordered_corner_points

        length_t = MathUtils.distance(tr, tl)
        length_b = MathUtils.distance(br, bl)
        length_r = MathUtils.distance(tr, br)
        length_l = MathUtils.distance(tl, bl)

        # compute the width of the new image, which will be the
        max_width = max(int(length_t), int(length_b))

        # compute the height of the new image, which will be the
        max_height = max(int(length_r), int(length_l))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image

        destination_points = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )

        rectangle_dimensions = (max_width, max_height)
        return destination_points, rectangle_dimensions

    @staticmethod
    def split_patch_contour_on_corners(patch_corners, bounding_contour=None):
        ordered_patch_corners = MathUtils.order_four_points(
            patch_corners, dtype="float32"
        )
        tl, tr, br, bl = ordered_patch_corners
        # First element of each contour should necessarily start & end with a corner point
        edge_contours_map = {
            EdgeType.TOP: [tl],
            EdgeType.RIGHT: [tr],
            EdgeType.BOTTOM: [br],
            EdgeType.LEFT: [bl],
        }

        # TODO: loop over boundary points in the bounding_contour and split them according to given corner points

        # Note: Each contour's points should be in the clockwise order
        # TODO: Need clockwise edge contours: top, right, bottom, left
        # TODO: Split the page_contour into 4 lines using the corner points
        # Each contour will at-least contain the two corner points

        # Can also readily generate reference points as per the given corner points(non-shifted)

        # Assure contour always covers the edge points
        edge_contours_map[EdgeType.TOP].append(tr)
        edge_contours_map[EdgeType.RIGHT].append(br)
        edge_contours_map[EdgeType.BOTTOM].append(bl)
        edge_contours_map[EdgeType.LEFT].append(tl)

        return ordered_patch_corners, edge_contours_map

    @staticmethod
    def get_vstack_image_grid(debug_vstack):
        padded_hstack = [
            ImageUtils.get_padded_hstack(hstack) for hstack in debug_vstack
        ]
        return ImageUtils.get_padded_vstack(padded_hstack)

    @staticmethod
    def get_padded_hstack(hstack):
        max_height = max(image.shape[0] for image in hstack)
        padded_hstack = [
            ImageUtils.pad_image_to_height(image, max_height) for image in hstack
        ]

        return np.hstack(padded_hstack)

    @staticmethod
    def get_padded_vstack(vstack):
        max_width = max(image.shape[1] for image in vstack)
        padded_vstack = [
            ImageUtils.pad_image_to_width(image, max_width) for image in vstack
        ]

        return np.vstack(padded_vstack)

    @staticmethod
    def pad_image_to_height(image, max_height, value=CLR_WHITE):
        return cv2.copyMakeBorder(
            image,
            0,
            max_height - image.shape[0],
            0,
            0,
            cv2.BORDER_CONSTANT,
            value,
        )

    @staticmethod
    def pad_image_to_width(image, max_width, value=CLR_WHITE):
        return cv2.copyMakeBorder(
            image,
            0,
            0,
            0,
            max_width - image.shape[1],
            cv2.BORDER_CONSTANT,
            value,
        )

    def pad_image_from_center(image, padding_width, padding_height=0, value=255):
        input_width, input_height = image.shape[:2]
        bounding_box = [
            padding_width,
            padding_width + input_width,
            padding_height,
            padding_height + input_height,
        ]
        white = value * np.ones(
            (padding_width * 2 + input_width, padding_height * 2 + input_height),
            np.uint8,
        )
        white[
            bounding_box[0] : bounding_box[1], bounding_box[2] : bounding_box[3]
        ] = image

        return white, bounding_box

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
    def draw_box(
        image,
        position,
        box_dimensions,
        color=None,
        style="BOX_HOLLOW",
        thickness_factor=1 / 12,
        border=3,
    ):
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

        if style == "BOX_HOLLOW":
            if color is None:
                color = CLR_GRAY
        elif style == "BOX_FILLED":
            if color is None:
                color = CLR_DARK_GRAY
            border = -1
        ImageUtils.draw_box_diagonal(
            image,
            position,
            position_diagonal,
            color,
            border,
        )

    @staticmethod
    def draw_text(
        image,
        text_value,
        position,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
        text_size=TEXT_SIZE,
        color=CLR_BLACK,
        thickness=2,
        # available LineTypes: FILLED, LINE_4, LINE_8, LINE_AA
        line_type=cv2.LINE_AA,
    ):
        if callable(position):
            size_x, size_y = cv2.getTextSize(
                text_value,
                font_face,
                text_size,
                thickness,
            )[0]
            position = position(size_x, size_y)

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
