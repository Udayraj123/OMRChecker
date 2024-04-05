import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils.constants import CLR_BLACK, CLR_DARK_GRAY, CLR_GRAY, CLR_WHITE, TEXT_SIZE
from src.utils.logger import logger

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
    def get_four_destination_points(ordered_corner_points):
        (tl, tr, br, bl) = ordered_corner_points

        # compute the width of the new image, which will be the
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        max_width = max(int(width_a), int(width_b))
        # max_width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))

        # compute the height of the new image, which will be the
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        # max_height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-br)))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        destination_points = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )
        return destination_points, max_width, max_height

    @staticmethod
    def four_point_transform(image, corner_points):
        # obtain a consistent order of the points and unpack them
        # individually
        ordered_corner_points = ImageUtils.order_four_points(
            corner_points, dtype="float32"
        )

        (
            destination_points,
            max_width,
            max_height,
        ) = ImageUtils.get_four_destination_points(ordered_corner_points)
        transform_matrix = cv2.getPerspectiveTransform(
            ordered_corner_points, destination_points
        )
        warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

        # return the warped image
        return warped

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
        return len(approx) == 4 and ImageUtils.check_max_cosine(approx.reshape(4, 2))

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
            cosine = abs(ImageUtils.angle(approx[i % 4], approx[i - 2], approx[i - 1]))
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

        cv2.rectangle(
            image,
            position,
            position_diagonal,
            color,
            border,
        )
        return position,position_diagonal

    
    @staticmethod
    def draw_text(
        image,
        text_value,
        position,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        text_size=TEXT_SIZE,
        color=CLR_BLACK,
        thickness=2,
    ):
        if callable(position):
            size_x, size_y = cv2.getTextSize(
                text_value,
                font,
                text_size,
                thickness,
            )[0]
            position = position(size_x, size_y)

        cv2.putText(
            image,
            text_value,
            position,
            font,
            text_size,
            color,
            thickness,
        )

    @staticmethod
    def draw_symbol(
        image,
        symbol,
        position,
        position_diagonal,
        color=CLR_BLACK
     ):
        
        center_position = lambda size_x , size_y : (
            (position[0]+position_diagonal[0]-size_x)//2,
            (position[1]+position_diagonal[1]+size_y)//2
        )
        

        ImageUtils.draw_text(
            image,
            symbol,
            center_position,
            color=color
        )
        

        

