"""
https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
"""
import cv2
import numpy as np

from src.constants.image_processing import (
    APPROX_POLY_EPSILON_FACTOR,
    CANNY_PARAMS,
    DEFAULT_CONTOUR_COLOR,
    DEFAULT_CONTOUR_FILL_COLOR,
    DEFAULT_CONTOUR_FILL_WIDTH,
    DEFAULT_CONTOUR_LINE_WIDTH,
    DEFAULT_GAUSSIAN_BLUR_KERNEL,
    MAX_COSINE_THRESHOLD,
    MIN_PAGE_AREA_THRESHOLD,
    PAGE_THRESHOLD_PARAMS,
)
from src.logger import logger
from src.processors.interfaces.ImagePreprocessor import ImagePreprocessor
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils


def normalize(image):
    return cv2.normalize(image, 0, 255, norm_type=cv2.NORM_MINMAX)


def check_max_cosine(approx):
    # assumes 4 pts present
    max_cosine = 0
    min_cosine = 1.5
    for i in range(2, 5):
        cosine = abs(angle(approx[i % 4], approx[i - 2], approx[i - 1]))
        max_cosine = max(cosine, max_cosine)
        min_cosine = min(cosine, min_cosine)

    if max_cosine >= MAX_COSINE_THRESHOLD:
        logger.warning("Quadrilateral is not a rectangle.")
        return False
    return True


def validate_rect(approx):
    return len(approx) == 4 and check_max_cosine(approx.reshape(4, 2))


def angle(p_1, p_2, p_0):
    dx1 = float(p_1[0] - p_0[0])
    dy1 = float(p_1[1] - p_0[1])
    dx2 = float(p_2[0] - p_0[0])
    dy2 = float(p_2[1] - p_0[1])
    return (dx1 * dx2 + dy1 * dy2) / np.sqrt(
        (dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10
    )


class CropPage(ImagePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cropping_ops = self.options
        self.morph_kernel = tuple(
            int(x) for x in cropping_ops.get("morphKernel", [10, 10])
        )

    def apply_filter(self, image, file_path):
        image = normalize(cv2.GaussianBlur(image, DEFAULT_GAUSSIAN_BLUR_KERNEL, 0))

        # Resize should be done with another preprocessor is needed
        sheet = self.find_page(image, file_path)
        if len(sheet) == 0:
            logger.error(
                f"\tError: Paper boundary not found for: '{file_path}'\nHave you accidentally included CropPage preprocessor?"
            )
            return None

        logger.info(f"Found page corners: \t {sheet.tolist()}")

        # Warp layer 1
        image = ImageUtils.four_point_transform(image, sheet)

        # Return preprocessed image
        return image

    def find_page(self, image, file_path):
        config = self.tuning_config

        image = normalize(image)

        _ret, image = cv2.threshold(
            image,
            PAGE_THRESHOLD_PARAMS["threshold_value"],
            PAGE_THRESHOLD_PARAMS["max_pixel_value"],
            cv2.THRESH_TRUNC,
        )
        image = normalize(image)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel)

        # Close the small holes, i.e. Complete the edges on canny image
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        edge = cv2.Canny(
            closed, CANNY_PARAMS["lower_threshold"], CANNY_PARAMS["upper_threshold"]
        )

        if config.outputs.show_image_level >= 5:
            InteractionUtils.show("edge", edge, config=config)

        # findContours returns outer boundaries in CW and inner ones, ACW.
        cnts = ImageUtils.grab_contours(
            cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        )
        # convexHull to resolve disordered curves due to noise
        cnts = [cv2.convexHull(c) for c in cnts]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        sheet = []
        for c in cnts:
            if cv2.contourArea(c) < MIN_PAGE_AREA_THRESHOLD:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(
                c, epsilon=APPROX_POLY_EPSILON_FACTOR * peri, closed=True
            )
            if validate_rect(approx):
                sheet = np.reshape(approx, (4, -1))
                cv2.drawContours(
                    image,
                    [approx],
                    -1,
                    DEFAULT_CONTOUR_COLOR,
                    DEFAULT_CONTOUR_LINE_WIDTH,
                )
                cv2.drawContours(
                    edge,
                    [approx],
                    -1,
                    DEFAULT_CONTOUR_FILL_COLOR,
                    DEFAULT_CONTOUR_FILL_WIDTH,
                )
                break

        return sheet
