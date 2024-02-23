"""
https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
"""
import cv2
import numpy as np

from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger

MIN_PAGE_AREA = 80000


def normalize(image):
    return cv2.normalize(image, 0, 255, norm_type=cv2.NORM_MINMAX)


class CropPage(ImageTemplatePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cropping_ops = self.options
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, tuple(cropping_ops.get("morphKernel", (10, 10)))
        )

    def apply_filter(self, image, _template, file_path):
        image = normalize(cv2.GaussianBlur(image, (3, 3), 0))
        # Resize should be done with another preprocessor is needed
        sheet = self.find_page(image, file_path)
        if len(sheet) == 0:
            logger.error(f"Error: Paper boundary not found for: '{file_path}'")
            logger.warning(
                f"Have you accidentally included CropPage preprocessor?\nIf no, increase the processing dimensions from config. Current image size used: {image.shape[:2]}"
            )
            raise Exception("Paper boundary not found")

        logger.info(f"Found page corners: \t {sheet.tolist()}")

        # Warp layer 1
        image = ImageUtils.four_point_transform(image, sheet)

        # Return preprocessed image
        return image

    def find_page(self, image, file_path):
        config = self.tuning_config

        image = normalize(image)

        _ret, image = cv2.threshold(image, 200, 255, cv2.THRESH_TRUNC)
        image = normalize(image)

        # Close the small holes, i.e. Complete the edges on canny image
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.morph_kernel)

        # TODO: parametrize these tuning params
        canny_edge = cv2.Canny(closed, 185, 55)

        # findContours returns outer boundaries in CW and inner ones, ACW.
        cnts = ImageUtils.grab_contours(
            cv2.findContours(canny_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        )
        # convexHull to resolve disordered curves due to noise
        cnts = [cv2.convexHull(c) for c in cnts]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        sheet = []
        for c in cnts:
            if cv2.contourArea(c) < MIN_PAGE_AREA:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon=0.025 * peri, closed=True)
            if ImageUtils.validate_rect(approx):
                sheet = np.reshape(approx, (4, -1))
                cv2.drawContours(canny_edge, [approx], -1, (255, 255, 255), 10)
                break

        if config.outputs.show_image_level >= 6:
            hstack = ImageUtils.get_padded_hstack([image, closed, canny_edge])

            InteractionUtils.show("Page edges detection", hstack, config=config)

        return sheet
