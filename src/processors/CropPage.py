"""
https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
"""

import cv2
import numpy as np

from src.processors.interfaces.CropOnIndexPoints import CropOnIndexPoints
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger

MIN_PAGE_AREA = 80000


class CropPage(CropOnIndexPoints):
    __is_internal_preprocessor__ = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        options = self.options
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, tuple(options.get("morphKernel", (10, 10)))
        )

    def __str__(self):
        return f"CropPage"

    def prepare_image(self, image):
        return ImageUtils.normalize(image)

    def find_four_corners(self, image, file_path):
        config = self.tuning_config

        _ret, image = cv2.threshold(image, 200, 255, cv2.THRESH_TRUNC)
        image = ImageUtils.normalize(image)

        # Close the small holes, i.e. Complete the edges on canny image
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.morph_kernel)

        # TODO: self.append_save_image(2, closed)

        # TODO: parametrize these tuning params
        canny_edge = cv2.Canny(closed, 185, 55)

        # TODO: self.append_save_image(3, canny_edge)

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

                # TODO: self.append_save_image(2, canny_edge)
                break

        if config.outputs.show_image_level >= 6:
            hstack = ImageUtils.get_padded_hstack([image, closed, canny_edge])

            InteractionUtils.show("Page edges detection", hstack, config=config)

        if len(sheet) == 0:
            logger.error(f"Error: Paper boundary not found for: '{file_path}'")
            logger.warning(
                f"Have you accidentally included CropPage preprocessor?\nIf no, increase the processing dimensions from config. Current image size used: {image.shape[:2]}"
            )
            raise Exception("Paper boundary not found")

        logger.info(f"Found page corners: \t {sheet.tolist()}")
        return sheet
