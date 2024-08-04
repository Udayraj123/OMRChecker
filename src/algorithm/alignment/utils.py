import cv2
import numpy as np

from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils


def show_displacement_overlay(
    block_gray_alignment_image, block_gray_image, shifted_block_image
):
    # ..
    overlay = ImageUtils.overlay_image(block_gray_alignment_image, block_gray_image)
    overlay_shifted = ImageUtils.overlay_image(
        block_gray_alignment_image, shifted_block_image
    )

    InteractionUtils.show(
        "Alignment For Field Block",
        ImageUtils.get_padded_hstack(
            [
                block_gray_alignment_image,
                block_gray_image,
                shifted_block_image,
                overlay,
                overlay_shifted,
            ]
        ),
    )


def hough_circles(gray):
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        1,
        rows / 10,
        param1=100,
        param2=30,
        minRadius=1,
        maxRadius=30,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # logger.info("circles", circles)
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(gray, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(gray, center, radius, (255, 0, 255), 3)
    return gray
