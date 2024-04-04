"""
https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
"""

import cv2
import numpy as np

from src.processors.constants import EDGE_TYPES_IN_ORDER
from src.processors.internal.WarpOnPointsCommon import WarpOnPointsCommon
from src.utils.constants import CLR_WHITE
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils

MIN_PAGE_AREA = 80000


class CropPage(WarpOnPointsCommon):
    __is_internal_preprocessor__ = False

    def validate_and_remap_options_schema(self, options):
        parsed_options = {"enableCropping": True, **options}
        return parsed_options

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

    def extract_control_destination_points(self, image, file_path):
        config = self.tuning_config
        options = self.options
        max_points_per_side = options.get("maxPointsPerSide", None)

        _ret, image = cv2.threshold(image, 200, 255, cv2.THRESH_TRUNC)
        image = ImageUtils.normalize(image)

        # Close the small holes, i.e. Complete the edges on canny image
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.morph_kernel)

        # TODO: self.append_save_image(2, closed)

        # TODO: parametrize these tuning params
        canny_edge = cv2.Canny(closed, 185, 55)

        # TODO: self.append_save_image(3, canny_edge)

        # findContours returns outer boundaries in CW and inner ones, ACW.
        all_contours = ImageUtils.grab_contours(
            cv2.findContours(canny_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        )
        # convexHull to resolve disordered curves due to noise
        all_contours = [
            cv2.convexHull(bounding_contour) for bounding_contour in all_contours
        ]
        all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)[:5]
        sheet = []
        page_contour = None
        for bounding_contour in all_contours:
            if cv2.contourArea(bounding_contour) < MIN_PAGE_AREA:
                continue
            peri = cv2.arcLength(bounding_contour, True)
            approx = cv2.approxPolyDP(
                bounding_contour, epsilon=0.025 * peri, closed=True
            )
            if MathUtils.validate_rect(approx):
                sheet = np.reshape(approx, (4, -1))
                page_contour = bounding_contour
                ImageUtils.draw_contour(
                    canny_edge, approx, color=CLR_WHITE, thickness=10
                )
                ImageUtils.draw_contour(
                    self.debug_image, approx, color=CLR_WHITE, thickness=10
                )

                # TODO: self.append_save_image(2, canny_edge)
                break

        if config.outputs.show_image_level >= 6:
            hstack = ImageUtils.get_padded_hstack([image, closed, canny_edge])

            InteractionUtils.show("Page edges detection", hstack, config=config)

        if page_contour is None:
            logger.error(f"Error: Paper boundary not found for: '{file_path}'")
            logger.warning(
                f"Have you accidentally included CropPage preprocessor?\nIf no, increase the processing dimensions from config. Current image size used: {image.shape[:2]}"
            )
            raise Exception("Paper boundary not found")

        (
            ordered_page_corners,
            edge_contours_map,
        ) = ImageUtils.split_patch_contour_on_corners(sheet, page_contour)
        destination_page_corners = self.get_cropped_rectangle_destination_points(
            ordered_page_corners
        )

        logger.info(f"Found page corners: \t {ordered_page_corners}")
        # Store control points in order
        control_points, destination_points = [], []
        for edge_type in EDGE_TYPES_IN_ORDER:
            source_contour = edge_contours_map[edge_type]
            destination_line = MathUtils.select_edge_from_rectangle(
                destination_page_corners, edge_type
            )
            # Extrapolates the destination_line to get approximate destination points
            (
                edge_control_points,
                edge_destination_points,
            ) = ImageUtils.get_control_destination_points_from_contour(
                source_contour, destination_line, max_points_per_side
            )
            control_points += edge_control_points
            destination_points += edge_destination_points

        return control_points, destination_points

    @staticmethod
    def get_cropped_rectangle_destination_points(ordered_page_corners):
        # Note: This utility would just find a good size ratio for the cropped image to look more realistic
        # but since we're anyway resizing the image, it doesn't make much sense to use these calculations
        (tl, tr, br, bl) = ordered_page_corners

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

        return destination_points
