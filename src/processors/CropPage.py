import cv2
import numpy as np

from src.processors.constants import EDGE_TYPES_IN_ORDER, WarpMethod
from src.processors.internal.WarpOnPointsCommon import WarpOnPointsCommon
from src.utils.constants import CLR_WHITE, hsv_white_high, hsv_white_low
from src.utils.drawing import DrawingUtils
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils

MIN_PAGE_AREA = 80000

"""
ref: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
"""


class CropPage(WarpOnPointsCommon):
    __is_internal_preprocessor__ = False

    def get_class_name(self):
        return "CropPage"

    def validate_and_remap_options_schema(self, options):
        tuning_options = options.get("tuningOptions", {})

        parsed_options = {
            "morphKernel": options.get("morphKernel"),
            "useColoredCanny": options.get("useColoredCanny"),
            "enableCropping": True,
            "tuningOptions": {
                "warpMethod": tuning_options.get(
                    "warpMethod", WarpMethod.PERSPECTIVE_TRANSFORM
                ),
                "normalizeConfig": [],
                "cannyConfig": [],
            },
        }
        return parsed_options

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        options = self.options
        self.use_colored_canny = options.get("useColoredCanny", False)

        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, tuple(options.get("morphKernel", (10, 10)))
        )

    def __str__(self):
        return f"CropPage"

    def prepare_image(self, image):
        return ImageUtils.normalize(image)

    def extract_control_destination_points(self, image, colored_image, file_path):
        options = self.options

        sheet, page_contour = self.find_page_contour_and_corners(
            image, colored_image, file_path
        )
        (
            ordered_page_corners,
            edge_contours_map,
        ) = ImageUtils.split_patch_contour_on_corners(sheet, page_contour)

        logger.debug(f"Found page corners: \t {ordered_page_corners}")
        (
            destination_page_corners,
            _,
        ) = ImageUtils.get_cropped_warped_rectangle_points(ordered_page_corners)

        if self.warp_method in [
            WarpMethod.DOC_REFINE,
            WarpMethod.PERSPECTIVE_TRANSFORM,
        ]:
            return ordered_page_corners, destination_page_corners, edge_contours_map
        else:
            # TODO: remove this if REMAP method is removed, see if homography REALLY needs page contour points
            max_points_per_edge = options.get("maxPointsPerEdge", None)

            control_points, destination_points = [], []
            for edge_type in EDGE_TYPES_IN_ORDER:
                destination_line = MathUtils.select_edge_from_rectangle(
                    destination_page_corners, edge_type
                )
                # Extrapolates the destination_line to get approximate destination points
                (
                    edge_control_points,
                    edge_destination_points,
                ) = ImageUtils.get_control_destination_points_from_contour(
                    edge_contours_map[edge_type], destination_line, max_points_per_edge
                )
                # Note: edge-wise duplicates would get added here
                # TODO: see if we can avoid duplicates at source itself
                control_points += edge_control_points
                destination_points += edge_destination_points

            return control_points, destination_points, edge_contours_map

    def find_page_contour_and_corners(self, image, colored_image, file_path):
        config = self.tuning_config

        if self.use_colored_canny and not config.outputs.colored_outputs_enabled:
            logger.warning(
                f"Cannot process colored image for CropPage. useColoredCanny is true but colored_outputs_enabled is false."
            )

        _ret, image = cv2.threshold(image, 200, 255, cv2.THRESH_TRUNC)
        image = ImageUtils.normalize(image)

        self.append_save_image("Truncate Threshold", [1, 4, 5, 6], image)

        if self.use_colored_canny and config.outputs.colored_outputs_enabled:
            hsv = cv2.cvtColor(colored_image, cv2.COLOR_BGR2HSV)
            # Mask image to only select white-ish zone
            mask = cv2.inRange(hsv, hsv_white_low, hsv_white_high)
            mask_result = cv2.bitwise_and(image, image, mask=mask)
            self.append_save_image("Mask Result", range(3, 7), mask_result)
            # TODO: get hsv mask working for colored separation
            # TODO: test this on more samples
            # InteractionUtils.show("hsv", hsv, 0)
            # InteractionUtils.show("colored_image", colored_image, 0)
            # InteractionUtils.show("mask_result", mask_result, 1)

            # TODO: self.append_save_image(2, mask_result)

            canny_edge = cv2.Canny(mask_result, 185, 55)
        else:
            _ret, image = cv2.threshold(image, 200, 255, cv2.THRESH_TRUNC)

            image = ImageUtils.normalize(image)

            # Close the small holes, i.e. Complete the edges on canny image
            closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.morph_kernel)

            self.append_save_image("Morph Page", range(3, 7), closed)

            # TODO: self.append_save_image(2, closed)

            # TODO: parametrize these tuning params
            canny_edge = cv2.Canny(closed, 185, 55)

        self.append_save_image("Canny Edges", range(5, 7), canny_edge)

        # findContours returns outer boundaries in CW and inner ones, ACW.
        all_contours = ImageUtils.grab_contours(
            cv2.findContours(
                canny_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )  # , cv2.CHAIN_APPROX_NONE)
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
                page_contour = np.vstack(bounding_contour).squeeze()
                DrawingUtils.draw_contour(
                    canny_edge, approx, color=CLR_WHITE, thickness=10
                )
                DrawingUtils.draw_contour(
                    self.debug_image, approx, color=CLR_WHITE, thickness=10
                )

                self.append_save_image("Bounding Contour", range(1, 7), canny_edge)
                break

        if config.outputs.show_image_level >= 6 or (
            page_contour is None and config.outputs.show_image_level >= 1
        ):
            hstack = ImageUtils.get_padded_hstack([image, closed, canny_edge])

            InteractionUtils.show("Page edges detection", hstack)

        if page_contour is None:
            logger.error(f"Error: Paper boundary not found for: '{file_path}'")
            logger.warning(
                f"Have you accidentally included CropPage preprocessor?\nIf no, increase the processing dimensions from config. Current image size used: {image.shape[:2]}"
            )
            raise Exception("Paper boundary not found")
        return sheet, page_contour
