import cv2

from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger

# TODO: apply a more generic way to transform the document into better warped configuration
# https://github.com/gwxie/Document-Dewarping-with-Control-Points

# Or even generic to de-warp the document


# Internal Processor for separation of code
class CropOnIndexPoints(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: add validation
        # self.type = IndexPointsType.FOUR_POINTS
        # - add validation for the points
        # - If type is displacement: single point
        # - If type is page_crop: 4 points as a convex polygon,
        # - If type is page_crop_smooth: >= 4 points as a convex polygon
        # - If type is template_align: max displacement should be controlled by scanbox?

    def exclude_files(self):
        return []

    def prepare_image(self, image):
        return image

    def apply_filter(self, image, colored_image, _template, file_path):
        config = self.tuning_config
        cropping_enabled = config.get("enableCropping", False)

        self.debug_image = image.copy()
        self.debug_hstack = []
        self.debug_vstack = []

        image = self.prepare_image(image)

        # TODO: Save intuitive meta data
        # self.append_save_image(3,warped_image)

        # cropping_enabled: destination points based on relative origin(extracted, topLeftDot) and a bounding box
        ordered_corner_points, edge_contours_map = self.find_corners_and_edges(
            image, file_path
        )

        # For page contour: split the contour between the approx corners somehow
        (
            control_points,
            destination_points,
            max_width,
            max_height,
        ) = ImageUtils.get_control_and_destination_points(
            ordered_corner_points, edge_contours_map
        )
        logger.info(f"destination_points={destination_points}, max_width={max_width}")

        # Find and pass control points in a defined order
        # For dots/lines: clockwise direction, with radially closer points first
        transform_matrix = cv2.getPerspectiveTransform(
            control_points, destination_points
        )

        # Crop the image
        warped_image = cv2.warpPerspective(
            image, transform_matrix, (max_width, max_height)
        )

        if config.outputs.show_colored_outputs:
            colored_image = cv2.warpPerspective(
                colored_image, transform_matrix, (max_width, max_height)
            )

        # self.append_save_image(1,warped_image)

        if config.outputs.show_image_level >= 4:
            hstack = ImageUtils.get_padded_hstack([self.debug_image, warped_image])
            InteractionUtils.show(
                f"warped_image: {file_path}", hstack, 1, 1, config=config
            )

        return warped_image, colored_image, _template
