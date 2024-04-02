import cv2
import numpy as np

from src.processors.constants import HomographyMethod
from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils


# Internal Processor for separation of code
class WarpOnPointsCommon(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = True

    homography_method_map = {
        HomographyMethod.INTER_LINEAR: cv2.INTER_LINEAR,
        HomographyMethod.INTER_CUBIC: cv2.INTER_CUBIC,
    }

    def validate_and_remap_options_schema(self):
        raise Exception(f"Not implemented")

    def __init__(self, options, relative_dir, image_instance_ops):
        # TODO: think of a better method in class designs :think:
        self.tuning_config = image_instance_ops.tuning_config
        parsed_options = self.validate_and_remap_options_schema(options)
        super().__init__(parsed_options, relative_dir, image_instance_ops)
        options = self.options
        self.homography_method = self.homography_method_map.get(
            options.get("homographyMethod", "INTER_LINEAR")
        )

    def exclude_files(self):
        return []

    def prepare_image(self, image):
        return image

    def apply_filter(self, image, colored_image, _template, file_path):
        config = self.tuning_config

        self.debug_image = image.copy()
        self.debug_hstack = []
        self.debug_vstack = []

        image = self.prepare_image(image)

        # TODO: Save intuitive meta data
        # self.append_save_image(3,warped_image)

        (
            control_points,
            destination_points,
        ) = self.extract_control_destination_points(image, file_path)
        (
            parsed_destination_points,
            warped_dimensions,
        ) = self.parse_destination_points_for_image(image, destination_points)

        logger.info(
            f"control_points={control_points}, parsed_destination_points={parsed_destination_points}, warped_dimensions={warped_dimensions}"
        )
        # TODO: check if float32 is really needed
        parsed_control_points = np.float32(control_points)
        parsed_destination_points = np.float32(parsed_destination_points)

        if len(parsed_control_points) == 4:
            # Find and pass control points in a defined order
            transform_matrix = cv2.getPerspectiveTransform(
                parsed_control_points, parsed_destination_points
            )
        else:
            # Getting the homography.
            homography, mask = cv2.findHomography(
                parsed_control_points,
                parsed_destination_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )
            # TODO: print mask for debugging
            # TODO: check if float32 is really needed
            transform_matrix = np.float32(homography)
        # elif TODO: try remap as well
        # elif TODO: try warpAffine as well for non cropped Alignment!!

        logger.info(f"transform_matrix={transform_matrix}")
        # Crop the image
        warped_image = cv2.warpPerspective(
            image, transform_matrix, warped_dimensions, flags=self.homography_method
        )

        if config.outputs.show_colored_outputs:
            colored_image = cv2.warpPerspective(
                colored_image, transform_matrix, warped_dimensions
            )

        # self.append_save_image(1,warped_image)

        if config.outputs.show_image_level >= 4:
            hstack = ImageUtils.get_padded_hstack([self.debug_image, warped_image])
            InteractionUtils.show(
                f"warped_image: {file_path}", hstack, 1, 1, config=config
            )

        return warped_image, colored_image, _template

    def parse_destination_points_for_image(self, image, destination_points):
        options = self.options
        h, w = image.shape[:2]
        parsed_destination_points, warped_dimensions = destination_points, (w, h)
        enable_cropping = options.get("enableCropping", False)
        if enable_cropping:
            # TODO: exclude the destination points marked with excludeFromCropping (using a small class for each point?)
            # Also exclude corresponding points from control points (Note: may need to be done in a second pass after alignment warping)
            # But if warping supports alignment of negative points, this will work as-is (try it!)

            # TODO: Give a warning if the destination_points do not form a convex polygon!

            #   get bounding box on the destination points (with validation?)
            (
                destination_box,
                rectangle_dimensions,
            ) = MathUtils.get_bounding_box_of_points(destination_points)
            warped_dimensions = rectangle_dimensions
            # Shift the destination points to enable the cropping
            parsed_destination_points = MathUtils.shift_origin_for_points(
                destination_box[0], destination_points
            )

            # Note: control points remain the same (wrt image shape!)

        return (
            parsed_destination_points,
            warped_dimensions,
        )
