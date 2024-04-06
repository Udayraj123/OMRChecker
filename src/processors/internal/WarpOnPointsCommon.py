import cv2
import numpy as np

from src.processors.constants import WarpMethod, WarpMethodFlags
from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils
from src.utils.parsing import OVERRIDE_MERGER


# Internal Processor for separation of code
class WarpOnPointsCommon(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = True

    warp_method_flags_map = {
        WarpMethodFlags.INTER_LINEAR: cv2.INTER_LINEAR,
        WarpMethodFlags.INTER_CUBIC: cv2.INTER_CUBIC,
        WarpMethodFlags.INTER_NEAREST: cv2.INTER_NEAREST,
    }

    def validate_and_remap_options_schema(self):
        raise Exception(f"Not implemented")

    def __init__(
        self, options, relative_dir, image_instance_ops, default_processing_image_shape
    ):
        # TODO: need to fix this (self attributes will be overridden by parent and may cause inconsistency)
        self.tuning_config = image_instance_ops.tuning_config

        parsed_options = self.validate_and_remap_options_schema(options)
        # Processor tuningOptions defaults
        parsed_options = OVERRIDE_MERGER.merge(
            {
                "tuningOptions": options.get("tuningOptions", {}),
            },
            parsed_options,
        )

        super().__init__(
            parsed_options,
            relative_dir,
            image_instance_ops,
            default_processing_image_shape,
        )
        options = self.options
        tuning_options = self.tuning_options
        self.enable_cropping = options.get("enableCropping", False)

        self.warp_method = tuning_options.get(
            "warpMethod",
            (
                WarpMethod.PERSPECTIVE_TRANSFORM
                if self.enable_cropping
                else WarpMethod.HOMOGRAPHY
            ),
        )
        self.warp_method_flag = self.warp_method_flags_map.get(
            tuning_options.get("warpMethodFlag", "INTER_LINEAR")
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

        (
            control_points,
            destination_points,
        ) = self.extract_control_destination_points(image, file_path)

        # TODO: add support for isOptionalFeaturePoint(maybe filter the 'None' control points!)
        (
            parsed_control_points,
            parsed_destination_points,
            warped_dimensions,
        ) = self.parse_control_destination_points_for_image(
            image, control_points, destination_points
        )

        logger.info(
            f"Cropping Enabled: {self.enable_cropping}\n parsed_control_points={parsed_control_points} \n parsed_destination_points={parsed_destination_points} \n warped_dimensions={warped_dimensions}"
        )

        if self.warp_method == WarpMethod.PERSPECTIVE_TRANSFORM:
            transform_matrix, warped_dimensions = self.get_perspective_transform_matrix(
                parsed_control_points, parsed_destination_points
            )
            warped_image, warped_colored_image = self.warp_perspective(
                image, colored_image, transform_matrix, warped_dimensions
            )

        elif self.warp_method == WarpMethod.HOMOGRAPHY:
            transform_matrix, _matches_mask = self.get_homography_matrix(
                parsed_control_points, parsed_destination_points
            )
            warped_image, warped_colored_image = self.warp_perspective(
                image, colored_image, transform_matrix, warped_dimensions
            )
        # elif TODO: try remap as well
        # elif TODO: try warpAffine as well for non cropped Alignment!!

        if config.outputs.show_image_level >= 4:
            title = "Warped Image"
            if self.enable_cropping:
                title = "Cropped Image"
                # Draw the convex hull of all control points
                ImageUtils.draw_contour(
                    self.debug_image, cv2.convexHull(parsed_control_points)
                )
            if config.outputs.show_image_level >= 5:
                InteractionUtils.show("Anchor Points", self.debug_image, pause=False)

            matched_lines = ImageUtils.draw_matches(
                image,
                parsed_control_points,
                warped_image,
                parsed_destination_points,
            )

            InteractionUtils.show(
                f"{title} with Match Lines: {file_path}",
                matched_lines,
                pause=True,
                resize_to_height=True,
                config=config,
            )

        return warped_image, warped_colored_image, _template

    def get_perspective_transform_matrix(
        self, parsed_control_points, _parsed_destination_points
    ):
        if len(parsed_control_points) > 4:
            logger.critical(f"Too many parsed_control_points={parsed_control_points}")
            raise Exception(
                f"Expected 4 control points for perspective transform. Found {len(parsed_control_points)}"
            )
        # TODO: order the points from outside in parsing itself
        parsed_control_points, ordered_indices = MathUtils.order_four_points(
            parsed_control_points, dtype="float32"
        )
        # TODO: fix use _parsed_destination_points and make it work?
        # parsed_destination_points = _parsed_destination_points[ordered_indices]
        (
            parsed_destination_points,
            warped_dimensions,
        ) = ImageUtils.get_cropped_rectangle_destination_points(parsed_control_points)

        transform_matrix = cv2.getPerspectiveTransform(
            parsed_control_points, parsed_destination_points
        )
        return transform_matrix, warped_dimensions

    def get_homography_matrix(self, parsed_control_points, parsed_destination_points):
        # Note: the robust methods cv2.RANSAC or cv2.LMEDS are not used as they will
        # take a subset of the destination points(inliers) which is not desired for our use-case

        # Getting the homography.
        homography, matches_mask = cv2.findHomography(
            parsed_control_points,
            parsed_destination_points,
            method=0,
            # Note: ransacReprojThreshold is literally the the pixel distance in our coordinates
            # ransacReprojThreshold=3.0,
        )
        # TODO: check if float32 is really needed for the matrix
        transform_matrix = np.float32(homography)
        return transform_matrix, matches_mask

    def warp_perspective(
        self, image, colored_image, transform_matrix, warped_dimensions
    ):
        config = self.tuning_config

        # Crop the image
        warped_image = cv2.warpPerspective(
            image, transform_matrix, warped_dimensions, flags=self.warp_method_flag
        )

        # TODO: Save intuitive meta data
        # self.append_save_image(3,warped_image)

        if config.outputs.show_colored_outputs:
            warped_colored_image = cv2.warpPerspective(
                colored_image, transform_matrix, warped_dimensions
            )

        # self.append_save_image(1,warped_image)

        return warped_image, warped_colored_image

    def parse_control_destination_points_for_image(
        self, image, control_points, destination_points
    ):
        parsed_control_points, parsed_destination_points = [], []

        # de-dupe
        control_points_set = set()
        for control_point, destination_point in zip(control_points, destination_points):
            control_point_tuple = tuple(control_point)
            if control_point_tuple not in control_points_set:
                control_points_set.add(control_point_tuple)
                parsed_control_points.append(control_point)
                parsed_destination_points.append(destination_point)

        # TODO: do an ordering of the points

        h, w = image.shape[:2]
        warped_dimensions = (w, h)
        if self.enable_cropping:
            # TODO: exclude the 'excludeFromCropping' destination points(and corresponding control points, after using for alignments)
            # TODO: use a small class for each point?
            # TODO: Give a warning if the destination_points do not form a convex polygon!

            #   get bounding box on the destination points (with validation?)
            (
                destination_box,
                rectangle_dimensions,
            ) = MathUtils.get_bounding_box_of_points(parsed_destination_points)
            warped_dimensions = rectangle_dimensions

            # Cropping means the bounding destination points need to become the bounding box!
            # >> Rest of the points need to scale according to that grid!?

            # TODO: find a way to get the bounding box to control points mapping
            # parsed_destination_points = destination_box[[1,2,0,3]]

            # Shift the destination points to enable the cropping
            from_origin = -1 * destination_box[0]
            parsed_destination_points = MathUtils.shift_points_from_origin(
                from_origin, parsed_destination_points
            )
            # Note: control points remain the same (wrt image shape!)

        # Note: the inner elements may already be floats returned by scan area detections
        parsed_control_points = np.float32(parsed_control_points)
        parsed_destination_points = np.float32(parsed_destination_points)

        return (
            parsed_control_points,
            parsed_destination_points,
            warped_dimensions,
        )
