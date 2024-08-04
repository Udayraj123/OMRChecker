import cv2
import numpy as np
from scipy.interpolate import griddata

from src.processors.constants import WarpMethod, WarpMethodFlags
from src.processors.helpers.rectify import rectify
from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.drawing import DrawingUtils
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
        self, options, relative_dir, save_image_ops, default_processing_image_shape
    ):
        # TODO: need to fix this (self attributes will be overridden by parent and may cause inconsistency)
        self.tuning_config = save_image_ops.tuning_config

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
            save_image_ops,
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
            edge_contours_map,
        ) = self.extract_control_destination_points(image, colored_image, file_path)

        (
            parsed_control_points,
            parsed_destination_points,
            warped_dimensions,
        ) = self.parse_control_destination_points_for_image(
            image, control_points, destination_points
        )

        logger.debug(
            f"Cropping Enabled: {self.enable_cropping}\n parsed_control_points={parsed_control_points} \n parsed_destination_points={parsed_destination_points} \n warped_dimensions={warped_dimensions}"
        )

        if self.warp_method == WarpMethod.PERSPECTIVE_TRANSFORM:
            (
                transform_matrix,
                warped_dimensions,
                parsed_destination_points,
            ) = self.get_perspective_transform_matrix(
                parsed_control_points,  # parsed_destination_points
            )
            warped_image, warped_colored_image = self.warp_perspective(
                image, colored_image, transform_matrix, warped_dimensions
            )

        # elif TODO: support for warpAffine as well for non cropped Alignment!!
        elif self.warp_method == WarpMethod.HOMOGRAPHY:
            transform_matrix, _matches_mask = self.get_homography_matrix(
                parsed_control_points, parsed_destination_points
            )
            warped_image, warped_colored_image = self.warp_perspective(
                image, colored_image, transform_matrix, warped_dimensions
            )
        elif self.warp_method == WarpMethod.REMAP_GRIDDATA:
            warped_image, warped_colored_image = self.remap_with_griddata(
                image,
                colored_image,
                parsed_control_points,
                parsed_destination_points,
            )
        elif self.warp_method == WarpMethod.DOC_REFINE:
            warped_image, warped_colored_image = self.remap_with_doc_refine_rectify(
                image, colored_image, edge_contours_map
            )

        if config.outputs.show_image_level >= 4:
            title_prefix = "Warped Image"
            if self.enable_cropping:
                title_prefix = "Cropped Image"
                # Draw the convex hull of all control points
                DrawingUtils.draw_contour(
                    self.debug_image, cv2.convexHull(parsed_control_points)
                )
            if config.outputs.show_image_level >= 5:
                InteractionUtils.show("Anchor Points", self.debug_image, pause=False)

            matched_lines = DrawingUtils.draw_matches(
                image,
                parsed_control_points,
                warped_image,
                parsed_destination_points,
            )

            InteractionUtils.show(
                f"{title_prefix} with Match Lines: {file_path}",
                matched_lines,
                pause=True,
                # resize_to_height=True,
                config=config,
            )

        self.append_save_image(
            f"Warped Image(no resize): {self}",
            range(4, 7),
            warped_image,
            warped_colored_image,
        )

        if str(self) == "CropPage":
            self.append_save_image(
                f"Anchor Points: {self}", range(6, 7), self.debug_image
            )
        else:
            self.append_save_image(
                f"Anchor Points: {self}", range(3, 7), self.debug_image
            )
        if self.output:
            InteractionUtils.show(
                f"{title_prefix} Preview of Warp: {file_path}",
                warped_image,
                pause=True,
            )

        return warped_image, warped_colored_image, _template

    def get_perspective_transform_matrix(
        self,
        parsed_control_points,  # _parsed_destination_points
    ):
        if len(parsed_control_points) > 4:
            logger.critical(f"Too many parsed_control_points={parsed_control_points}")
            raise Exception(
                f"Expected 4 control points for perspective transform, found {len(parsed_control_points)}. If you want to use a different method, pass it in tuningOptions['warpMethod']"
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
        ) = ImageUtils.get_cropped_warped_rectangle_points(parsed_control_points)

        transform_matrix = cv2.getPerspectiveTransform(
            parsed_control_points, parsed_destination_points
        )
        return transform_matrix, warped_dimensions, parsed_destination_points

    def get_homography_matrix(self, parsed_control_points, parsed_destination_points):
        # Note: the robust methods cv2.RANSAC or cv2.LMEDS are not used as they will
        # take a subset of the destination points(inliers) which is not desired for our use-case

        # Getting the homography.
        homography, matches_mask = cv2.findHomography(
            parsed_control_points,
            parsed_destination_points,
            method=0,
            # method=cv2.RANSAC,
            # Note: ransacReprojThreshold is literally the the pixel distance in our coordinates
            # ransacReprojThreshold=3,
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

        warped_colored_image = None
        if config.outputs.colored_outputs_enabled:
            warped_colored_image = cv2.warpPerspective(
                colored_image, transform_matrix, warped_dimensions
            )

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

        # TODO: add support for isOptionalFeaturePoint(maybe filter the 'None' control points!)
        # TODO: do an ordering of the points

        h, w = image.shape[:2]
        warped_dimensions = (w, h)
        if self.enable_cropping:
            # TODO: exclude the 'excludeFromCropping' destination points(and corresponding control points, after using for alignments)
            # TODO: use a small class for each point?

            #   get bounding box on the destination points (with validation?)
            (
                destination_box,
                rectangle_dimensions,
            ) = MathUtils.get_bounding_box_of_points(parsed_destination_points)
            # TODO: Give a warning if the destination_points do not form a convex polygon!
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

        # Note: the inner elements may already be floats returned by scan zone detections
        parsed_control_points = np.float32(parsed_control_points)
        parsed_destination_points = np.float32(parsed_destination_points)

        return (
            parsed_control_points,
            parsed_destination_points,
            warped_dimensions,
        )

    def remap_with_griddata(
        self, image, colored_image, parsed_control_points, parsed_destination_points
    ):
        config = self.tuning_config

        assert image.shape[:2] == colored_image.shape[:2]

        if self.enable_cropping:
            # TODO: >> get this more reliably - use minZoneRect instead?
            _, (w, h) = MathUtils.get_bounding_box_of_points(parsed_destination_points)
        else:
            h, w = image.shape[:2]

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
        # For us: D=2, n=len(parsed_control_points)
        # meshgrid == all integer coordinates of destination image ( [0, h] x [0, w])
        grid_y, grid_x = np.mgrid[0 : h - 1 : complex(h), 0 : w - 1 : complex(w)]

        # We make use of griddata's ability to map n-d data points in a continuous function
        grid_z = griddata(
            # Input points
            points=parsed_destination_points,
            # Expected values
            values=parsed_control_points,
            # Points at which to interpolate data (inside convex hull of the points)
            xi=(grid_x, grid_y),
            method="cubic",
        )
        grid_z = grid_z.astype("float32")

        warped_image = cv2.remap(
            image, map1=grid_z, map2=None, interpolation=cv2.INTER_CUBIC
        )
        warped_colored_image = None
        if config.outputs.colored_outputs_enabled:
            warped_colored_image = cv2.remap(
                colored_image,
                map1=grid_z,
                map2=None,
                interpolation=cv2.INTER_CUBIC,
            )

        return warped_image, warped_colored_image

    def remap_with_doc_refine_rectify(self, image, colored_image, edge_contours_map):
        config = self.tuning_config

        # TODO: adapt this contract in the remap framework (use griddata vs scanline interchangeably?)
        scaled_map = rectify(
            edge_contours_map=edge_contours_map, enable_cropping=self.enable_cropping
        )

        warped_image = cv2.remap(
            image,
            map1=scaled_map,
            map2=None,
            interpolation=cv2.INTER_NEAREST,  # cv2.INTER_CUBIC
        )
        if config.outputs.show_image_level >= 1:
            InteractionUtils.show("warped_image", warped_image, 0)

        warped_colored_image = None
        if config.outputs.colored_outputs_enabled:
            warped_colored_image = cv2.remap(
                colored_image, map1=scaled_map, map2=None, interpolation=cv2.INTER_CUBIC
            )
            if config.outputs.show_image_level >= 1:
                InteractionUtils.show("warped_colored_image", warped_colored_image, 0)

        return warped_image, warped_colored_image
