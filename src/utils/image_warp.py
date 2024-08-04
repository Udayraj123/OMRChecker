import cv2
import numpy as np

from src.utils.constants import CLR_DARK_GREEN, CLR_DARK_RED
from src.utils.drawing import DrawingUtils
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils


class ImageWarpUtils:
    # TODO: add subclasses/submodules to ImageUtils to organize better.
    @staticmethod
    def warp_triangle_inplace(
        image, warped_image, source_triangle, warped_triangle, show_image_level=False
    ):
        if MathUtils.check_collinear_points(*source_triangle):
            logger.critical(
                f"Found collinear points. Skipping warp step for the source triangle {source_triangle}"
            )
            return
        if MathUtils.check_collinear_points(*warped_triangle):
            logger.critical(
                f"Found collinear points. Skipping warp step for the warped triangle {source_triangle}"
            )
            return

        # Find bounding box and crop input image
        (
            source_tl,
            _source_tr,
            source_br,
            _source_bl,
        ), source_box_dimensions = MathUtils.get_bounding_box_of_points(source_triangle)
        (
            warped_tl,
            _warped_tr,
            warped_br,
            _warped_bl,
        ), warped_box_dimensions = MathUtils.get_bounding_box_of_points(warped_triangle)

        source_shifted_triangle = MathUtils.shift_points_to_origin(
            source_tl, source_triangle
        )
        warped_shifted_triangle = MathUtils.shift_points_to_origin(
            warped_tl, warped_triangle
        )
        logger.info("source_shifted_triangle", source_shifted_triangle)
        logger.info("warped_shifted_triangle", warped_shifted_triangle)
        # Given a pair of triangles, find the affine transform.
        triangle_affine_matrix = cv2.getAffineTransform(
            np.float32(source_shifted_triangle),
            np.float32(warped_shifted_triangle),
        )

        # Crop input image
        source_triangle_box = image[
            source_tl[1] : source_br[1], source_tl[0] : source_br[0]
        ]
        logger.info("source_triangle_box", source_triangle_box.shape)
        logger.info("source_triangle", source_triangle)
        logger.info("warped_triangle", warped_triangle)
        logger.info("triangle_affine_matrix", triangle_affine_matrix)
        # Apply the Affine Transform just found to the src image
        warped_triangle_box = cv2.warpAffine(
            source_triangle_box,
            triangle_affine_matrix,
            warped_box_dimensions,
            None,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        # Note: the warped image dimensions will match to that of the warped triangle's bounding box(with black filling)
        assert warped_triangle_box.shape == tuple(
            reversed(warped_box_dimensions)
        ), f"{warped_triangle_box.shape} != {tuple(reversed(warped_box_dimensions))}"

        logger.info(source_triangle_box.shape, warped_triangle_box.shape)

        colored_source_triangle_box = cv2.cvtColor(
            source_triangle_box, cv2.COLOR_GRAY2BGR
        )

        DrawingUtils.draw_polygon(
            colored_source_triangle_box, source_shifted_triangle, color=CLR_DARK_RED
        )
        DrawingUtils.draw_polygon(
            colored_source_triangle_box, warped_shifted_triangle, color=CLR_DARK_GREEN
        )

        (
            background_from_source_image,
            triangle_from_source_image,
            triangle_from_warped_image,
        ) = ImageWarpUtils.replace_triangle_inplace(
            warped_image,
            warped_shifted_triangle,
            warped_triangle_box,
            warped_tl,
            warped_br,
            warped_box_dimensions,
        )
        if show_image_level >= 5:
            colored_warped_triangle_box = cv2.cvtColor(
                warped_image[warped_tl[1] : warped_br[1], warped_tl[0] : warped_br[0]],
                cv2.COLOR_GRAY2BGR,
            )
            DrawingUtils.draw_polygon(
                colored_warped_triangle_box,
                warped_shifted_triangle,
                color=CLR_DARK_GREEN,
            )

            InteractionUtils.show(
                f"Warped Triangle Patching -{(warped_tl, warped_box_dimensions)}",
                ImageUtils.get_padded_hstack(
                    [
                        colored_source_triangle_box,
                        cv2.cvtColor(background_from_source_image, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(triangle_from_source_image, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(triangle_from_warped_image, cv2.COLOR_GRAY2BGR),
                        colored_warped_triangle_box,
                    ]
                ),
                0,
            )

    @staticmethod
    def replace_triangle_inplace(
        source_image,
        shifted_triangle,
        warped_triangle_box,
        warped_tl,
        warped_br,
        warped_box_dimensions,
    ):
        logger.info("shifted_triangle", shifted_triangle)
        logger.info(
            "source_image",
            source_image.shape,
            "warped_triangle_box",
            warped_triangle_box.shape,
        )
        channels = (
            1 if len(warped_triangle_box.shape) == 2 else warped_triangle_box.shape[2]
        )
        tl, br, dest_w, dest_h = (
            warped_tl,
            warped_br,
            warped_box_dimensions[0],
            warped_box_dimensions[1],
        )
        if channels == 3:
            # Get a white triangle mask
            # Note: we use shifted triangle to reduce mask size (as outside the box will have zeroes anyway)
            white_triangle = np.zeros((dest_h, dest_w, channels), dtype=np.float32)
            cv2.fillConvexPoly(
                white_triangle,
                np.int32(shifted_triangle),
                (1.0, 1.0, 1.0),
                cv2.LINE_AA,
                shift=0,
            )

            # Get a black triangle mask
            black_triangle = (1.0, 1.0, 1.0) - white_triangle

        else:
            # Get a white triangle mask
            # Note: we use shifted triangle to reduce mask size (as outside the box will have zeroes anyway)
            white_triangle = np.zeros((dest_h, dest_w), dtype=np.float32)
            cv2.fillConvexPoly(
                white_triangle,
                np.int32(shifted_triangle),
                1.0,
                cv2.LINE_AA,
                shift=0,
            )

            # Get a black triangle mask
            black_triangle = 1.0 - white_triangle

        # Extract the triangle-only warped_triangle_box using the mask
        triangle_from_warped_image = (warped_triangle_box * white_triangle).astype(
            np.uint8
        )
        source_triangle_box = source_image[tl[1] : br[1], tl[0] : br[0]]

        background_from_source_image = (source_triangle_box * black_triangle).astype(
            np.uint8
        )

        # TODO: Temp debugging
        triangle_from_source_image = (source_triangle_box * white_triangle).astype(
            np.uint8
        )

        # Add the triangle-only area with the non-masked triangle image
        source_image[tl[1] : br[1], tl[0] : br[0]] = (
            background_from_source_image + triangle_from_warped_image
        )
        return (
            background_from_source_image,
            triangle_from_source_image,
            triangle_from_warped_image,
        )
