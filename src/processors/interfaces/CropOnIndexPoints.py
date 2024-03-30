# import cv2
# import numpy as np
# from src.utils.logger import logger

from dotmap import DotMap

from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils

# TODO: apply a more generic way to transform the document into better warped configuration
# https://github.com/gwxie/Document-Dewarping-with-Control-Points

# Or even generic to de-warp the document

IndexPointsType = DotMap(
    {
        "FOUR_POINTS": "FOUR_POINTS",
        "FOUR_POINTS_AND_CONTOUR": "FOUR_POINTS_AND_CONTOUR",
    },
    _dynamic=False,
)


# Internal Processor for separation of code
class CropOnIndexPoints(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: add validation
        self.type = IndexPointsType.FOUR_POINTS
        # - add validation for the points
        # - If type is displacement: single point
        # - If type is page_crop: 4 points as a convex polygon,
        # - If type is page_crop_smooth: >= 4 points as a convex polygon
        # - If type is template_align: max displacement should be controlled by scanbox?

    def exclude_files(self):
        return []

    def prepare_image(self, image):
        return image

    def apply_filter(self, *args):
        if self.type == IndexPointsType.FOUR_POINTS:
            return self.apply_filter_four_points(*args)
        elif self.type == IndexPointsType.FOUR_POINTS_AND_CONTOUR:
            return self.apply_filter_contour(*args)
        else:  # Default
            return self.apply_filter_contour(*args)

    def apply_filter_four_points(self, image, colored_image, _template, file_path):
        config = self.tuning_config

        self.debug_image = image.copy()
        self.debug_hstack = []
        self.debug_vstack = []

        image = self.prepare_image(image)

        # TODO: Save intuitive meta data
        # self.append_save_image(3,warped_image)

        four_corners = self.find_four_corners(image, file_path)

        # Crop the image
        warped_image = ImageUtils.four_point_transform(image, four_corners)

        if config.outputs.show_colored_outputs:
            colored_image = ImageUtils.four_point_transform(colored_image, four_corners)

        # self.append_save_image(1,warped_image)

        if config.outputs.show_image_level >= 4:
            hstack = ImageUtils.get_padded_hstack([self.debug_image, warped_image])
            InteractionUtils.show(
                f"warped_image: {file_path}", hstack, 1, 1, config=config
            )

        return warped_image, colored_image, _template

    def apply_filter_contour(self, image, colored_image, _template, file_path):
        config = self.tuning_config

        self.debug_image = image.copy()
        self.debug_hstack = []
        self.debug_vstack = []

        image = self.prepare_image(image)
        # TODO: contour mapping (3 steps)
        warped_image = image
        # self.append_save_image(1,warped_image)

        if config.outputs.show_image_level >= 4:
            hstack = ImageUtils.get_padded_hstack([self.debug_image, warped_image])
            InteractionUtils.show(
                f"warped_image: {file_path}", hstack, 1, 1, config=config
            )

        return warped_image, colored_image, _template
