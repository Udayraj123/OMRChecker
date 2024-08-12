import cv2

from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger


class AutoRotate(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = False

    def get_class_name(self):
        return "AutoRotate"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = self.get_relative_path(self.options["referenceImage"])
        self.reference_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.marker_dimensions = self.options.get("markerDimensions", None)
        self.resized_reference = self.reference_image
        self.threshold = self.options.get("threshold", None)
        if self.marker_dimensions:
            self.resized_reference = ImageUtils.resize_to_dimensions(
                self.marker_dimensions, self.reference_image
            )

    def apply_filter(self, image, colored_image, _template, file_path):
        config = self.tuning_config
        best_val, best_rotation = -1, None
        rotations = [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ]
        for rotation in rotations:
            rotated_img = image
            if rotation is not None:
                rotated_img = ImageUtils.rotate(
                    image, rotation, keep_original_shape=True
                )
            # TODO: find a better suited template matching for white images.
            res = cv2.matchTemplate(
                rotated_img, self.resized_reference, cv2.TM_CCOEFF_NORMED
            )
            if config.outputs.show_image_level >= 4:
                InteractionUtils.show(f"Image for rotation: {rotation}", rotated_img, 0)
                InteractionUtils.show(
                    f"Reference for rotation: {rotation}", self.resized_reference, 0
                )
                InteractionUtils.show(
                    f"Template Matching Result for rotation: {rotation}", res
                )
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # logger.info(rotation, max_val)
            if max_val > best_val:
                best_val = max_val
                best_rotation = rotation

        if self.threshold is not None:
            if self.threshold["value"] > best_val:
                if self.threshold["passthrough"]:
                    logger.warning(
                        "The autorotate score is below threshold. Continuing due to passthrough flag."
                    )
                else:
                    logger.error(
                        "The autorotate score is below threshold. Adjust your threshold or check the reference marker and input image."
                    )
                    raise Exception(f"The autorotate score is below threshold")
        logger.info(
            "AutoRotate Applied with rotation", best_rotation, "best value", best_val
        )
        if best_rotation is None:
            return image, colored_image, _template

        image = ImageUtils.rotate(image, best_rotation, keep_original_shape=True)
        if self.tuning_config.outputs.colored_outputs_enabled:
            colored_image = ImageUtils.rotate(
                colored_image, best_rotation, keep_original_shape=True
            )
        return image, colored_image, _template

    def exclude_files(self):
        path = self.get_relative_path(self.options["referenceImage"])
        return [path]
