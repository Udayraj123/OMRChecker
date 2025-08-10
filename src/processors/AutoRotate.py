from pathlib import Path

import cv2

from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger


class AutoRotate(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = False

    def get_class_name(self) -> str:
        return "AutoRotate"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        path = self.get_relative_path(self.options["referenceImage"])
        if not path.exists():
            msg = f"Reference image for AutoRotate not found at {path}"
            raise Exception(msg)
        self.reference_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        self.marker_dimensions = self.options.get("markerDimensions", None)
        self.resized_reference = self.reference_image
        self.threshold = self.options.get("threshold", None)
        if self.marker_dimensions:
            self.resized_reference = ImageUtils.resize_to_dimensions(
                self.marker_dimensions, self.reference_image
            )

    def apply_filter(self, image, colored_image, _template, _file_path):
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
            _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(res)

            if config.outputs.show_image_level >= 5:
                hstack = ImageUtils.get_padded_hstack(
                    [rotated_img, self.resized_reference]
                )
                InteractionUtils.show("Template matching result", res, 0)
                InteractionUtils.show(
                    f"Template Matching for rotation: {'No Rotation' if rotation is None else 90 * (1 + rotation)} ({max_val:.2f})",
                    hstack,
                )

            if max_val > best_val:
                best_val = max_val
                best_rotation = rotation

        if self.threshold is not None and self.threshold["value"] > best_val:
            if self.threshold["passthrough"]:
                logger.warning(
                    "The autorotate score is below threshold. Continuing due to passthrough flag."
                )
            else:
                logger.error(
                    "The autorotate score is below threshold. Adjust your threshold or check the reference marker and input image."
                )
                msg = "The autorotate score is below threshold"
                raise Exception(msg)

        logger.info(
            f"AutoRotate Applied with rotation {best_rotation} and value {best_val}"
        )
        if best_rotation is None:
            return image, colored_image, _template

        image = ImageUtils.rotate(image, best_rotation, keep_original_shape=True)

        if config.outputs.show_image_level >= 4:
            InteractionUtils.show(
                f"Image after AutoRotate: rotation - {best_rotation}",
                image,
                1,
            )
        if self.tuning_config.outputs.colored_outputs_enabled:
            colored_image = ImageUtils.rotate(
                colored_image, best_rotation, keep_original_shape=True
            )
        return image, colored_image, _template

    def exclude_files(self) -> list[Path]:
        path = self.get_relative_path(self.options["referenceImage"])
        return [path]
