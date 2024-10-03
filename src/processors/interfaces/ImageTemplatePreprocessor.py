# Use all imports relative to root directory
import os

from src.processors.internal.Processor import Processor
from src.utils.image import ImageUtils


class ImageTemplatePreprocessor(Processor):
    """Base class for an extension that applies some preprocessing to the input image"""

    def __init__(
        self, options, relative_dir, save_image_ops, default_processing_image_shape
    ):
        super().__init__(
            options,
            relative_dir,
        )
        self.append_save_image = save_image_ops.append_save_image
        self.tuning_config = save_image_ops.tuning_config

        # Note: we're taking this at preProcessor level because it represents
        # the need of a preProcessor's coordinate system(e.g. zone selectors)
        self.processing_image_shape = options.get(
            "processingImageShape", default_processing_image_shape
        )
        self.output = options.get("output")

    def get_relative_path(self, path):
        return os.path.join(self.relative_dir, path)

    def apply_filter(self, _image, _colored_image, _template, _file_path):
        """Apply filter to the image and returns modified image"""
        raise NotImplementedError

    def resize_and_apply_filter(self, in_image, colored_image, _template, _file_path):
        config = self.tuning_config

        in_image = ImageUtils.resize_to_shape(self.processing_image_shape, in_image)

        if config.outputs.colored_outputs_enabled:
            colored_image = ImageUtils.resize_to_shape(
                self.processing_image_shape,
                colored_image,
            )

        out_image, colored_image, _template = self.apply_filter(
            in_image, colored_image, _template, _file_path
        )

        return out_image, colored_image, _template

    @staticmethod
    def exclude_files():
        """Returns a list of file paths that should be excluded from processing"""
        return []
