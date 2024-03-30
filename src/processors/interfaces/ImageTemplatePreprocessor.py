# Use all imports relative to root directory
from src.processors.manager import Processor
from src.utils.image import ImageUtils


class ImageTemplatePreprocessor(Processor):
    """Base class for an extension that applies some preprocessing to the input image"""

    def __init__(self, options, relative_dir, image_instance_ops):
        super().__init__(
            options,
            relative_dir,
        )
        self.append_save_image = image_instance_ops.append_save_image
        self.tuning_config = image_instance_ops.tuning_config
        # Note: we're taking this at preProcessor level instead of tuningOptions because
        # it represents the need of a preProcessor's coordinate system(e.g. area selectors)
        self.processing_image_shape = options.get(
            "processingImageShape",
            self.tuning_config.dimensions.processing_image_shape,
        )

    def apply_filter(self, _image, _colored_image, _template, _file_path):
        """Apply filter to the image and returns modified image"""
        raise NotImplementedError

    def resize_and_apply_filter(self, in_image, colored_image, _template, _file_path):
        config = self.tuning_config
        processing_height, processing_width = self.processing_image_shape

        in_image = ImageUtils.resize_util(
            in_image,
            processing_width,
            processing_height,
        )

        if config.outputs.show_colored_outputs:
            colored_image = ImageUtils.resize_util(
                colored_image,
                processing_width,
                processing_height,
            )

        out_image, colored_image, _template = self.apply_filter(
            in_image, colored_image, _template, _file_path
        )

        return out_image, colored_image, _template

    @staticmethod
    def exclude_files():
        """Returns a list of file paths that should be excluded from processing"""
        return []
