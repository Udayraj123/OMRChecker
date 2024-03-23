# Use all imports relative to root directory
from src.processors.manager import Processor
from src.utils.image import ImageUtils
from src.utils.logger import logger


class ImageTemplatePreprocessor(Processor):
    """Base class for an extension that applies some preprocessing to the input image"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        out_image, colored_image, _template = self.apply_filter(
            in_image, colored_image, _template, _file_path
        )

        if out_image.shape != self.processing_image_shape:
            logger.warning(
                f"The shape of the image returned by {str(self)} is not matching with expected processing dimensions. Resizing automatically..."
            )
            out_image = ImageUtils.resize_util(
                out_image,
                processing_width,
                processing_height,
            )
            if config.outputs.show_colored_outputs:
                colored_image = ImageUtils.resize_util(
                    colored_image,
                    processing_width,
                    processing_height,
                )

        return out_image, colored_image, _template

    @staticmethod
    def exclude_files():
        """Returns a list of file paths that should be excluded from processing"""
        return []
