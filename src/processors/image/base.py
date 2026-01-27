# Use all imports relative to root directory
from pathlib import Path
from typing import Any, Never

from cv2.typing import MatLike

from src.processors.base import ProcessingContext, Processor
from src.utils.image import ImageUtils
from src.utils.logger import logger


class ImageTemplatePreprocessor(Processor):
    """Base class for image preprocessing.

    All image preprocessors inherit from this class and implement the
    unified Processor interface with process(context) method.
    """

    def __init__(
        self, options, relative_dir, save_image_ops, default_processing_image_shape
    ) -> None:
        # Initialize processor-specific attributes
        self.options = options
        self.tuning_options = options.get("tuning_options", {})
        self.relative_dir = Path(relative_dir)
        self.description = "UNKNOWN"

        # Image preprocessing specific attributes
        self.append_save_image = save_image_ops.append_save_image
        self.tuning_config = save_image_ops.tuning_config

        # Note: we're taking this at preProcessor level because it represents
        # the need of a preProcessor's coordinate system(e.g. zone selectors)
        self.processing_image_shape = options.get(
            "processing_image_shape", default_processing_image_shape
        )
        self.output = options.get("output")

    def get_relative_path(self, path):
        return self.relative_dir.joinpath(path)

    def apply_filter(
        self, _image, _colored_image, _template, _file_path
    ) -> tuple[MatLike, MatLike, Any]:
        """Apply filter to the image and returns modified image."""
        raise NotImplementedError

    def get_class_name(self) -> Never:
        """Get the class name for this preprocessor."""
        raise NotImplementedError

    def get_name(self) -> str:
        """Get the name of this processor (required by unified Processor interface)."""
        return self.get_class_name()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Process images using the unified processor interface.

        This is the interface that all processors must implement.

        Args:
            context: Processing context with images and state

        Returns:
            Updated context with processed images
        """
        logger.debug(f"Starting {self.get_name()} processor")

        gray_image = context.gray_image
        colored_image = context.colored_image
        template = context.template
        file_path = context.file_path

        # Resize images to preprocessor's processing shape
        gray_image = ImageUtils.resize_to_shape(self.processing_image_shape, gray_image)

        if self.tuning_config.outputs.colored_outputs_enabled:
            colored_image = ImageUtils.resize_to_shape(
                self.processing_image_shape, colored_image
            )

        # Apply the specific filter
        gray_image, colored_image, template = self.apply_filter(
            gray_image, colored_image, template, file_path
        )

        # Update context
        context.gray_image = gray_image
        context.colored_image = colored_image
        context.template = template

        logger.debug(f"Completed {self.get_name()} processor")

        return context

    def exclude_files(self) -> list[Path]:
        """Return a list of file paths that should be excluded from processing."""
        return []
