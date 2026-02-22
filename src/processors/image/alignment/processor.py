"""Alignment Processor for template alignment."""

from src.processors.image.alignment.template_alignment import (
    apply_template_alignment,
)
from src.processors.base import ProcessingContext, Processor
from src.utils.logger import logger


class AlignmentProcessor(Processor):
    """Processor that applies template alignment to images.

    This processor performs feature-based alignment if a reference image
    is provided in the template configuration.
    """

    def __init__(self, template) -> None:
        """Initialize the alignment processor.

        Args:
            template: The template containing alignment configuration
        """
        self.template = template
        self.tuning_config = template.tuning_config

    def get_name(self) -> str:
        """Get the name of this processor."""
        return "Alignment"

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Execute alignment on the images.

        Args:
            context: Processing context with preprocessed images

        Returns:
            Updated context with aligned images and template
        """
        logger.debug(f"Starting {self.get_name()} processor")

        gray_image = context.gray_image
        colored_image = context.colored_image
        template = context.template

        # Only apply alignment if images are valid and alignment is configured
        if gray_image is not None and "gray_alignment_image" in template.alignment:
            gray_image, colored_image, template = apply_template_alignment(
                gray_image, colored_image, template, self.tuning_config
            )

            # Update context with aligned images
            context.gray_image = gray_image
            context.colored_image = colored_image
            context.template = template

        logger.debug(f"Completed {self.get_name()} processor")

        return context
