"""Image processing coordinator for the unified processor architecture."""

from src.processors.base import ProcessingContext, Processor
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger


class PreprocessingCoordinator(Processor):
    """Coordinates all image preprocessing steps in sequence.

    This is NOT an individual preprocessor. It orchestrates all preprocessors
    defined in template.template_layout.pre_processors.

    Responsibilities:
    1. Creates a copy of the template layout for mutation
    2. Resizes images to processing dimensions
    3. Runs all preprocessors in sequence (they implement Processor interface)
    4. Optionally resizes to output dimensions
    5. Shows before/after diffs when configured
    """

    def __init__(self, template) -> None:
        """Initialize the preprocessing processor.

        Args:
            template: The template containing preprocessors and configuration
        """
        self.template = template
        self.tuning_config = template.tuning_config

    def get_name(self) -> str:
        """Get the name of this processor."""
        return "Preprocessing"

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Execute all preprocessing steps.

        Args:
            context: Processing context with input images

        Returns:
            Updated context with preprocessed images and updated template
        """
        logger.debug(f"Starting {self.get_name()} processor")

        # Get a copy of the template layout for mutation
        next_template_layout = context.template.template_layout.get_copy_for_shifting()

        # Reset the shifts in the copied template layout
        next_template_layout.reset_all_shifts()

        gray_image = context.gray_image
        colored_image = context.colored_image

        # Resize to conform to common preprocessor input requirements
        gray_image = ImageUtils.resize_to_shape(
            next_template_layout.processing_image_shape, gray_image
        )
        if self.tuning_config.outputs.colored_outputs_enabled:
            colored_image = ImageUtils.resize_to_shape(
                next_template_layout.processing_image_shape, colored_image
            )

        show_preprocessors_diff = self.tuning_config.outputs.show_preprocessors_diff

        # Update context for preprocessors
        context.gray_image = gray_image
        context.colored_image = colored_image
        context.template.template_layout = next_template_layout

        # Run preprocessors in sequence using their process() method
        for pre_processor in next_template_layout.pre_processors:
            pre_processor_name = pre_processor.get_name()

            # Show Before Preview
            if show_preprocessors_diff.get(pre_processor_name, False):
                InteractionUtils.show(
                    f"Before {pre_processor_name}: {context.file_path}",
                    (
                        context.colored_image
                        if self.tuning_config.outputs.colored_outputs_enabled
                        else context.gray_image
                    ),
                )

            # Process using unified interface - preprocessors now implement process(context)
            context = pre_processor.process(context)

            # Show After Preview
            if show_preprocessors_diff.get(pre_processor_name, False):
                InteractionUtils.show(
                    f"After {pre_processor_name}: {context.file_path}",
                    (
                        context.colored_image
                        if self.tuning_config.outputs.colored_outputs_enabled
                        else context.gray_image
                    ),
                )

        # Resize to output requirements if specified
        template_layout = context.template.template_layout
        if template_layout.output_image_shape:
            context.gray_image = ImageUtils.resize_to_shape(
                template_layout.output_image_shape, context.gray_image
            )
            if self.tuning_config.outputs.colored_outputs_enabled:
                context.colored_image = ImageUtils.resize_to_shape(
                    template_layout.output_image_shape, context.colored_image
                )

        logger.debug(f"Completed {self.get_name()} processor")

        return context
