from copy import copy as shallowcopy
from copy import deepcopy

from src.processors.constants import FieldDetectionType
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger


class TemplatePreprocessing:
    def apply_preprocessors(
        self, file_path, gray_image, colored_image, original_template
    ):
        config = self.tuning_config

        # Copy template for this instance op
        template = shallowcopy(original_template)
        # Make deepcopy for only parts that are mutated by Processor
        template.field_blocks = deepcopy(template.field_blocks)

        # Reset the shifts in the copied template
        template.reset_all_shifts()

        # resize to conform to common preprocessor input requirements
        gray_image = ImageUtils.resize_to_shape(
            gray_image, template.processing_image_shape
        )
        if config.outputs.colored_outputs_enabled:
            colored_image = ImageUtils.resize_to_shape(
                colored_image, template.processing_image_shape
            )

        show_preprocessors_diff = config.outputs.show_preprocessors_diff
        # run pre_processors in sequence
        for pre_processor in template.pre_processors:
            pre_processor_name = pre_processor.get_class_name()

            # Show Before Preview
            if show_preprocessors_diff[pre_processor_name]:
                InteractionUtils.show(
                    f"Before {pre_processor_name}: {file_path}",
                    (
                        colored_image
                        if config.outputs.colored_outputs_enabled
                        else gray_image
                    ),
                )

            # Apply filter
            (
                out_omr,
                colored_image,
                next_template,
            ) = pre_processor.resize_and_apply_filter(
                gray_image, colored_image, template, file_path
            )
            gray_image = out_omr
            template = next_template

            # Show After Preview
            if show_preprocessors_diff[pre_processor_name]:
                InteractionUtils.show(
                    f"After {pre_processor_name}: {file_path}",
                    (
                        colored_image
                        if config.outputs.colored_outputs_enabled
                        else gray_image
                    ),
                )

        if template.output_image_shape:
            # resize to output requirements
            gray_image = ImageUtils.resize_to_shape(
                gray_image, template.output_image_shape
            )
            if config.outputs.colored_outputs_enabled:
                colored_image = ImageUtils.resize_to_shape(
                    colored_image, template.output_image_shape
                )

        return gray_image, colored_image, template
