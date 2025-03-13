import cv2

from src.algorithm.evaluation.config import EvaluationConfigForSet
from src.utils.constants import MARKED_TEMPLATE_TRANSPARENCY
from src.utils.drawing import DrawingUtils
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils


class TemplateDrawing:
    def __init__(self, template):
        self.template = template

    def draw_template_layout(self, gray_image, colored_image, config, *args, **kwargs):
        template = self.template
        # TODO: extract field_id_to_interpretation from template itself
        return TemplateDrawingUtils.draw_template_layout(
            gray_image, colored_image, template, config, *args, **kwargs
        )

    def draw_only_field_blocks(
        self, image, shifted=True, shouldCopy=True, thickness=3, border=3
    ):
        template = self.template
        return TemplateDrawing.draw_field_blocks_layout_util(
            template, image, shifted, shouldCopy, thickness, border
        )

    @staticmethod
    def draw_field_blocks_layout_util(
        template, image, shifted=True, shouldCopy=True, thickness=3, border=3
    ):
        marked_image = image.copy() if shouldCopy else image
        for field_block in template.field_blocks:
            field_block.drawing.draw_field_block(
                marked_image, shifted, thickness, border
            )

        return marked_image


class TemplateDrawingUtils:
    @staticmethod
    def draw_template_layout(
        gray_image, colored_image, template, config, *args, **kwargs
    ):
        final_marked = TemplateDrawingUtils.draw_template_layout_util(
            gray_image, "GRAYSCALE", template, config, *args, **kwargs
        )

        colored_final_marked = colored_image
        if config.outputs.colored_outputs_enabled:
            colored_final_marked = TemplateDrawingUtils.draw_template_layout_util(
                colored_final_marked,
                "COLORED",
                template,
                config,
                *args,
                **kwargs,
            )

        if config.outputs.show_image_level >= 1:
            InteractionUtils.show(
                "Final Marked Template",
                final_marked,
                pause=(not config.outputs.colored_outputs_enabled),
                resize_to_height=True,
                config=config,
            )
            if config.outputs.colored_outputs_enabled:
                InteractionUtils.show(
                    "Final Marked Template (Colored)",
                    colored_final_marked,
                    1,
                    resize_to_height=True,
                    config=config,
                )

        template.save_image_ops.append_save_image(
            "Marked Template", range(1, 7), final_marked, colored_final_marked
        )

        return final_marked, colored_final_marked

    @staticmethod
    def draw_template_layout_util(
        image,
        image_type,
        template,
        config,
        field_id_to_interpretation=None,
        evaluation_meta=None,
        evaluation_config_for_response=None,
        shifted=False,
        border=-1,
    ):
        marked_image = ImageUtils.resize_to_dimensions(
            template.template_dimensions, image
        )

        transparent_layer = marked_image.copy()

        if field_id_to_interpretation is None:
            marked_image = template.drawing.draw_only_field_blocks(
                marked_image, shifted, shouldCopy=False, border=border
            )
            return marked_image
        else:
            if config.outputs.save_image_level >= 1:
                # Create a copy of the marked image for saving
                marked_image_copy = marked_image.copy()

                # Draw marked bubbles without evaluation meta
                marked_image_copy = TemplateDrawingUtils.draw_all_fields(
                    marked_image_copy,
                    image_type,
                    template,
                    field_id_to_interpretation,
                    evaluation_meta=None,
                    evaluation_config_for_response=None,
                )
                if image_type == "GRAYSCALE":
                    template.save_image_ops.append_save_image(
                        f"Marked Image", range(2, 7), marked_image_copy
                    )
                else:
                    template.save_image_ops.append_save_image(
                        f"Marked Image", range(2, 7), colored_image=marked_image_copy
                    )

            marked_image = TemplateDrawingUtils.draw_all_fields(
                marked_image,
                image_type,
                template,
                field_id_to_interpretation,
                evaluation_meta,
                evaluation_config_for_response,
            )

            # Draw evaluation summary
            if evaluation_meta is not None:
                marked_image = TemplateDrawingUtils.draw_evaluation_summary(
                    marked_image, evaluation_meta, evaluation_config_for_response
                )

        # Translucent
        cv2.addWeighted(
            marked_image,
            MARKED_TEMPLATE_TRANSPARENCY,
            transparent_layer,
            1 - MARKED_TEMPLATE_TRANSPARENCY,
            0,
            marked_image,
        )

        return marked_image

    @staticmethod
    def draw_all_fields(
        marked_image,
        image_type,
        template,
        field_id_to_interpretation,
        evaluation_meta,
        evaluation_config_for_response,
    ):
        for field in template.all_fields:
            field_interpretation = field_id_to_interpretation[field.id]
            field_interpretation.drawing.draw_field_interpretation(
                marked_image,
                image_type,
                evaluation_meta,
                evaluation_config_for_response,
            )

        return marked_image

    @staticmethod
    def draw_evaluation_summary(
        marked_image,
        evaluation_meta,
        evaluation_config_for_response: EvaluationConfigForSet,
    ):
        if evaluation_config_for_response.draw_answers_summary["enabled"]:
            (
                formatted_answers_summary,
                position,
                size,
                thickness,
            ) = evaluation_config_for_response.get_formatted_answers_summary()

            DrawingUtils.draw_text(
                marked_image,
                formatted_answers_summary,
                position,
                text_size=size,
                thickness=thickness,
            )

        if evaluation_config_for_response.draw_score["enabled"]:
            (
                formatted_score,
                position,
                size,
                thickness,
            ) = evaluation_config_for_response.get_formatted_score(
                evaluation_meta["score"]
            )
            DrawingUtils.draw_text(
                marked_image,
                formatted_score,
                position,
                text_size=size,
                thickness=thickness,
            )

        return marked_image
