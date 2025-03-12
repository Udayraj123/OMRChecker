from typing import List

import cv2

from src.algorithm.evaluation.config import AnswerMatcher, EvaluationConfigForSet
from src.algorithm.template.detection.bubbles_threshold.interpretation import (
    BubbleInterpretation,
)
from src.processors.constants import FieldDetectionType
from src.schemas.constants import AnswerType
from src.utils.constants import (
    CLR_GRAY,
    CLR_NEAR_BLACK,
    CLR_WHITE,
    MARKED_TEMPLATE_TRANSPARENCY,
    TEXT_SIZE,
)
from src.utils.drawing import DrawingUtils
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils


class TemplateDrawing:
    def __init__(self, template):
        self.template = template

    def draw_template_layout(self, gray_image, colored_image, config, *args, **kwargs):
        template = self.template
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
                    0,
                    resize_to_height=True,
                    config=config,
                )
                InteractionUtils.show(
                    "Final Marked Template (Colored)",
                    colored_final_marked,
                    1,
                    resize_to_height=True,
                    config=config,
                )
        elif config.outputs.show_image_level >= 1:
            InteractionUtils.show(
                "Final Marked Template",
                final_marked,
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
        field_id_to_interpretations=None,
        evaluation_meta=None,
        evaluation_config_for_response=None,
        shifted=False,
        border=-1,
    ):
        marked_image = ImageUtils.resize_to_dimensions(
            template.template_dimensions, image
        )

        transparent_layer = marked_image.copy()

        if field_id_to_interpretations is None:
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
                    field_id_to_interpretations,
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
                field_id_to_interpretations,
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
        field_id_to_interpretations,
        evaluation_meta,
        evaluation_config_for_response,
    ):
        should_draw_question_verdicts = (
            evaluation_meta is not None and evaluation_config_for_response is not None
        )
        for field in template.all_fields:
            # TODO: draw OCR detections separately
            # if field.field_detection_type == FieldDetectionType.OCR:
            if field.field_detection_type != FieldDetectionType.BUBBLES_THRESHOLD:
                continue

            field_label = field.field_label
            field_bubble_interpretations = field_id_to_interpretations[field.id]
            question_has_verdict = (
                evaluation_meta is not None
                and field_label in evaluation_meta["questions_meta"]
            )

            # linked_custom_labels = [custom_label if (field_label in field_labels) else None for (custom_label, field_labels) in template.custom_labels.items()]
            # is_part_of_custom_label = len(linked_custom_labels) > 0
            # TODO: replicate verdict: question_has_verdict = len([if field_label in questions_meta else None for field_label in linked_custom_labels])

            if (
                should_draw_question_verdicts
                and question_has_verdict
                and evaluation_config_for_response.draw_question_verdicts["enabled"]
            ):
                question_meta = evaluation_meta["questions_meta"][field_label]
                # Draw answer key items
                TemplateDrawingUtils.draw_field_with_question_meta(
                    marked_image,
                    image_type,
                    field_bubble_interpretations,
                    question_meta,
                    evaluation_config_for_response,
                )
            else:
                TemplateDrawingUtils.draw_scan_boxes_and_detections(
                    marked_image,
                    field_bubble_interpretations,
                    evaluation_config_for_response,
                )

        return marked_image

    @staticmethod
    def draw_field_with_question_meta(
        marked_image,
        image_type,
        field_bubble_interpretations: List[BubbleInterpretation],
        question_meta,
        evaluation_config_for_response,
    ):
        for scan_box_interpretation in field_bubble_interpretations:
            scan_box_interpretation.drawing.draw_interpretation(
                marked_image, image_type
            )

        if evaluation_config_for_response.draw_answer_groups["enabled"]:
            TemplateDrawingUtils.draw_answer_groups(
                marked_image,
                image_type,
                question_meta,
                field_bubble_interpretations,
                evaluation_config_for_response,
            )

    @staticmethod
    def draw_scan_boxes_and_detections(
        marked_image,
        field_bubble_interpretations: List[BubbleInterpretation],
        evaluation_config_for_response: EvaluationConfigForSet,
    ):
        # TODO: make this generic, consume FieldInterpretation
        for scan_box_interpretation in field_bubble_interpretations:
            bubble = scan_box_interpretation.item_reference
            bubble_dimensions = bubble.dimensions
            shifted_position = tuple(bubble.get_shifted_position())
            bubble_value = str(bubble.bubble_value)

            if scan_box_interpretation.is_marked:
                DrawingUtils.draw_box(
                    marked_image,
                    shifted_position,
                    bubble_dimensions,
                    color=CLR_GRAY,
                    style="BOX_FILLED",
                    thickness_factor=1 / 12,
                )
                if (
                    # Note: this mimics the default true behavior for draw_detected_bubble_texts
                    evaluation_config_for_response is None
                    or evaluation_config_for_response.draw_detected_bubble_texts[
                        "enabled"
                    ]
                ):
                    DrawingUtils.draw_text(
                        marked_image,
                        bubble_value,
                        shifted_position,
                        text_size=TEXT_SIZE,
                        color=CLR_NEAR_BLACK,
                        thickness=int(1 + 3.5 * TEXT_SIZE),
                    )
            else:
                DrawingUtils.draw_box(
                    marked_image,
                    shifted_position,
                    bubble_dimensions,
                    style="BOX_HOLLOW",
                    thickness_factor=1 / 10,
                )

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

    @staticmethod
    def draw_answer_groups(
        marked_image,
        image_type,
        question_meta,
        field_bubble_interpretations: List[BubbleInterpretation],
        evaluation_config_for_response: EvaluationConfigForSet,
    ):
        # Note: currently draw_answer_groups is limited for questions with upto 4 bubbles
        answer_type = question_meta["answer_type"]
        if answer_type == AnswerType.STANDARD:
            return
        box_edges = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
        color_sequence = evaluation_config_for_response.draw_answer_groups[
            "color_sequence"
        ]
        if image_type == "GRAYSCALE":
            color_sequence = [CLR_WHITE] * len(color_sequence)
        for scan_box_interpretation in field_bubble_interpretations:
            bubble = scan_box_interpretation.item_reference
            bubble_dimensions = bubble.dimensions
            shifted_position = tuple(bubble.get_shifted_position())
            bubble_value = str(bubble.bubble_value)
            matched_groups = AnswerMatcher.get_matched_answer_groups(
                question_meta, bubble_value
            )
            for answer_index in matched_groups:
                box_edge = box_edges[answer_index % 4]
                color = color_sequence[answer_index % 4]
                DrawingUtils.draw_group(
                    marked_image, shifted_position, bubble_dimensions, box_edge, color
                )
