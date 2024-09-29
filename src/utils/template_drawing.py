from typing import List

import cv2

from src.algorithm.evaluation.config import EvaluationConfigForSet
from src.algorithm.template.detection.bubbles_threshold.interpretation import (
    BubbleInterpretation,
)
from src.algorithm.template.template_layout import FieldBlock
from src.processors.constants import FieldDetectionType
from src.schemas.constants import AnswerType
from src.utils.constants import (
    CLR_BLACK,
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
    # TODO: move this class into template.drawing_handler?
    @staticmethod
    def draw_template_layout(
        gray_image, colored_image, template, config, *args, **kwargs
    ):
        final_marked = TemplateDrawing.draw_template_layout_util(
            gray_image, "GRAYSCALE", template, config, *args, **kwargs
        )

        colored_final_marked = colored_image
        if config.outputs.colored_outputs_enabled:
            colored_final_marked = TemplateDrawing.draw_template_layout_util(
                colored_final_marked,
                "COLORED",
                template,
                config,
                *args,
                **kwargs,
            )
            if config.outputs.show_image_level >= 1:
                InteractionUtils.show(
                    "Final Marked Bubbles",
                    final_marked,
                    0,
                    resize_to_height=True,
                    config=config,
                )
                InteractionUtils.show(
                    "Final Marked Bubbles (Colored)",
                    colored_final_marked,
                    1,
                    resize_to_height=True,
                    config=config,
                )
        elif config.outputs.show_image_level >= 1:
            InteractionUtils.show(
                "Final Marked Bubbles",
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
        field_number_to_scan_box_interpretation=None,
        evaluation_meta=None,
        evaluation_config_for_response=None,
        shifted=False,
        border=-1,
    ):
        marked_image = ImageUtils.resize_to_dimensions(
            template.template_dimensions, image
        )

        transparent_layer = marked_image.copy()
        should_draw_field_block_rectangles = (
            field_number_to_scan_box_interpretation is None
        )
        should_draw_marked_bubbles = field_number_to_scan_box_interpretation is not None
        should_draw_question_verdicts = (
            should_draw_marked_bubbles and evaluation_meta is not None
        )

        if should_draw_field_block_rectangles:
            marked_image = TemplateDrawing.draw_field_blocks_layout(
                marked_image, template, shifted, shouldCopy=False, border=border
            )
            return marked_image

        if should_draw_marked_bubbles:
            if config.outputs.save_image_level >= 1:
                marked_image_copy = marked_image.copy()
                marked_image_copy = (
                    TemplateDrawing.draw_marked_bubbles_with_evaluation_meta(
                        marked_image_copy,
                        image_type,
                        template,
                        field_number_to_scan_box_interpretation,
                        evaluation_meta=None,
                        evaluation_config_for_response=None,
                    )
                )
                if image_type == "GRAYSCALE":
                    template.save_image_ops.append_save_image(
                        f"Marked Image", range(2, 7), marked_image_copy
                    )
                else:
                    template.save_image_ops.append_save_image(
                        f"Marked Image", range(2, 7), colored_image=marked_image_copy
                    )

            marked_image = TemplateDrawing.draw_marked_bubbles_with_evaluation_meta(
                marked_image,
                image_type,
                template,
                field_number_to_scan_box_interpretation,
                evaluation_meta,
                evaluation_config_for_response,
            )

        if should_draw_question_verdicts:
            marked_image = TemplateDrawing.draw_evaluation_summary(
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
    def draw_field_blocks_layout(
        image, template, shifted=True, shouldCopy=True, thickness=3, border=3
    ):
        marked_image = image.copy() if shouldCopy else image
        for field_block in template.field_blocks:
            TemplateDrawing.draw_field_block(
                field_block, marked_image, shifted, thickness, border
            )

        return marked_image

    @staticmethod
    def draw_field_block(
        field_block, marked_image, shifted=True, thickness=3, border=3
    ):
        (
            field_block_name,
            bounding_box_origin,
            bounding_box_dimensions,
        ) = map(
            lambda attr: getattr(field_block, attr),
            [
                "name",
                "bounding_box_origin",
                "bounding_box_dimensions",
            ],
        )

        # TODO: get this field block using a bounding box of all bubbles instead. (remove shift at field block level)
        block_position = (
            field_block.get_shifted_origin() if shifted else bounding_box_origin
        )
        if not shifted:
            # Field block bounding rectangle
            DrawingUtils.draw_box(
                marked_image,
                block_position,
                bounding_box_dimensions,
                color=CLR_BLACK,
                style="BOX_HOLLOW",
                thickness_factor=0,
                border=border,
            )

        for field in field_block.fields:
            scan_boxes = field.scan_boxes
            for unit_bubble in scan_boxes:
                shifted_position = unit_bubble.get_shifted_position(field_block.shifts)
                dimensions = unit_bubble.dimensions
                DrawingUtils.draw_box(
                    marked_image,
                    shifted_position,
                    dimensions,
                    thickness_factor=1 / 10,
                    border=border,
                )

        if shifted:
            text_position = lambda size_x, size_y: (
                int(block_position[0] + bounding_box_dimensions[0] - size_x),
                int(block_position[1] - size_y),
            )
            text = f"({field_block.shifts}){field_block_name}"
            DrawingUtils.draw_text(
                marked_image, text, text_position, thickness=thickness
            )

        return

    @staticmethod
    def draw_marked_bubbles_with_evaluation_meta(
        marked_image,
        image_type,
        template,
        field_number_to_scan_box_interpretation,
        evaluation_meta,
        evaluation_config_for_response,
    ):
        should_draw_question_verdicts = (
            evaluation_meta is not None and evaluation_config_for_response is not None
        )
        absolute_field_number = 0
        for field_block in template.field_blocks:
            for field in field_block.fields:
                # TODO: draw OCR detections separately (after fixing absolute_field_number)
                # if field.field_detection_type == FieldDetectionType.OCR:
                if field.field_detection_type != FieldDetectionType.BUBBLES_THRESHOLD:
                    continue

                field_label = field.field_label
                field_bubble_interpretations = field_number_to_scan_box_interpretation[
                    absolute_field_number
                ]
                absolute_field_number += 1
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
                    TemplateDrawing.draw_field_with_question_meta(
                        marked_image,
                        image_type,
                        field_bubble_interpretations,
                        field_block,
                        question_meta,
                        evaluation_config_for_response,
                    )
                else:
                    TemplateDrawing.draw_scan_boxes_and_detections(
                        marked_image,
                        field_bubble_interpretations,
                        field_block,
                        evaluation_config_for_response,
                    )

        return marked_image

    @staticmethod
    def draw_field_with_question_meta(
        marked_image,
        image_type,
        field_bubble_interpretations: List[BubbleInterpretation],
        field_block: FieldBlock,
        question_meta,
        evaluation_config_for_response,
    ):
        bonus_type = question_meta["bonus_type"]
        for scan_box_interpretation in field_bubble_interpretations:
            bubble = scan_box_interpretation.item_reference
            bubble_dimensions = bubble.dimensions
            shifted_position = tuple(bubble.get_shifted_position(field_block.shifts))
            bubble_value = str(bubble.bubble_value)

            # Enhanced bounding box for expected answer:
            if TemplateDrawing.is_part_of_some_answer(question_meta, bubble_value):
                DrawingUtils.draw_box(
                    marked_image,
                    shifted_position,
                    bubble_dimensions,
                    CLR_BLACK,
                    style="BOX_HOLLOW",
                    thickness_factor=0,
                )

            # Filled box in case of marked bubble or bonus case
            if scan_box_interpretation.is_marked or bonus_type is not None:
                (
                    verdict_symbol,
                    verdict_color,
                    verdict_symbol_color,
                    thickness_factor,
                ) = evaluation_config_for_response.get_evaluation_meta_for_question(
                    question_meta, scan_box_interpretation, image_type
                )

                # Bounding box for marked bubble or bonus bubble
                if verdict_color != "":
                    position, position_diagonal = DrawingUtils.draw_box(
                        marked_image,
                        shifted_position,
                        bubble_dimensions,
                        color=verdict_color,
                        style="BOX_FILLED",
                        thickness_factor=thickness_factor,
                    )

                # Symbol for the marked bubble or bonus bubble
                if verdict_symbol != "":
                    DrawingUtils.draw_symbol(
                        marked_image,
                        verdict_symbol,
                        position,
                        position_diagonal,
                        color=verdict_symbol_color,
                    )

                # Symbol of the field value for marked bubble
                if (
                    scan_box_interpretation.is_marked
                    and evaluation_config_for_response.draw_detected_bubble_texts[
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

        if evaluation_config_for_response.draw_answer_groups["enabled"]:
            TemplateDrawing.draw_answer_groups(
                marked_image,
                image_type,
                question_meta,
                field_bubble_interpretations,
                field_block,
                evaluation_config_for_response,
            )

    @staticmethod
    def draw_scan_boxes_and_detections(
        marked_image,
        field_bubble_interpretations: List[BubbleInterpretation],
        field_block: FieldBlock,
        evaluation_config_for_response: EvaluationConfigForSet,
    ):
        # TODO: make this generic, consume FieldInterpretation
        for scan_box_interpretation in field_bubble_interpretations:
            bubble = scan_box_interpretation.item_reference
            bubble_dimensions = bubble.dimensions
            shifted_position = tuple(bubble.get_shifted_position(field_block.shifts))
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
    def is_part_of_some_answer(question_meta, bubble_value):
        if question_meta["bonus_type"] is not None:
            return True
        matched_groups = TemplateDrawing.get_matched_answer_groups(
            question_meta, bubble_value
        )
        return len(matched_groups) > 0

    @staticmethod
    def get_matched_answer_groups(question_meta, bubble_value):
        matched_groups = []
        answer_type, answer_item = map(
            question_meta.get, ["answer_type", "answer_item"]
        )

        if answer_type == AnswerType.STANDARD:
            # Note: implicit check on concatenated answer
            if bubble_value in str(answer_item):
                matched_groups.append(0)
        if answer_type == AnswerType.MULTIPLE_CORRECT:
            for answer_index, allowed_answer in enumerate(answer_item):
                if bubble_value in allowed_answer:
                    matched_groups.append(answer_index)
        elif answer_type == AnswerType.MULTIPLE_CORRECT_WEIGHTED:
            for answer_index, (allowed_answer, score) in enumerate(answer_item):
                if bubble_value in allowed_answer and score > 0:
                    matched_groups.append(answer_index)
        return matched_groups

    @staticmethod
    def draw_answer_groups(
        marked_image,
        image_type,
        question_meta,
        field_bubble_interpretations: List[BubbleInterpretation],
        field_block: FieldBlock,
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
            shifted_position = tuple(bubble.get_shifted_position(field_block.shifts))
            bubble_value = str(bubble.bubble_value)
            matched_groups = TemplateDrawing.get_matched_answer_groups(
                question_meta, bubble_value
            )
            for answer_index in matched_groups:
                box_edge = box_edges[answer_index % 4]
                color = color_sequence[answer_index % 4]
                DrawingUtils.draw_group(
                    marked_image, shifted_position, bubble_dimensions, box_edge, color
                )
