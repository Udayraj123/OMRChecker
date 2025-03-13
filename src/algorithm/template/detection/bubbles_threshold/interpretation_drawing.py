from typing import List

from src.algorithm.evaluation.config import AnswerMatcher, EvaluationConfigForSet
from src.algorithm.template.detection.base.interpretation import FieldInterpretation
from src.algorithm.template.detection.base.interpretation_drawing import (
    FieldInterpretationDrawing,
)
from src.processors.constants import FieldDetectionType
from src.schemas.constants import AnswerType
from src.utils.constants import (
    CLR_BLACK,
    CLR_GRAY,
    CLR_NEAR_BLACK,
    CLR_WHITE,
    TEXT_SIZE,
)
from src.utils.drawing import DrawingUtils


class BubblesFieldInterpretationDrawing(FieldInterpretationDrawing):
    def __init__(self, field_interpretation):
        super().__init__(field_interpretation)

    def draw_field_interpretation(
        self, marked_image, image_type, evaluation_meta, evaluation_config_for_response
    ):
        field_label = self.field.field_label
        bubble_interpretations = self.field_interpretation.bubble_interpretations
        should_draw_question_verdicts = (
            evaluation_meta is not None and evaluation_config_for_response is not None
        )
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
            self.draw_bubbles_and_detections_with_verdicts(
                marked_image,
                image_type,
                bubble_interpretations,
                question_meta,
                evaluation_config_for_response,
            )
        else:
            self.draw_bubbles_and_detections_without_verdicts(
                marked_image,
                bubble_interpretations,
                evaluation_config_for_response,
            )

    @staticmethod
    def draw_bubbles_and_detections_with_verdicts(
        marked_image,
        image_type,
        bubble_interpretations: List[FieldInterpretation],
        question_meta,
        evaluation_config_for_response,
    ):
        for bubble_interpretation in bubble_interpretations:
            BubblesFieldInterpretationDrawing.draw_unit_bubble_interpretation_with_verdicts(
                bubble_interpretation,
                marked_image,
                evaluation_config_for_response,
                question_meta,
                image_type,
            )

        if evaluation_config_for_response.draw_answer_groups["enabled"]:
            BubblesFieldInterpretationDrawing.draw_answer_groups_for_bubbles(
                marked_image,
                image_type,
                question_meta,
                bubble_interpretations,
                evaluation_config_for_response,
            )

    @staticmethod
    def draw_bubbles_and_detections_without_verdicts(
        marked_image,
        bubble_interpretations: List[FieldInterpretation],
        evaluation_config_for_response: EvaluationConfigForSet,
    ):
        # TODO: make this generic, consume FieldInterpretation
        for field_interpretation in bubble_interpretations:
            bubble = field_interpretation.item_reference
            bubble_dimensions = bubble.dimensions
            shifted_position = tuple(bubble.get_shifted_position())
            bubble_value = str(bubble.bubble_value)

            if field_interpretation.is_marked:
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
    def draw_unit_bubble_interpretation_with_verdicts(
        bubble_interpretation,
        marked_image,
        evaluation_config_for_response,
        question_meta,
        image_type,
        thickness_factor=1 / 12,
    ):
        bonus_type = question_meta["bonus_type"]

        bubble = bubble_interpretation.item_reference
        bubble_dimensions = bubble.dimensions
        shifted_position = tuple(bubble.get_shifted_position())
        bubble_value = str(bubble.bubble_value)

        # Enhanced bounding box for expected answer:
        if AnswerMatcher.is_part_of_some_answer(question_meta, bubble_value):
            DrawingUtils.draw_box(
                marked_image,
                shifted_position,
                bubble_dimensions,
                CLR_BLACK,
                style="BOX_HOLLOW",
                thickness_factor=0,
            )

        # Filled box in case of marked bubble or bonus case
        if bubble_interpretation.is_marked or bonus_type is not None:
            (
                verdict_symbol,
                verdict_color,
                verdict_symbol_color,
                thickness_factor,
            ) = evaluation_config_for_response.get_evaluation_meta_for_question(
                question_meta, bubble_interpretation, image_type
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
                bubble_interpretation.is_marked
                and evaluation_config_for_response.draw_detected_bubble_texts["enabled"]
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
    def draw_answer_groups_for_bubbles(
        marked_image,
        image_type,
        question_meta,
        bubble_interpretations: List[FieldInterpretation],
        evaluation_config_for_response: EvaluationConfigForSet,
    ):
        # Note: currently draw_answer_groups is limited for questions with upto 4 values
        answer_type = question_meta["answer_type"]
        if answer_type == AnswerType.STANDARD:
            return
        box_edges = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
        color_sequence = evaluation_config_for_response.draw_answer_groups[
            "color_sequence"
        ]
        if image_type == "GRAYSCALE":
            color_sequence = [CLR_WHITE] * len(color_sequence)

        for field_interpretation in bubble_interpretations:
            field = field_interpretation.field
            if field.field_detection_type != FieldDetectionType.BUBBLES_THRESHOLD:
                continue

            bubble = field_interpretation.item_reference
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
