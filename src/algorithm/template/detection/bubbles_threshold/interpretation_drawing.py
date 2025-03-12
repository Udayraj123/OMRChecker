from src.algorithm.evaluation.config import AnswerMatcher
from src.algorithm.template.detection.base.interpretation_drawing import (
    FieldInterpretationDrawing,
)
from src.utils.constants import CLR_BLACK, CLR_NEAR_BLACK, TEXT_SIZE
from src.utils.drawing import DrawingUtils


class BubblesFieldInterpretationDrawing(FieldInterpretationDrawing):
    def __init__(self, field_interpretation):
        super().__init__(field_interpretation)

    # Field-wise drawing with shifts support?
    def draw_interpretation(
        self,
        marked_image,
        evaluation_config_for_response,
        question_meta,
        image_type,
        thickness_factor,
    ):
        bonus_type = question_meta["bonus_type"]

        field_interpretation = self.field_interpretation
        bubble = field_interpretation.item_reference
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
        if field_interpretation.is_marked or bonus_type is not None:
            (
                verdict_symbol,
                verdict_color,
                verdict_symbol_color,
                thickness_factor,
            ) = evaluation_config_for_response.get_evaluation_meta_for_question(
                question_meta, field_interpretation, image_type
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
                field_interpretation.is_marked
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
