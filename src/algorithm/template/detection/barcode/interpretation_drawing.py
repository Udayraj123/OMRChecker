from src.algorithm.template.detection.base.interpretation_drawing import (
    FieldInterpretationDrawing,
)
from src.utils.constants import CLR_BLACK
from src.utils.drawing import DrawingUtils
from src.utils.math import MathUtils


class BarcodeFieldInterpretationDrawing(FieldInterpretationDrawing):
    def __init__(self, field_interpretation) -> None:
        super().__init__(field_interpretation)

    def draw_field_interpretation(
        self, marked_image, image_type, evaluation_meta, evaluation_config_for_response
    ) -> None:
        field_label = self.field.field_label
        field_interpretation = self.field_interpretation
        if len(field_interpretation.interpretations) == 0:
            return

        # Draw all bounding boxes
        all_bounding_box_points = []
        for interpretation in field_interpretation.interpretations:
            bounding_box = interpretation.text_detection.bounding_box
            DrawingUtils.draw_contour(marked_image, bounding_box, color=CLR_BLACK)
            all_bounding_box_points.extend(bounding_box)

        combined_bounding_box, _dimensions = MathUtils.get_bounding_box_of_points(
            all_bounding_box_points
        )

        bounding_box_start = combined_bounding_box[0]
        text_position = MathUtils.add_points(
            bounding_box_start, (0, max(-10, -1 * bounding_box_start[1]))
        )
        combined_bounding_box_color = CLR_BLACK

        should_draw_question_verdicts = (
            evaluation_meta is not None and evaluation_config_for_response is not None
        )
        if should_draw_question_verdicts:
            question_has_verdict = field_label in evaluation_meta["questions_meta"]
            if question_has_verdict:
                question_meta = evaluation_meta["questions_meta"][field_label]
                bonus_type = question_meta["bonus_type"]

                # Filled box in case of marked bubble or bonus case
                if field_interpretation.is_attempted or bonus_type is not None:
                    (
                        _verdict_symbol,
                        verdict_color,
                        _verdict_symbol_color,
                        _thickness_factor,
                    ) = evaluation_config_for_response.get_evaluation_meta_for_question(
                        question_meta, field_interpretation, image_type
                    )
                    combined_bounding_box_color = verdict_color

        DrawingUtils.draw_contour(
            marked_image, combined_bounding_box, color=combined_bounding_box_color
        )

        # This string is the interpreted text from the Barcode detection
        interpreted_text = field_interpretation.get_field_interpretation_string()
        DrawingUtils.draw_text_responsive(
            marked_image, interpreted_text, text_position, color=CLR_BLACK, thickness=3
        )
