from src.algorithm.template.detection.base.interpretation import FieldInterpretation
from src.algorithm.template.template_layout import Field


class OCRFieldInterpretation(FieldInterpretation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_detected_string(self):
        marked_bubbles = [
            bubble_interpretation.bubble_value
            for bubble_interpretation in self.field_bubble_interpretations
            if bubble_interpretation.is_marked
        ]
        # Empty value logic
        if len(marked_bubbles) == 0:
            return self.empty_value

        # Concatenation logic
        return "".join(marked_bubbles)

    def run_interpretation(
        self,
        field: Field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ):
        # field_label = field.field_label
        # self.process_field_bubble_means()
        self.update_common_interpretations()

    def update_common_interpretations(self):
        # TODO: can we move it to a common wrapper since is_multi_marked is independent of field detection type?
        marked_bubbles = [
            bubble_interpretation.bubble_value
            for bubble_interpretation in self.field_bubble_interpretations
            if bubble_interpretation.is_marked
        ]
        self.is_multi_marked = len(marked_bubbles) > 1
