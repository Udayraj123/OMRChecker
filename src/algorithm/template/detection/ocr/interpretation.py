from typing import List

from src.algorithm.template.detection.base.interpretation import FieldInterpretation
from src.algorithm.template.template_layout import Field


class OCRInterpretation:
    def __init__(self):
        self.is_marked = None

    def get_value():
        return "todo"


class OCRFieldInterpretation(FieldInterpretation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_detected_string(self):
        marked_interpretations = [
            interpretation.get_value()
            for interpretation in self.interpretations
            if interpretation.is_marked
        ]
        # Empty value logic
        if len(marked_interpretations) == 0:
            return self.empty_value

        # TODO: if self.interpretation_config.concatenatation.enabled:
        # return "".join(marked_interpretations)

        return marked_interpretations[0]

    def run_interpretation(
        self,
        field: Field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ):
        # field_label = field.field_label
        self.interpretations: List[OCRInterpretation] = []
        # self.process_field_bubble_means()
        self.update_common_interpretations()

    def update_common_interpretations(self):
        # TODO: can we move it to a common wrapper since is_multi_marked is independent of field detection type?
        marked_interpretations = [
            interpretation.get_value()
            for interpretation in self.interpretations
            if interpretation.is_marked
        ]
        self.is_multi_marked = len(marked_interpretations) > 1
