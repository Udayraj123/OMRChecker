from typing import List

from src.algorithm.template.detection.base.interpretation import FieldInterpretation
from src.algorithm.template.template_layout import Field


class OCRInterpretation:
    def __init__(self, detected_text):
        self.is_marked = True
        self.detected_text = detected_text

    def get_value(self):
        return self.detected_text


class OCRFieldInterpretation(FieldInterpretation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_field_interpretation_string(self):
        marked_interpretations = [
            interpretation.get_value()
            for interpretation in self.interpretations
            if interpretation.is_marked
        ]
        # Empty value logic
        if len(marked_interpretations) == 0:
            return self.empty_value

        # TODO: if self.interpretation_config.concatenation.enabled:
        # return "".join(marked_interpretations)

        return marked_interpretations[0]

    def run_interpretation(
        self,
        field: Field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ):
        self.initialize_from_file_level_aggregates(
            field, file_level_detection_aggregates, file_level_interpretation_aggregates
        )
        self.update_common_interpretations()

    def initialize_from_file_level_aggregates(
        self,
        field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ):
        field_label = field.field_label

        field_level_detection_aggregates = file_level_detection_aggregates[
            "field_label_wise_aggregates"
        ][field_label]

        # map detections to interpretations
        self.interpretations: List[OCRInterpretation] = [
            OCRInterpretation(detected_text)
            for detected_text in field_level_detection_aggregates["detected_texts"]
        ]

    def update_common_interpretations(self):
        # TODO: can we move it to a common wrapper since is_multi_marked is independent of field detection type?
        for interpretation in self.interpretations:
            if interpretation.get_value() == "":
                interpretation.is_marked = False
        marked_interpretations = [
            interpretation.get_value()
            for interpretation in self.interpretations
            if interpretation.is_marked
        ]
        self.is_multi_marked = len(marked_interpretations) > 1
