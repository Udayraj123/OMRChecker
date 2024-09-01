from src.algorithm.template.detection.base.field_interpretation import (
    FieldInterpretation,
)
from src.processors.constants import FieldDetectionType


class OCRInterpretation(FieldInterpretation):
    def __init__(
        self,
        tuning_config,
        field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ):
        self.field_detection_type = FieldDetectionType.OCR
        super().__init__(tuning_config, field)

        self.empty_value = field.empty_value
        self.interpret_detection(
            field, file_level_detection_aggregates, file_level_interpretation_aggregates
        )

    def interpret_detection(
        self,
        field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ):
        pass

    def get_detected_string(self):
        detected_strings = self.detected_strings

        # TODO: Concatenation logic based on config
        # if self.interpretation_config.concatenatation.enabled:
        # detected_string = interpretation_config.concatenatation.separator.join(detected_strings)

        detected_string = detected_strings[0]

        if detected_string == "":
            return self.empty_value

        return detected_string

    def update_common_interpretations(self):
        pass
