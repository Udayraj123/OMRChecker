from src.processors.detection.base.interpretation import (
    BaseInterpretation,
    FieldInterpretation,
)
from src.processors.detection.ocr.interpretation_drawing import (
    OCRFieldInterpretationDrawing,
)
from src.processors.layout.field.base import Field
from src.utils.logger import logger


class OCRInterpretation(BaseInterpretation):
    def __init__(self, detection) -> None:
        super().__init__(detection)


class OCRFieldInterpretation(FieldInterpretation):
    def __init__(self, *args, **kwargs) -> None:
        self.interpretations: list[OCRInterpretation] = None
        super().__init__(*args, **kwargs)

    def get_drawing_instance(self):
        return OCRFieldInterpretationDrawing(self)

    def get_field_interpretation_string(self):
        # TODO: update this logic
        marked_interpretations = [
            interpretation.get_value()
            for interpretation in self.interpretations
            if interpretation.is_attempted
        ]
        # Empty value logic
        if len(marked_interpretations) == 0:
            return self.empty_value

        # TODO: if not self.interpretation_config.concatenation.enabled:
        #   return marked_interpretations[0]
        return "".join(marked_interpretations)

    def run_interpretation(
        self,
        field: Field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ) -> None:
        self.initialize_from_file_level_aggregates(
            field, file_level_detection_aggregates, file_level_interpretation_aggregates
        )
        self.update_common_interpretations()

    def initialize_from_file_level_aggregates(
        self,
        field,
        file_level_detection_aggregates,
        _file_level_interpretation_aggregates,
    ) -> None:
        field_label = field.field_label

        # Use new typed models pipeline via ocr_fields
        if "ocr_fields" not in file_level_detection_aggregates:
            msg = f"ocr_fields not found in file_level_detection_aggregates for field {field_label}"
            raise KeyError(msg)

        ocr_fields = file_level_detection_aggregates["ocr_fields"]
        if field_label not in ocr_fields:
            msg = f"Field {field_label} not found in ocr_fields"
            raise KeyError(msg)
        ocr_result = ocr_fields[field_label]
        detections = ocr_result.detections

        # map detections to interpretations
        self.interpretations = [
            OCRInterpretation(detection) for detection in detections
        ]

        if len(self.interpretations) == 0:
            logger.warning(f"No OCR detection for field: {self.field.id}")

    def update_common_interpretations(self) -> None:
        marked_interpretations = [
            interpretation.get_value()
            for interpretation in self.interpretations
            if interpretation.is_attempted
        ]
        self.is_attempted = len(marked_interpretations) > 0
        self.is_multi_marked = len(marked_interpretations) > 1
