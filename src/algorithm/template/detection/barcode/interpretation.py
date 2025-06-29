from src.algorithm.template.detection.barcode.interpretation_drawing import (
    BarcodeFieldInterpretationDrawing,
)
from src.algorithm.template.detection.base.interpretation import FieldInterpretation
from src.algorithm.template.layout.field.base import Field
from src.utils.logger import logger


class BarcodeInterpretation:
    def __init__(self, detection):
        self.detection = detection
        self.is_marked = detection is not None
        self.detected_text = detection.detected_text if self.is_marked else ""

    def get_value(self):
        return self.detected_text


class BarcodeFieldInterpretation(FieldInterpretation):
    def __init__(self, *args, **kwargs):
        self.interpretations: list[BarcodeInterpretation] = None
        super().__init__(*args, **kwargs)

    def get_drawing_instance(self):
        return BarcodeFieldInterpretationDrawing(self)

    def get_field_interpretation_string(self):
        # TODO: update this logic
        marked_interpretations = [
            interpretation.get_value()
            for interpretation in self.interpretations
            if interpretation.is_marked
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
        self.interpretations: list[BarcodeInterpretation] = [
            BarcodeInterpretation(detection)
            for detection in field_level_detection_aggregates["detections"]
        ]

        if len(self.interpretations) == 0:
            logger.warning(f"No Barcode detection for field: {self.field.id}")

    def update_common_interpretations(self):
        # TODO: can we move it to a common wrapper since is_multi_marked is independent of field detection type?
        marked_interpretations = [
            interpretation.get_value()
            for interpretation in self.interpretations
            if interpretation.is_marked
        ]
        self.is_marked = len(marked_interpretations) > 0
        self.is_multi_marked = len(marked_interpretations) > 1
