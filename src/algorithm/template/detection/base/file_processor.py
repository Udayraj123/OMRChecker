from src.algorithm.template.detection.base.detection_pass import FieldTypeDetectionPass
from src.algorithm.template.detection.base.interpretation_pass import (
    FieldTypeInterpretationPass,
)
from src.algorithm.template.template_layout import Field

"""
FieldTypeFileProcessor contains the external contract to be used by TemplateDetector for each of the field_detection_types
It is static per template instance. Instantiated once per field type from the template.json

"""


class FieldTypeFileProcessor:
    def __init__(
        self,
        tuning_config,
        field_detection_type,
        detection_pass: FieldTypeDetectionPass,
        interpretation_pass: FieldTypeInterpretationPass,
    ):
        self.tuning_config = tuning_config
        self.field_detection_type = field_detection_type
        self.detection_pass = detection_pass
        self.interpretation_pass = interpretation_pass

    # Common wrappers
    def initialize_directory_level_aggregates(self):
        self.detection_pass.initialize_directory_level_aggregates()
        self.interpretation_pass.initialize_directory_level_aggregates()

    # TODO: see if this is really needed?
    # def get_aggregates_data(self):
    #     return {
    #         "detection": {
    #             "directory_level_aggregates": self.detection_pass.get_directory_level_aggregates(),
    #             # ...
    #         },
    #         "interpretation": {
    #             "directory_level_aggregates": self.interpretation_pass.get_directory_level_aggregates(),
    #             # ...
    #         },
    #         # More passes in future?
    #     }

    # Detection: Field Level
    def get_field_level_detection_aggregates(self):
        return self.detection_pass.get_field_level_aggregates()

    def run_field_level_detection(self, field, gray_image, colored_image):
        self.detection_pass.initialize_field_level_aggregates(field)

        field_detection = self.detection_pass.get_field_detection(
            field, gray_image, colored_image
        )

        self.detection_pass.update_aggregates_on_processed_field_detection(
            field, field_detection
        )

    # Detection: File Level
    def initialize_file_level_detection_aggregates(self, file_path):
        return self.detection_pass.initialize_file_level_aggregates(file_path)

    def get_file_level_detection_aggregates(self):
        return self.detection_pass.get_file_level_aggregates()

    def update_detection_aggregates_on_processed_file(self, file_path):
        return self.detection_pass.update_aggregates_on_processed_file(file_path)

    # Interpretation: Field Level
    def get_field_level_interpretation_aggregates(self):
        return self.interpretation_pass.get_field_level_aggregates()

    def run_field_level_interpretation(self, field: Field, _gray_image, _colored_image):
        self.interpretation_pass.initialize_field_level_aggregates(field)
        file_level_detection_aggregates = (
            self.detection_pass.get_file_level_aggregates()
        )
        file_level_interpretation_aggregates = (
            self.interpretation_pass.get_file_level_aggregates()
        )
        # TODO: directly call .run_field_interpretation()
        field_interpretation = self.interpretation_pass.get_field_interpretation(
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

        self.interpretation_pass.update_aggregates_on_processed_field_interpretation(
            field, field_interpretation
        )

        detected_string = field_interpretation.get_detected_string()

        return detected_string

    # Interpretation: File Level
    def initialize_file_level_interpretation_aggregates(
        self,
        file_path,
        field_detection_type_wise_detection_aggregates,
        field_label_wise_detection_aggregates,
    ):
        return self.interpretation_pass.initialize_file_level_aggregates(
            file_path,
            field_detection_type_wise_detection_aggregates,
            field_label_wise_detection_aggregates,
        )

    def get_file_level_interpretation_aggregates(self):
        return self.interpretation_pass.get_file_level_aggregates()

    def update_interpretation_aggregates_on_processed_file(self, file_path):
        return self.interpretation_pass.update_aggregates_on_processed_file(file_path)
