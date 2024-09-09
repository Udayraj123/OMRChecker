from src.algorithm.template.detection.base.interpretation_pass import (
    FieldTypeInterpretationPass,
)
from src.algorithm.template.detection.ocr.interpretation import OCRFieldInterpretation
from src.algorithm.template.template_layout import Field


class OCRInterpretationPass(FieldTypeInterpretationPass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Note: This is used by parent to generate the interpretation: detected string etc
    def get_field_interpretation(
        self,
        field: Field,
        file_level_detection_aggregates,
        file_level_aggregates,
    ) -> OCRFieldInterpretation:
        tuning_config = self.tuning_config
        return OCRFieldInterpretation(
            tuning_config,
            field,
            file_level_detection_aggregates,
            file_level_aggregates,
        )

    def initialize_file_level_aggregates(
        self,
        file_path,
        field_detection_type_wise_detection_aggregates,
        field_label_wise_detection_aggregates,
    ):
        super().initialize_file_level_aggregates(
            file_path,
            field_detection_type_wise_detection_aggregates,
            field_label_wise_detection_aggregates,
        )
        self.insert_file_level_aggregates(
            {
                # TODO: check if any insert needed
            }
        )

    def update_field_level_aggregates_on_processed_field_interpretation(
        self, field: Field, field_interpretation: OCRFieldInterpretation
    ):
        super().update_field_level_aggregates_on_processed_field_interpretation(
            field, field_interpretation
        )
        # TODO: get this object from field_interpretation through a function?
        # TODO: move is_multi_marked logic to a parent class
        self.insert_field_level_aggregates(
            {
                "is_multi_marked": field_interpretation.is_multi_marked,
            }
        )

    def update_file_level_aggregates_on_processed_field_interpretation(
        self, field, field_interpretation, field_level_aggregates
    ):
        super().update_file_level_aggregates_on_processed_field_interpretation(
            field, field_interpretation, field_level_aggregates
        )
