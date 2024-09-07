from abc import abstractmethod

from src.algorithm.template.detection.base.common_pass import FilePassAggregates
from src.algorithm.template.detection.base.interpretation import FieldInterpretation
from src.algorithm.template.template_layout import Field


class FieldTypeInterpretationPass(FilePassAggregates):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_field_interpretation(
        self,
        field: Field,
        file_level_detection_aggregates,
        file_level_aggregates,
    ) -> FieldInterpretation:
        raise Exception("Not implemented")

    def initialize_field_level_aggregates(self, field):
        super().initialize_field_level_aggregates(field)
        self.insert_file_level_aggregates(
            {
                "confidence_metrics_for_field": {},
            }
        )

    def initialize_file_level_aggregates(
        self,
        file_path,
        field_detection_type_wise_detection_aggregates,
        field_label_wise_detection_aggregates,
    ):
        super().initialize_file_level_aggregates(file_path)
        self.insert_file_level_aggregates(
            {
                "confidence_metrics_for_file": {},
            }
        )

    def update_aggregates_on_processed_field_interpretation(
        self, field: Field, field_interpretation: FieldInterpretation
    ):
        self.update_field_level_aggregates_on_processed_field_interpretation(
            field, field_interpretation
        )
        super().update_field_level_aggregates_on_processed_field(field)
        field_level_aggregates = self.get_field_level_aggregates()
        self.update_file_level_aggregates_on_processed_field_interpretation(
            field, field_interpretation, field_level_aggregates
        )
        super().update_file_level_aggregates_on_processed_field(
            field, field_level_aggregates
        )

    def update_field_level_aggregates_on_processed_field_interpretation(
        self, field: Field, field_interpretation: FieldInterpretation
    ):
        self.insert_field_level_aggregates(
            {
                "confidence_metrics_for_field": field_interpretation.confidence_metrics_for_field,
            }
        )

    def update_file_level_aggregates_on_processed_field_interpretation(
        self,
        field: Field,
        field_interpretation: FieldInterpretation,
        field_level_aggregates,
    ):
        self.file_level_aggregates["confidence_metrics_for_file"][
            field.field_label
        ] = field_interpretation.confidence_metrics_for_field
