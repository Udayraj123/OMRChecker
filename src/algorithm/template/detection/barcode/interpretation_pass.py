from src.algorithm.template.detection.barcode.interpretation import (
    BarcodeFieldInterpretation,
)
from src.algorithm.template.detection.base.interpretation_pass import (
    FieldTypeInterpretationPass,
)
from src.algorithm.template.layout.field.base import Field


class BarcodeInterpretationPass(FieldTypeInterpretationPass):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # Note: This is used by parent to generate the interpretation: detected string etc
    def get_field_interpretation(
        self,
        field: Field,
        file_level_detection_aggregates,
        file_level_aggregates,
    ) -> BarcodeFieldInterpretation:
        tuning_config = self.tuning_config
        return BarcodeFieldInterpretation(
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
    ) -> None:
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
        self, field: Field, field_interpretation: BarcodeFieldInterpretation
    ) -> None:
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
    ) -> None:
        super().update_file_level_aggregates_on_processed_field_interpretation(
            field, field_interpretation, field_level_aggregates
        )
