from abc import abstractmethod

from src.algorithm.template.template_layout import Field


class FieldInterpretation:
    def __init__(
        self,
        tuning_config,
        field: Field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ):
        self.tuning_config = tuning_config
        self.field = field
        # self.field_block = field.field_block
        self.empty_value = field.empty_value
        self.confidence_metrics_for_field = {}

        self.run_interpretation(
            field, file_level_detection_aggregates, file_level_interpretation_aggregates
        )

    @abstractmethod
    def run_interpretation(
        self,
        field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ):
        raise Exception(f"Not implemented")

    @abstractmethod
    def get_detected_string():
        raise Exception(f"Not implemented")

    @abstractmethod
    def get_field_level_confidence_metrics():
        raise Exception(f"Not implemented")
