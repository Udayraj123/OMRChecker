from abc import abstractmethod
from typing import Never

from src.algorithm.template.layout.field.base import Field


class FieldInterpretation:
    def __init__(
        self,
        tuning_config,
        field: Field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ) -> None:
        self.tuning_config = tuning_config
        self.field = field

        # TODO: replace is_marked with is_attempted
        self.is_marked = None
        # self.field_block = field.field_block
        self.empty_value = field.empty_value
        self.field_level_confidence_metrics = {}
        self.drawing = self.get_drawing_instance()

        self.run_interpretation(
            field, file_level_detection_aggregates, file_level_interpretation_aggregates
        )

    @abstractmethod
    def get_drawing_instance(self) -> Never:
        msg = "Not implemented"
        raise Exception(msg)

    @abstractmethod
    def run_interpretation(
        self,
        field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ) -> Never:
        msg = "Not implemented"
        raise Exception(msg)

    @abstractmethod
    def get_field_interpretation_string() -> Never:
        msg = "Not implemented"
        raise Exception(msg)

    def get_field_level_confidence_metrics(self):
        return self.field_level_confidence_metrics

    def insert_field_level_confidence_metrics(
        self, next_field_level_confidence_metrics
    ) -> None:
        self.field_level_confidence_metrics = {
            **self.field_level_confidence_metrics,
            **next_field_level_confidence_metrics,
        }
