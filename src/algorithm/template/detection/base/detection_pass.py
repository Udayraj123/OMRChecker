from abc import abstractmethod

from src.algorithm.template.detection.base.common_pass import FilePassAggregates
from src.algorithm.template.detection.base.detection import FieldDetection
from src.algorithm.template.template_layout import Field


class FieldTypeDetectionPass(FilePassAggregates):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_field_detection(
        self, field: Field, gray_image, colored_image
    ) -> FieldDetection:
        raise Exception("Not implemented")

    def update_aggregates_on_processed_field_detection(
        self, field: Field, field_detection: FieldDetection
    ):
        self.update_field_level_aggregates_on_processed_field_detection(
            field, field_detection
        )
        super().update_field_level_aggregates_on_processed_field(field)

        field_level_aggregates = self.get_field_level_aggregates()

        self.update_file_level_aggregates_on_processed_field_detection(
            field, field_detection, field_level_aggregates
        )
        super().update_file_level_aggregates_on_processed_field(
            field, field_level_aggregates
        )

    def update_field_level_aggregates_on_processed_field_detection(
        self, field, field_detection
    ):
        pass

    def update_file_level_aggregates_on_processed_field_detection(
        self, field, field_detection, field_level_aggregates
    ):
        # detection confidence metrics?
        pass
