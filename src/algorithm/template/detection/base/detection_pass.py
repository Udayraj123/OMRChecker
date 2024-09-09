from abc import abstractmethod

from src.algorithm.template.detection.base.common_pass import FilePassAggregates
from src.algorithm.template.detection.base.detection import FieldDetection
from src.algorithm.template.template_layout import Field
from src.utils.stats import StatsByLabel


class FieldTypeDetectionPass(FilePassAggregates):
    def __init__(self, tuning_config, field_detection_type):
        self.field_detection_type = field_detection_type
        super().__init__(tuning_config)

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

        field_level_aggregates = self.get_field_level_aggregates()

        self.update_file_level_aggregates_on_processed_field_detection(
            field, field_detection, field_level_aggregates
        )

        self.update_directory_level_aggregates_on_processed_field_detection(
            field, field_detection, field_level_aggregates
        )

    def update_field_level_aggregates_on_processed_field_detection(
        self, field, field_detection
    ):
        super().update_field_level_aggregates_on_processed_field(field)

    def update_file_level_aggregates_on_processed_field_detection(
        self, field, field_detection, field_level_aggregates
    ):
        super().update_file_level_aggregates_on_processed_field(
            field, field_level_aggregates
        )
        # detection confidence metrics?

    def update_directory_level_aggregates_on_processed_field_detection(
        self, field, field_detection, field_level_aggregates
    ):
        super().update_directory_level_aggregates_on_processed_field(
            field, field_level_aggregates
        )
        # (if needed) update_directory_level_aggregates_on_processed_field


class TemplateDetectionPass(FilePassAggregates):
    def initialize_directory_level_aggregates(
        self, initial_directory_path, all_field_detection_types
    ):
        super().initialize_directory_level_aggregates(initial_directory_path)
        self.insert_directory_level_aggregates(
            {
                "files_by_label_count": StatsByLabel("processed", "multi_marked"),
                "field_detection_type_wise_aggregates": {
                    key: {"fields_count": StatsByLabel("processed")}
                    for key in all_field_detection_types
                },
            }
        )

    def initialize_file_level_aggregates(self, file_path, all_field_detection_types):
        super().initialize_file_level_aggregates(file_path)
        self.insert_file_level_aggregates(
            {
                "field_detection_type_wise_aggregates": {
                    key: {"fields_count": StatsByLabel("processed")}
                    for key in all_field_detection_types
                },
            }
        )

    def update_aggregates_on_processed_field_detection(
        self, field: Field, field_detection: FieldDetection
    ):
        self.update_field_level_aggregates_on_processed_field_detection(
            field, field_detection
        )

        # TODO: can we move this getter to parent class?
        field_level_aggregates = self.get_field_level_aggregates()

        self.update_file_level_aggregates_on_processed_field_detection(
            field, field_detection, field_level_aggregates
        )
        self.update_directory_level_aggregates_on_processed_field_detection(
            field, field_detection, field_level_aggregates
        )

    def update_field_level_aggregates_on_processed_field_detection(
        self, field, field_detection
    ):
        super().update_field_level_aggregates_on_processed_field(field)

    def update_file_level_aggregates_on_processed_field_detection(
        self, field, field_detection, field_level_aggregates
    ):
        super().update_file_level_aggregates_on_processed_field(
            field, field_level_aggregates
        )
        # detection confidence metrics?

    def update_directory_level_aggregates_on_processed_field_detection(
        self, field: Field, field_detection, field_level_aggregates
    ):
        super().update_directory_level_aggregates_on_processed_field(
            field, field_level_aggregates
        )
        field_detection_type = field.field_detection_type

        field_detection_type_wise_aggregates = self.directory_level_aggregates[
            "field_detection_type_wise_aggregates"
        ][field_detection_type]
        # Update the processed field count for that runner
        field_detection_type_wise_aggregates["fields_count"].push("processed")

    # TODO: check if passing runners is really needed or not here -
    def update_aggregates_on_processed_file(
        self, file_path, field_detection_type_runners
    ):
        super().update_aggregates_on_processed_file(file_path)

        field_detection_type_wise_aggregates = self.file_level_aggregates[
            "field_detection_type_wise_aggregates"
        ]
        for field_detection_type_runner in field_detection_type_runners.values():
            field_detection_type_wise_aggregates[
                field_detection_type_runner.field_detection_type
            ] = field_detection_type_runner.get_file_level_detection_aggregates()

        # TODO: uncomment if needed for directory level graphs
        # self.directory_level_aggregates["field_label_wise_aggregates"][file_path] = self.field_label_wise_aggregates
