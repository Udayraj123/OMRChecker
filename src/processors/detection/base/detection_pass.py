from abc import abstractmethod

from src.processors.detection.base.common_pass import FilePassAggregates
from src.processors.detection.base.detection import FieldDetection
from src.processors.layout.field.base import Field
from src.utils.logger import Logger
from src.utils.stats import StatsByLabel


class FieldTypeDetectionPass(FilePassAggregates):
    """FieldTypeDetectionPass implements detection pass for specific field types, managing the detection-related aggregates.".

    It is responsible for executing detection logic on the image from the field information.
    It does not determine the actual field values, that is left to the interpretation pass
    which can make use of aggregate data collected during the detection pass.
    """

    def __init__(self, tuning_config, field_detection_type) -> None:
        self.field_detection_type = field_detection_type
        super().__init__(tuning_config)

    @abstractmethod
    def get_field_detection(
        self, field: Field, gray_image, colored_image
    ) -> FieldDetection:
        # Not implemented
        raise NotImplementedError

    def update_aggregates_on_processed_field_detection(
        self, field: Field, field_detection: FieldDetection
    ) -> None:
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
        self, field, _field_detection
    ) -> None:
        super().update_field_level_aggregates_on_processed_field(field)

    def update_file_level_aggregates_on_processed_field_detection(
        self, field, _field_detection, field_level_aggregates
    ) -> None:
        super().update_file_level_aggregates_on_processed_field(
            field, field_level_aggregates
        )
        # detection confidence metrics?

    def update_directory_level_aggregates_on_processed_field_detection(
        self, field, _field_detection, field_level_aggregates
    ) -> None:
        super().update_directory_level_aggregates_on_processed_field(
            field, field_level_aggregates
        )
        # (if needed) update_directory_level_aggregates_on_processed_field


class TemplateDetectionPass(FilePassAggregates):
    def initialize_directory_level_aggregates(
        self, initial_directory_path, all_field_detection_types
    ) -> None:
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

    def initialize_file_level_aggregates(
        self, file_path, all_field_detection_types
    ) -> None:
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
    ) -> None:
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
        self, field, _field_detection
    ) -> None:
        super().update_field_level_aggregates_on_processed_field(field)

    def update_file_level_aggregates_on_processed_field_detection(
        self, field, _field_detection, field_level_aggregates
    ) -> None:
        super().update_file_level_aggregates_on_processed_field(
            field, field_level_aggregates
        )
        # detection confidence metrics?

    def update_directory_level_aggregates_on_processed_field_detection(
        self, field: Field, _field_detection, field_level_aggregates
    ) -> None:
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
        self, file_path, field_detection_type_file_runners
    ) -> None:
        super().update_aggregates_on_processed_file(file_path)

        field_detection_type_wise_aggregates = self.file_level_aggregates[
            "field_detection_type_wise_aggregates"
        ]
        for (
            field_detection_type_file_runner
        ) in field_detection_type_file_runners.values():
            field_detection_type_wise_aggregates[
                field_detection_type_file_runner.field_detection_type
            ] = field_detection_type_file_runner.get_file_level_detection_aggregates()

        # Populate typed field results from repository
        # Get repository from any field type runner (they all share the same repository)
        any_runner = next(iter(field_detection_type_file_runners.values()), None)
        if any_runner and hasattr(any_runner, "repository") and any_runner.repository:
            try:
                file_results = any_runner.repository.get_file_results(file_path)

                # Map all field types by field_label for interpretation access
                bubble_fields_by_label = {
                    result.field_label: result
                    for result in file_results.bubble_fields.values()
                }
                ocr_fields_by_label = {
                    result.field_label: result
                    for result in file_results.ocr_fields.values()
                }
                barcode_fields_by_label = {
                    result.field_label: result
                    for result in file_results.barcode_fields.values()
                }

                self.file_level_aggregates["bubble_fields"] = bubble_fields_by_label
                self.file_level_aggregates["ocr_fields"] = ocr_fields_by_label
                self.file_level_aggregates["barcode_fields"] = barcode_fields_by_label
            except KeyError as e:
                # File not yet finalized in repository - this should not happen
                # but if it does, log and re-raise to surface the issue
                logger = Logger(__name__)
                logger.error(
                    f"File {file_path} not found in repository after finalize_file(). "
                    f"This indicates a bug in the repository lifecycle management. Error: {e}"
                )
                # Re-raise to surface the issue during development
                raise
            except Exception as e:
                # Other unexpected errors
                logger = Logger(__name__)
                logger.error(
                    f"Unexpected error populating field results from repository for {file_path}: {e}"
                )
                # Initialize empty dicts to prevent KeyError in interpretation
                self.file_level_aggregates.setdefault("bubble_fields", {})
                self.file_level_aggregates.setdefault("ocr_fields", {})
                self.file_level_aggregates.setdefault("barcode_fields", {})
