from src.algorithm.template.layout.field.base import Field
from src.utils.stats import StatsByLabel


class FilePassAggregates:
    """Interface defining methods for collecting and managing data during file processing at various levels (field, file, directory)."""

    def __init__(self, tuning_config) -> None:
        self.tuning_config = tuning_config

    def initialize_directory_level_aggregates(self, initial_directory_path) -> None:
        self.directory_level_aggregates = {
            "initial_directory_path": initial_directory_path,
            "file_wise_aggregates": {},
            "files_count": StatsByLabel("processed"),
        }

    def get_directory_level_aggregates(self):
        return self.directory_level_aggregates

    def insert_directory_level_aggregates(
        self, next_directory_level_aggregates
    ) -> None:
        self.directory_level_aggregates = {
            **self.directory_level_aggregates,
            **next_directory_level_aggregates,
        }

    def initialize_file_level_aggregates(self, file_path) -> None:
        self.file_level_aggregates = {
            "file_path": file_path,
            "fields_count": StatsByLabel("processed"),
            "field_label_wise_aggregates": {},
        }

    def get_file_level_aggregates(self):
        return self.file_level_aggregates

    def insert_file_level_aggregates(self, next_file_level_aggregates) -> None:
        self.file_level_aggregates = {
            **self.file_level_aggregates,
            **next_file_level_aggregates,
        }

    def update_aggregates_on_processed_file(self, file_path) -> None:
        # Nested access
        self.directory_level_aggregates["file_wise_aggregates"][file_path] = (
            self.file_level_aggregates
        )
        self.directory_level_aggregates["files_count"].push("processed")

    def initialize_field_level_aggregates(self, field: Field) -> None:
        self.field_level_aggregates = {
            "field": field,
        }

    def get_field_level_aggregates(self):
        return self.field_level_aggregates

    def insert_field_level_aggregates(self, next_field_level_aggregates) -> None:
        self.field_level_aggregates = {
            **self.field_level_aggregates,
            **next_field_level_aggregates,
        }

    # To be called by the child classes as per consumer needs
    def update_field_level_aggregates_on_processed_field(self, field: Field) -> None:
        pass

    # To be called by the child classes as per consumer needs
    def update_file_level_aggregates_on_processed_field(
        self, field: Field, field_level_aggregates
    ) -> None:
        field_label = field.field_label
        # TODO: convert into field_id_wise_aggregates
        self.file_level_aggregates["field_label_wise_aggregates"][field_label] = (
            field_level_aggregates
        )

        self.file_level_aggregates["fields_count"].push("processed")

    def update_directory_level_aggregates_on_processed_field(
        self, field: Field, field_level_aggregates
    ) -> None:
        pass
