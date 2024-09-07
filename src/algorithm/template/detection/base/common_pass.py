from src.algorithm.template.template_layout import Field
from src.utils.stats import StatsByLabel


class FilePassAggregates:
    def __init__(self, tuning_config, field_detection_type):
        self.tuning_config = tuning_config
        self.field_detection_type = field_detection_type

    def initialize_directory_level_aggregates(self):
        self.directory_level_aggregates = {
            "file_wise_aggregates": {},
            "files_count": StatsByLabel("processed"),
            # "file_wise_detection_aggregates": {},
            # "file_wise_interpretation_aggregates": {},
        }

    def get_directory_level_aggregates(self):
        return self.directory_level_aggregates

    def insert_directory_level_aggregates(self, next_directory_level_aggregates):
        self.directory_level_aggregates = {
            **self.directory_level_aggregates,
            **next_directory_level_aggregates,
        }

    def initialize_file_level_aggregates(self, file_path):
        self.file_level_aggregates = {
            "file_path": file_path,
            "fields_count": StatsByLabel("processed"),
            "field_level_aggregates": {},
        }

    def get_file_level_aggregates(self):
        return self.file_level_aggregates

    def insert_file_level_aggregates(self, next_file_level_aggregates):
        self.file_level_aggregates = {
            **self.file_level_aggregates,
            **next_file_level_aggregates,
        }

    def update_aggregates_on_processed_file(self, file_path):
        self.directory_level_aggregates["file_wise_aggregates"][
            file_path
        ] = self.file_level_aggregates
        self.directory_level_aggregates["files_count"].push("processed")

    def initialize_field_level_aggregates(self, field: Field):
        self.field_level_aggregates = {
            "field": field,
        }

    def get_field_level_aggregates(self):
        return self.field_level_aggregates

    def insert_field_level_aggregates(self, next_field_level_aggregates):
        self.field_level_aggregates = {
            **self.field_level_aggregates,
            **next_field_level_aggregates,
        }

    # Extensible by the class
    def update_field_level_aggregates_on_processed_field(self, field: Field):
        pass

    # Extensible by the class
    def update_file_level_aggregates_on_processed_field(
        self, field: Field, field_level_aggregates
    ):
        field_label = field.field_label
        self.file_level_aggregates["field_level_aggregates"][
            field_label
        ] = field_level_aggregates
        # fields count++ for field_detection_type(self)
        self.file_level_aggregates["fields_count"].push("processed")
