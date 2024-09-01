"""
The field type detector base class finds the scan zone and performs the detection for the field at runtime.
It is static per template instance. Instantiated once per field type in the template.json
"""


class FieldTypeDetector:
    def __init__(self, tuning_config):
        # TODO: use local_threshold from here
        # self.local_threshold = None
        self.tuning_config = tuning_config
        # Expected to be set by the child class
        self.field_detection_type = None

    def get_field_level_detection(self):
        raise Exception("Not implemented")

    def initialize_directory_level_aggregates(self):
        self.directory_level_aggregates = {
            "file_wise_detection_aggregates": {},
            "file_wise_interpretation_aggregates": {},
        }

    def initialize_file_level_detection_aggregates(self, file_path):
        self.file_level_detection_aggregates = {
            # "files_count": StatsByLabel("processed"),
            "file_path": file_path,
            "field_level_detection_aggregates": {},
        }

    def initialize_field_level_detection_aggregates(self):
        self.field_level_detection_aggregates = {}

    def get_field_level_detection_aggregates(self):
        return self.field_level_detection_aggregates

    def get_file_level_detection_aggregates(self):
        return self.file_level_detection_aggregates

    def run_field_level_detection(self, field, gray_image, colored_image):
        self.initialize_field_level_detection_aggregates()

        field_detection = self.generate_field_level_detection(
            field, gray_image, colored_image
        )

        self.update_detection_aggregates_on_processed_field(field, field_detection)

    def run_field_level_interpretation(self, field, _gray_image, _colored_image):
        self.initialize_field_level_interpretation_aggregates()

        field_interpretation = self.generate_field_level_interpretation(
            field,
            self.get_file_level_detection_aggregates(),
            self.get_file_level_interpretation_aggregates(),
        )

        self.update_field_level_interpretation_aggregates(field, field_interpretation)

        detected_string = field_interpretation.get_detected_string()

        return detected_string

    def initialize_field_level_interpretation_aggregates(self):
        self.field_level_interpretation_aggregates = {}

    def get_field_level_interpretation_aggregates(self):
        return self.field_level_interpretation_aggregates

    def get_file_level_interpretation_aggregates(self):
        return self.file_level_interpretation_aggregates
