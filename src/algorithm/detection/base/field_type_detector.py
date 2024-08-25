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

    # TODO: update the skeleton as per latest info
    def setup_directory_metrics():
        raise Exception(f"Not implemented")

    def read_field():
        raise Exception(f"Not implemented")

    # def read_field_second_pass():
    #     raise Exception(f"Not implemented")

    def reset_field_type_aggregates():
        raise Exception(f"Not implemented")

    def update_field_level_aggregates():
        raise Exception(f"Not implemented")

    def get_field_interpretation():
        # Returns an instance of FieldInterpretation
        raise Exception(f"Not implemented")
