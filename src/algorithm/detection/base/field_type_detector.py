"""
The field type detector base class finds the scan zone and performs the detection for the field at runtime.
It is static per template instance. Instantiated once per field type in the template.json
"""


class FieldTypeDetector:
    def __init__(self, config):
        # TODO: use local_threshold from here
        # self.local_threshold = None
        self.config = config

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
        # Returns an instance of FieldInterpreter
        raise Exception(f"Not implemented")
