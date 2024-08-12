"""
The field type detector base class finds the scan zone and performs the detection for the field at runtime.
It is static per template instance. Instantiated once per field type in the template.json
"""


class FieldTypeDetector:
    def __init__(self):
        # TODO: use local_threshold from here
        # self.local_threshold = None
        pass

    def setup_directory_metrics():
        raise Exception(f"Not implemented")

    def read_field():
        raise Exception(f"Not implemented")

    # def read_field_second_pass():
    #     raise Exception(f"Not implemented")

    def reset_field_type_aggregates():
        raise Exception(f"Not implemented")

    def get_field_interpretation():
        # Returns an instance of FieldInterpreter
        raise Exception(f"Not implemented")


"""
The field detection base class stores all metrics on for the scan zone of the field
Instantiated for every field for each image at runtime.
"""


class FieldInterpreter:
    def __init__(self, field, field_block):
        self.field = field
        self.field_block = field_block
        self.confidence_metrics = {}

    def get_detected_string():
        raise Exception(f"Not implemented")

    def get_field_level_confidence_metrics():
        raise Exception(f"Not implemented")

    def update_field_level_aggregates():
        raise Exception(f"Not implemented")
