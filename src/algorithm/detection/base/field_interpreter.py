
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
