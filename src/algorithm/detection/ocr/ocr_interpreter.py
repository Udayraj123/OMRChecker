from src.algorithm.detection.field import FieldInterpreter


class OCRInterpreter(FieldInterpreter):
    def __init__(self, field, field_block):
        super().__init__(field, field_block)

    def get_detected_string(self):
        # TODO: Empty value logic
        # TODO: Concatenation logic
        return self.detected_string

    def get_field_level_confidence_metrics(self):
        return {"OCR_METRICS": "TODO"}
