from src.algorithm.detection.base.field_interpretation import FieldInterpretation


class BarcodeInterpretation(FieldInterpretation):
    def __init__(self, field, field_block):
        super().__init__(field, field_block)

    def get_detected_string(self):
        return self.detected_string

    def get_field_level_confidence_metrics(self):
        return {"Barcode": "TODO"}
