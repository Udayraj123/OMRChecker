
from src.algorithm.detection.base.field_type_detector import FieldTypeDetector


class BarcodeQRDetector(FieldTypeDetector):
    def __init__(self):
        super().__init__()

    def read_field(self, field, gray_image, colored_image, file_aggregate_params):
        self.detected_string = "TODO_Barcode"  # pyzbar
