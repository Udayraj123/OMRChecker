from src.algorithm.template.detection.base.detection import FieldDetection


class OCRFieldDetection(FieldDetection):

    """
    Here we find the scan zone and perform the detection for the field at runtime.
    """

    def run_detection(self, field, gray_image, _colored_image):
        # field_bubbles = field.field_bubbles
        # TODO: connect ocr and field config
        self.detections = []
