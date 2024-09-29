from src.algorithm.template.detection.base.detection import FieldDetection
from src.algorithm.template.detection.ocr.lib.easyocr import EasyOCR
from src.algorithm.template.template_layout import Field, OCRField
from src.utils.shapes import ShapeUtils


class OCRFieldDetection(FieldDetection):
    """
    Here we find the scan zone and perform the detection for the field at runtime.
    """

    def __init__(self, field: Field, gray_image, colored_image):
        super().__init__(field, gray_image, colored_image)

    # Note: run_detection is called from the parent constructor
    def run_detection(self, field: OCRField, gray_image, _colored_image):
        scan_box = field.scan_boxes[0]
        zone_label = scan_box.zone_description["label"]
        scan_zone_rectangle = scan_box.scan_zone_rectangle
        # Can use "channels" of different versions of the image
        # On "ocr" channel we would have different set of contrasts and filters to enhance the handwriting
        image_zone = ShapeUtils.extract_image_from_zone_rectangle(
            gray_image, zone_label, scan_zone_rectangle
        )
        detected_text = EasyOCR.read_single_text(image_zone, confidence_threshold=0.8)

        self.detected_texts = [detected_text]

        return self.detected_texts
