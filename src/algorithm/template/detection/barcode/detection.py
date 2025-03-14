from src.algorithm.template.detection.barcode.lib.pyzbar import PyZBar
from src.algorithm.template.detection.base.detection import FieldDetection

# TODO: add BarcodeField
from src.algorithm.template.layout.field.base import BarcodeField, Field
from src.utils.shapes import ShapeUtils


class BarcodeFieldDetection(FieldDetection):
    """
    Here we find the scan zone and perform the detection for the field at runtime.
    """

    def __init__(self, field: Field, gray_image, colored_image):
        super().__init__(field, gray_image, colored_image)

    # Note: run_detection is called from the parent constructor
    def run_detection(self, field: BarcodeField, gray_image, _colored_image):
        scan_box = field.scan_boxes[0]
        zone_label = scan_box.zone_description["label"]
        scan_zone_rectangle = scan_box.scan_zone_rectangle
        # Can use "channels" of different versions of the image
        # On "Barcode" channel we would have different set of contrasts and filters to enhance the handwriting
        image_zone = ShapeUtils.extract_image_from_zone_rectangle(
            gray_image, zone_label, scan_zone_rectangle
        )
        text_detection = PyZBar.get_single_text_detection(
            image_zone, confidence_threshold=0.8
        )

        self.detections = [text_detection]

        return self.detections
