from dotmap import DotMap

from src.algorithm.template.detection.base.detection import (
    FieldDetection,
    TextDetection,
)
from src.algorithm.template.detection.ocr.lib.easyocr import EasyOCR
from src.algorithm.template.layout.field.base import Field
from src.algorithm.template.layout.field.ocr_field import OCRField
from src.utils.math import MathUtils
from src.utils.shapes import ShapeUtils

OCR_LIBS = DotMap({"EASY_OCR": "EASY_OCR"})


class OCRDetection(TextDetection):
    def __init__(
        self,
        scan_zone_rectangle,
        detected_text,
        bounding_box,
        rotated_rectangle,
        confident_score,
    ) -> None:
        self.library = OCR_LIBS.EASY_OCR
        self.scan_zone_rectangle = scan_zone_rectangle
        super().__init__(
            detected_text, bounding_box, rotated_rectangle, confident_score
        )

    @staticmethod
    def from_scan_zone_detection(scan_zone_rectangle, text_detection: TextDetection):
        zone_start = scan_zone_rectangle[0]
        absolute_bounding_box = MathUtils.shift_points_from_origin(
            zone_start, text_detection.bounding_box
        )

        absolute_rotated_rectangle = MathUtils.shift_points_from_origin(
            zone_start, text_detection.rotated_rectangle
        )
        return OCRDetection(
            scan_zone_rectangle,
            text_detection.detected_text,
            absolute_bounding_box,
            absolute_rotated_rectangle,
            text_detection.confident_score,
        )


class OCRFieldDetection(FieldDetection):
    """Here we find the scan zone and perform the detection for the field at runtime."""

    def __init__(self, field: Field, gray_image, colored_image) -> None:
        self.detections: list[OCRDetection] = None
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

        # TODO: access field config to determine which lib to use + lib level config
        text_detection = EasyOCR.get_single_text_detection(
            image_zone, confidence_threshold=0.8
        )

        self.detections = []
        if text_detection is not None:
            self.detections = [
                OCRDetection.from_scan_zone_detection(
                    scan_zone_rectangle, text_detection
                )
            ]

        return self.detections
