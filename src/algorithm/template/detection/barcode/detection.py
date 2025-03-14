from typing import List

from dotmap import DotMap

from src.algorithm.template.detection.barcode.lib.pyzbar import PyZBar
from src.algorithm.template.detection.base.detection import (
    FieldDetection,
    TextDetection,
)
from src.algorithm.template.layout.field.barcode_field import BarcodeField
from src.algorithm.template.layout.field.base import Field
from src.utils.math import MathUtils
from src.utils.shapes import ShapeUtils

BARCODE_LIBS = DotMap({"PYZBAR": "PYZBAR"})


class BarcodeDetection(TextDetection):
    def __init__(
        self,
        scan_zone_rectangle,
        detected_text,
        bounding_box,
        rotated_rectangle,
        confident_score,
    ):
        self.library = BARCODE_LIBS.PYZBAR
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
        return BarcodeDetection(
            scan_zone_rectangle,
            text_detection.detected_text,
            absolute_bounding_box,
            absolute_rotated_rectangle,
            text_detection.confident_score,
        )


class BarcodeFieldDetection(FieldDetection):
    """
    Here we find the scan zone and perform the detection for the field at runtime.
    """

    def __init__(self, field: Field, gray_image, colored_image):
        self.detections: List[BarcodeDetection] = None
        super().__init__(field, gray_image, colored_image)

    # Note: run_detection is called from the parent constructor
    def run_detection(self, field: BarcodeField, gray_image, _colored_image):
        scan_box = field.scan_boxes[0]
        zone_label = scan_box.zone_description["label"]
        scan_zone_rectangle = scan_box.scan_zone_rectangle
        image_zone = ShapeUtils.extract_image_from_zone_rectangle(
            gray_image, zone_label, scan_zone_rectangle
        )
        # InteractionUtils.show("image_zone", image_zone)
        # TODO: access field config to determine which lib to use + lib level config
        # if self.library == BARCODE_LIBS.PYZBAR:
        text_detection = PyZBar.get_single_text_detection(
            image_zone, confidence_threshold=0.8
        )

        self.detections = []
        if text_detection is not None:
            self.detections = [
                BarcodeDetection.from_scan_zone_detection(
                    scan_zone_rectangle, text_detection
                )
            ]

        return self.detections
