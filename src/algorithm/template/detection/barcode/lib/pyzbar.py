# Todo: import pyzbar only if class is instantia
from pyzbar.pyzbar import decode as decode_barcode

from src.algorithm.template.detection.barcode.lib.text_barcode import TextBarcode
from src.algorithm.template.detection.base.detection import TextDetection
from src.utils.math import MathUtils


class PyZBar(TextBarcode):
    @staticmethod
    def get_all_text_detections(image, confidence_threshold=0.8, clear_whitespace=True):
        filtered_texts_with_boxes = PyZBar.read_texts_with_boxes(
            image, confidence_threshold
        )
        filtered_detections = [
            PyZBar.convert_to_text_detection(
                box, text, score, polygon, clear_whitespace
            )
            for (box, text, score, polygon) in filtered_texts_with_boxes
        ]
        return filtered_detections

    @staticmethod
    def get_single_text_detection(
        image, confidence_threshold=0.8, clear_whitespace=True
    ):
        filtered_texts_with_boxes = PyZBar.read_texts_with_boxes(
            image, confidence_threshold, sort_by_score=True
        )

        if len(filtered_texts_with_boxes) == 0:
            return None

        box, text, score, polygon = filtered_texts_with_boxes[0]
        if score <= confidence_threshold:
            return None

        return PyZBar.convert_to_text_detection(
            box, text, score, polygon, clear_whitespace
        )

    @staticmethod
    def parse_result_to_standard_form(result):
        rect = result.rect
        bounding_box = MathUtils.get_rectangle_points(
            rect.top, rect.left, rect.width, rect.height
        )
        detected_text = str(result.data)
        score = float(result.quality)
        rotated_rectangle = [[point.x, point.y] for point in result.polygon]

        return (bounding_box, detected_text, score, rotated_rectangle)

    @staticmethod
    def read_texts_with_boxes(image, confidence_threshold=0.8, sort_by_score=True):
        text_results = decode_barcode(image)
        if len(text_results) == 0:
            return []

        filtered_texts_with_boxes = [
            PyZBar.parse_result_to_standard_form(result)
            for result in text_results
            if result.quality >= confidence_threshold
        ]

        if sort_by_score:
            return sorted(filtered_texts_with_boxes, key=lambda x: x[1], reverse=True)
        else:
            return filtered_texts_with_boxes

    @staticmethod
    def convert_to_text_detection(box, text, score, polygon, clear_whitespace):
        ordered_box, _ordered_indices = MathUtils.order_four_points(box)

        rotated_rect = polygon

        # Process to a cv2 printable text
        processed_text = TextBarcode.postprocess_text(text, clear_whitespace)

        return TextDetection(processed_text, ordered_box, rotated_rect, score)
