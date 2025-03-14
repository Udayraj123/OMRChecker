# Todo: import pyzbar only if class is instantia
from pyzbar.pyzbar import decode as decode_barcode

from src.algorithm.template.detection.barcode.lib.text_barcode import TextBarcode


class PyZBar(TextBarcode):
    @staticmethod
    def get_all_text_detections(image, confidence_threshold=0.8):
        filtered_texts_with_boxes = PyZBar.read_texts_with_boxes(
            image, confidence_threshold
        )
        filtered_texts = [
            text for (text, score, box, polygon) in filtered_texts_with_boxes
        ]
        return filtered_texts

    @staticmethod
    def get_single_text_detection(
        image, confidence_threshold=0.8, clear_whitespace=True
    ):
        filtered_texts_with_boxes = PyZBar.read_texts_with_boxes(
            image, confidence_threshold, sort_by_score=True
        )

        if len(filtered_texts_with_boxes) == 0:
            return ""

        text, score, _box, _polygon = filtered_texts_with_boxes[0]
        if score <= confidence_threshold:
            return ""

        processed_text = TextBarcode.postprocess_text(text, clear_whitespace)
        return processed_text

    @staticmethod
    def read_texts_with_boxes(image, confidence_threshold=0.8, sort_by_score=True):
        text_results = decode_barcode(image)
        if len(text_results) == 0:
            return []

        filtered_texts_with_boxes = [
            (result.data, result.quality, result.rect, result.polygon)
            for result in text_results
            if result.quality >= confidence_threshold
        ]

        if sort_by_score:
            return sorted(filtered_texts_with_boxes, key=lambda x: x[1], reverse=True)
        else:
            return filtered_texts_with_boxes
