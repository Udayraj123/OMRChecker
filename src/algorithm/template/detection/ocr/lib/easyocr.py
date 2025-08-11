import operator

from src.algorithm.template.detection.base.detection import TextDetection
from src.algorithm.template.detection.ocr.lib.text_ocr import TextOCR
from src.utils.logger import logger
from src.utils.math import MathUtils


class EasyOCR(TextOCR):
    reader = None

    @staticmethod
    def initialize() -> None:
        # ruff: noqa: PLC0415
        import easyocr

        # this needs to run only once to load the model into memory
        EasyOCR.reader = easyocr.Reader(["en"], gpu=True)

    @staticmethod
    def get_all_text_detections(image, confidence_threshold=0.8, clear_whitespace=True):
        filtered_texts_with_boxes = EasyOCR.read_texts_with_boxes(
            image, confidence_threshold, sort_by_score=True
        )
        return [
            EasyOCR.convert_to_text_detection(box, text, score, clear_whitespace)
            for (box, text, score) in filtered_texts_with_boxes
        ]

    @staticmethod
    def get_single_text_detection(
        image, confidence_threshold=0.8, clear_whitespace=True
    ):
        filtered_texts_with_boxes = EasyOCR.read_texts_with_boxes(
            image, confidence_threshold, sort_by_score=True
        )

        # print("filtered_texts_with_boxes", filtered_texts_with_boxes)
        if len(filtered_texts_with_boxes) == 0:
            return None

        # TODO: support concatenating nearby boxes within a distance param? (concatenateWithinDistance, order: topLeftDiagonal)

        # We will currently rely on easyocr that it is able to cluster nearby texts despite font issues.
        box, text, score = filtered_texts_with_boxes[0]

        if score <= confidence_threshold:
            return None

        return EasyOCR.convert_to_text_detection(box, text, score, clear_whitespace)

    @staticmethod
    def read_texts_with_boxes(image, confidence_threshold=0.8, sort_by_score=True):
        # Lazy load the reader when needed
        if EasyOCR.reader is None:
            logger.info("Initializing EasyOCR reader...")
            EasyOCR.initialize()
        text_results = EasyOCR.reader.readtext(image)
        if len(text_results) == 0:
            return []

        filtered_texts_with_boxes = [
            (box, text, score)
            for (box, text, score) in text_results
            if score >= confidence_threshold
        ]

        if sort_by_score:
            return sorted(
                filtered_texts_with_boxes, key=operator.itemgetter(2), reverse=True
            )
        return filtered_texts_with_boxes

    @staticmethod
    def read_text_for_matching_rule(
        _image, _text_to_find, _confidence_threshold=0.3
    ) -> None:
        return None
        # Firstly postprocess all texts before searching.

        # TODO: add (fuzzy?) search logic or regex match

        # Note: the caller should take care of extracting center vs corners of the box

    @staticmethod
    def convert_to_text_detection(box, text, score, clear_whitespace):
        ordered_box, _ordered_indices = MathUtils.order_four_points(box)

        # Since easyocr doesn't return rotated rectangle, we pass box as the closest value.
        rotated_rect = ordered_box

        # Process to a cv2 printable text
        processed_text = TextOCR.postprocess_text(text, clear_whitespace)

        return TextDetection(processed_text, ordered_box, rotated_rect, score)
