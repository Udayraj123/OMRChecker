# TODO support pytesseract methods as well.
from src.algorithm.template.detection.ocr.lib.textocr import TextOCR


class TesseractOCR(TextOCR):
    @staticmethod
    def read_texts(image, confidence_threshold=0.8):
        # TODO: pytesseract.image_to_string
        raise Exception(f"To be implemented")

    @staticmethod
    def read_single_text(image, confidence_threshold=0.8, clear_whitespace=True):
        raise Exception(f"To be implemented")

    @staticmethod
    def read_texts_with_boxes(image, confidence_threshold=0.8, sort_by_score=True):
        raise Exception(f"To be implemented")
