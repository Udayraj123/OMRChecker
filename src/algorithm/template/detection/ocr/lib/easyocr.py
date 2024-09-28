import easyocr

from src.algorithm.template.detection.ocr.lib.textocr import TextOCR


class EasyOCR(TextOCR):
    # this needs to run only once to load the model into memory
    reader = easyocr.Reader(["en"])

    @staticmethod
    def read_texts(image, confidence_threshold=0.8):
        filtered_texts_with_boxes = EasyOCR.read_texts_with_boxes(
            image, confidence_threshold
        )
        filtered_texts = [text for (box, text, score) in filtered_texts_with_boxes]
        return filtered_texts

    @staticmethod
    def read_single_text(image, confidence_threshold=0.8, clear_whitespace=True):
        filtered_texts_with_boxes = EasyOCR.read_texts_with_boxes(
            image, confidence_threshold, sort_by_score=True
        )

        print("filtered_texts_with_boxes", filtered_texts_with_boxes)
        if len(filtered_texts_with_boxes) == 0:
            return ""

        # TODO: support concatenating nearby boxes within a distance param?
        # We will currently rely on easyocr that it is able to cluster nearby texts despite font issues.
        _box, text, score = filtered_texts_with_boxes[0]
        if score <= confidence_threshold:
            return ""

        processed_text = TextOCR.postprocess_text(text, clear_whitespace)
        return processed_text

    @staticmethod
    def read_texts_with_boxes(image, confidence_threshold=0.8, sort_by_score=True):
        all_texts = EasyOCR.reader.readtext(image)
        if len(all_texts) == 0:
            return []

        filtered_texts_with_boxes = [
            (box, text, score)
            for (box, text, score) in all_texts
            if score >= confidence_threshold
        ]

        if sort_by_score:
            return sorted(filtered_texts_with_boxes, key=lambda x: x[2], reverse=True)
        else:
            return filtered_texts_with_boxes

    @staticmethod
    def find_text(image, text_to_find, confidence_threshold=0.3):
        found_box = None
        # Firstly postprocess all texts before searching.

        # TODO: add (fuzzy?) search logic

        # Note: the caller should take care of extracting center vs corners of the box
        return found_box
