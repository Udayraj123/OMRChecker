import re


class TextBarcode:
    @staticmethod
    def get_all_text_detections(image, confidence_threshold=0.8):
        raise Exception(f"Not implemented")

    @staticmethod
    def get_single_text_detection(
        image, confidence_threshold=0.8, clear_whitespace=True
    ):
        raise Exception(f"Not implemented")

    @staticmethod
    def read_texts_with_boxes(image, confidence_threshold=0.8, sort_by_score=True):
        raise Exception(f"Not implemented")

    @staticmethod
    def cleanup_text(text):
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV
        return "".join([c for c in text if ord(c) < 128])

    @staticmethod
    def filter_text(text, charset):
        return "".join([c for c in text if c in charset])

    @staticmethod
    def postprocess_text(text, clear_whitespace=False, max_length=None, charset=None):
        stripped_text = text.strip()
        printable_text = TextBarcode.cleanup_text(stripped_text)

        if clear_whitespace:
            cleaned_text = re.sub("\\s{2,}", " ", printable_text)

        if charset is not None:
            cleaned_text = TextBarcode.filter_text(cleaned_text, charset)

        # Clip the given text to max length
        if max_length is not None and len(cleaned_text) > max_length:
            cleaned_text = cleaned_text[:max_length]

        return cleaned_text
