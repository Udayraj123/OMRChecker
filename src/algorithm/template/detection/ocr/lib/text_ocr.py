import re
import string
from typing import ClassVar, Never


class TextOCR:
    digits_set: ClassVar = set(string.digits)
    letters_set: ClassVar = set(string.ascii_letters)
    lowercase_letters_set: ClassVar = set(string.ascii_lowercase)
    uppercase_letters_set: ClassVar = set(string.ascii_uppercase)
    alphanumeric_set: ClassVar = set(string.ascii_letters + string.digits)
    url_symbols_set: ClassVar = set("@$-_.+!*'(),")

    @staticmethod
    def get_all_text_detections(_image, _confidence_threshold=0.8) -> Never:
        msg = "Not implemented"
        raise Exception(msg)

    @staticmethod
    def get_single_text_detection(
        _image, _confidence_threshold=0.8, _clear_whitespace=True
    ) -> Never:
        msg = "Not implemented"
        raise Exception(msg)

    @staticmethod
    def read_texts_with_boxes(
        _image, _confidence_threshold=0.8, _sort_by_score=True
    ) -> Never:
        msg = "Not implemented"
        raise Exception(msg)

    @staticmethod
    def cleanup_text(text: str) -> str:
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV
        return "".join([c for c in text if ord(c) < 128])

    @staticmethod
    def filter_text(text, charset) -> str:
        return "".join([c for c in text if c in charset])

    @staticmethod
    def postprocess_text(
        text, clear_whitespace=False, max_length=None, charset=None
    ) -> str:
        stripped_text = text.strip()
        printable_text = TextOCR.cleanup_text(stripped_text)

        if clear_whitespace:
            cleaned_text = re.sub("\\s{2,}", " ", printable_text)

        if charset is not None:
            cleaned_text = TextOCR.filter_text(cleaned_text, charset)

        # Clip the given text to max length
        if max_length is not None and len(cleaned_text) > max_length:
            cleaned_text = cleaned_text[:max_length]

        return cleaned_text
