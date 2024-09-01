import re
import string


class TextOCR:
    digits_set = set(string.digits)
    letters_set = set(string.ascii_letters)
    lowercase_letters_set = set(string.ascii_lowercase)
    uppercase_letters_set = set(string.ascii_uppercase)
    alphanumeric_set = set(string.ascii_letters + string.digits)
    url_symbols_set = set("@$-_.+!*'(),")

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
        printable_text = TextOCR.cleanup_text(stripped_text)

        if clear_whitespace:
            cleaned_text = re.sub("\\s{2,}", " ", printable_text)

        if charset is not None:
            cleaned_text = TextOCR.filter_text(cleaned_text, charset)

        # Clip the given text to max length
        if max_length is not None and len(cleaned_text) > max_length:
            cleaned_text = cleaned_text[:max_length]

        return cleaned_text
