import re
import string
import easyocr


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
            cleaned_text = re.sub("\s{2,}", " ", printable_text)

        if charset is not None:
            cleaned_text = TextOCR.filter_text(cleaned_text, charset)

        # Clip the given text to max length
        if max_length is not None and len(cleaned_text) > max_length:
            cleaned_text = cleaned_text[:max_length]

        return cleaned_text


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
        
        # TODO: add search logic 

        # Note: the caller should take care of extracting center vs corners of the box
        return found_box



# TODO support pytesseract methods as well.
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
    