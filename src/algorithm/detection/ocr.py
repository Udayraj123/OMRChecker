import easyocr


class OCR:
    # this needs to run only once to load the model into memory
    reader = easyocr.Reader(["en"])

    @staticmethod
    def cleanup_text(text):
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()

    @staticmethod
    def read_texts(image, confidence_threshold=0.8):
        all_texts = OCR.reader.readtext(image)
        if len(all_texts) == 0:
            return []
        filtered_texts = [
            text for (box, text, score) in all_texts if score >= confidence_threshold
        ]
        return filtered_texts

    @staticmethod
    def read_single_text(image, confidence_threshold=0.8):
        all_texts = OCR.reader.readtext(image)
        if len(all_texts) == 0:
            return ""
        sorted_texts = sorted(all_texts, key=lambda s: s[2], reverse=True)
        text, score = sorted_texts[0]
        if score <= confidence_threshold:
            return ""
        return text
