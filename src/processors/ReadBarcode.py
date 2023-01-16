import cv2
from pyzbar.pyzbar import decode

from src.logger import logger

from .interfaces.ImagePreprocessor import ImagePreprocessor


class ReadBarcode(ImagePreprocessor):
    def __init__(self, options, _args):
        self.top_left = options.get("top_left")
        self.top_right = options.get("top_right")
        self.qr_to_output = options.get("qr_to_output_directory")
        self.input_sorting = options.get("input_sorting", False)
        self.y1 = self.top_left[1]
        self.y2 = self.top_right[1]
        self.x1 = self.top_left[0]
        self.x2 = self.top_right[1]


    def detect(image):
        size = 0
        data = None
        for i, barcode in enumerate(decode(image)):
            data = barcode.data.decode("utf-8")
            size = i + 1
        if size > 1:
            logger.error("\tError: Multiple QR found, size=", size)
            data = None
        elif data is None:
            logger.error(
                "\tError: QR not found :Have you accidentally included ReadBarcode plugin?"
            )
        return data

    def apply_filter(self, img, args):
        img1 = img[self.y1 : self.y2, self.x1 : self.x2]

        data = ReadBarcode.detect(img1)
        if self.qr_to_output !={}:
            data_1 = self.qr_to_output[data]
        else:
            data_1 = data
        return str(data_1) + "/", self.input_sorting
