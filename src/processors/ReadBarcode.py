import cv2
from pyzbar.pyzbar import decode

from src.logger import logger

from .interfaces.ImagePreprocessor import ImagePreprocessor


class ReadBarcode(ImagePreprocessor):
    def __init__(self, options, _args):
        self.x1 = options.get("x1")
        self.x2 = options.get("x2")
        self.y1 = options.get("y1")
        self.y2 = options.get("y2")
        self.qr_to_output = options.get("qr_to_output_directory")
        self.input_sorting = options.get("input_sorting", False)


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
            data="error"
        return data

    def apply_filter(self, img, args):
        img1 = img[self.x1 : self.x2, self.y1 : self.y2]

        data = ReadBarcode.detect(img1)
        if self.qr_to_output is not None:
            data_1 = self.qr_to_output[data]
        else:
            data_1 = data
        return str(data_1) + "/", self.input_sorting
