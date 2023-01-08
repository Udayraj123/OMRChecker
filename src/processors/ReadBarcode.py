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
        self.output_sorting = options.get("output_sorting", False)
        self.input_sorting = options.get("input_sorting", False)

    def apply_filter(self, img, args):
        img1 = img[self.x1 : self.x2, self.y1 : self.y2]
        cv2.imshow("cropped", img1)
        cv2.waitKey(0)

        def detect(image):
            # image = cv2.resize(image, (5000, 5000))
            b = 0
            data = None
            for i, barcode in enumerate(decode(image)):
                data = barcode.data.decode("utf-8")
                b = i + 1
            return data, b

        data, size = detect(img1)
        if size > 1:
            logger.error("\tError: Multiple QR found, size=", size)
            data = ""
        if data is None:
            logger.error(
                "\tError: QR not found :Have you accidentally included ReadBarcode plugin?"
            )
            data = ""
        if self.qr_to_output is not None:
            data_1 = self.qr_to_output[data]
        else:
            data_1 = data
        return str(data_1) + "/", self.input_sorting
