import cv2
import numpy as np

from .interfaces.ImagePreprocessor import ImagePreprocessor
from pyzbar.pyzbar import decode
from src.logger import logger
import os

class Levels(ImagePreprocessor):
    def __init__(self, options, _args):
        def output_level(value, low, high, gamma):
            if value <= low:
                return 0
            if value >= high:
                return 255
            inv_gamma = 1.0 / gamma
            return (((value - low) / (high - low)) ** inv_gamma) * 255
            
        self.gamma = np.array(
            [
                output_level(
                    i,
                    int(255 * options.get("low", 0)),
                    int(255 * options.get("high", 1)),
                    options.get("gamma", 1.0),
                )
                for i in np.arange(0, 256)
            ]
        ).astype("uint8")

    def apply_filter(self, image, _args):
        return cv2.LUT(image, self.gamma)


class MedianBlur(ImagePreprocessor):
    def __init__(self, options, _args):
        self.kSize = options.get("kSize", 5)

    def apply_filter(self, image, _args):
        return cv2.medianBlur(
                            image,
                            self.kSize)


class GaussianBlur(ImagePreprocessor):
    def apply_filter(self, image, _args):
        return cv2.GaussianBlur(
            image,
            tuple(self.options.get("kSize", (3, 3))),
            self.options.get("sigmax", 0),
        )


class ReadBarcode(ImagePreprocessor):

    def __init__(self, options, _args):
        self.x1 = options.get("x1")
        self.x2 = options.get("x2")
        self.y1 = options.get("y1")
        self.y2 = options.get("y2")

    def apply_filter(self,img,args,save_dir):
        img1=img[self.x1 : self.x2 ,self.y1 : self.y2]
        cv2.imshow("cropped", img1)

        def detect(image):
            # image = cv2.resize(image, (5000, 5000))
            for barcode in decode(image):
                data = barcode.data.decode('utf-8')
                return data

        def make_folders(path, data):
            b=0
            for file in os.listdir(path):
                if (file == str(data)):
                    break
            else:
                os.mkdir(path+'/'+str(data))

        data=detect(img1)
        if data== None:
            data='error'
        path=str(save_dir[:-1])
        logger.info(path)
        make_folders(path,data)
        return str(data) +'/'
