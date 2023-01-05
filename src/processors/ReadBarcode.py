import cv2
import numpy as np

from .interfaces.ImagePreprocessor import ImagePreprocessor
from pyzbar.pyzbar import decode
from src.logger import logger
import os


class ReadBarcode(ImagePreprocessor):

    def __init__(self, options, _args):
        self.x1 = options.get("x1")
        self.x2 = options.get("x2")
        self.y1 = options.get("y1")
        self.y2 = options.get("y2")
        self.qr_to_output=options.get("qr_to_output_directory")
        self.output_sorting=options.get("output_sorting",False)
        logger.info(self.qr_to_output)

    def apply_filter(self,img,args,save_dir):
        img1=img[self.x1 : self.x2 ,self.y1 : self.y2]
        cv2.imshow("cropped", img1)

        def detect(image):
            # image = cv2.resize(image, (5000, 5000))
            for barcode in decode(image):
                data = barcode.data.decode('utf-8')
                return data

        def make_folders(path, data):
            for file in os.listdir(path):
                if (file == str(data)):
                    break
            else:
                os.mkdir(path+'/'+str(data))

        data=detect(img1)
        if data== None:
            data='error'
        path=str(save_dir[:-1])
        # logger.info(self.qr_to_output[data])
        if self.qr_to_output is not None:
            data_1=self.qr_to_output[data]
        else:
            data_1=data
        make_folders(path,data_1)
        return str(data_1) +'/',self.output_sorting
