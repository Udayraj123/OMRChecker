import os
import cv2
import numpy as np
from .interfaces.ImagePreprocessor import ImagePreprocessor


class Levels(ImagePreprocessor):
    def __init__(self, options, args):

        def output_level(input, low, high, gamma):
            if input <= low:
                return 0
            elif input >= high:
                return 255
            else:
                invGamma = 1.0 / gamma
                return (((input-low)/(high-low)) ** invGamma) * 255

        self.gamma = np.array(
            [output_level(i, 
                          int(255 * options.get("low", 0)), 
                          int(255 * options.get("high", 1)),
                          options.get("gamma", 1.0)) 
             for i in np.arange(0, 256)]).astype("uint8")

    def apply_filter(self, image, filename):
        return cv2.LUT(image, self.gamma)


class GaussianBlur(ImagePreprocessor):
    def apply_filter(self, image, filename):
        return cv2.GaussianBlur(image, 
                                tuple(self.options.get("kSize", (3, 3))),
                                self.options.get("sigmax", 0))



