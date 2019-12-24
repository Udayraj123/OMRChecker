import os
import cv2
import numpy as np
from extension import ImagePreprocessor


class Levels(ImagePreprocessor):
    def __init__(self, options, args):

        def output_level(input, low, high, gamma):
            invGamma = 1.0 / gamma
            if input <= low:
                return 0
            elif input >= high:
                return 255
            else:
                return (((input-low)/(high-low)) ** invGamma) * (high-low)

        self.gamma = np.array(
            [output_level(i, 
                          int(255 * options.get("Low", 0)), 
                          int(255 * options.get("High", 1)),
                          options.get("Gamma", 1.0)) 
             for i in np.arange(0, 256)]).astype("uint8")

    def apply_filter(self, image, filename):
        return cv2.LUT(image, self.gamma)


class GaussianBlur(ImagePreprocessor):
    def __init__(self, options, args):
        self.options = options

    def apply_filter(self, image, filename):
        return cv2.GaussianBlur(image, 
                                tuple(self.options.get("kSize", (3, 3))),
                                self.options.get("sigmaX", 0))
