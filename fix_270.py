# path/to/existing/file.py

# Imports
import cv2
import numpy as np
from src.constants.image_processing import *
from src.processors.CropOnMarkers import CropOnMarkers
from src.processors.CropPage import CropPage

# Adding new methods to existing classes
class CropOnMarkers:
    def __init__(self, template):
        self.template = template

    def get_anchor_points(self, image):
        # Implement your logic here to get anchor points
        pass

    def crop_on_markers(self, image):
        anchor_points = self.get_anchor_points(image)
        # Implement your logic here to crop on markers
        pass

class CropPage:
    def __init__(self, template):
        self.template = template

    def crop_page(self, image):
        # Implement your logic here to crop page
        pass

# Test cases
def test_crop_on_markers():
    template = {
        "markers": [
            {"name": "marker1", "coordinates": [10, 20]},
            {"name": "marker2", "coordinates": [30, 40]},
        ]
    }
    image = cv2.imread("path/to/image.png")
    crop_on_markers = CropOnMarkers(template)
    cropped_image = crop_on_markers.crop_on_markers(image)
    assert cropped_image.shape == (100, 100, 3)  # replace with expected shape

def test_crop_page():
    template = {
        "page": {"coordinates": [50, 60, 70, 80]}
    }
    image = cv2.imread("path/to/image.png")
    crop_page = CropPage(template)
    cropped_image = crop_page.crop_page(image)
    assert cropped_image.shape == (100, 100, 3)  # replace with expected shape