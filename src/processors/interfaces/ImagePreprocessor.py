# Use all imports relative to root directory
from src.processors.manager import Processor


class ImagePreprocessor(Processor):
    """Base class for an extension that applies some preprocessing to the input image"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_filter(self, image, filename):
        """Apply filter to the image and returns modified image"""
        raise NotImplementedError

    @staticmethod
    def exclude_files():
        """Returns a list of file paths that should be excluded from processing"""
        return []
