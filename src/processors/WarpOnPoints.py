from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.processors.internal.CropOnCustomMarkers import CropOnCustomMarkers
from src.processors.internal.CropOnDotLines import CropOnDotLines


# TODO: temp file, replace with CropOnMarkers and other wrappers
class WarpOnPoints(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.options["type"] == "CUSTOM_MARKER":
            self.instance = CropOnCustomMarkers(*args, **kwargs)
        else:
            self.instance = CropOnDotLines(*args, **kwargs)

    def exclude_files(self):
        return self.instance.exclude_files()

    def __str__(self):
        return self.instance.__str__()

    def apply_filter(self, *args, **kwargs):
        return self.instance.apply_filter(*args, **kwargs)
