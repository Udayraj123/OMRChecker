from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.processors.internal.CropOnCustomMarkers import CropOnCustomMarkers
from src.processors.internal.CropOnDotLines import CropOnDotLines


class CropOnMarkers(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.options["type"] == "FOUR_MARKERS":
            self.instance = CropOnCustomMarkers(*args, **kwargs)
        else:
            # TODO: convex hull method for the sparse blobs
            self.instance = CropOnDotLines(*args, **kwargs)

    def exclude_files(self):
        return self.instance.exclude_files()

    def __str__(self) -> str:
        return self.instance.__str__()

    def get_class_name(self) -> str:
        return "CropOnMarkers"

    def apply_filter(self, *args, **kwargs):
        image, coloured_image, template = self.instance.apply_filter(*args, **kwargs)
        return image, coloured_image, template
