from pathlib import Path

from src.processors.image.base import (
    ImageTemplatePreprocessor,
)
from src.processors.image.crop_on_patches import (
    CropOnCustomMarkers,
    CropOnDotLines,
    CropOnLMarkers,
)
from src.utils.exceptions import TemplateConfigurationError

_DOT_LINE_TYPES = {
    "ONE_LINE_TWO_DOTS",
    "TWO_DOTS_ONE_LINE",
    "TWO_LINES",
    "TWO_LINES_HORIZONTAL",
    "FOUR_DOTS",
}


class CropOnMarkers(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        type_str = self.options["type"]
        if type_str == "FOUR_MARKERS":
            self.instance = CropOnCustomMarkers(*args, **kwargs)
        elif type_str == "L_MARKERS":
            self.instance = CropOnLMarkers(*args, **kwargs)
        elif type_str in _DOT_LINE_TYPES:
            # TODO: convex hull method for the sparse blobs
            self.instance = CropOnDotLines(*args, **kwargs)
        else:
            msg = f"Unknown CropOnMarkers type: '{type_str}'. Valid types: FOUR_MARKERS, L_MARKERS, {', '.join(sorted(_DOT_LINE_TYPES))}"
            raise TemplateConfigurationError(msg, type_str=type_str)

    def exclude_files(self) -> list[Path]:
        return self.instance.exclude_files()

    def __str__(self) -> str:
        return self.instance.__str__()

    def get_class_name(self) -> str:
        return "CropOnMarkers"

    def apply_filter(self, *args, **kwargs):
        image, coloured_image, template = self.instance.apply_filter(*args, **kwargs)
        return image, coloured_image, template
