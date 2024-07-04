from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.processors.internal.CropOnCustomMarkers import CropOnCustomMarkers
from src.processors.internal.CropOnDotLines import CropOnDotLines
from src.utils.interaction import InteractionUtils

class CropOnMarkers(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.options["type"] == "FOUR_MARKERS":
            self.instance = CropOnCustomMarkers(*args, **kwargs)
        else:
            self.instance = CropOnDotLines(*args, **kwargs)

    def exclude_files(self):
        return self.instance.exclude_files()

    def __str__(self):
        return self.instance.__str__()

    def apply_filter(self, *args, **kwargs):
        if self.tuning_config.outputs.show_preview:
            InteractionUtils.show("Before Crop on markers",image)
        image , coloured_image, template = self.instance.apply_filter(*args, **kwargs)
        if self.tuning_config.outputs.show_preview:
            InteractionUtils.show("Crop on markers",image)
        return image, coloured_image, template
        
        
