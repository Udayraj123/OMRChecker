import cv2

from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.interaction import InteractionUtils

class GaussianBlur(ImageTemplatePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        options = self.options
        self.kSize = tuple(int(x) for x in options.get("kSize", (3, 3)))
        self.sigmaX = int(options.get("sigmaX", 0))

    def apply_filter(self, image, _colored_image, _template, _file_path):
        if self.tuning_config.outputs.show_preview:
            InteractionUtils.show("Before GaussianBlur",image)
        image = cv2.GaussianBlur(image, self.kSize, self.sigmaX)
        if self.tuning_config.outputs.show_preview:
            InteractionUtils.show("GaussianBlur", image)
        return image,_colored_image, _template
