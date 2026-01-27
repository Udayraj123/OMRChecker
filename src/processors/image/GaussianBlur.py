import cv2

from src.processors.image.base import (
    ImageTemplatePreprocessor,
)


class GaussianBlur(ImageTemplatePreprocessor):
    def get_class_name(self) -> str:
        return "GaussianBlur"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        options = self.options
        self.kSize = tuple(int(x) for x in options.get("k_size", (3, 3)))
        self.sigmaX = int(options.get("sigma_x", 0))

    def apply_filter(self, image, _colored_image, _template, _file_path):
        image = cv2.GaussianBlur(image, self.kSize, self.sigmaX)
        return image, _colored_image, _template
