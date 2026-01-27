import cv2

from src.processors.image.base import (
    ImageTemplatePreprocessor,
)


class MedianBlur(ImageTemplatePreprocessor):
    def get_class_name(self) -> str:
        return "MedianBlur"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        options = self.options
        self.kSize = int(options.get("k_size", 5))

    def apply_filter(self, image, _colored_image, _template, _file_path):
        return cv2.medianBlur(image, self.kSize), _colored_image, _template
