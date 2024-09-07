from abc import abstractmethod

from src.algorithm.template.template_layout import Field


class FieldDetection:
    def __init__(self, field: Field, gray_image, colored_image):
        self.field = field
        self.gray_image = gray_image
        self.colored_image = colored_image
        # Note: field object can have the corresponding runtime config for the detection
        self.run_detection(field, gray_image, colored_image)

    @abstractmethod
    def run_detection(self):
        raise Exception("Not implemented")
