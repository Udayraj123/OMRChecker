from abc import abstractmethod
from typing import Never

from src.algorithm.template.layout.field.base import Field


class TextDetection:
    # Note: This is a base class for all text detections
    # It is used to store the detected text, bounding box, rotated rectangle and confident score
    # The boxes are relative to the image that was used.
    def __init__(
        self, detected_text, bounding_box, rotated_rectangle, confident_score
    ) -> None:
        self.detected_text = detected_text
        self.bounding_box = bounding_box
        self.rotated_rectangle = rotated_rectangle
        self.confident_score = confident_score

    def is_null(self):
        return self.detected_text is None


class FieldDetection:
    def __init__(self, field: Field, gray_image, colored_image) -> None:
        self.field = field
        self.gray_image = gray_image
        self.colored_image = colored_image
        # Note: field object can have the corresponding runtime config for the detection
        self.run_detection(field, gray_image, colored_image)

    @abstractmethod
    def run_detection(self) -> Never:
        msg = "Not implemented"
        raise Exception(msg)
