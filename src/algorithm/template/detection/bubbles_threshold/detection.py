from typing import List

import cv2
import numpy as np

from src.algorithm.template.detection.base.detection import FieldDetection
from src.algorithm.template.detection.bubbles_threshold.stats import MeanValueItem
from src.algorithm.template.template_layout import Field, FieldBubble
from src.utils.parsing import default_dump


# TODO: merge with FieldInterpretation/BubbleInterpretation?
class BubbleMeanValue(MeanValueItem):
    def __init__(self, mean_value, unit_bubble: FieldBubble):
        super().__init__(mean_value, unit_bubble)

    def to_json(self):
        # TODO: mini util for this loop (for export metrics)
        return {
            key: default_dump(getattr(self, key))
            for key in [
                # "is_marked",
                # "shifted_position": unit_bubble.item_reference.get_shifted_position(field_block.shifts),
                # "item_reference_name",
                "mean_value",
            ]
        }

    def __str__(self):
        return f"{self.item_reference} : {round(self.mean_value, 2)} {'*' if self.is_marked else ''}"


class BubblesFieldDetection(FieldDetection):

    """
    Here we find the scan zone and perform the detection for the field at runtime.
    """

    def run_detection(self, field, gray_image, _colored_image):
        field_bubbles = field.field_bubbles
        self.field_bubble_means = []
        for unit_bubble in field_bubbles:
            # TODO: cross/check mark detection support (#167)
            # detectCross(gray_image, rect) ? 0 : 255
            bubble_mean_value = self.read_bubble_mean_value(unit_bubble, gray_image)
            self.field_bubble_means.append(bubble_mean_value)

    @staticmethod
    def read_bubble_mean_value(unit_bubble: FieldBubble, gray_image):
        box_w, box_h = unit_bubble.bubble_dimensions
        x, y = unit_bubble.get_shifted_position()
        rect = [y, y + box_h, x, x + box_w]
        mean_value = cv2.mean(gray_image[rect[0] : rect[1], rect[2] : rect[3]])[0]
        bubble_mean_value = BubbleMeanValue(mean_value, unit_bubble)
        return bubble_mean_value


class FieldStdMeanValue(MeanValueItem):
    def __init__(
        self, field_bubble_means: List[BubbleMeanValue], item_reference: Field
    ):
        mean_value = np.std([item.mean_value for item in field_bubble_means])

        super().__init__(mean_value, item_reference)

    def to_json(self):
        return {
            key: default_dump(getattr(self, key))
            for key in [
                # "item_reference_name",
                "mean_value",
            ]
        }
