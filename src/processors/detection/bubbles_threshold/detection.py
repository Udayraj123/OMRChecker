"""Bubble field detection using new typed models.

Refactored to use BubbleFieldDetectionResult instead of nested dictionaries.
"""

import cv2

from src.processors.detection.base.detection import FieldDetection
from src.processors.detection.models.detection_results import (
    BubbleFieldDetectionResult,
    BubbleMeanValue,
)
from src.processors.layout.field.base import Field
from src.processors.layout.field.bubble_field import BubblesScanBox


class BubblesFieldDetection(FieldDetection):
    """Detects bubble values and returns strongly-typed result.

    Replaces dictionary-based aggregates with BubbleFieldDetectionResult.
    """

    def run_detection(self, field: Field, gray_image, _colored_image) -> None:
        """Run detection and create typed result.

        Args:
            field: Field to detect
            gray_image: Grayscale image
            _colored_image: Colored image (unused for bubble detection)
        """
        bubble_means = []

        for unit_bubble in field.scan_boxes:
            # TODO: cross/check mark detection support (#167)
            bubble_mean_value = self.read_bubble_mean_value(unit_bubble, gray_image)
            bubble_means.append(bubble_mean_value)

        # Create strongly-typed result
        # Properties like std_deviation and scan_quality are auto-calculated
        self.result = BubbleFieldDetectionResult(
            field_id=field.id,
            field_label=field.field_label,
            bubble_means=bubble_means,
        )

    @staticmethod
    def read_bubble_mean_value(
        unit_bubble: BubblesScanBox, gray_image
    ) -> BubbleMeanValue:
        """Read mean intensity value for a single bubble.

        Args:
            unit_bubble: Bubble scan box
            gray_image: Grayscale image

        Returns:
            BubbleMeanValue with mean intensity
        """
        box_w, box_h = unit_bubble.bubble_dimensions
        x, y = unit_bubble.get_shifted_position()
        rect = [y, y + box_h, x, x + box_w]
        mean_value = cv2.mean(gray_image[rect[0] : rect[1], rect[2] : rect[3]], None)[0]
        return BubbleMeanValue(
            mean_value=mean_value, unit_bubble=unit_bubble, position=(x, y)
        )
