
# TODO: use this or merge it
import numpy as np

from src.algorithm.detection.thresholding import BubbleMeanValue, FieldStdMeanValue


class FieldDetection:
    def __init__(self, field, field_block):
        self.field = field
        self.field_block = field_block

class BubblesField(FieldDetection):
    def __init__(self, field, field_block):
        super().__init__(field, field_block)

        self.field_block = field_block
        self.field_bubble_means = field_bubble_means
            # TODO: move std calculation inside the class
        field_bubble_means_std = round(
            np.std([item.mean_value for item in field_bubble_means]), 2
        )
        self.field_bubble_means_std = FieldStdMeanValue(field_bubble_means_std, field_block)

    def evaluate_detection_params(self, gray_image):
        box_w, box_h = self.field_block.bubble_dimensions
        field_bubbles = self.field.field_bubbles

        field_bubble_means = []
        for unit_bubble in field_bubbles:
            x, y = unit_bubble.get_shifted_position(self.shifts)
            rect = [y, y + box_h, x, x + box_w]
            # TODO: get this from within the BubbleDetection class
            mean_value = cv2.mean(
                gray_image[rect[0] : rect[1], rect[2] : rect[3]]
            )[0]
            field_bubble_means.append(
                BubbleMeanValue(mean_value, unit_bubble)
                # TODO: cross/check mark detection support (#167)
                # detectCross(gray_image, rect) ? 0 : 255
            )
        self.field_bubble_means = field_bubble_means
        return self.field_bubble_means

    def get_field_bubble_means(self):
        return self.field_bubble_means
    def get_field_bubble_means_std(self):
        return self.field_bubble_means_std
class OCRField(FieldDetection):
    pass
class BarcodeField(FieldDetection):
    pass

class FieldDetector:
    def __init__(self, field, field_block, confidence):
        self.field = field
        self.shifts = field_block.shifts
        self.confidence = confidence
        # TODO: use local_threshold from here
        # self.local_threshold = None



# TODO: move the detection utils in this file.
