from src.algorithm.template.detection.bubbles_threshold.detection import BubbleMeanValue


class BubbleInterpretation:
    def __init__(self, field_bubble_mean: BubbleMeanValue, local_threshold) -> None:
        self.is_attempted = None
        # self.field_bubble_mean = field_bubble_mean
        self.mean_value = field_bubble_mean.mean_value
        self.bubble_value = field_bubble_mean.item_reference.bubble_value
        self.local_threshold = local_threshold
        # TODO: decouple this -  needed for drawing (else not needed here?)
        self.item_reference = field_bubble_mean.item_reference
        # self.unit_bubble = field_bubble_mean.item_reference

        self.update_interpretation(local_threshold)

    def update_interpretation(self, local_threshold):
        is_attempted = local_threshold > self.mean_value
        self.is_attempted = is_attempted
        return is_attempted

    def __str__(self) -> str:
        return f"{self.item_reference} : {round(self.mean_value, 2)} {'*' if self.is_attempted else ''}"
