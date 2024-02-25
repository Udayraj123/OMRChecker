import functools


@functools.total_ordering
class MeanValueItem:
    def __init__(self, mean_value, item_reference):
        self.mean_value = mean_value
        self.item_reference = item_reference

    def __str__(self):
        return f"{self.item_reference} : {round(self.mean_value, 2)}"

    def _is_valid_operand(self, other):
        return hasattr(other, "mean_value") and hasattr(other, "item_reference")

    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplementedError
        return self.mean_value == other.mean_value

    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplementedError
        return self.mean_value < other.mean_value


# TODO: merge with FieldDetection
class BubbleMeanValue(MeanValueItem):
    def __init__(self, mean_value, item_reference):
        super().__init__(mean_value, item_reference)
        self.is_marked = None
        self.local_threshold = None


# TODO: see if this one can be merged in above
class FieldStdMeanValue(MeanValueItem):
    def __init__(self, mean_value, item_reference):
        super().__init__(mean_value, item_reference)


class FieldDetection:
    def __init__(self, field, confidence):
        self.field = field
        self.confidence = confidence
