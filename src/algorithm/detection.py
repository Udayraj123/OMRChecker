"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

import functools

from src.utils.parsing import default_dump


@functools.total_ordering
class MeanValueItem:
    def __init__(self, mean_value, item_reference):
        self.mean_value = mean_value
        self.item_reference = item_reference
        self.item_reference_name = item_reference.name

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
    def __init__(self, mean_value, unit_bubble):
        super().__init__(mean_value, unit_bubble)
        # TODO: move this into FieldDetection/give reference to it
        self.is_marked = None

    def to_json(self):
        # TODO: mini util for this loop
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "is_marked",
                # "shifted_position": unit_bubble.item_reference.get_shifted_position(field_block.shifts),
                "item_reference_name",
                "mean_value",
            ]
        }

    def __str__(self):
        return f"{self.item_reference} : {round(self.mean_value, 2)} {'*' if self.is_marked else ''}"


# TODO: see if this one can be merged in above
class FieldStdMeanValue(MeanValueItem):
    def __init__(self, mean_value, item_reference):
        super().__init__(mean_value, item_reference)

    def to_json(self):
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "item_reference_name",
                "mean_value",
            ]
        }


# TODO: use this or merge it
class FieldDetection:
    def __init__(self, field, confidence):
        self.field = field
        self.confidence = confidence
        # TODO: use local_threshold from here
        # self.local_threshold = None


# TODO: move the detection utils in this file.
