import functools


@functools.total_ordering
class MeanValueItem:
    def __init__(self, mean_value, item_reference) -> None:
        self.mean_value = mean_value
        self.item_reference = item_reference
        # self.item_reference_name = item_reference.name

    def __str__(self) -> str:
        return f"{self.item_reference} : {round(self.mean_value, 2)}"

    def validate_and_extract_value(self, other) -> float | int | None:
        if hasattr(other, "mean_value") and hasattr(other, "item_reference"):
            return other.mean_value
        if isinstance(other, (float, int)):
            return other
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        value = self.validate_and_extract_value(other)
        return self.mean_value == value

    def __lt__(self, other) -> bool:
        value = self.validate_and_extract_value(other)
        return self.mean_value < value
