import functools
from typing import Any, Generic, TypeVar

ItemReferenceT = TypeVar("ItemReferenceT", bound=Any)


@functools.total_ordering
class MeanValueItem(Generic[ItemReferenceT]):
    def __init__(self, mean_value: float, item_reference: ItemReferenceT) -> None:
        self.mean_value = mean_value
        self.item_reference = item_reference
        # self.item_reference_name = item_reference.name

    def __str__(self) -> str:
        return f"{self.item_reference} : {round(self.mean_value, 2)}"

    def __hash__(self) -> int:
        return super().__hash__()

    def validate_and_extract_value(self, other) -> float | int:
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
