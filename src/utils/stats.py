import json
from typing import Any

from src.utils.parsing import default_dump


class StatsByLabel:
    def __init__(self, *labels) -> None:
        self.label_counts = dict.fromkeys(labels, 0)

    def push(self, label, number=1) -> None:
        if label not in self.label_counts:
            msg = f"Unknown label passed to stats by label: {label}, allowed labels: {self.label_counts.keys()}"
            raise Exception(msg)

        self.label_counts[label] += number

    def to_json(self) -> dict[str, Any]:
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "label_counts",
            ]
        }

    def __str__(self) -> str:
        return json.dumps(self.to_json())


class NumberAggregate:
    def __init__(self) -> None:
        self.collection = []
        self.running_sum = 0
        self.running_average = 0

    def push(self, number_like, label) -> None:
        self.collection.append([number_like, label])
        self.running_sum += number_like
        self.running_average = self.running_sum / len(self.collection)

    def merge(self, other_aggregate) -> None:
        self.collection += other_aggregate.collection
        self.running_sum += other_aggregate.running_sum
        self.running_average = self.running_sum / len(self.collection)

    def to_json(self) -> dict[str, Any]:
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "collection",
                "running_sum",
                "running_average",
            ]
        }
