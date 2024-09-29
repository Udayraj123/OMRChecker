import json

from src.utils.parsing import default_dump


class StatsByLabel:
    def __init__(self, *labels):
        self.label_counts = {label: 0 for label in labels}

    def push(self, label, number=1):
        if label not in self.label_counts:
            raise Exception(
                f"Unknown label passed to stats by label: {label}, allowed labels: {self.label_counts.keys()}"
            )
            # self.label_counts[label] = []

        self.label_counts[label] += number

    def to_json(self):
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "label_counts",
            ]
        }

    def __str__(self):
        return json.dumps(self.to_json())


class NumberAggregate:
    def __init__(self):
        self.collection = []
        self.running_sum = 0
        self.running_average = 0

    def push(self, number_like, label):
        self.collection.append([number_like, label])
        # if isinstance(number_like, MeanValueItem):
        #     self.running_sum += number_like.mean_value
        # else:
        self.running_sum += number_like
        self.running_average = self.running_sum / len(self.collection)

    def merge(self, other_aggregate):
        self.collection += other_aggregate.collection
        self.running_sum += other_aggregate.running_sum
        self.running_average = self.running_sum / len(self.collection)

    def to_json(self):
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "collection",
                "running_sum",
                "running_average",
            ]
        }
