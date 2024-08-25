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
