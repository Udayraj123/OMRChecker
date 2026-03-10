from pathlib import Path


class Processor:
    """Base class that each processor must inherit from."""

    def __init__(
        self,
        options,
        relative_dir,
    ) -> None:
        self.options = options
        self.tuning_options = options.get("tuningOptions", {})
        self.relative_dir = Path(relative_dir)
        self.description = "UNKNOWN"

    def __str__(self) -> str:
        return self.__module__
