"""Bubble detection pass using new typed models and repository.

Refactored to use DetectionRepository instead of nested dictionaries.
Much cleaner and shorter!
"""

from src.processors.detection.base.detection_pass import FieldTypeDetectionPass
from src.processors.detection.bubbles_threshold.detection import (
    BubblesFieldDetection,
)
from src.processors.layout.field.base import Field
from src.processors.repositories.detection_repository import (
    DetectionRepository,
)
from src.utils.logger import Logger
from src.utils.stats import NumberAggregate


class BubblesThresholdDetectionPass(FieldTypeDetectionPass):
    """Detection pass for bubble fields using repository pattern.

    Stores results in DetectionRepository with typed models.
    """

    def __init__(self, *args, repository: DetectionRepository, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.repository = repository

    def get_field_detection(
        self, field: Field, gray_image, colored_image
    ) -> BubblesFieldDetection:
        """Create field detection (called by parent)."""
        return BubblesFieldDetection(field, gray_image, colored_image)

    def initialize_directory_level_aggregates(self, initial_directory_path) -> None:
        """Initialize directory-level aggregates."""
        super().initialize_directory_level_aggregates(initial_directory_path)
        self.insert_directory_level_aggregates(
            {
                "file_wise_thresholds": NumberAggregate(),
            }
        )

        # Note: Repository is initialized by TemplateFileRunner before calling this method

    def initialize_file_level_aggregates(self, file_path) -> None:
        """Initialize file-level aggregates."""
        super().initialize_file_level_aggregates(file_path)
        self.insert_file_level_aggregates(
            {
                "global_max_jump": None,
                "all_field_bubble_means": [],
                "all_field_bubble_means_std": [],
            }
        )

        # Note: Repository is initialized by TemplateFileRunner before calling this method

    def update_field_level_aggregates_on_processed_field_detection(
        self, field: Field, field_detection: BubblesFieldDetection
    ) -> None:
        """Update field-level aggregates after detection."""
        super().update_field_level_aggregates_on_processed_field_detection(
            field, field_detection
        )

        # Save to repository
        if field_detection.result is None:
            logger = Logger(__name__)
            logger.error(
                f"field_detection.result is None for field {field.id}. "
                f"Detection may have failed or result was not created."
            )
            raise ValueError(f"field_detection.result is None for field {field.id}")
        self.repository.save_bubble_field(field.id, field_detection.result)

        # Use result for aggregates
        field_bubble_means = field_detection.result.bubble_means
        std_deviation = field_detection.result.std_deviation

        self.insert_field_level_aggregates(
            {
                "field_bubble_means": field_bubble_means,
                "field_bubble_means_std": std_deviation,
            }
        )

    def update_file_level_aggregates_on_processed_field_detection(
        self,
        field: Field,
        field_detection: BubblesFieldDetection,
        field_level_aggregates,
    ) -> None:
        """Update file-level aggregates after field detection."""
        # Skip base class update_file_level_aggregates_on_processed_field
        # which would populate field_label_wise_aggregates. Just update fields_count.
        file_level_aggregates = self.get_file_level_aggregates()
        file_level_aggregates["fields_count"].push("processed")

        # Use result from field detection
        field_bubble_means = field_level_aggregates["field_bubble_means"]
        field_bubble_means_std = field_level_aggregates["field_bubble_means_std"]

        file_level_aggregates["all_field_bubble_means"].extend(field_bubble_means)
        file_level_aggregates["all_field_bubble_means_std"].append(
            field_bubble_means_std
        )
