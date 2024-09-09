from src.algorithm.template.detection.base.detection_pass import FieldTypeDetectionPass
from src.algorithm.template.detection.bubbles_threshold.detection import (
    BubblesFieldDetection,
    FieldStdMeanValue,
)
from src.algorithm.template.template_layout import Field
from src.utils.stats import NumberAggregate


class BubblesThresholdDetectionPass(FieldTypeDetectionPass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Note: This is used by parent to generate the detection
    def get_field_detection(
        self, field: Field, gray_image, colored_image
    ) -> BubblesFieldDetection:
        return BubblesFieldDetection(field, gray_image, colored_image)

    def initialize_directory_level_aggregates(self, initial_directory_path):
        super().initialize_directory_level_aggregates(initial_directory_path)
        self.insert_directory_level_aggregates(
            {
                "file_wise_thresholds": NumberAggregate(),
            }
        )

    def initialize_file_level_aggregates(self, file_path):
        super().initialize_file_level_aggregates(file_path)
        self.insert_file_level_aggregates(
            {
                "global_max_jump": None,
                "all_field_bubble_means": [],
                "all_field_bubble_means_std": [],
            }
        )

    def update_field_level_aggregates_on_processed_field_detection(
        self, field: Field, field_detection: BubblesFieldDetection
    ):
        super().update_field_level_aggregates_on_processed_field_detection(
            field, field_detection
        )
        field_bubble_means = field_detection.field_bubble_means
        # self.file_level_aggregates["fields_count"].push("processed")

        field_bubble_means_std = FieldStdMeanValue(field_bubble_means, field)

        self.insert_field_level_aggregates(
            {
                # Note: "field" key is injected from parent
                "field_bubble_means": field_bubble_means,
                "field_bubble_means_std": field_bubble_means_std,
            }
        )

    def update_file_level_aggregates_on_processed_field_detection(
        self,
        field: Field,
        field_detection: BubblesFieldDetection,
        field_level_aggregates,
    ):
        super().update_file_level_aggregates_on_processed_field_detection(
            field, field_detection, field_level_aggregates
        )

        field_bubble_means = field_level_aggregates["field_bubble_means"]
        field_bubble_means_std = field_level_aggregates["field_bubble_means_std"]

        self.file_level_aggregates["all_field_bubble_means"].extend(field_bubble_means)
        self.file_level_aggregates["all_field_bubble_means_std"].append(
            field_bubble_means_std
        )
        # fields count++ for field_detection_type(self) and bubble_field_type
        # self.file_level_aggregates["fields_count"].push(field.bubble_field_type)

        # TODO ...
