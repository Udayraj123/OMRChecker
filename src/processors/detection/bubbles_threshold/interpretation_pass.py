from src.processors.detection.base.interpretation_pass import (
    FieldTypeInterpretationPass,
)
from src.processors.detection.bubbles_threshold.interpretation import (
    BubblesFieldInterpretation,
)
from src.processors.layout.field.base import Field
from src.processors.threshold.global_threshold import GlobalThresholdStrategy
from src.processors.threshold.threshold_result import ThresholdConfig
from src.utils.logger import logger
from src.utils.stats import NumberAggregate


class BubblesThresholdInterpretationPass(FieldTypeInterpretationPass):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # Note: This is used by parent to generate the interpretation: detected string etc
    def get_field_interpretation(
        self,
        field: Field,
        file_level_detection_aggregates,
        file_level_aggregates,
    ) -> BubblesFieldInterpretation:
        tuning_config = self.tuning_config
        return BubblesFieldInterpretation(
            # TODO: [think] on what should be the place for file level thresholds - interpretation vs detection (or middle)
            # ... As file_level_aggregates["field_label_wise_aggregates"] is not filled yet!
            tuning_config,
            field,
            file_level_detection_aggregates,
            file_level_aggregates,
        )

    def initialize_file_level_aggregates(
        self,
        file_path,
        field_detection_type_wise_detection_aggregates,
        field_label_wise_detection_aggregates,
    ) -> None:
        super().initialize_file_level_aggregates(
            file_path,
            field_detection_type_wise_detection_aggregates,
            field_label_wise_detection_aggregates,
        )
        # Note: we also have access to other detectors aggregates if for any conditionally interpretation in future.
        own_file_level_detection_aggregates = (
            field_detection_type_wise_detection_aggregates[self.field_detection_type]
        )
        all_outlier_deviations = own_file_level_detection_aggregates[
            "all_field_bubble_means_std"
        ]
        outlier_deviation_threshold_for_file = self.get_outlier_deviation_threshold(
            all_outlier_deviations
        )

        field_wise_means_and_refs = own_file_level_detection_aggregates[
            "all_field_bubble_means"
        ]
        file_level_fallback_threshold, global_max_jump = self.get_fallback_threshold(
            field_wise_means_and_refs
        )

        logger.debug(
            f"Thresholding: \t file_level_fallback_threshold: {round(file_level_fallback_threshold, 2)} \tglobal_std_THR: {round(outlier_deviation_threshold_for_file, 2)}\t{'(Looks like a Xeroxed OMR)' if (file_level_fallback_threshold == 255) else ''}"
        )
        # TODO: uncomment for more granular threshold detection insights (per bubbleFieldType)
        # bubble_field_type_wise_thresholds = {}
        # for field_detection_aggregates in field_label_wise_detection_aggregates.values():
        #     field = field_detection_aggregates["field"]
        #     bubble_field_type = field.bubble_field_type
        #     field_wise_means_and_refs = field_detection_aggregates["field_bubble_means"].values()
        #
        #     if bubble_field_type not in bubble_field_type_wise_thresholds:
        #         bubble_field_type_wise_thresholds[bubble_field_type] = NumberAggregate()

        #     bubble_field_type_level_fallback_threshold, _ = self.get_fallback_threshold(file_path, field_wise_means_and_refs)
        #     bubble_field_type_wise_thresholds[bubble_field_type].push(bubble_field_type_level_fallback_threshold)

        # TODO: return bubble_field_type_wise(MCQ/INT/CUSTOM_1/CUSTOM_2) thresholds too - for interpretation?
        self.insert_file_level_aggregates(
            {
                "file_level_fallback_threshold": file_level_fallback_threshold,
                "global_max_jump": global_max_jump,
                "outlier_deviation_threshold_for_file": outlier_deviation_threshold_for_file,
                "field_label_wise_local_thresholds": {},
                "bubble_field_type_wise_thresholds": {},
                "all_fields_local_thresholds": NumberAggregate(),
                "field_wise_confidence_metrics": {},
            }
        )

    def get_outlier_deviation_threshold(
        self,
        all_outlier_deviations,
    ):
        config = self.tuning_config
        # ruff: noqa: N806
        MIN_JUMP_STD = config.thresholding.MIN_JUMP_STD
        GLOBAL_PAGE_THRESHOLD_STD = config.thresholding.GLOBAL_PAGE_THRESHOLD_STD

        # Use GlobalThresholdStrategy instead of static method
        strategy = GlobalThresholdStrategy()
        threshold_config = ThresholdConfig(
            min_jump=MIN_JUMP_STD,
            default_threshold=GLOBAL_PAGE_THRESHOLD_STD,
        )

        # all_outlier_deviations is already a list of floats (std deviations)
        result = strategy.calculate_threshold(all_outlier_deviations, threshold_config)
        return result.threshold_value

    def get_fallback_threshold(
        self,
        field_wise_means_and_refs,
    ):
        config = self.tuning_config
        # ruff: noqa: N806
        GLOBAL_PAGE_THRESHOLD = config.thresholding.GLOBAL_PAGE_THRESHOLD
        MIN_JUMP = config.thresholding.MIN_JUMP

        # Use GlobalThresholdStrategy instead of static method
        strategy = GlobalThresholdStrategy()
        threshold_config = ThresholdConfig(
            min_jump=MIN_JUMP,
            default_threshold=GLOBAL_PAGE_THRESHOLD,
        )

        # field_wise_means_and_refs is a list of BubbleMeanValue objects
        bubble_values = [item.mean_value for item in field_wise_means_and_refs]

        result = strategy.calculate_threshold(bubble_values, threshold_config)

        # Approximate global_max_jump from result
        global_max_jump = result.max_jump

        return result.threshold_value, global_max_jump

    def update_field_level_aggregates_on_processed_field_interpretation(
        self, field: Field, field_interpretation: BubblesFieldInterpretation
    ) -> None:
        super().update_field_level_aggregates_on_processed_field_interpretation(
            field, field_interpretation
        )
        # TODO: get this object from field_interpretation through a function?
        self.insert_field_level_aggregates(
            {
                # TODO: move is_multi_marked logic to a parent class(or make copy in parent)
                "is_multi_marked": field_interpretation.is_multi_marked,
                "local_threshold_for_field": field_interpretation.local_threshold_for_field,
                "bubble_interpretations": field_interpretation.bubble_interpretations,
                # Needed for exporting?
                # "field_bubble_means": field_interpretation.field_bubble_means,
            }
        )

    def update_file_level_aggregates_on_processed_field_interpretation(
        self, field, field_interpretation, field_level_aggregates
    ) -> None:
        super().update_file_level_aggregates_on_processed_field_interpretation(
            field, field_interpretation, field_level_aggregates
        )
        self.file_level_aggregates["all_fields_local_thresholds"].push(
            field_interpretation.local_threshold_for_field, field
        )

        # TODO: update bubble_field_type_wise_thresholds
