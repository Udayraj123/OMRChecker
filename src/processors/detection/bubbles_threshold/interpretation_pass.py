from src.processors.detection.base.interpretation_pass import (
    FieldTypeInterpretationPass,
)
from src.processors.detection.bubbles_threshold.interpretation import (
    BubblesFieldInterpretation,
)
from src.processors.layout.field.base import Field
from src.processors.detection.detection_repository import DetectionRepository
from src.processors.detection.threshold.global_threshold import GlobalThresholdStrategy
from src.processors.detection.threshold.threshold_result import ThresholdConfig
from src.utils.logger import logger
from src.utils.stats import NumberAggregate


class BubblesThresholdInterpretationPass(FieldTypeInterpretationPass):
    def __init__(self, *args, repository: DetectionRepository, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.repository = repository

    # Note: This is used by parent to generate the interpretation: detected string etc
    def get_field_interpretation(
        self,
        field: Field,
        file_level_detection_aggregates,
        file_level_aggregates,
    ) -> BubblesFieldInterpretation:
        tuning_config = self.tuning_config
        return BubblesFieldInterpretation(
            tuning_config,
            field,
            file_level_detection_aggregates,
            file_level_aggregates,
        )

    def initialize_file_level_aggregates(
        self,
        file_path,
    ) -> None:
        super().initialize_file_level_aggregates(file_path)
        # Get bubble means from repository
        all_bubble_means = self.repository.get_all_bubble_means_for_current_file()

        # Calculate std deviations from repository results
        all_outlier_deviations = []
        for (
            field_result
        ) in self.repository.get_all_bubble_fields_for_current_file().values():
            all_outlier_deviations.append(field_result.std_deviation)

        field_wise_means_and_refs = all_bubble_means

        outlier_deviation_threshold_for_file = self.get_outlier_deviation_threshold(
            all_outlier_deviations
        )

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
    ) -> float:
        """Calculate outlier deviation threshold using threshold strategy.

        Args:
            all_outlier_deviations: List of standard deviations from all fields

        Returns:
            Calculated threshold value
        """
        config = self.tuning_config.thresholding
        threshold_config = ThresholdConfig(
            min_jump=config.min_jump_std,
            default_threshold=config.global_page_threshold_std,
        )

        strategy = GlobalThresholdStrategy()
        result = strategy.calculate_threshold(all_outlier_deviations, threshold_config)
        return result.threshold_value

    def get_fallback_threshold(
        self,
        field_wise_means_and_refs,
    ) -> tuple[float, float]:
        """Calculate fallback threshold using threshold strategy.

        Args:
            field_wise_means_and_refs: List of BubbleMeanValue objects from all fields

        Returns:
            Tuple of (fallback_threshold, global_max_jump)
        """
        config = self.tuning_config.thresholding
        threshold_config = ThresholdConfig(
            min_jump=config.min_jump,
            default_threshold=config.global_page_threshold,
        )

        # field_wise_means_and_refs is a list of BubbleMeanValue objects
        bubble_values = [item.mean_value for item in field_wise_means_and_refs]

        strategy = GlobalThresholdStrategy()
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
        file_level_aggregates = self.get_file_level_aggregates()
        file_level_aggregates["all_fields_local_thresholds"].push(
            field_interpretation.local_threshold_for_field, field
        )

        # TODO: update bubble_field_type_wise_thresholds
