import functools

import cv2
import numpy as np

from src.algorithm.detection.base.field_type_detector import FieldTypeDetector
from src.algorithm.detection.bubbles_threshold.bubbles_threshold_interpreter import (
    BubblesFieldInterpretation,
)
from src.processors.constants import FieldDetectionType
from src.utils.logger import logger
from src.utils.parsing import default_dump
from src.utils.stats import StatsByLabel


@functools.total_ordering
class MeanValueItem:
    def __init__(self, mean_value, item_reference):
        self.mean_value = mean_value
        self.item_reference = item_reference
        # self.item_reference_name = item_reference.name

    def __str__(self):
        return f"{self.item_reference} : {round(self.mean_value, 2)}"

    def _is_valid_operand(self, other):
        return hasattr(other, "mean_value") and hasattr(other, "item_reference")

    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplementedError
        return self.mean_value == other.mean_value

    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplementedError
        return self.mean_value < other.mean_value


# TODO: merge with FieldInterpretation/BubbleInterpretation?
class BubbleMeanValue(MeanValueItem):
    def __init__(self, mean_value, unit_bubble):
        super().__init__(mean_value, unit_bubble)

    def to_json(self):
        # TODO: mini util for this loop (for export metrics)
        return {
            key: default_dump(getattr(self, key))
            for key in [
                # "is_marked",
                # "shifted_position": unit_bubble.item_reference.get_shifted_position(field_block.shifts),
                # "item_reference_name",
                "mean_value",
            ]
        }

    def __str__(self):
        return f"{self.item_reference} : {round(self.mean_value, 2)} {'*' if self.is_marked else ''}"


class FieldStdMeanValue(MeanValueItem):
    def __init__(self, field_bubble_means, item_reference):
        mean_value = round(np.std([item.mean_value for item in field_bubble_means]), 2)

        super().__init__(mean_value, item_reference)

    def to_json(self):
        return {
            key: default_dump(getattr(self, key))
            for key in [
                # "item_reference_name",
                "mean_value",
            ]
        }


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


# TODO: "lift up" some skeleton of this detector to be used by other detectors
class BubblesThresholdDetector(FieldTypeDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.field_detection_type = FieldDetectionType.BUBBLES_THRESHOLD

    def reset_directory_level_aggregates(self):
        self.directory_level_aggregates = {
            "files_count": StatsByLabel("processed"),
            "file_wise_thresholds": NumberAggregate(),
        }

    def reset_file_level_detection_aggregates(self, file_path):
        self.file_level_detection_aggregates = {
            # "files_count": StatsByLabel("processed"),
            "file_path": file_path,
            "global_max_jump": None,
            "all_field_bubble_means": [],
            "all_field_bubble_means_std": [],
            "field_level_detection_aggregates": {},
        }

    def get_file_level_detection_aggregates(self):
        return self.file_level_detection_aggregates

    def finalize_file_level_detection_aggregates(self):
        return self.file_level_detection_aggregates

    def get_file_level_interpretation_aggregates(self):
        return self.file_level_interpretation_aggregates

    # TODO: move into FieldInterpretation?
    def reinitialize_file_level_interpretation_aggregates(
        self,
        file_path,
        field_detection_type_wise_detection_aggregates,
        field_label_wise_detection_aggregates,
    ):
        # Note: we also have access to other detectors aggregates if for any conditionally interpretation in future.
        own_file_level_detection_aggregates = (
            field_detection_type_wise_detection_aggregates[self.field_detection_type]
        )
        all_outlier_deviations = own_file_level_detection_aggregates[
            "all_field_bubble_means_std"
        ]
        outlier_deviation_threshold_for_file = self.get_outlier_deviation_threshold(
            file_path, all_outlier_deviations
        )

        field_wise_means_and_refs = own_file_level_detection_aggregates[
            "all_field_bubble_means"
        ]
        file_level_fallback_threshold, global_max_jump = self.get_fallback_threshold(
            file_path, field_wise_means_and_refs
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
        self.file_level_interpretation_aggregates = {
            "file_level_fallback_threshold": file_level_fallback_threshold,
            "global_max_jump": global_max_jump,
            "outlier_deviation_threshold_for_file": outlier_deviation_threshold_for_file,
            "field_label_wise_local_thresholds": {},
            "bubble_field_type_wise_thresholds": {},
            "all_fields_local_thresholds": NumberAggregate(),
            "field_wise_confidence_metrics": None,
            "confidence_metrics_for_file": None,
            "fields_count": StatsByLabel("processed"),
            "field_level_interpretation_aggregates": {},
        }

    def get_outlier_deviation_threshold(self, file_path, all_outlier_deviations):
        config = self.tuning_config
        (
            MIN_JUMP_STD,
            JUMP_DELTA_STD,
            GLOBAL_PAGE_THRESHOLD_STD,
        ) = map(
            config.thresholding.get,
            [
                "MIN_JUMP_STD",
                "JUMP_DELTA_STD",
                "GLOBAL_PAGE_THRESHOLD_STD",
            ],
        )
        (
            outlier_deviation_threshold_for_file,
            _,
            _,
        ) = BubblesFieldInterpretation.get_global_threshold(
            all_outlier_deviations,
            GLOBAL_PAGE_THRESHOLD_STD,
            MIN_JUMP=MIN_JUMP_STD,
            JUMP_DELTA=JUMP_DELTA_STD,
            plot_title=f"Field-wise Std-dev Plot for {file_path}",
            plot_show=config.outputs.show_image_level >= 6,
            sort_in_plot=True,
        )
        return outlier_deviation_threshold_for_file

    def get_fallback_threshold(self, file_path, field_wise_means_and_refs):
        config = self.tuning_config
        (
            GLOBAL_PAGE_THRESHOLD,
            MIN_JUMP,
            JUMP_DELTA,
        ) = map(
            config.thresholding.get,
            [
                "GLOBAL_PAGE_THRESHOLD",
                "MIN_JUMP",
                "JUMP_DELTA",
            ],
        )

        # Note: Plotting takes Significant times here --> Change Plotting args
        # to support show_image_level
        (
            file_level_fallback_threshold,
            j_low,
            j_high,
        ) = BubblesFieldInterpretation.get_global_threshold(
            field_wise_means_and_refs,  # , looseness=4
            GLOBAL_PAGE_THRESHOLD,
            plot_title=f"Mean Intensity Barplot: {file_path}",
            MIN_JUMP=MIN_JUMP,
            JUMP_DELTA=JUMP_DELTA,
            plot_show=config.outputs.show_image_level >= 6,
            sort_in_plot=True,
            looseness=4,
        )
        global_max_jump = j_high - j_low

        return file_level_fallback_threshold, global_max_jump

    def run_field_level_detection(self, field, gray_image, _colored_image):
        self.reset_field_level_detection_aggregates()
        field_bubbles = field.field_bubbles

        field_bubble_means = []
        for unit_bubble in field_bubbles:
            # TODO: cross/check mark detection support (#167)
            # detectCross(gray_image, rect) ? 0 : 255
            bubble_mean_value = self.read_bubble_mean_value(unit_bubble, gray_image)

            field_bubble_means.append(bubble_mean_value)

        self.update_detection_aggregates_on_processed_field(field, field_bubble_means)

    def read_bubble_mean_value(self, unit_bubble, gray_image):
        box_w, box_h = unit_bubble.bubble_dimensions
        x, y = unit_bubble.get_shifted_position()
        rect = [y, y + box_h, x, x + box_w]
        mean_value = cv2.mean(gray_image[rect[0] : rect[1], rect[2] : rect[3]])[0]
        bubble_mean_value = BubbleMeanValue(mean_value, unit_bubble)
        return bubble_mean_value

    def reset_field_level_detection_aggregates(self):
        self.field_level_detection_aggregates = {}

    def get_field_level_detection_aggregates(self):
        return self.field_level_detection_aggregates

    def update_detection_aggregates_on_processed_field(self, field, field_bubble_means):
        field_label = field.field_label
        # self.file_level_detection_aggregates["fields_count"].push("processed")

        field_bubble_means_std = FieldStdMeanValue(field_bubble_means, field)

        # update_field_level_detection_aggregates
        self.field_level_detection_aggregates = {
            "field": field,
            "field_bubble_means": field_bubble_means,
            "field_bubble_means_std": field_bubble_means_std,
        }
        self.file_level_detection_aggregates["field_level_detection_aggregates"][
            field_label
        ] = self.field_level_detection_aggregates
        self.file_level_detection_aggregates["all_field_bubble_means"].extend(
            field_bubble_means
        )
        self.file_level_detection_aggregates["all_field_bubble_means_std"].append(
            field_bubble_means_std
        )
        # TODO ...

    def run_field_level_interpretation(self, field, _gray_image, _colored_image):
        tuning_config = self.tuning_config
        self.reset_field_level_interpretation_aggregates()

        # TODO: instantiate this class somewhere more appropriate?
        field_interpretation = BubblesFieldInterpretation(
            # TODO: [think] on what should be the place for file level thresholds - interpretation vs detection (or middle)
            # ... As file_level_interpretation_aggregates["field_level_interpretation_aggregates"] is not filled yet!
            tuning_config,
            field,
            self.file_level_detection_aggregates,
            self.file_level_interpretation_aggregates,
        )

        self.update_field_level_interpretation_aggregates(field, field_interpretation)

        detected_string = field_interpretation.get_detected_string()

        return detected_string

    def reset_field_level_interpretation_aggregates(self):
        self.field_level_interpretation_aggregates = {}

    def get_field_level_interpretation_aggregates(self):
        return self.field_level_interpretation_aggregates

    def finalize_file_level_interpretation_aggregates(self, file_path):
        confidence_metrics_for_file = {}
        # TODO: confidence_metrics_for_file
        self.file_level_interpretation_aggregates[
            "confidence_metrics_for_file"
        ] = confidence_metrics_for_file

        # self.directory_level_aggregates["file_level_interpretation_aggregates"][file_path] = self.file_level_interpretation_aggregates

    def update_field_level_interpretation_aggregates(self, field, field_interpretation):
        # TODO: move this into the same class? or call this function from inside field_interpretation?
        self.field_level_interpretation_aggregates = {
            "field": field,
            "is_multi_marked": field_interpretation.is_multi_marked,
            "confidence_metrics_for_field": field_interpretation.confidence_metrics_for_field,
            "local_threshold_for_field": field_interpretation.local_threshold_for_field,
            "field_bubble_interpretations": field_interpretation.field_bubble_interpretations,
            # Needed for exporting?
            # "field_bubble_means": field_interpretation.field_bubble_means,
        }

        self.file_level_interpretation_aggregates["all_fields_local_thresholds"].push(
            field_interpretation.local_threshold_for_field, field
        )

        # TODO: update field_label_wise_local_thresholds
        # TODO: update bubble_field_type_wise_thresholds

        # self.file_level_interpretation_aggregates[][field_label] = self.field_level_interpretation_aggregates

        # fields count++ for field_detection_type(self) and bubble_field_type
        self.file_level_interpretation_aggregates["fields_count"].push("processed")
        # self.file_level_interpretation_aggregates["fields_count"].push(field.bubble_field_type)
