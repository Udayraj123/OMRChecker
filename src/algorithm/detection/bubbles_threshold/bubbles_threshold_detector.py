import functools
import math
import random
import re

import cv2
import numpy as np
from matplotlib import colormaps, pyplot

from src.algorithm.detection.base.field_interpreter import FieldInterpretation
from src.utils.logger import logger
from src.utils.parsing import default_dump
from src.processors.constants import FieldDetectionType
from src.algorithm.template.template_detector import StatsByLabel
from src.algorithm.detection.base.field_type_detector import FieldTypeDetector


@functools.total_ordering
class MeanValueItem:
    def __init__(self, mean_value, item_reference):
        self.mean_value = mean_value
        self.item_reference = item_reference
        self.item_reference_name = item_reference.name

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


# TODO: merge with FieldInterpreter?
class BubbleMeanValue(MeanValueItem):
    def __init__(self, mean_value, unit_bubble):
        super().__init__(mean_value, unit_bubble)

    def to_json(self):
        # TODO: mini util for this loop
        return {
            key: default_dump(getattr(self, key))
            for key in [
                # "is_marked",
                # "shifted_position": unit_bubble.item_reference.get_shifted_position(field_block.shifts),
                "item_reference_name",
                "mean_value",
            ]
        }

    def __str__(self):
        return f"{self.item_reference} : {round(self.mean_value, 2)} {'*' if self.is_marked else ''}"


# TODO: see if this one can be merged in above
class FieldStdMeanValue(MeanValueItem):
    def __init__(self, mean_value, item_reference):
        super().__init__(mean_value, item_reference)

    def to_json(self):
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "item_reference_name",
                "mean_value",
            ]
        }


class NumberAggregate:
    def __init__(self):
        self.collection = []
        self.running_sum = 0
        self.running_average = 0
    
    def push(self, number, label):
        self.collection.push([number, label])
        self.running_sum += number
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

class BubblesThresholdDetector(FieldTypeDetector):
    def __init__(self):
        self.detection_type = FieldDetectionType.BUBBLES_THRESHOLD
        super().__init__()
        # TODO: use local_threshold from here
        # self.local_threshold = None

    def reset_directory_level_aggregates(self):
        self.directory_level_aggregates = {
            "files_count": StatsByLabel("processed"),
            "file_wise_thresholds": NumberAggregate(),
        }

    def reinitialize_file_level_detection_aggregates(self):
        self.field_wise_means_and_refs, self.all_outlier_deviations = {}, {}

    def generate_file_level_detection_aggregates(self, file_path):

    # def get_file_level_detection_aggregates(self):
        return {
            "file_path": file_path,
            "field_wise_means_and_refs": self.field_wise_means_and_refs, 
            "all_outlier_deviations": self.all_outlier_deviations,
        }
    
    # TODO: move into FieldInterpretation?
    def reinitialize_file_level_interpretation_aggregates(self, file_path, all_file_level_detection_aggregates):
        config = self.config
        # Note: we also have access to other detectors aggregates if for any conditionally interpretation in future.
        own_detection_aggregates = all_file_level_detection_aggregates["field_detection_type_wise_aggregates"][self.detection_type]
        field_wise_means_and_refs, all_outlier_deviations = own_detection_aggregates["field_wise_means_and_refs"].values(), own_detection_aggregates["all_outlier_deviations"].values()
        
        # Calculate thresholds here
        (
            GLOBAL_PAGE_THRESHOLD,
            MIN_JUMP,
            JUMP_DELTA,
            MIN_JUMP_STD,
            JUMP_DELTA_STD,
            GLOBAL_PAGE_THRESHOLD_STD,
        ) = map(
            config.thresholding.get,
            [
                "GLOBAL_PAGE_THRESHOLD",
                "MIN_JUMP",
                "JUMP_DELTA",
                "MIN_JUMP_STD",
                "JUMP_DELTA_STD",
                "GLOBAL_PAGE_THRESHOLD_STD",
            ],
        )
        (
            all_fields_outlier_deviation_threshold_for_file,
            _,
            _,
        ) = self.get_global_threshold(
            all_outlier_deviations,
            GLOBAL_PAGE_THRESHOLD_STD,
            MIN_JUMP=MIN_JUMP_STD,
            JUMP_DELTA=JUMP_DELTA_STD,
            plot_title="Q-wise Std-dev Plot",
            plot_show=config.outputs.show_image_level >= 6,
            sort_in_plot=True,
        )

        # Note: Plotting takes Significant times here --> Change Plotting args
        # to support show_image_level
        all_fields_threshold_for_file, j_low, j_high = self.get_global_threshold(
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

        logger.debug(
            f"Thresholding: \t all_fields_threshold_for_file: {round(all_fields_threshold_for_file, 2)} \tglobal_std_THR: {round(all_fields_outlier_deviation_threshold_for_file, 2)}\t{'(Looks like a Xeroxed OMR)' if (all_fields_threshold_for_file == 255) else ''}"
        )
    
        # TODO: return field_type(MCQ/INT/CUSTOM_1/CUSTOM_2) wise thresholds too - for interpretation?

        self.file_level_interpretation_aggregates = {
            # 
            "all_fields_threshold_for_file": all_fields_threshold_for_file,
            "all_fields_outlier_deviation_threshold_for_file": all_fields_outlier_deviation_threshold_for_file,
            "global_max_jump": global_max_jump,

            "field_wise_local_thresholds": NumberAggregate(),
            "field_wise_confidence_metrics": None,
            "fields_count": StatsByLabel("processed"),
            "field_type_wise_thresholds": NumberAggregate(),
            "file_level_threshold": own_detection_aggregates["all_fields_threshold_for_file"],
        }
        # This is the metric we use as the fallback in old code
        # self.file_level_interpretation_aggregates["field_wise_local_thresholds"].running_average

    def generate_file_level_interpretation_aggregates(self):
        
        # TODO: produce metrics here.

        return {
            # "field_wise_local_thresholds"
            # "field_wise_confidence_metrics"
            # "file_level_threshold"
            # "field_type_wise_aggregates"
        }

    def run_field_level_detection(self, field, gray_image, _colored_image):
        field_bubbles = field.field_bubbles

        field_bubble_means = []
        for unit_bubble in field_bubbles:
            # TODO: cross/check mark detection support (#167)
            # detectCross(gray_image, rect) ? 0 : 255
            bubble_mean_value = self.read_bubble_mean_value(unit_bubble, gray_image)
            
            field_bubble_means.append(
                bubble_mean_value
            )

        self.update_all_detection_aggregates(field, field_bubble_means)
    
    def read_bubble_mean_value(self, unit_bubble, gray_image):
        box_w, box_h = unit_bubble.bubble_dimensions 
        x, y = unit_bubble.get_shifted_position()
        rect = [y, y + box_h, x, x + box_w]
        mean_value = cv2.mean(gray_image[rect[0] : rect[1], rect[2] : rect[3]])[0]
        bubble_mean_value = BubbleMeanValue(mean_value, unit_bubble)
        return bubble_mean_value

    def update_all_detection_aggregates(self, field, field_bubble_means):
        field_label = field.field_label
        
        # update_field_level_detection_aggregates

        field_bubble_means_std = round(
            np.std([item.mean_value for item in field_bubble_means]), 2
        )

        field_bubble_means_std = FieldStdMeanValue(
            field_bubble_means_std, field
        )

        self.all_outlier_deviations[field_label] = field_bubble_means_std
        self.field_wise_means_and_refs[field_label] = field_bubble_means

    def run_field_level_interpretation(self, field, gray_image, _colored_image):
        field_bubbles = field.field_bubbles

        field_bubble_means = []
        # Use aggregates-
        # for unit_bubble in field_bubbles:
            

        self.update_all_interpretation_aggregates(field, field_bubble_means)
    
    def update_all_interpretation_aggregates(self, field, field_bubble_means):
        # fields count++
        pass
    
    def reset_field_type_level_aggregates(self):
        # The field_type is about MCQ vs INT etc including user defined types
        # The field_detection_type is about BUBBLES vs OCR etc
        
        field_type_aggregates = {
            "field_type_wise_thresholds": NumberAggregate(),
        }
        file_level_interpretation_aggregates["files_count"] = StatsByLabel("processed")

        # "local_thresholds": {
        #     "processed_fields_count": 0,
        #     # TODO: both directory and file level
        #     "field_wise": {},
        #     # TODO: both directory and file level
        #     "field_type_wise": {},
        #     # FieldType -> { "running_average", "processed_fields_count"}
        # },
        "confidence_metrics_for_file": {},
        "all_fields_threshold_for_file": all_fields_threshold_for_file,
        "all_fields_outlier_deviation_threshold_for_file": all_fields_outlier_deviation_threshold_for_file,
    # }

