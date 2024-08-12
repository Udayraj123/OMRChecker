import functools
import math
import random
import re

import cv2
import numpy as np
from matplotlib import colormaps, pyplot

from src.algorithm.detection.field import FieldTypeDetector
from src.processors.constants import FieldDetectionType
from src.utils.parsing import default_dump


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


# TODO: merge with FieldInterpreter
class BubbleMeanValue(MeanValueItem):
    def __init__(self, mean_value, unit_bubble):
        super().__init__(mean_value, unit_bubble)
        # TODO: move this into FieldInterpreter/give reference to it
        self.is_marked = None

    def to_json(self):
        # TODO: mini util for this loop
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "is_marked",
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


class BubblesThresholdDetector(FieldTypeDetector):
    def __init__(self):
        self.type = FieldDetectionType.BUBBLES_THRESHOLD
        super().__init__()
        # TODO: use local_threshold from here
        # self.local_threshold = None

    def setup_directory_metrics(self, directory_aggregates):
        directory_aggregates[self.type] = {
            "per_omr_thresholds": [],
            "omr_thresholds_sum": 0,
            "running_omr_threshold": 0,
        }

    def get_field_bubble_means(self):
        return self.field_bubble_means

    def get_field_bubble_means_std(self):
        return self.field_bubble_means_std

    def update_field_type_aggregates(self, field_label):
        self.all_outlier_deviations[field_label] = self.field_bubble_means_std
        self.global_bubble_means_and_refs[field_label] = self.field_bubble_means

    def reset_field_type_aggregates(self):
        self.global_bubble_means_and_refs, self.all_outlier_deviations = {}, {}

    def read_field(self, field, gray_image, colored_image, file_aggregate_params=None):
        # Make use of file_aggregate_params if available (in second+ passes)
        if file_aggregate_params is not None:
            # override fallback threshold as per current running one(based on some config: NONE, FILE_LEVEL, FIELD_BLOCK_LEVEL, FIELD_LEVEL)
            pass

        # TODO: get bubble dimensions at field level during parsing-
        # box_w, box_h = self.field.bubble_dimensions
        box_w, box_h = self.field_block.bubble_dimensions

        field_bubbles = self.field.field_bubbles

        field_bubble_means = []
        for unit_bubble in field_bubbles:
            x, y = unit_bubble.get_shifted_position(self.shifts)
            rect = [y, y + box_h, x, x + box_w]
            # TODO: get this from within the BubbleDetection class
            mean_value = cv2.mean(gray_image[rect[0] : rect[1], rect[2] : rect[3]])[0]
            field_bubble_means.append(
                BubbleMeanValue(mean_value, unit_bubble)
                # TODO: cross/check mark detection support (#167)
                # detectCross(gray_image, rect) ? 0 : 255
            )
        self.field_bubble_means = field_bubble_means

        # TODO: move std calculation inside the class
        field_bubble_means_std = round(
            np.std([item.mean_value for item in field_bubble_means]), 2
        )
        self.field_bubble_means_std = FieldStdMeanValue(
            field_bubble_means_std, field_block
        )

        # TODO return interpretation?
        return self.field_bubble_means

    def get_field_interpretation(
        self,
        field,
        # TODO: self + aggregate params
        field_bubble_means,
        config,
        all_fields_threshold_for_file,
        global_max_jump,
        all_fields_outlier_deviation_threshold_for_file,
    ):
        # self_deviation = all_outlier_deviations[absolute_field_number].mean_value
        self_deviation = self.field_bubble_means_deviation
        # All Black or All White case
        no_outliers = (
            # TODO: rename mean_value in parent class to suit better
            self_deviation
            < all_fields_outlier_deviation_threshold_for_file
        )
        (
            local_threshold_for_field,
            local_max_jump,
        ) = self.get_local_threshold(
            field_bubble_means,
            all_fields_threshold_for_file,
            no_outliers,
            plot_title=f"Mean Intensity Barplot for {key}.{field.field_label}.block{block_field_number}",
            plot_show=config.outputs.show_image_level >= 7,
            config=config,
        )
        # TODO: move get_local_threshold into FieldInterpreter
        field.local_threshold = local_threshold_for_field
        per_omr_threshold_avg += local_threshold_for_field

        # TODO: see if deepclone is really needed given parent's instance
        # field_bubble_means = [
        #     deepcopy(bubble) for bubble in field_bubble_means
        # ]

        # Main detection logic:
        for bubble in field_bubble_means:
            local_bubble_is_marked = local_threshold_for_field > bubble.mean_value
            # TODO: refactor this mutation to a more appropriate place
            bubble.is_marked = local_bubble_is_marked

        # TODO: call from FieldInterpreter
        confidence_metrics = self.interpreter.get_field_level_confidence_metrics(
            field,
            field_bubble_means,
            config,
            local_threshold_for_field,
            all_fields_threshold_for_file,
            local_max_jump,
            global_max_jump,
        )

        return field_bubble_means, confidence_metrics

    def populate_field_type_aggregates():
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
            global_bubble_means_and_refs,  # , looseness=4
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
        field_type_aggregates = {
            **field_type_aggregates,
            #
            "local_thresholds": {
                # TODO: make stats util to "push" a value
                "running_total": 0,
                "running_average": 0,
                "processed_fields_count": 0,
                # TODO: both directory and file level
                "field_wise": {},
                # TODO: both directory and file level
                "field_type_wise": {},
                # FieldType -> { "running_average", "processed_fields_count"}
            },
            "confidence_metrics_for_file": {},
            "all_fields_threshold_for_file": all_fields_threshold_for_file,
            "all_fields_outlier_deviation_threshold_for_file": all_fields_outlier_deviation_threshold_for_file,
        }

    @staticmethod
    def get_global_threshold(
        bubble_means_and_refs,
        global_default_threshold,
        MIN_JUMP,
        JUMP_DELTA,
        plot_title,
        plot_show,
        sort_in_plot=True,
        looseness=1,
    ):
        """
        Note: Cannot assume qStrip has only-gray or only-white bg (no jump).
            So it's assumed that there will be either 1 or 2 jumps.
        1 Jump :
                ......
                ||||||
                ||||||  <-- risky THR
                ||||||  <-- safe THR
            ....||||||
            ||||||||||

        2 Jumps :
                ......
                |||||| <-- wrong THR
            ....||||||
            |||||||||| <-- safe THR
          ..||||||||||
          ||||||||||||
        """
        # Sort the Q bubbleValues
        sorted_bubble_means_and_refs = sorted(
            bubble_means_and_refs,
        )
        sorted_bubble_means = [item.mean_value for item in sorted_bubble_means_and_refs]

        # Find the FIRST LARGE GAP and set it as threshold:
        ls = (looseness + 1) // 2
        l = len(sorted_bubble_means) - ls
        max1, thr1 = MIN_JUMP, global_default_threshold
        for i in range(ls, l):
            jump = sorted_bubble_means[i + ls] - sorted_bubble_means[i - ls]
            if jump > max1:
                max1 = jump
                thr1 = sorted_bubble_means[i - ls] + jump / 2

        all_fields_threshold_for_file, j_low, j_high = (
            thr1,
            thr1 - max1 // 2,
            thr1 + max1 // 2,
        )

        # TODO: make use of outliers using percentile logic and report the benchmarks
        # Make use of the fact that the JUMP_DELTA(Vertical gap ofc) between
        # values at detected jumps would be atleast 20
        max2, thr2 = MIN_JUMP, global_default_threshold
        # Requires atleast 1 gray box to be present (Roll field will ensure this)
        for i in range(ls, l):
            jump = sorted_bubble_means[i + ls] - sorted_bubble_means[i - ls]
            new_thr = sorted_bubble_means[i - ls] + jump / 2
            if jump > max2 and abs(thr1 - new_thr) > JUMP_DELTA:
                max2 = jump
                thr2 = new_thr
        # TODO: deprecate thr2 and thus JUMP_DELTA (used only in the plotting)
        # all_fields_threshold_for_file = min(thr1,thr2)

        # TODO: maybe use plot_create flag to add plots in append_save_image
        if plot_show:
            plot_means_and_refs = (
                sorted_bubble_means_and_refs if sort_in_plot else bubble_means_and_refs
            )
            BubblesThresholdDetector.plot_for_global_threshold(
                plot_means_and_refs, plot_title, all_fields_threshold_for_file, thr2
            )

        return all_fields_threshold_for_file, j_low, j_high

    @staticmethod
    def plot_for_global_threshold(
        plot_means_and_refs, plot_title, all_fields_threshold_for_file, thr2
    ):
        _, ax = pyplot.subplots()
        # TODO: move into individual utils

        plot_values = [x.mean_value for x in plot_means_and_refs]
        original_bin_names = [
            x.item_reference.plot_bin_name for x in plot_means_and_refs
        ]
        plot_labels = [x.item_reference_name for x in plot_means_and_refs]

        # TODO: move into individual utils
        sorted_unique_bin_names, unique_label_indices = np.unique(
            original_bin_names, return_inverse=True
        )

        plot_color_sampler = colormaps["Spectral"].resampled(
            len(sorted_unique_bin_names)
        )

        shuffled_color_indices = random.sample(
            list(unique_label_indices), len(unique_label_indices)
        )
        plot_colors = plot_color_sampler(
            [shuffled_color_indices[i] for i in unique_label_indices]
        )
        bar_container = ax.bar(
            range(len(plot_means_and_refs)),
            plot_values,
            color=plot_colors,
            label=plot_labels,
        )

        # TODO: move into individual utils
        low = min(plot_values)
        high = max(plot_values)
        margin_factor = 0.1
        pyplot.ylim(
            [
                math.ceil(low - margin_factor * (high - low)),
                math.ceil(high + margin_factor * (high - low)),
            ]
        )

        # Show field labels
        ax.bar_label(bar_container, labels=plot_labels)
        handles, labels = ax.get_legend_handles_labels()
        # Naturally sorted unique legend labels https://stackoverflow.com/a/27512450/6242649
        ax.legend(
            *zip(
                *sorted(
                    [
                        (h, l)
                        for i, (h, l) in enumerate(zip(handles, labels))
                        if l not in labels[:i]
                    ],
                    key=lambda s: [
                        int(t) if t.isdigit() else t.lower()
                        for t in re.split("(\\d+)", s[1])
                    ],
                )
            )
        )
        ax.set_title(plot_title)
        ax.axhline(
            all_fields_threshold_for_file, color="green", ls="--", linewidth=5
        ).set_label("Global Threshold")
        ax.axhline(thr2, color="red", ls=":", linewidth=3).set_label("THR2 Line")
        # ax.axhline(j_low,color='red',ls='-.', linewidth=3)
        # ax.axhline(j_high,color='red',ls='-.', linewidth=3).set_label("Boundary Line")
        # ax.set_ylabel("Mean Intensity")
        ax.set_ylabel("Values")
        ax.set_xlabel("Position")

        pyplot.title(plot_title)
        pyplot.show()

    @staticmethod
    def get_local_threshold(
        bubble_means_and_refs,
        all_fields_threshold_for_file,
        no_outliers,
        plot_title,
        plot_show,
        config,
    ):
        """
        TODO: Update this documentation too-

        0 Jump :
                        <-- safe THR?
               .......
            ...|||||||
            ||||||||||  <-- safe THR?

        => Will fallback to all_fields_threshold_for_file

        How to decide it is this case of 0 jumps

        1 Jump :
                ......
                ||||||
                ||||||  <-- risky THR
                ||||||  <-- safe THR
            ....||||||
            ||||||||||

        => Will use safe local threshold

        """
        # Sort the Q bubbleValues
        sorted_bubble_means_and_refs = sorted(
            bubble_means_and_refs,
        )
        sorted_bubble_means = [item.mean_value for item in sorted_bubble_means_and_refs]
        # Small no of pts cases:
        # base case: 1 or 2 pts
        if len(sorted_bubble_means) < 3:
            max1, thr1 = config.thresholding.MIN_JUMP, (
                all_fields_threshold_for_file
                if np.max(sorted_bubble_means) - np.min(sorted_bubble_means)
                < config.thresholding.MIN_GAP_TWO_BUBBLES
                else np.mean(sorted_bubble_means)
            )
        else:
            l = len(sorted_bubble_means) - 1
            max1, thr1 = config.thresholding.MIN_JUMP, 255
            for i in range(1, l):
                jump = sorted_bubble_means[i + 1] - sorted_bubble_means[i - 1]
                if jump > max1:
                    max1 = jump
                    thr1 = sorted_bubble_means[i - 1] + jump / 2

            confident_jump = (
                config.thresholding.MIN_JUMP
                + config.thresholding.MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK
            )

            # TODO: seek improvement here because of the empty cases failing here(boundary walls)
            # Can see erosion make a lot of sense here?
            # If not confident, then only take help of all_fields_threshold_for_file
            if max1 < confident_jump:
                # Threshold hack: local can never be 255
                if no_outliers or thr1 == 255:
                    # All Black or All White case
                    thr1 = all_fields_threshold_for_file
                else:
                    # TODO: Low confidence parameters here
                    pass

        # TODO: Make a common plot util to show local and global thresholds
        if plot_show:
            BubblesThresholdDetector.plot_for_local_threshold(
                sorted_bubble_means, thr1, all_fields_threshold_for_file, plot_title
            )
        return thr1, max1

    @staticmethod
    def plot_for_local_threshold(
        sorted_bubble_means, thr1, all_fields_threshold_for_file, plot_title
    ):
        # TODO: add plot labels via the util
        _, ax = pyplot.subplots()
        ax.bar(range(len(sorted_bubble_means)), sorted_bubble_means)
        thrline = ax.axhline(thr1, color="green", ls=("-."), linewidth=3)
        thrline.set_label("Local Threshold")
        thrline = ax.axhline(
            all_fields_threshold_for_file, color="red", ls=":", linewidth=5
        )
        thrline.set_label("Global Threshold")
        ax.set_title(plot_title)
        ax.set_ylabel("Bubble Mean Intensity")
        ax.set_xlabel("Bubble Number(sorted)")
        ax.legend()
        pyplot.show()
