import math
import random
import re
from typing import List

import numpy as np
from matplotlib import colormaps, pyplot

from src.algorithm.template.detection.base.interpretation import FieldInterpretation
from src.algorithm.template.detection.bubbles_threshold.detection import BubbleMeanValue
from src.algorithm.template.template_layout import Field
from src.utils.logger import logger


class BubbleInterpretation:
    def __init__(self, field_bubble_mean: BubbleMeanValue, local_threshold):
        self.is_marked = None
        # self.field_bubble_mean = field_bubble_mean
        self.mean_value = field_bubble_mean.mean_value
        self.bubble_value = field_bubble_mean.item_reference.bubble_value
        self.local_threshold = local_threshold
        # TODO: decouple this -  needed for drawing (else not needed here?)
        self.item_reference = field_bubble_mean.item_reference
        # self.unit_bubble = field_bubble_mean.item_reference

        self.update_interpretation(local_threshold)

    def update_interpretation(self, local_threshold):
        is_marked = local_threshold > self.mean_value
        self.is_marked = is_marked
        return is_marked

    def __str__(self):
        return f"{self.item_reference} : {round(self.mean_value, 2)} {'*' if self.is_marked else ''}"


class BubblesFieldInterpretation(FieldInterpretation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_field_interpretation_string(self):
        marked_bubbles = [
            bubble_interpretation.bubble_value
            for bubble_interpretation in self.field_bubble_interpretations
            if bubble_interpretation.is_marked
        ]
        # Empty value logic
        if len(marked_bubbles) == 0:
            return self.empty_value

        # Concatenation logic
        return "".join(marked_bubbles)

    def run_interpretation(
        self,
        field: Field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ):
        self.initialize_from_file_level_aggregates(
            field, file_level_detection_aggregates, file_level_interpretation_aggregates
        )
        self.process_field_bubble_means()

    def initialize_from_file_level_aggregates(
        self,
        field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ):
        field_label = field.field_label

        field_level_detection_aggregates = file_level_detection_aggregates[
            "field_label_wise_aggregates"
        ][field_label]
        self.field_bubble_means = field_level_detection_aggregates["field_bubble_means"]

        self.field_bubble_means_std = field_level_detection_aggregates[
            "field_bubble_means_std"
        ]

        self.file_level_fallback_threshold = file_level_interpretation_aggregates[
            "file_level_fallback_threshold"
        ]

        self.outlier_deviation_threshold_for_file = (
            file_level_interpretation_aggregates["outlier_deviation_threshold_for_file"]
        )
        self.global_max_jump = file_level_interpretation_aggregates["global_max_jump"]

        # TODO: try out using additional fallbacks available
        # file_level_average_threshold = file_level_interpretation_aggregates["all_fields_local_thresholds"].running_average
        # bubble_field_type_average_threshold = file_level_interpretation_aggregates["bubble_field_type_wise_thresholds"][field.bubble_field_type].running_average
        # directory_level_bubble_field_type_average_threshold = directory_level_interpretation_aggregates["bubble_field_type_wise_thresholds"][field.bubble_field_type].running_average
        # directory_level_field_label_average_threshold = directory_level_interpretation_aggregates["field_label_wise_local_thresholds"][field_label].running_average

    def process_field_bubble_means(
        self,
    ):
        self.update_local_threshold_for_field()
        self.update_interpretations_for_field()

        # TODO: see if parent can call this function -
        self.update_common_interpretations()
        self.update_field_level_confidence_metrics()

    def update_local_threshold_for_field(self):
        field = self.field
        config = self.tuning_config

        # All Black or All White case
        no_outliers = (
            self.field_bubble_means_std < self.outlier_deviation_threshold_for_file
        )

        (
            self.local_threshold_for_field,
            self.local_max_jump,
        ) = self.get_local_threshold(
            self.field_bubble_means,
            self.file_level_fallback_threshold,
            no_outliers,
            config=config,
            plot_title=f"Mean Intensity Barplot for {field.field_label}.block",
            plot_show=config.outputs.show_image_level >= 7,
        )

    def update_interpretations_for_field(self):
        self.field_bubble_interpretations: List[BubbleInterpretation] = []

        # Main detection/thresholding logic here:
        for field_bubble_mean in self.field_bubble_means:
            self.field_bubble_interpretations.append(
                BubbleInterpretation(field_bubble_mean, self.local_threshold_for_field)
            )

    def update_common_interpretations(self):
        # TODO: can we move it to a common wrapper since is_multi_marked is independent of field detection type?
        marked_bubbles = [
            bubble_interpretation.bubble_value
            for bubble_interpretation in self.field_bubble_interpretations
            if bubble_interpretation.is_marked
        ]
        self.is_multi_marked = len(marked_bubbles) > 1

    def update_field_level_confidence_metrics(self):
        field = self.field
        config = self.tuning_config

        field_level_confidence_metrics = self.calculate_field_level_confidence_metrics(
            field,
            self.field_bubble_means,
            config,
            self.local_threshold_for_field,
            self.file_level_fallback_threshold,
            self.local_max_jump,
            self.global_max_jump,
        )
        self.insert_field_level_confidence_metrics(field_level_confidence_metrics)

    # TODO: see if this one should move to detector
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

        file_level_fallback_threshold, j_low, j_high = (
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
        # file_level_fallback_threshold = min(thr1,thr2)

        # TODO: maybe use plot_create flag to add plots in append_save_image
        if plot_show:
            plot_means_and_refs = (
                sorted_bubble_means_and_refs if sort_in_plot else bubble_means_and_refs
            )
            BubblesFieldInterpretation.plot_for_global_threshold(
                plot_means_and_refs, plot_title, file_level_fallback_threshold, thr2
            )

        return file_level_fallback_threshold, j_low, j_high

    @staticmethod
    def plot_for_global_threshold(
        plot_means_and_refs, plot_title, file_level_fallback_threshold, thr2
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
            file_level_fallback_threshold, color="green", ls="--", linewidth=5
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
        file_level_fallback_threshold,
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

        => Will fallback to file_level_fallback_threshold

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
                file_level_fallback_threshold
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
            # If not confident, then only take help of file_level_fallback_threshold
            if max1 < confident_jump:
                # Threshold hack: local can never be 255
                if no_outliers or thr1 == 255:
                    # All Black or All White case
                    thr1 = file_level_fallback_threshold
                else:
                    # TODO: Low confidence parameters here
                    pass

        # TODO: Make a common plot util to show local and global thresholds
        if plot_show:
            BubblesFieldInterpretation.plot_for_local_threshold(
                sorted_bubble_means, thr1, file_level_fallback_threshold, plot_title
            )
        return thr1, max1

    @staticmethod
    def plot_for_local_threshold(
        sorted_bubble_means, thr1, file_level_fallback_threshold, plot_title
    ):
        # TODO: add plot labels via the util?
        _, ax = pyplot.subplots()
        ax.bar(range(len(sorted_bubble_means)), sorted_bubble_means)
        thr_line = ax.axhline(thr1, color="green", ls=("-."), linewidth=3)
        thr_line.set_label("Local Threshold")
        thr_line = ax.axhline(
            file_level_fallback_threshold, color="red", ls=":", linewidth=5
        )
        thr_line.set_label("Global Threshold")
        ax.set_title(plot_title)
        ax.set_ylabel("Bubble Mean Intensity")
        ax.set_xlabel("Bubble Number(sorted)")
        ax.legend()
        pyplot.show()

    @staticmethod
    def calculate_field_level_confidence_metrics(
        field: Field,
        # TODO: call from FieldInterpretation? using field_bubble_interpretations?
        field_bubble_means: List[BubbleMeanValue],
        config,
        local_threshold_for_field,
        file_level_fallback_threshold,
        local_max_jump,
        global_max_jump,
    ):
        if not config.outputs.show_confidence_metrics:
            return {}

        (
            MIN_JUMP,
            GLOBAL_THRESHOLD_MARGIN,
            MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK,
            CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY,
        ) = map(
            config.thresholding.get,
            [
                "MIN_JUMP",
                "GLOBAL_THRESHOLD_MARGIN",
                "MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK",
                "CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY",
            ],
        )
        bubbles_in_doubt = {
            "by_disparity": [],
            "by_jump": [],
            "global_higher": [],
            "global_lower": [],
        }

        # Main detection logic:
        for bubble in field_bubble_means:
            global_bubble_is_marked = file_level_fallback_threshold > bubble.mean_value
            local_bubble_is_marked = local_threshold_for_field > bubble.mean_value
            # 1. Disparity in global/local threshold output
            if global_bubble_is_marked != local_bubble_is_marked:
                bubbles_in_doubt["by_disparity"].append(bubble)
        # 5. High confidence if the gap is very large compared to MIN_JUMP
        is_global_jump_confident = (
            global_max_jump > MIN_JUMP + CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY
        )
        is_local_jump_confident = (
            local_max_jump > MIN_JUMP + CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY
        )

        thresholds_string = f"global={round(file_level_fallback_threshold, 2)} local={round(local_threshold_for_field, 2)} global_margin={GLOBAL_THRESHOLD_MARGIN}"
        jumps_string = f"global_max_jump={round(global_max_jump, 2)} local_max_jump={round(local_max_jump, 2)} MIN_JUMP={MIN_JUMP} SURPLUS={CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY}"
        if len(bubbles_in_doubt["by_disparity"]) > 0:
            logger.warning(
                f"found disparity in field: {field.field_label}",
                list(map(str, bubbles_in_doubt["by_disparity"])),
                thresholds_string,
            )
            # 5.2 if the gap is very large compared to MIN_JUMP, but still there is disparity
            if is_global_jump_confident:
                logger.warning(
                    f"is_global_jump_confident but still has disparity",
                    jumps_string,
                )
            elif is_local_jump_confident:
                logger.warning(
                    f"is_local_jump_confident but still has disparity",
                    jumps_string,
                )
        else:
            # Note: debug logs are disabled by default
            logger.debug(
                f"party_matched for field: {field.field_label}",
                thresholds_string,
            )

            # 5.1 High confidence if the gap is very large compared to MIN_JUMP
            if is_local_jump_confident:
                # Higher weightage for confidence
                logger.debug(
                    f"is_local_jump_confident => increased confidence",
                    jumps_string,
                )
            # No output disparity, but -
            # 2.1 global threshold is "too close" to lower bubbles
            bubbles_in_doubt["global_lower"] = [
                bubble
                for bubble in field_bubble_means
                if GLOBAL_THRESHOLD_MARGIN
                > max(
                    GLOBAL_THRESHOLD_MARGIN,
                    file_level_fallback_threshold - bubble.mean_value,
                )
            ]

            if len(bubbles_in_doubt["global_lower"]) > 0:
                logger.warning(
                    'bubbles_in_doubt["global_lower"]',
                    list(map(str, bubbles_in_doubt["global_lower"])),
                )
            # 2.2 global threshold is "too close" to higher bubbles
            bubbles_in_doubt["global_higher"] = [
                bubble
                for bubble in field_bubble_means
                if GLOBAL_THRESHOLD_MARGIN
                > max(
                    GLOBAL_THRESHOLD_MARGIN,
                    bubble.mean_value - file_level_fallback_threshold,
                )
            ]

            if len(bubbles_in_doubt["global_higher"]) > 0:
                logger.warning(
                    'bubbles_in_doubt["global_higher"]',
                    list(map(str, bubbles_in_doubt["global_higher"])),
                )

            # 3. local jump outliers are close to the configured min_jump but below it.
            # Note: This factor indicates presence of cases like partially filled bubbles,
            # mis-aligned box boundaries or some form of unintentional marking over the bubble
            if len(field_bubble_means) > 1:
                jumps_in_bubble_means = (
                    BubblesFieldInterpretation.get_jumps_in_bubble_means(
                        field_bubble_means
                    )
                )
                bubbles_in_doubt["by_jump"] = [
                    bubble
                    for jump, bubble in jumps_in_bubble_means
                    if MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK
                    > max(
                        MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK,
                        MIN_JUMP - jump,
                    )
                ]

                if len(bubbles_in_doubt["by_jump"]) > 0:
                    logger.warning(
                        'bubbles_in_doubt["by_jump"]',
                        list(map(str, bubbles_in_doubt["by_jump"])),
                    )
                    logger.warning(
                        list(map(str, jumps_in_bubble_means)),
                    )

                # TODO: aggregate the bubble metrics into the Field objects
                # collect_bubbles_in_doubt(bubbles_in_doubt["by_disparity"], bubbles_in_doubt["global_higher"], bubbles_in_doubt["global_lower"], bubbles_in_doubt["by_jump"])
        field_level_confidence_metrics = {
            "bubbles_in_doubt": bubbles_in_doubt,
            "is_global_jump_confident": is_global_jump_confident,
            "is_local_jump_confident": is_local_jump_confident,
            "local_max_jump": local_max_jump,
            "field_label": field.field_label,
        }
        return field_level_confidence_metrics

    @staticmethod
    def get_jumps_in_bubble_means(field_bubble_means: List[BubbleMeanValue]):
        # get sorted array
        sorted_field_bubble_means = sorted(
            field_bubble_means,
        )
        # get jumps
        jumps_in_bubble_means = []
        previous_bubble = sorted_field_bubble_means[0]
        previous_mean = previous_bubble.mean_value
        for i in range(1, len(sorted_field_bubble_means)):
            bubble = sorted_field_bubble_means[i]
            current_mean = bubble.mean_value
            jumps_in_bubble_means.append(
                [
                    round(current_mean - previous_mean, 2),
                    previous_bubble,
                ]
            )
            previous_bubble = bubble
            previous_mean = current_mean
        return jumps_in_bubble_means
