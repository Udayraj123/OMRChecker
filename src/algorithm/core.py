"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

import math
import random
import re
from copy import copy as shallowcopy
from copy import deepcopy

import cv2
import numpy as np
from matplotlib import colormaps, pyplot

from src.algorithm.detection import BubbleMeanValue, FieldStdMeanValue
from src.utils.image import ImageUtils
from src.utils.logger import logger


class ImageInstanceOps:
    """Class to hold fine-tuned utilities for a group of images. One instance for each processing directory."""

    def __init__(self, tuning_config):
        super().__init__()
        self.tuning_config = tuning_config

    def apply_preprocessors(
        self, file_path, gray_image, colored_image, original_template
    ):
        config = self.tuning_config

        # Copy template for this instance op
        template = shallowcopy(original_template)

        # Make deepcopy for only parts that are mutated by Processor
        template.field_blocks = deepcopy(template.field_blocks)

        # resize to conform to common preprocessor input requirements
        gray_image = ImageUtils.resize_to_shape(
            gray_image, template.processing_image_shape
        )
        if config.outputs.colored_outputs_enabled:
            colored_image = ImageUtils.resize_to_shape(
                colored_image, template.processing_image_shape
            )

        # run pre_processors in sequence
        for pre_processor in template.pre_processors:
            (
                out_omr,
                colored_image,
                next_template,
            ) = pre_processor.resize_and_apply_filter(
                gray_image, colored_image, template, file_path
            )
            gray_image = out_omr
            template = next_template

        if template.output_image_shape:
            # resize to output requirements
            gray_image = ImageUtils.resize_to_shape(
                gray_image, template.output_image_shape
            )
            if config.outputs.colored_outputs_enabled:
                colored_image = ImageUtils.resize_to_shape(
                    colored_image, template.output_image_shape
                )

        return gray_image, colored_image, template

    # TODO: move algorithm to detection.py
    def read_omr_response(self, input_gray_image, colored_image, template, file_path):
        config = self.tuning_config

        gray_image = input_gray_image.copy()

        gray_image = ImageUtils.resize_to_dimensions(
            gray_image, template.template_dimensions
        )
        if colored_image is not None:
            colored_image = ImageUtils.resize_to_dimensions(
                colored_image, template.template_dimensions
            )
        # Resize to template dimensions for saved outputs
        template.save_image_ops.append_save_image(
            f"Resized Image", range(3, 7), gray_image, colored_image
        )

        if gray_image.max() > gray_image.min():
            gray_image = ImageUtils.normalize(gray_image)

        # Move them to data class if needed
        omr_response = {}
        multi_marked, multi_roll = False, False

        # Get mean bubbleValues n other stats
        (
            global_bubble_means_and_refs,
            field_number_to_field_bubble_means,
            global_field_bubble_means_stds,
        ) = (
            [],
            [],
            [],
        )
        for field_block in template.field_blocks:
            #  TODO: support for if field_block.field_type == "BARCODE":

            field_bubble_means_stds = []
            box_w, box_h = field_block.bubble_dimensions
            for field in field_block.fields:
                field_bubbles = field.field_bubbles
                field_bubble_means = []
                for unit_bubble in field_bubbles:
                    x, y = unit_bubble.get_shifted_position(field_block.shifts)
                    rect = [y, y + box_h, x, x + box_w]
                    # TODO: get this from within the BubbleDetection class
                    mean_value = cv2.mean(
                        gray_image[rect[0] : rect[1], rect[2] : rect[3]]
                    )[0]
                    field_bubble_means.append(
                        BubbleMeanValue(mean_value, unit_bubble)
                        # TODO: cross/check mark detection support (#167)
                        # detectCross(gray_image, rect) ? 0 : 255
                    )

                # TODO: move std calculation inside the class
                field_std = round(
                    np.std([item.mean_value for item in field_bubble_means]), 2
                )
                field_bubble_means_stds.append(
                    FieldStdMeanValue(field_std, field_block)
                )

                field_number_to_field_bubble_means.append(field_bubble_means)
                global_bubble_means_and_refs.extend(field_bubble_means)
            global_field_bubble_means_stds.extend(field_bubble_means_stds)

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
        global_std_thresh, _, _ = self.get_global_threshold(
            global_field_bubble_means_stds,
            GLOBAL_PAGE_THRESHOLD_STD,
            MIN_JUMP=MIN_JUMP_STD,
            JUMP_DELTA=JUMP_DELTA_STD,
            plot_title="Q-wise Std-dev Plot",
            plot_show=config.outputs.show_image_level >= 6,
            sort_in_plot=True,
        )

        # Note: Plotting takes Significant times here --> Change Plotting args
        # to support show_image_level
        global_threshold_for_template, j_low, j_high = self.get_global_threshold(
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
            f"Thresholding: \t global_threshold_for_template: {round(global_threshold_for_template, 2)} \tglobal_std_THR: {round(global_std_thresh, 2)}\t{'(Looks like a Xeroxed OMR)' if (global_threshold_for_template == 255) else ''}"
        )

        per_omr_threshold_avg, absolute_field_number = 0, 0
        global_field_confidence_metrics = []
        for field_block in template.field_blocks:
            block_field_number = 1
            key = field_block.name[:3]
            box_w, box_h = field_block.bubble_dimensions

            for field in field_block.fields:
                field_bubbles = field.field_bubbles
                # All Black or All White case
                no_outliers = (
                    # TODO: rename mean_value in parent class to suit better
                    global_field_bubble_means_stds[absolute_field_number].mean_value
                    < global_std_thresh
                )

                field_bubble_means = field_number_to_field_bubble_means[
                    absolute_field_number
                ]

                (
                    local_threshold_for_field,
                    local_max_jump,
                ) = self.get_local_threshold(
                    field_bubble_means,
                    global_threshold_for_template,
                    no_outliers,
                    plot_title=f"Mean Intensity Barplot for {key}.{field.field_label}.block{block_field_number}",
                    plot_show=config.outputs.show_image_level >= 7,
                )
                # TODO: move get_local_threshold into FieldDetection
                field.local_threshold = local_threshold_for_field
                per_omr_threshold_avg += local_threshold_for_field

                field_bubble_means, confidence_metrics = self.apply_field_detection(
                    field,
                    field_bubble_means,
                    config,
                    local_threshold_for_field,
                    global_threshold_for_template,
                    local_max_jump,
                    global_max_jump,
                )
                global_field_confidence_metrics.append(confidence_metrics)

                detected_bubbles = [
                    bubble_detection
                    for bubble_detection in field_bubble_means
                    if bubble_detection.is_marked
                ]
                for bubble_detection in detected_bubbles:
                    bubble = bubble_detection.item_reference
                    field_label, field_value = (
                        bubble.field_label,
                        bubble.field_value,
                    )
                    multi_marked_local = field_label in omr_response
                    # Apply concatenation
                    omr_response[field_label] = (
                        (omr_response[field_label] + field_value)
                        if multi_marked_local
                        else field_value
                    )
                    # TODO: support for multi_marked bucket based on identifier config
                    # multi_roll = multi_marked_local and "Roll" in str(q)
                    multi_marked = multi_marked or multi_marked_local

                # Empty value logic
                if len(detected_bubbles) == 0:
                    field_label = field.field_label
                    omr_response[field_label] = field_block.empty_val

                block_field_number += 1
                absolute_field_number += 1
                # TODO: refactor all_c_box_vals

            # /for field_block

        # TODO: aggregate with weightages
        # overall_confidence = self.get_confidence_metrics(fields_confidence)
        # TODO: Make the plot for underconfident_fields
        # underconfident_fields = filter(lambda x: x.confidence < 0.8, fields_confidence)

        per_omr_threshold_avg /= absolute_field_number
        per_omr_threshold_avg = round(per_omr_threshold_avg, 2)

        # TODO: if config.outputs.show_image_level >= 5: f"Bubble Intensity by question type for {name}"

        return (
            omr_response,
            multi_marked,
            multi_roll,
            field_number_to_field_bubble_means,
            global_threshold_for_template,
            global_field_confidence_metrics,
        )

    def get_global_threshold(
        self,
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

        global_threshold_for_template, j_low, j_high = (
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
        # global_threshold_for_template = min(thr1,thr2)

        # TODO: maybe use plot_create flag to add plots in append_save_image
        if plot_show:
            plot_means_and_refs = (
                sorted_bubble_means_and_refs if sort_in_plot else bubble_means_and_refs
            )
            self.plot_for_global_threshold(
                plot_means_and_refs, plot_title, global_threshold_for_template, thr2
            )

        return global_threshold_for_template, j_low, j_high

    @staticmethod
    def plot_for_global_threshold(
        plot_means_and_refs, plot_title, global_threshold_for_template, thr2
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
            global_threshold_for_template, color="green", ls="--", linewidth=5
        ).set_label("Global Threshold")
        ax.axhline(thr2, color="red", ls=":", linewidth=3).set_label("THR2 Line")
        # ax.axhline(j_low,color='red',ls='-.', linewidth=3)
        # ax.axhline(j_high,color='red',ls='-.', linewidth=3).set_label("Boundary Line")
        # ax.set_ylabel("Mean Intensity")
        ax.set_ylabel("Values")
        ax.set_xlabel("Position")

        pyplot.title(plot_title)
        pyplot.show()

    def get_local_threshold(
        self,
        bubble_means_and_refs,
        global_threshold_for_template,
        no_outliers,
        plot_title,
        plot_show,
    ):
        """
        TODO: Update this documentation too-

        0 Jump :
                        <-- safe THR?
               .......
            ...|||||||
            ||||||||||  <-- safe THR?

        => Will fallback to global_threshold_for_template

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
        config = self.tuning_config
        # Sort the Q bubbleValues
        sorted_bubble_means_and_refs = sorted(
            bubble_means_and_refs,
        )
        sorted_bubble_means = [item.mean_value for item in sorted_bubble_means_and_refs]
        # Small no of pts cases:
        # base case: 1 or 2 pts
        if len(sorted_bubble_means) < 3:
            max1, thr1 = config.thresholding.MIN_JUMP, (
                global_threshold_for_template
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
            # If not confident, then only take help of global_threshold_for_template
            if max1 < confident_jump:
                # Threshold hack: local can never be 255
                if no_outliers or thr1 == 255:
                    # All Black or All White case
                    thr1 = global_threshold_for_template
                else:
                    # TODO: Low confidence parameters here
                    pass

        # TODO: Make a common plot util to show local and global thresholds
        if plot_show:
            self.plot_for_local_threshold(
                sorted_bubble_means, thr1, global_threshold_for_template, plot_title
            )
        return thr1, max1

    @staticmethod
    def plot_for_local_threshold(
        sorted_bubble_means, thr1, global_threshold_for_template, plot_title
    ):
        # TODO: add plot labels via the util
        _, ax = pyplot.subplots()
        ax.bar(range(len(sorted_bubble_means)), sorted_bubble_means)
        thrline = ax.axhline(thr1, color="green", ls=("-."), linewidth=3)
        thrline.set_label("Local Threshold")
        thrline = ax.axhline(
            global_threshold_for_template, color="red", ls=":", linewidth=5
        )
        thrline.set_label("Global Threshold")
        ax.set_title(plot_title)
        ax.set_ylabel("Bubble Mean Intensity")
        ax.set_xlabel("Bubble Number(sorted)")
        ax.legend()
        pyplot.show()

    @staticmethod
    def apply_field_detection(
        field,
        field_bubble_means,
        config,
        local_threshold_for_field,
        global_threshold_for_template,
        local_max_jump,
        global_max_jump,
    ):
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
        # TODO: see if deepclone is really needed given parent's instance
        # field_bubble_means = [
        #     deepcopy(bubble) for bubble in field_bubble_means
        # ]

        bubbles_in_doubt = {
            "by_disparity": [],
            "by_jump": [],
            "global_higher": [],
            "global_lower": [],
        }

        # Main detection logic:
        for bubble in field_bubble_means:
            global_bubble_is_marked = global_threshold_for_template > bubble.mean_value
            local_bubble_is_marked = local_threshold_for_field > bubble.mean_value
            # TODO: refactor this mutation to a more appropriate place
            bubble.is_marked = local_bubble_is_marked
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

        # TODO: FieldDetection.bubbles = field_bubble_means
        thresholds_string = f"global={round(global_threshold_for_template, 2)} local={round(local_threshold_for_field, 2)} global_margin={GLOBAL_THRESHOLD_MARGIN}"
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
                    global_threshold_for_template - bubble.mean_value,
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
                    bubble.mean_value - global_threshold_for_template,
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
                # TODO: move util
                def get_jumps_in_bubble_means(field_bubble_means):
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

                jumps_in_bubble_means = get_jumps_in_bubble_means(field_bubble_means)
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
        confidence_metrics = {
            "bubbles_in_doubt": bubbles_in_doubt,
            "is_global_jump_confident": is_global_jump_confident,
            "is_local_jump_confident": is_local_jump_confident,
            "local_max_jump": local_max_jump,
            "field_label": field.field_label,
        }
        return field_bubble_means, confidence_metrics
