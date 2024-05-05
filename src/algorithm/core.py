"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

import math
import os
import random
import re
from collections import defaultdict
from copy import copy as shallowcopy
from copy import deepcopy
from typing import Any
import cv2
import numpy as np
from matplotlib import colormaps, pyplot

from src.algorithm.detection import BubbleMeanValue, FieldStdMeanValue
from src.algorithm.evaluation import get_evaluation_meta_for_question
from src.schemas.constants import Verdict
from src.utils.constants import (
    BONUS_SYMBOL,
    CLR_BLACK,
    CLR_GRAY,
    CLR_WHITE,
    MARKED_TEMPLATE_TRANSPARENCY,
    TEXT_SIZE,
    GROUP_THICKNESS
)
from src.schemas.constants import AnswerType
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger


class ImageInstanceOps:
    """Class to hold fine-tuned utilities for a group of images. One instance for each processing directory."""

    save_img_list: Any = defaultdict(list)

    def __init__(self, tuning_config):
        super().__init__()
        self.tuning_config = tuning_config
        self.save_image_level = tuning_config.outputs.save_image_level

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
        if config.outputs.show_colored_outputs:
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
            if config.outputs.show_colored_outputs:
                colored_image = ImageUtils.resize_to_shape(
                    colored_image, template.output_image_shape
                )

        return gray_image, colored_image, template

    def read_omr_response(self, image, template, file_path):
        config = self.tuning_config

        img = image.copy()
        # origDim = img.shape[:2]
        img = ImageUtils.resize_to_dimensions(img, template.template_dimensions)
        if img.max() > img.min():
            img = ImageUtils.normalize(img)

        # Move them to data class if needed
        omr_response = {}
        multi_marked, multi_roll = 0, 0

        # TODO Make this part useful for visualizing status checks
        # blackVals=[0]
        # whiteVals=[255]

        # if config.outputs.show_image_level >= 5:
        #     all_c_box_vals = {"int": [], "mcq": []}
        #     # TODO: simplify this logic
        #     q_nums = {"int": [], "mcq": []}

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
            field_bubble_means_stds = []
            box_w, box_h = field_block.bubble_dimensions
            for field in field_block.fields:
                field_bubbles = field.field_bubbles
                field_bubble_means = []
                for unit_bubble in field_bubbles:
                    x, y = unit_bubble.get_shifted_position(field_block.shifts)
                    rect = [y, y + box_h, x, x + box_w]
                    mean_value = cv2.mean(img[rect[0] : rect[1], rect[2] : rect[3]])[0]
                    field_bubble_means.append(
                        BubbleMeanValue(mean_value, unit_bubble)
                        # TODO: cross/check mark detection support (#167)
                        # detectCross(img, rect) ? 0 : 255
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
            GLOBAL_THRESHOLD_MARGIN,
            MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK,
            CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY,
        ) = map(
            config.thresholding.get,
            [
                "GLOBAL_PAGE_THRESHOLD",
                "MIN_JUMP",
                "JUMP_DELTA",
                "GLOBAL_THRESHOLD_MARGIN",
                "MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK",
                "CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY",
            ],
        )
        # TODO: see if this is needed, then take from config.json
        MIN_JUMP_STD = 15
        JUMP_DELTA_STD = 5
        global_default_std_threshold = 10
        global_std_thresh, _, _ = self.get_global_threshold(
            global_field_bubble_means_stds,
            global_default_std_threshold,
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

        logger.info(
            f"Thresholding:\t global_threshold_for_template: {round(global_threshold_for_template, 2)} \tglobal_std_THR: {round(global_std_thresh, 2)}\t{'(Looks like a Xeroxed OMR)' if (global_threshold_for_template == 255) else ''}"
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
                # print(absolute_field_number, field.field_label,
                #   global_field_bubble_means_stds[absolute_field_number].mean_value, "no_outliers:", no_outliers)

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
                # print(field.field_label,key,block_field_number, "THR: ",
                #   round(local_threshold_for_field,2))
                per_omr_threshold_avg += local_threshold_for_field

                # TODO: @staticmethod
                def apply_field_detection(
                    field,
                    field_bubble_means,
                    local_threshold_for_field,
                    global_threshold_for_template,
                ):
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

                    for bubble in field_bubble_means:
                        global_bubble_is_marked = (
                            global_threshold_for_template > bubble.mean_value
                        )
                        local_bubble_is_marked = (
                            local_threshold_for_field > bubble.mean_value
                        )
                        # TODO: refactor this mutation to a more appropriate place
                        bubble.is_marked = local_bubble_is_marked
                        # 1. Disparity in global/local threshold output
                        if global_bubble_is_marked != local_bubble_is_marked:
                            bubbles_in_doubt["by_disparity"].append(bubble)

                    # 5. High confidence if the gap is very large compared to MIN_JUMP
                    is_global_jump_confident = (
                        global_max_jump
                        > MIN_JUMP + CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY
                    )
                    is_local_jump_confident = (
                        local_max_jump > MIN_JUMP + CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY
                    )

                    # TODO: FieldDetection.bubbles = field_bubble_means
                    thresholds_string = f"global={round(global_threshold_for_template,2)} local={round(local_threshold_for_field,2)} global_margin={GLOBAL_THRESHOLD_MARGIN}"
                    jumps_string = f"global_max_jump={round(global_max_jump,2)} local_max_jump={round(local_max_jump,2)} MIN_JUMP={MIN_JUMP} SURPLUS={CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY}"
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
                        # TODO: reduce the noise for parity logs
                        skip_extra_logs = True  # temp
                        if not skip_extra_logs:
                            logger.info(
                                f"party_matched for field: {field.field_label}",
                                thresholds_string,
                            )

                        # 5.1 High confidence if the gap is very large compared to MIN_JUMP
                        if is_local_jump_confident:
                            # Higher weightage for confidence
                            if not skip_extra_logs:
                                logger.info(
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

                            jumps_in_bubble_means = get_jumps_in_bubble_means(
                                field_bubble_means
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
                    confidence_metrics = {
                        "bubbles_in_doubt": bubbles_in_doubt,
                        "is_global_jump_confident": is_global_jump_confident,
                        "is_local_jump_confident": is_local_jump_confident,
                        "local_max_jump": local_max_jump,
                        "field_label": field.field_label,
                    }
                    return field_bubble_means, confidence_metrics

                field_bubble_means, confidence_metrics = apply_field_detection(
                    field,
                    field_bubble_means,
                    local_threshold_for_field,
                    global_threshold_for_template,
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
                    # TODO: generalize this into rolls -> identifier
                    # Only send rolls multi-marked in the directory ()
                    # multi_roll = multi_marked_local and "Roll" in str(q)

                    multi_marked = multi_marked or multi_marked_local

                # Empty value logic
                if len(detected_bubbles) == 0:
                    field_label = field.field_label
                    omr_response[field_label] = field_block.empty_val

                # TODO: fix after all_c_box_vals is refactored
                # if config.outputs.show_image_level >= 5:
                #     if key in all_c_box_vals:
                #         q_nums[key].append(f"{key[:2]}_c{str(block_field_number)}")
                #         all_c_box_vals[key].append(field_number_to_field_bubble_means[absolute_field_number])

                block_field_number += 1
                absolute_field_number += 1
            # /for field_block

        # TODO: aggregate with weightages
        # overall_confidence = self.get_confidence_metrics(fields_confidence)
        # underconfident_fields = filter(lambda x: x.confidence < 0.8, fields_confidence)

        # TODO: Make the plot for underconfident_fields
        # logger.info(name, overall_confidence, underconfident_fields)

        per_omr_threshold_avg /= absolute_field_number
        per_omr_threshold_avg = round(per_omr_threshold_avg, 2)

        # TODO: refactor all_c_box_vals
        # Box types
        # if config.outputs.show_image_level >= 5:
        #     # pyplot.draw()
        #     f, axes = pyplot.subplots(len(all_c_box_vals), sharey=True)
        #     f.canvas.manager.set_window_title(
        #         f"Bubble Intensity by question type for {name}"
        #     )
        #     ctr = 0
        #     type_name = {
        #         "int": "Integer",
        #         "mcq": "MCQ",
        #         "med": "MED",
        #         "rol": "Roll",
        #     }
        #     for k, boxvals in all_c_box_vals.items():
        #         axes[ctr].title.set_text(type_name[k] + " Type")
        #         axes[ctr].boxplot(boxvals)
        #         # thrline=axes[ctr].axhline(per_omr_threshold_avg,color='red',ls='--')
        #         # thrline.set_label("Average THR")
        #         axes[ctr].set_ylabel("Intensity")
        #         axes[ctr].set_xticklabels(q_nums[k])
        #         # axes[ctr].legend()
        #         ctr += 1
        #     # imshow will do the waiting
        #     pyplot.tight_layout(pad=0.5)
        #     pyplot.show()

        return (
            omr_response,
            multi_marked,
            multi_roll,
            field_number_to_field_bubble_means,
            global_threshold_for_template,
            global_field_confidence_metrics,
        )

    def draw_template_layout(
        self, gray_image, colored_image, template, *args, **kwargs
    ):
        config = self.tuning_config
        final_marked = self.draw_template_layout_util(
            gray_image, "GRAYSCALE", template, *args, **kwargs
        )

        colored_final_marked = colored_image
        if config.outputs.show_colored_outputs:
            save_marked_dir = kwargs.get("save_marked_dir", None)
            kwargs["save_marked_dir"] = (
                save_marked_dir.joinpath("colored")
                if save_marked_dir is not None
                else None
            )
            colored_final_marked = self.draw_template_layout_util(
                colored_final_marked,
                "COLORED",
                template,
                *args,
                **kwargs,
            )

            InteractionUtils.show(
                "final_marked", final_marked, 0, resize_to_height=True, config=config
            )
            InteractionUtils.show(
                "colored_final_marked",
                colored_final_marked,
                1,
                resize_to_height=True,
                config=config,
            )
        else:
            InteractionUtils.show(
                "final_marked", final_marked, 1, resize_to_height=True, config=config
            )

        return final_marked, colored_final_marked

    def draw_template_layout_util(
        self,
        image,
        image_type,
        template,
        file_id=None,
        field_number_to_field_bubble_means=None,
        save_marked_dir=None,
        evaluation_meta=None,
        evaluation_config=None,
        shifted=False,
        border=-1,
    ):
        config = self.tuning_config

        marked_image = ImageUtils.resize_to_dimensions(
            image, template.template_dimensions
        )
        transparent_layer = marked_image.copy()
        should_draw_field_block_rectangles = field_number_to_field_bubble_means is None
        should_draw_marked_bubbles = field_number_to_field_bubble_means is not None
        should_draw_question_verdicts = (
            should_draw_marked_bubbles and evaluation_meta is not None
        )

        should_save_detections = (
            (image_type == "COLORED" or image_type == "GRAYSCALE")
            and config.outputs.save_detections
            and save_marked_dir is not None
        )

        if should_draw_field_block_rectangles:
            marked_image = self.draw_field_blocks_layout(
                marked_image, template, shifted, shouldCopy=False, border=border
            )
            return marked_image

        # TODO: move indent into smaller function
        if should_draw_marked_bubbles:
            marked_image = self.draw_marked_bubbles_with_evaluation_meta(
                marked_image,
                image_type,
                template,
                evaluation_meta,
                evaluation_config,
                field_number_to_field_bubble_means,
            )

        if should_save_detections:
            # TODO: migrate support for multi_marked bucket based on identifier config
            # if multi_roll:
            #     save_marked_dir = save_marked_dir.joinpath("_MULTI_")
            image_path = str(save_marked_dir.joinpath(file_id))
            ImageUtils.save_img(image_path, marked_image)

        # if config.outputs.show_colored_outputs:
        # TODO: add colored counterparts

        if should_draw_question_verdicts:
            marked_image = self.draw_evaluation_summary(
                marked_image, evaluation_meta, evaluation_config
            )

        # Prepare save images
        if should_save_detections:
            self.append_save_image(2, marked_image)

        # Translucent
        cv2.addWeighted(
            marked_image,
            MARKED_TEMPLATE_TRANSPARENCY,
            transparent_layer,
            1 - MARKED_TEMPLATE_TRANSPARENCY,
            0,
            marked_image,
        )

        if should_save_detections:
            for i in range(config.outputs.save_image_level):
                self.save_image_stacks(i + 1, file_id, save_marked_dir)

        if config.outputs.show_image_level >= 2 and file_id is not None:
            InteractionUtils.show(
                f"Final Marked Bubbles({image_type}) : '{file_id}'",
                marked_image,
                pause=True,
                resize_to_height=True,
                config=config,
            )

        return marked_image

    def draw_field_blocks_layout(
        self, image, template, shifted=True, shouldCopy=True, thickness=3, border=3
    ):
        marked_image = image.copy() if shouldCopy else image
        for field_block in template.field_blocks:
            field_block_name, origin, dimensions, bubble_dimensions = map(
                lambda attr: getattr(field_block, attr),
                [
                    "name",
                    "origin",
                    "dimensions",
                    "bubble_dimensions",
                ],
            )
            block_position = field_block.get_shifted_origin() if shifted else origin

            # Field block bounding rectangle
            ImageUtils.draw_box(
                marked_image,
                block_position,
                dimensions,
                color=CLR_BLACK,
                style="BOX_HOLLOW",
                thickness_factor=0,
                border=border,
            )

            for field in field_block.fields:
                field_bubbles = field.field_bubbles
                for unit_bubble in field_bubbles:
                    shifted_position = unit_bubble.get_shifted_position(
                        field_block.shifts
                    )
                    ImageUtils.draw_box(
                        marked_image,
                        shifted_position,
                        bubble_dimensions,
                        thickness_factor=1 / 10,
                        border=border,
                    )

            if shifted:
                text_position = lambda size_x, size_y: (
                    int(block_position[0] + dimensions[0] - size_x),
                    int(block_position[1] - size_y),
                )
                text = f"({field_block.shifts}){field_block_name}"
                ImageUtils.draw_text(marked_image, text, text_position, thickness)

        return marked_image

    def draw_marked_bubbles_with_evaluation_meta(
        self,
        marked_image,
        # TODO: make cases for colors for image_type == "COLORED" and box shapes for image_type == "GRAYSCALE"
        image_type,
        template,
        evaluation_meta,
        evaluation_config,
        field_number_to_field_bubble_means,
    ):
        should_draw_question_verdicts = evaluation_meta is not None
        absolute_field_number = 0
        for field_block in template.field_blocks:
            for field in field_block.fields:
                field_label = field.field_label
                field_bubble_means = field_number_to_field_bubble_means[
                    absolute_field_number
                ]
                absolute_field_number += 1

                question_has_verdict = (
                    should_draw_question_verdicts
                    and field_label in evaluation_meta["questions_meta"]
                )

                # linked_custom_labels = [custom_label if (field_label in field_labels) else None for (custom_label, field_labels) in template.custom_labels.items()]
                # is_part_of_custom_label = len(linked_custom_labels) > 0
                # TODO: replicate verdict: question_has_verdict = len([if field_label in questions_meta else None for field_label in linked_custom_labels])

                if (
                    evaluation_config.draw_question_verdicts["enabled"]
                    and question_has_verdict
                ):
                    question_meta = evaluation_meta["questions_meta"][field_label]
                    # Draw answer key items
                    self.draw_field_with_question_meta(
                        marked_image,
                        image_type,
                        field_bubble_means,
                        field_block,
                        question_meta,
                        evaluation_config,
                    )
                else:
                    self.draw_field_bubbles_and_detections(
                        marked_image, field_bubble_means, field_block, evaluation_config
                    )

        return marked_image

    def draw_field_bubbles_and_detections(
        self, marked_image, field_bubble_means, field_block, evaluation_config
    ):
        bubble_dimensions = tuple(field_block.bubble_dimensions)
        for bubble_detection in field_bubble_means:
            bubble = bubble_detection.item_reference
            shifted_position = tuple(bubble.get_shifted_position(field_block.shifts))
            field_value = str(bubble.field_value)

            if bubble_detection.is_marked:
                ImageUtils.draw_box(
                    marked_image,
                    shifted_position,
                    bubble_dimensions,
                    color=CLR_GRAY,
                    style="BOX_FILLED",
                    # TODO: pass verdict_color here and insert symbol mapping here ( +, -, *)
                    thickness_factor=1 / 12,
                )
                if evaluation_config.draw_detected_bubble_texts["enabled"]:
                    ImageUtils.draw_text(
                        marked_image,
                        field_value,
                        shifted_position,
                        text_size=TEXT_SIZE,
                        color=(20, 20, 10),
                        thickness=int(1 + 3.5 * TEXT_SIZE),
                    )
            else:
                ImageUtils.draw_box(
                    marked_image,
                    shifted_position,
                    bubble_dimensions,
                    style="BOX_HOLLOW",
                    thickness_factor=1 / 10,
                )

    @staticmethod
    def is_part_of_some_answer(question_meta, field_value):
        bonus_type, answer_type, answer_item = map(
            question_meta.get, ["bonus_type", "answer_type", "answer_item"]
        )
        if bonus_type:
            return True
        if answer_type == AnswerType.STANDARD:
            return field_value in str(answer_item)
        elif answer_type == AnswerType.MULTIPLE_CORRECT:
            for allowed_answer in answer_item:
                if field_value in allowed_answer:
                    return True
            return False
        else:
            for allowed_answer, score in answer_item:
                if score > 0 and field_value in allowed_answer:
                    return True
            return False

    def draw_field_with_question_meta(
        self,
        marked_image,
        image_type,
        field_bubble_means,
        field_block,
        question_meta,
        evaluation_config,
    ):
        bubble_dimensions = tuple(field_block.bubble_dimensions)
        verdict_colors = evaluation_config.verdict_colors
        verdict_symbol_colors = evaluation_config.verdict_symbol_colors
        draw_answer_groups = evaluation_config.draw_answer_groups
        for bubble_detection in field_bubble_means:
            bubble = bubble_detection.item_reference
            shifted_position = tuple(bubble.get_shifted_position(field_block.shifts))
            field_value = str(bubble.field_value)
            if self.is_part_of_some_answer(question_meta, field_value):
                ImageUtils.draw_box(
                    marked_image,
                    shifted_position,
                    bubble_dimensions,
                    CLR_BLACK,
                    style="BOX_HOLLOW",
                    thickness_factor=0,
                )
            if question_meta["bonus_type"] == "BONUS_ON_ATTEMPT":
                if question_meta["question_schema_verdict"] == Verdict.UNMARKED:
                    if image_type == "GRAYSCALE":
                        position, position_diagonal = ImageUtils.draw_box(
                            marked_image,
                            shifted_position,
                            bubble_dimensions,
                            color=CLR_WHITE,
                            style="BOX_FILLED",
                            # TODO: pass verdict_color here and insert symbol mapping here ( +, -, *)
                            thickness_factor=1 / 12,
                        )

                        ImageUtils.draw_symbol(
                            marked_image,
                            BONUS_SYMBOL,
                            position,
                            position_diagonal,
                        )
                    else:
                        position, position_diagonal = ImageUtils.draw_box(
                            marked_image,
                            shifted_position,
                            bubble_dimensions,
                            color=verdict_colors["bonus"],
                            style="BOX_FILLED",
                            thickness_factor=1 / 12,
                        )

                        ImageUtils.draw_symbol(
                            marked_image,
                            BONUS_SYMBOL,
                            position,
                            position_diagonal,
                            color=verdict_symbol_colors["bonus"],
                        )

            if question_meta["bonus_type"] == "BONUS_FOR_ALL":
                if image_type == "GRAYSCALE":
                    position, position_diagonal = ImageUtils.draw_box(
                        marked_image,
                        shifted_position,
                        bubble_dimensions,
                        color=CLR_WHITE,
                        style="BOX_FILLED",
                        # TODO: pass verdict_color here and insert symbol mapping here ( +, -, *)
                        thickness_factor=1 / 12,
                    )

                    ImageUtils.draw_symbol(
                        marked_image,
                        "+",
                        position,
                        position_diagonal,
                    )
                else:
                    position, position_diagonal = ImageUtils.draw_box(
                        marked_image,
                        shifted_position,
                        bubble_dimensions,
                        color=verdict_colors["correct"],
                        style="BOX_FILLED",
                        # TODO: pass verdict_color here and insert symbol mapping here ( +, -, *)
                        thickness_factor=1 / 12,
                    )

                    ImageUtils.draw_symbol(
                        marked_image,
                        "+",
                        position,
                        position_diagonal,
                        color=verdict_symbol_colors["positive"],
                    )

            # TODO: take config for CROSS_TICKS vs BUBBLE_BOUNDARY and call appropriate util
            if (
                bubble_detection.is_marked
                or question_meta["bonus_type"] == "BONUS_FOR_ALL"
            ):
                symbol, verdict_color, verdict_symbol_color =  get_evaluation_meta_for_question(
                    question_meta, verdict_colors, verdict_symbol_colors
                )

                if image_type == "GRAYSCALE":
                    position, position_diagonal = ImageUtils.draw_box(
                        marked_image,
                        shifted_position,
                        bubble_dimensions,
                        color=CLR_WHITE,
                        style="BOX_FILLED",
                        thickness_factor=1 / 12,
                    )

                    ImageUtils.draw_symbol(
                        marked_image,
                        symbol,
                        position,
                        position_diagonal,
                    )
                else:
                    position, position_diagonal = ImageUtils.draw_box(
                        marked_image,
                        shifted_position,
                        bubble_dimensions,
                        color=verdict_color,
                        style="BOX_FILLED",
                        thickness_factor=1 / 12,
                    )

                    ImageUtils.draw_symbol(
                        marked_image,
                        symbol,
                        position,
                        position_diagonal,
                        color=verdict_symbol_color,
                    )
         
                if (
                    bubble_detection.is_marked
                    and evaluation_config.draw_detected_bubble_texts["enabled"]
                ):
                    ImageUtils.draw_text(
                        marked_image,
                        field_value,
                        shifted_position,
                        text_size=TEXT_SIZE,
                        color=(20, 20, 10),
                        thickness=int(1 + 3.5 * TEXT_SIZE),
                    )

            else:
                ImageUtils.draw_box(
                    marked_image,
                    shifted_position,
                    bubble_dimensions,
                    style="BOX_HOLLOW",
                    thickness_factor=1 / 10,
                )
        if (
            evaluation_config.draw_answer_groups["enabled"]
            and question_meta["answer_type"] != AnswerType.STANDARD
        ):
            self.draw_answer_groups(
                marked_image,
                image_type,
                question_meta,
                field_bubble_means,
                field_block,
                evaluation_config,
            )

    def get_matched_answer_groups(self, question_meta, field_value):
        occurences = []
        answer_type, answer_item = map(
            question_meta.get, ["answer_type", "answer_item"]
        )
        if answer_type == AnswerType.MULTIPLE_CORRECT:
            for i in range(0, len(answer_item)):
                if field_value in answer_item[i]:
                    occurences.append(i)
        else:
            for i in range(0, len(answer_item)):
                if (
                    field_value in answer_item[i][0]
                    and answer_item[i][1] > 0
                ):
                    occurences.append(i)
        return occurences

    def draw_answer_groups(
        self,
        marked_image,
        image_type,
        question_meta,
        field_bubble_means,
        field_block,
        evaluation_config,
    ):
        bubble_dimensions = field_block.bubble_dimensions
        draw_pos = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
        color_sequence = evaluation_config.draw_answer_groups["color_sequence"]
        if image_type == "GRAYSCALE":
            color_sequence = [CLR_WHITE] * len(color_sequence)
        for bubble_detection in field_bubble_means:
            bubble = bubble_detection.item_reference
            shifted_position = tuple(bubble.get_shifted_position(field_block.shifts))
            field_value = str(bubble.field_value)
            occurence = self.get_matched_answer_groups(question_meta, field_value)
            for pos in occurence:
                state = draw_pos[pos]
                color = color_sequence[pos]
                ImageUtils.draw_group(
                    marked_image, shifted_position, bubble_dimensions, state, color, GROUP_THICKNESS
                )

    def draw_evaluation_summary(self, marked_image, evaluation_meta, evaluation_config):
        if evaluation_config.draw_answers_summary["enabled"]:
            self.draw_answers_summary(
                marked_image, evaluation_config, evaluation_meta["score"]
            )

        if evaluation_config.draw_score["enabled"]:
            self.draw_score(marked_image, evaluation_config, evaluation_meta["score"])
        return marked_image

    def draw_answers_summary(self, marked_image, evaluation_config, score):
        (
            formatted_answers_summary,
            position,
            size,
            thickness,
        ) = evaluation_config.get_formatted_answers_summary()
        ImageUtils.draw_text(
            marked_image,
            formatted_answers_summary,
            position,
            text_size=size,
            thickness=thickness,
        )

    def draw_score(self, marked_image, evaluation_config, score):
        (
            formatted_score,
            position,
            size,
            thickness,
        ) = evaluation_config.get_formatted_score(score)
        ImageUtils.draw_text(
            marked_image, formatted_score, position, text_size=size, thickness=thickness
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
        Note: Cannot assume qStrip has only-gray or only-white bg
            (in which case there is only one jump).
        So there will be either 1 or 2 jumps.
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

        The abstract "First LARGE GAP" is perfect for this.
        Current code is considering ONLY TOP 2 jumps(>= MIN_GAP) to be big,
            gives the smaller one (looseness factor)

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

        # NOTE: thr2 is deprecated, thus is JUMP_DELTA
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
        # global_threshold_for_template = min(thr1,thr2)
        global_threshold_for_template, j_low, j_high = (
            thr1,
            thr1 - max1 // 2,
            thr1 + max1 // 2,
        )

        # TODO: maybe use plot_create flag when using plots in append_save_image
        if plot_show:
            _, ax = pyplot.subplots()
            # TODO: move into individual utils
            plot_means_and_refs = (
                sorted_bubble_means_and_refs if sort_in_plot else bubble_means_and_refs
            )
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
            # logger.info(list(zip(original_bin_names, shuffled_color_indices)))
            plot_colors = plot_color_sampler(
                [shuffled_color_indices[i] for i in unique_label_indices]
            )
            # plot_colors = plot_color_sampler(unique_label_indices)
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

        return global_threshold_for_template, j_low, j_high

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
        //No more - Assumption : Colwise background color is uniformly gray or white,
                but not alternating. In this case there is atmost one jump.

        0 Jump :
                        <-- safe THR?
            .......
            ...|||||||
            ||||||||||  <-- safe THR?
        // How to decide given range is above or below gray?
            -> global bubble_means_list shall absolutely help here. Just run same function
                on total bubble_means_list instead of colwise _//
        How to decide it is this case of 0 jumps

        1 Jump :
                ......
                ||||||
                ||||||  <-- risky THR
                ||||||  <-- safe THR
            ....||||||
            ||||||||||

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
                < config.thresholding.MIN_GAP
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
            # print(field_label,sorted_bubble_means,max1)

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
        return thr1, max1

    def append_save_image(self, key, img):
        if self.save_image_level >= int(key):
            self.save_img_list[key].append(img.copy())

    def save_image_stacks(self, key, filename, save_marked_dir):
        config = self.tuning_config
        if self.save_image_level >= int(key) and self.save_img_list[key] != []:
            display_height, display_width = config.outputs.display_image_dimensions
            name = os.path.splitext(filename)[0]
            result = np.hstack(
                tuple(
                    [
                        ImageUtils.resize_util(img, u_height=display_height)
                        for img in self.save_img_list[key]
                    ]
                )
            )
            result = ImageUtils.resize_util(
                result,
                min(
                    len(self.save_img_list[key]) * display_width // 3,
                    int(display_width * 2.5),
                ),
            )
            ImageUtils.save_img(
                f"{save_marked_dir}stack/{name}_{str(key)}_stack.jpg", result
            )

    def reset_all_save_img(self):
        for i in range(self.save_image_level):
            self.save_img_list[i + 1] = []
