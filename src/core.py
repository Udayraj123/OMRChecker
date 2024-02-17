import math
import os
import random
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

import src.constants as constants
from src.logger import logger
from src.utils.image import ImageUtils


class ImageInstanceOps:
    """Class to hold fine-tuned utilities for a group of images. One instance for each processing directory."""

    save_img_list: Any = defaultdict(list)

    def __init__(self, tuning_config):
        super().__init__()
        self.tuning_config = tuning_config
        self.save_image_level = tuning_config.outputs.save_image_level

    def apply_preprocessors(self, file_path, in_omr, template):
        tuning_config = self.tuning_config
        # resize to conform to common preprocessor input requirements
        in_omr = ImageUtils.resize_util(
            in_omr,
            tuning_config.dimensions.processing_width,
            tuning_config.dimensions.processing_height,
        )
        # Copy template for this instance op
        template = deepcopy(template)
        # run pre_processors in sequence
        for pre_processor in template.pre_processors:
            result = pre_processor.apply_filter(in_omr, template, file_path)
            out_omr, next_template = (
                result if type(result) is tuple else (result, template)
            )
            # resize the image if its shape is changed by the filter
            if out_omr.shape[:2] != in_omr.shape[:2]:
                out_omr = ImageUtils.resize_util(
                    out_omr,
                    tuning_config.dimensions.processing_width,
                    tuning_config.dimensions.processing_height,
                )
            in_omr = out_omr
            template = next_template
        return in_omr, template

    def read_omr_response(self, template, image, name, save_dir=None):
        config = self.tuning_config

        img = image.copy()
        # origDim = img.shape[:2]
        img = ImageUtils.resize_util(
            img, template.page_dimensions[0], template.page_dimensions[1]
        )
        if img.max() > img.min():
            img = ImageUtils.normalize_util(img)
        # Processing copies
        transp_layer = img.copy()
        final_marked = img.copy()

        # Move them to data class if needed
        # Overlay Transparencies
        alpha = 0.65
        omr_response = {}
        multi_marked, multi_roll = 0, 0

        # TODO Make this part useful for visualizing status checks
        # blackVals=[0]
        # whiteVals=[255]

        if config.outputs.show_image_level >= 5:
            all_c_box_vals = {"int": [], "mcq": []}
            # TODO: simplify this logic
            q_nums = {"int": [], "mcq": []}

        # Get mean bubbleValues n other stats
        global_bubble_means_and_refs, all_q_strip_arrs, global_std_vals_and_refs = (
            [],
            [],
            [],
        )
        total_q_strip_no = 0
        for field_block in template.field_blocks:
            field_block_std_vals_and_refs = []
            box_w, box_h = field_block.bubble_dimensions
            for field_block_bubbles in field_block.traverse_bubbles:
                q_strip_bubbles = []
                q_strip_bubble_means = []
                for field_bubble in field_block_bubbles:
                    # TODO: move this responsibility into the plugin(not pre-processor) (of shifting every point)
                    # shifted
                    x, y = (
                        field_bubble.x + field_block.shift_x,
                        field_bubble.y + field_block.shift_y,
                    )
                    rect = [y, y + box_h, x, x + box_w]
                    bubble_mean = cv2.mean(img[rect[0] : rect[1], rect[2] : rect[3]])[0]
                    q_strip_bubble_means.append(bubble_mean)
                    q_strip_bubbles.append(
                        [bubble_mean, field_bubble]
                        # TODO: cross/check mark detection support (#167)
                        # detectCross(img, rect) ? 0 : 255
                    )
                field_block_std = round(np.std(q_strip_bubble_means), 2)
                field_block_std_vals_and_refs.append([field_block_std, field_block])
                all_q_strip_arrs.append(q_strip_bubble_means)
                # _, _, _ = get_global_threshold(q_strip_bubbles, "QStrip Plot",
                #   plot_show=False, sort_in_plot=True)
                # hist = getPlotImg()
                # InteractionUtils.show("QStrip "+field_block_bubbles[0].field_label, hist, 0, 1,config=config)
                global_bubble_means_and_refs.extend(q_strip_bubbles)
                # print(total_q_strip_no, field_block_bubbles[0].field_label, field_block_std_vals_and_refs[len(field_block_std_vals_and_refs)-1])
                total_q_strip_no += 1
            global_std_vals_and_refs.extend(field_block_std_vals_and_refs)

        (
            PAGE_TYPE_FOR_THRESHOLD,
            GLOBAL_PAGE_THRESHOLD_WHITE,
            GLOBAL_PAGE_THRESHOLD_BLACK,
            MIN_JUMP,
            JUMP_DELTA,
        ) = map(
            config.threshold_params.get,
            [
                "PAGE_TYPE_FOR_THRESHOLD",
                "GLOBAL_PAGE_THRESHOLD_WHITE",
                "GLOBAL_PAGE_THRESHOLD_BLACK",
                "MIN_JUMP",
                "JUMP_DELTA",
            ],
        )
        global_default_threshold = (
            GLOBAL_PAGE_THRESHOLD_WHITE
            if PAGE_TYPE_FOR_THRESHOLD == "white"
            else GLOBAL_PAGE_THRESHOLD_BLACK
        )
        # TODO: see if this is needed, then take from config.json
        global_default_std_threshold = 10

        MIN_JUMP_STD = 30
        JUMP_DELTA_STD = 10
        global_std_thresh, _, _ = self.get_global_threshold(
            global_std_vals_and_refs,
            global_default_std_threshold,
            MIN_JUMP=MIN_JUMP_STD,
            JUMP_DELTA=JUMP_DELTA_STD,
            plot_title="Q-wise Std-dev Plot",
            plot_show=True,
            sort_in_plot=True,
        )
        # plt.show()
        # hist = getPlotImg()
        # InteractionUtils.show("StdHist", hist, 0, 1,config=config)

        # Note: Plotting takes Significant times here --> Change Plotting args
        # to support show_image_level
        global_thr, _, _ = self.get_global_threshold(
            global_bubble_means_and_refs,  # , looseness=4
            global_default_threshold,
            plot_title="Mean Intensity Histogram",
            MIN_JUMP=MIN_JUMP,
            JUMP_DELTA=JUMP_DELTA,
            plot_show=True,
            sort_in_plot=True,
            looseness=4,
        )

        logger.info(
            f"Thresholding:\tglobal_thr: {round(global_thr, 2)} \tglobal_std_THR: {round(global_std_thresh, 2)}\t{'(Looks like a Xeroxed OMR)' if (global_thr == 255) else ''}"
        )
        # plt.show()
        # hist = getPlotImg()
        # InteractionUtils.show("StdHist", hist, 0, 1,config=config)

        # if(config.outputs.show_image_level>=1):
        #     hist = getPlotImg()
        #     InteractionUtils.show("Hist", hist, 0, 1,config=config)
        #     appendSaveImg(4,hist)
        #     appendSaveImg(5,hist)
        #     appendSaveImg(2,hist)

        per_omr_threshold_avg, total_q_strip_no, total_q_box_no = 0, 0, 0
        for field_block in template.field_blocks:
            block_q_strip_no = 1
            key = field_block.name[:3]
            box_w, box_h = field_block.bubble_dimensions
            for field_block_bubbles in field_block.traverse_bubbles:
                # All Black or All White case
                no_outliers = (
                    global_std_vals_and_refs[total_q_strip_no][0] < global_std_thresh
                )
                # print(total_q_strip_no, field_block_bubbles[0].field_label,
                #   global_std_vals_and_refs[total_q_strip_no][0], "no_outliers:", no_outliers)
                per_q_strip_threshold = self.get_local_threshold(
                    all_q_strip_arrs[total_q_strip_no],
                    global_thr,
                    no_outliers,
                    f"Mean Intensity Histogram for {key}.{field_block_bubbles[0].field_label}.{block_q_strip_no}",
                    field_block_bubbles[0].field_label
                    in ["q72", "q52", "roll5"],  # Temp
                    # config.outputs.show_image_level >= 6,
                )
                # print(field_block_bubbles[0].field_label,key,block_q_strip_no, "THR: ",
                #   round(per_q_strip_threshold,2))
                per_omr_threshold_avg += per_q_strip_threshold

                # TODO: get rid of total_q_box_no
                detected_bubbles = []
                for bubble in field_block_bubbles:
                    bubble_is_marked = (
                        per_q_strip_threshold
                        > global_bubble_means_and_refs[total_q_box_no][0]
                    )
                    total_q_box_no += 1

                    x, y, field_value = (
                        bubble.x + field_block.shift_x,
                        bubble.y + field_block.shift_y,
                        bubble.field_value,
                    )
                    if bubble_is_marked:
                        detected_bubbles.append(bubble)
                        # Draw the shifted box
                        cv2.rectangle(
                            final_marked,
                            (int(x + box_w / 12), int(y + box_h / 12)),
                            (
                                int(x + box_w - box_w / 12),
                                int(y + box_h - box_h / 12),
                            ),
                            constants.CLR_DARK_GRAY,
                            3,
                        )

                        cv2.putText(
                            final_marked,
                            str(field_value),
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            constants.TEXT_SIZE,
                            (20, 20, 10),
                            int(1 + 3.5 * constants.TEXT_SIZE),
                        )
                    else:
                        cv2.rectangle(
                            final_marked,
                            (int(x + box_w / 10), int(y + box_h / 10)),
                            (
                                int(x + box_w - box_w / 10),
                                int(y + box_h - box_h / 10),
                            ),
                            constants.CLR_GRAY,
                            -1,
                        )

                for bubble in detected_bubbles:
                    field_label, field_value = (
                        bubble.field_label,
                        bubble.field_value,
                    )
                    # Only send rolls multi-marked in the directory
                    multi_marked_local = field_label in omr_response
                    omr_response[field_label] = (
                        (omr_response[field_label] + field_value)
                        if multi_marked_local
                        else field_value
                    )
                    # TODO: generalize this into identifier
                    # multi_roll = multi_marked_local and "Roll" in str(q)
                    multi_marked = multi_marked or multi_marked_local

                if len(detected_bubbles) == 0:
                    field_label = field_block_bubbles[0].field_label
                    omr_response[field_label] = field_block.empty_val

                if config.outputs.show_image_level >= 5:
                    if key in all_c_box_vals:
                        q_nums[key].append(f"{key[:2]}_c{str(block_q_strip_no)}")
                        all_c_box_vals[key].append(all_q_strip_arrs[total_q_strip_no])

                block_q_strip_no += 1
                total_q_strip_no += 1
            # /for field_block

        overall_confidence, fields_confidence = self.get_confidence_metrics()

        underconfident_fields = filter(lambda x: x.confidence < 0.8, fields_confidence)

        # TODO: Make the plot for underconfident_fields

        per_omr_threshold_avg /= total_q_strip_no
        per_omr_threshold_avg = round(per_omr_threshold_avg, 2)
        # Translucent
        cv2.addWeighted(final_marked, alpha, transp_layer, 1 - alpha, 0, final_marked)
        # Box types
        if config.outputs.show_image_level >= 5:
            # plt.draw()
            f, axes = plt.subplots(len(all_c_box_vals), sharey=True)
            f.canvas.manager.set_window_title(
                f"Bubble Intensity by question type for {name}"
            )
            ctr = 0
            type_name = {
                "int": "Integer",
                "mcq": "MCQ",
                "med": "MED",
                "rol": "Roll",
            }
            for k, boxvals in all_c_box_vals.items():
                axes[ctr].title.set_text(type_name[k] + " Type")
                axes[ctr].boxplot(boxvals)
                # thrline=axes[ctr].axhline(per_omr_threshold_avg,color='red',ls='--')
                # thrline.set_label("Average THR")
                axes[ctr].set_ylabel("Intensity")
                axes[ctr].set_xticklabels(q_nums[k])
                # axes[ctr].legend()
                ctr += 1
            # imshow will do the waiting
            plt.tight_layout(pad=0.5)
            plt.show()

        if config.outputs.save_detections and save_dir is not None:
            if multi_roll:
                save_dir = save_dir.joinpath("_MULTI_")
            image_path = str(save_dir.joinpath(name))
            ImageUtils.save_img(image_path, final_marked)

        self.append_save_img(2, final_marked)

        if save_dir is not None:
            for i in range(config.outputs.save_image_level):
                self.save_image_stacks(i + 1, name, save_dir)

        return omr_response, final_marked, multi_marked, multi_roll

    @staticmethod
    def draw_template_layout(img, template, shifted=True, draw_qvals=False, border=-1):
        img = ImageUtils.resize_util(
            img, template.page_dimensions[0], template.page_dimensions[1]
        )
        final_align = img.copy()
        for field_block in template.field_blocks:
            field_block_name, s, d, bubble_dimensions, shift_x, shift_y = map(
                lambda attr: getattr(field_block, attr),
                [
                    "name",
                    "origin",
                    "dimensions",
                    "bubble_dimensions",
                    "shift_x",
                    "shift_y",
                ],
            )
            box_w, box_h = bubble_dimensions

            if shifted:
                cv2.rectangle(
                    final_align,
                    (s[0] + shift_x, s[1] + shift_y),
                    (s[0] + shift_x + d[0], s[1] + shift_y + d[1]),
                    constants.CLR_BLACK,
                    3,
                )
            else:
                cv2.rectangle(
                    final_align,
                    (s[0], s[1]),
                    (s[0] + d[0], s[1] + d[1]),
                    constants.CLR_BLACK,
                    3,
                )
            for field_block_bubbles in field_block.traverse_bubbles:
                for field_bubble in field_block_bubbles:
                    x, y = (
                        (field_bubble.x + shift_x, field_bubble.y + shift_y)
                        if shifted
                        else (field_bubble.x, field_bubble.y)
                    )
                    cv2.rectangle(
                        final_align,
                        (int(x + box_w / 10), int(y + box_h / 10)),
                        (int(x + box_w - box_w / 10), int(y + box_h - box_h / 10)),
                        constants.CLR_GRAY,
                        border,
                    )
                    if draw_qvals:
                        rect = [y, y + box_h, x, x + box_w]
                        cv2.putText(
                            final_align,
                            f"{int(cv2.mean(img[rect[0] : rect[1], rect[2] : rect[3]])[0])}",
                            (rect[2] + 2, rect[0] + (box_h * 2) // 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            constants.CLR_BLACK,
                            2,
                        )
            if shifted:
                text_in_px = cv2.getTextSize(
                    field_block_name, cv2.FONT_HERSHEY_SIMPLEX, constants.TEXT_SIZE, 4
                )
                cv2.putText(
                    final_align,
                    field_block_name,
                    (
                        int(s[0] + shift_x + d[0] - text_in_px[0][0]),
                        int(s[1] + shift_y - text_in_px[0][1]),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    constants.TEXT_SIZE,
                    constants.CLR_BLACK,
                    4,
                )

        return final_align

    def get_global_threshold(
        self,
        bubble_means_and_refs,
        global_default_threshold,
        MIN_JUMP,
        JUMP_DELTA,
        plot_title=None,
        plot_show=True,
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
        sorted_bubble_means_and_refs = sorted(bubble_means_and_refs, key=lambda x: x[0])

        # Find the FIRST LARGE GAP and set it as threshold:
        sorted_bubble_means = [item[0] for item in sorted_bubble_means_and_refs]
        ls = (looseness + 1) // 2
        l = len(sorted_bubble_means) - ls
        max1, thr1 = MIN_JUMP, global_default_threshold
        for i in range(ls, l):
            jump = sorted_bubble_means[i + ls] - sorted_bubble_means[i - ls]
            if jump > max1:
                max1 = jump
                thr1 = sorted_bubble_means[i - ls] + jump / 2

        # NOTE: thr2 is deprecated, thus is JUMP_DELTA
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
        # global_thr = min(thr1,thr2)
        global_thr, j_low, j_high = thr1, thr1 - max1 // 2, thr1 + max1 // 2

        # # For normal images
        # thresholdRead =  116
        # if(thr1 > thr2 and thr2 > thresholdRead):
        #     print("Note: taking safer thr line.")
        #     global_thr, j_low, j_high = thr2, thr2 - max2//2, thr2 + max2//2

        if plot_title:
            _, ax = plt.subplots()
            plot_means_and_refs = (
                sorted_bubble_means_and_refs if sort_in_plot else bubble_means_and_refs
            )
            plot_values = list(map(lambda x: x[0], plot_means_and_refs))
            original_bin_names = list(
                map(lambda x: x[1].plot_bin_name, plot_means_and_refs)
            )
            plot_labels = list(map(lambda x: x[1].name, plot_means_and_refs))
            sorted_unique_bin_names, unique_label_indices = np.unique(
                original_bin_names, return_inverse=True
            )

            plot_color_sampler = colormaps["Spectral"].resampled(
                len(sorted_unique_bin_names)
            )

            shuffled_color_indices = random.sample(
                list(unique_label_indices), len(unique_label_indices)
            )
            logger.info(list(zip(original_bin_names, shuffled_color_indices)))
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
            plt.ylim(
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
                            for t in re.split("(\d+)", s[1])
                        ],
                    )
                )
            )
            ax.set_title(plot_title)
            ax.axhline(global_thr, color="green", ls="--", linewidth=5).set_label(
                "Global Threshold"
            )
            ax.axhline(thr2, color="red", ls=":", linewidth=3).set_label("THR2 Line")
            # ax.axhline(j_low,color='red',ls='-.', linewidth=3)
            # ax.axhline(j_high,color='red',ls='-.', linewidth=3).set_label("Boundary Line")
            # ax.set_ylabel("Mean Intensity")
            ax.set_ylabel("Values")
            ax.set_xlabel("Position")

            if plot_show:
                plt.title(plot_title)
                plt.show()

        return global_thr, j_low, j_high

    def get_local_threshold(
        self,
        bubble_means_list,
        global_thr,
        no_outliers,
        plot_title=None,
        plot_show=True,
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
        sorted_bubble_means = sorted(bubble_means_list)

        # Small no of pts cases:
        # base case: 1 or 2 pts
        if len(sorted_bubble_means) < 3:
            thr1 = (
                global_thr
                if np.max(sorted_bubble_means) - np.min(sorted_bubble_means)
                < config.threshold_params.MIN_GAP
                else np.mean(sorted_bubble_means)
            )
        else:
            # qmin, qmax, qmean, qstd = round(np.min(sorted_bubble_means),2), round(np.max(sorted_bubble_means),2),
            #   round(np.mean(sorted_bubble_means),2), round(np.std(sorted_bubble_means),2)
            # GVals = [round(abs(q-qmean),2) for q in sorted_bubble_means]
            # gmean, gstd = round(np.mean(GVals),2), round(np.std(GVals),2)
            # # DISCRETION: Pretty critical factor in reading response
            # # Doesn't work well for small number of values.
            # DISCRETION = 2.7 # 2.59 was closest hit, 3.0 is too far
            # L2MaxGap = round(max([abs(g-gmean) for g in GVals]),2)
            # if(L2MaxGap > DISCRETION*gstd):
            #     no_outliers = False

            # # ^Stackoverflow method
            # print(field_label, no_outliers,"qstd",round(np.std(sorted_bubble_means),2), "gstd", gstd,
            #   "Gaps in gvals",sorted([round(abs(g-gmean),2) for g in GVals],reverse=True),
            #   '\t',round(DISCRETION*gstd,2), L2MaxGap)

            # else:
            # Find the LARGEST GAP and set it as threshold: //(FIRST LARGE GAP)
            l = len(sorted_bubble_means) - 1
            max1, thr1 = config.threshold_params.MIN_JUMP, 255
            for i in range(1, l):
                jump = sorted_bubble_means[i + 1] - sorted_bubble_means[i - 1]
                if jump > max1:
                    max1 = jump
                    thr1 = sorted_bubble_means[i - 1] + jump / 2
            # print(field_label,sorted_bubble_means,max1)

            confident_jump = (
                config.threshold_params.MIN_JUMP
                + config.threshold_params.CONFIDENT_SURPLUS
            )
            # If not confident, then only take help of global_thr
            if max1 < confident_jump:
                if no_outliers:
                    # All Black or All White case
                    thr1 = global_thr
                else:
                    # TODO: Low confidence parameters here
                    pass

            # if(thr1 == 255):
            #     print("Warning: threshold is unexpectedly 255! (Outlier Delta issue?)",plot_title)

        # Make a common plot function to show local and global thresholds
        if plot_show and plot_title is not None:
            _, ax = plt.subplots()
            ax.bar(range(len(sorted_bubble_means)), sorted_bubble_means)
            thrline = ax.axhline(thr1, color="green", ls=("-."), linewidth=3)
            thrline.set_label("Local Threshold")
            thrline = ax.axhline(global_thr, color="red", ls=":", linewidth=5)
            thrline.set_label("Global Threshold")
            ax.set_title(plot_title)
            ax.set_ylabel("Bubble Mean Intensity")
            ax.set_xlabel("Bubble Number(sorted)")
            ax.legend()
            # TODO append QStrip to this plot-
            # appendSaveImg(6,getPlotImg())
            if plot_show:
                plt.show()
        return thr1

    def append_save_img(self, key, img):
        if self.save_image_level >= int(key):
            self.save_img_list[key].append(img.copy())

    def save_image_stacks(self, key, filename, save_dir):
        config = self.tuning_config
        if self.save_image_level >= int(key) and self.save_img_list[key] != []:
            name = os.path.splitext(filename)[0]
            result = np.hstack(
                tuple(
                    [
                        ImageUtils.resize_util_h(img, config.dimensions.display_height)
                        for img in self.save_img_list[key]
                    ]
                )
            )
            result = ImageUtils.resize_util(
                result,
                min(
                    len(self.save_img_list[key]) * config.dimensions.display_width // 3,
                    int(config.dimensions.display_width * 2.5),
                ),
            )
            ImageUtils.save_img(f"{save_dir}stack/{name}_{str(key)}_stack.jpg", result)

    def reset_all_save_img(self):
        for i in range(self.save_image_level):
            self.save_img_list[i + 1] = []
