"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

import os
from csv import QUOTE_NONNUMERIC
from pathlib import Path
from time import time

import cv2
import numpy as np
import pandas as pd

from src import constants
from src.core import ImageInstanceOps
from src.defaults import CONFIG_DEFAULTS, EVALUATION_DEFAULTS
from src.logger import logger
from src.processors.manager import ProcessorManager
from src.template import Template
from src.utils.file import Paths, setup_dirs_for_paths, setup_outputs_for_template
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils, Stats
from src.utils.parsing import (
    evaluate_concatenated_response,
    get_concatenated_response,
    open_config_with_defaults,
    open_evaluation_with_defaults,
)

# Load processors
PROCESSOR_MANAGER = ProcessorManager()
STATS = Stats()


def entry_point(root_dir, curr_dir, args):
    return process_dir(root_dir, curr_dir, args)


# TODO: make this function pure
def process_dir(
    root_dir,
    curr_dir,
    args,
    template=None,
    tuning_config=CONFIG_DEFAULTS,
    evaluation_config=EVALUATION_DEFAULTS,
    image_instance_ops=None,
):
    # Update local tuning_config (in current recursion stack)
    local_config_path = curr_dir.joinpath(constants.CONFIG_FILENAME)
    if os.path.exists(local_config_path):
        tuning_config = open_config_with_defaults(local_config_path)

    # Update local template (in current recursion stack)
    local_template_path = curr_dir.joinpath(constants.TEMPLATE_FILENAME)
    if os.path.exists(local_template_path):
        # todo: consider moving template inside image_instance_ops as an attribute
        image_instance_ops = ImageInstanceOps(tuning_config)
        template = Template(
            local_template_path,
            image_instance_ops,
            PROCESSOR_MANAGER.processors,
        )

    evaluation_config = {}
    local_evaluation_path = curr_dir.joinpath(constants.EVALUATION_FILENAME)
    if os.path.exists(local_evaluation_path):
        evaluation_config = open_evaluation_with_defaults(local_evaluation_path)

    # Look for subdirectories for processing
    subdirs = [d for d in curr_dir.iterdir() if d.is_dir()]

    paths = Paths(Path(args["output_dir"], curr_dir.relative_to(root_dir)))

    # look for images in current dir to process
    exts = ("*.png", "*.jpg")
    omr_files = sorted([f for ext in exts for f in curr_dir.glob(ext)])

    # Exclude images (take union over all pre_processors)
    excluded_files = []
    if template:
        for pp in template.pre_processors:
            excluded_files.extend(Path(p) for p in pp.exclude_files())

    omr_files = [f for f in omr_files if f not in excluded_files]

    if omr_files:
        if not template:
            logger.error(
                f'Found images, but no template in the directory tree \
                of "{curr_dir}". \nPlace {constants.TEMPLATE_FILENAME} in the \
                directory or specify a template using -t.'
            )
            return

        # TODO: get rid of args here
        args_local = args.copy()
        if "OverrideFlags" in template.options:
            args_local.update(template.options["OverrideFlags"])
        logger.info(
            "------------------------------------------------------------------"
        )
        logger.info(f'Processing directory "{curr_dir}" with settings- ')
        logger.info(f"\t{'Total images':<22}: {len(omr_files)}")
        logger.info(
            f"\t{'Cropping Enabled':<22}: {str('CropOnMarkers' in template.pre_processors)}"
        )
        logger.info(f"\t{'Auto Alignment':<22}: {str(args_local['autoAlign'])}")
        logger.info(f"\t{'Using Template':<22}: { str(template)}")
        logger.info(
            f"\t{'Using pre-processors':<22}: {[pp.__class__.__name__ for pp in template.pre_processors]}"
        )
        logger.info("")

        setup_dirs_for_paths(paths)
        outputs_namespace = setup_outputs_for_template(paths, template)

        process_files(
            omr_files,
            template,
            tuning_config,
            evaluation_config,
            args_local,
            outputs_namespace,
            image_instance_ops,
        )

    elif not subdirs:
        # Each subdirectory should have images or should be non-leaf
        logger.info(
            f"Note: No valid images or sub-folders found in {curr_dir}.\
            Empty directories not allowed."
        )

    # recursively process sub-folders
    for d in subdirs:
        process_dir(
            root_dir,
            d,
            args,
            template,
            tuning_config,
            evaluation_config,
            image_instance_ops,
        )


# TODO: take a look at 'outputs_namespace.paths'
def process_files(
    omr_files,
    template,
    tuning_config,
    evaluation_config,
    args,
    outputs_namespace,
    image_instance_ops,
):
    start_time = int(time())
    files_counter = 0
    STATS.files_not_moved = 0

    for file_path in omr_files:
        files_counter += 1

        file_name = file_path.name
        args["current_file"] = file_path

        in_omr = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        logger.info("")
        logger.info(
            f"({files_counter}) Opening image: \t{file_path}\tResolution: {in_omr.shape}"
        )

        # TODO: Get rid of saveImgList
        for i in range(image_instance_ops.save_image_level):
            image_instance_ops.reset_save_img(i + 1)

        image_instance_ops.append_save_img(1, in_omr)

        # resize to conform to template
        in_omr = ImageUtils.resize_util(
            in_omr,
            tuning_config.dimensions.processing_width,
            tuning_config.dimensions.processing_height,
        )

        # run pre_processors in sequence
        for pre_processor in template.pre_processors:
            in_omr = pre_processor.apply_filter(in_omr, args)

        if in_omr is None:
            # Error OMR case
            new_file_path = outputs_namespace.paths.errors_dir + file_name
            outputs_namespace.OUTPUT_SET.append(
                [file_name] + outputs_namespace.empty_resp
            )
            if check_and_move(
                constants.ERROR_CODES.NO_MARKER_ERR, file_path, new_file_path
            ):
                err_line = [
                    file_name,
                    file_path,
                    new_file_path,
                    "NA",
                ] + outputs_namespace.empty_resp
                pd.DataFrame(err_line, dtype=str).T.to_csv(
                    outputs_namespace.files_obj["Errors"],
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            continue

        if args["setLayout"]:
            template_layout = image_instance_ops.draw_template_layout(
                in_omr, template, shifted=False, border=2
            )
            InteractionUtils.show(
                "Template Layout", template_layout, 1, 1, config=tuning_config
            )
            continue

        # uniquify
        file_id = str(file_name)
        save_dir = outputs_namespace.paths.save_marked_dir
        (
            response_dict,
            final_marked,
            multi_marked,
            _,
        ) = image_instance_ops.read_omr_response(
            template,
            image=in_omr,
            name=file_id,
            save_dir=save_dir,
            auto_align=args["autoAlign"],
        )

        # concatenate roll nos, set unmarked responses, etc
        omr_response = get_concatenated_response(response_dict, template)
        logger.info(f"Read Response: \n{omr_response}")
        if tuning_config.outputs.show_image_level >= 2:
            InteractionUtils.show(
                f"Final Marked Bubbles : {file_id}",
                ImageUtils.resize_util_h(
                    final_marked, int(tuning_config.dimensions.display_height * 1.3)
                ),
                1,
                1,
                config=tuning_config,
            )

        # This evaluates and returns the score attribute
        score = evaluate_concatenated_response(omr_response, evaluation_config)

        resp_array = []
        for k in outputs_namespace.resp_cols:
            resp_array.append(omr_response[k])

        outputs_namespace.OUTPUT_SET.append([file_name] + resp_array)

        # TODO: Add roll number validation here
        if multi_marked == 0:
            STATS.files_not_moved += 1
            new_file_path = save_dir + file_id
            # Enter into Results sheet-
            results_line = [file_name, file_path, new_file_path, score] + resp_array
            # Write/Append to results_line file(opened in append mode)
            pd.DataFrame(results_line, dtype=str).T.to_csv(
                outputs_namespace.files_obj["Results"],
                mode="a",
                quoting=QUOTE_NONNUMERIC,
                header=False,
                index=False,
            )
            # Todo: Add score calculation/explanations
            # print(f"[/{files_counter}] Graded with score: {round(score, 2)}\t file_id: {file_id}")
            # print(files_counter,file_id,omr_response['Roll'],'score : ',score)
        else:
            # multi_marked file
            logger.info(f"[{files_counter}] Found multi-marked file: {file_id}")
            new_file_path = outputs_namespace.paths.multi_marked_dir + file_name
            if check_and_move(
                constants.ERROR_CODES.MULTI_BUBBLE_WARN, file_path, new_file_path
            ):
                mm_line = [file_name, file_path, new_file_path, "NA"] + resp_array
                pd.DataFrame(mm_line, dtype=str).T.to_csv(
                    outputs_namespace.files_obj["MultiMarked"],
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            # else:
            #     TODO:  Add appropriate record handling here
            #     pass

    print_stats(start_time, files_counter, tuning_config)

    # flush after every 20 files for a live view
    # if(files_counter % 20 == 0 or files_counter == len(omr_files)):
    #     for file_key in outputs_namespace.filesMap.keys():
    #         outputs_namespace.files_obj[file_key].flush()


def print_stats(start_time, files_counter, tuning_config):
    time_checking = round(time() - start_time, 2) if files_counter else 1
    log = logger.info
    log("")
    log(f"{'Total file(s) moved':<27}: {STATS.files_moved}")
    log(f"{'Total file(s) not moved':<27}: {STATS.files_not_moved}")
    log("--------------------------------")
    log(
        f"{'Total file(s) processed':<27}: {files_counter} ({'Sum Tallied!' if files_counter == (STATS.files_moved + STATS.files_not_moved) else 'Not Tallying!'})"
    )

    if tuning_config.outputs.show_image_level <= 0:
        log(
            f"\nFinished Checking {files_counter} file(s) in {round(time_checking, 1)} seconds i.e. ~{round(time_checking/60, 1)} minute(s)."
        )
        log(
            f"{'OMR Processing Rate':<27}:\t ~ {round(time_checking/files_counter,2)} seconds/OMR"
        )
        log(
            f"{'OMR Processing Speed':<27}:\t ~ {round((files_counter * 60) / time_checking, 2)} OMRs/minute"
        )
    else:
        log(f"\n{'Total script time':<27}: {time_checking} seconds")

    if tuning_config.outputs.show_image_level <= 1:
        log(
            "\nTip: To see some awesome visuals, open config.json and increase 'show_image_level'"
        )

    # evaluate_correctness(outputs_namespace)

    # Use this data to train as +ve feedback
    # if config.outputs.show_image_level >= 0 and files_counter > 10:
    #     for x in [thresholdCircles]:#,badThresholds,veryBadPoints, mws, mbs]:
    #         if(x != []):
    #             x = pd.DataFrame(x)
    #             print(x.describe())
    #             plt.plot(range(len(x)), x)
    #             plt.title("Mystery Plot")
    #             plt.show()
    #         else:
    #             print(x)


# Evaluate accuracy based on OMRDataset file generated through review
# portal on the same set of images
def evaluate_correctness(outputs_namespace):
    # TODO: test_file WOULD BE RELATIVE TO INPUT SUBDIRECTORY NOW-
    test_file = "inputs/OMRDataset.csv"
    if os.path.exists(test_file):
        logger.info(f"Starting evaluation for: '{test_file}'")

        test_cols = ["file_id"] + outputs_namespace.resp_cols
        y_df = (
            pd.read_csv(test_file, dtype=str)[test_cols]
            .replace(np.nan, "", regex=True)
            .set_index("file_id")
        )

        if np.any(y_df.index.duplicated):
            y_df_filtered = y_df.loc[~y_df.index.duplicated(keep="first")]
            logger.warning(
                f"WARNING: Found duplicate File-ids in file '{test_file}'. Removed {y_df.shape[0] - y_df_filtered.shape[0]} rows from testing data. Rows remaining: {y_df_filtered.shape[0]}"
            )
            y_df = y_df_filtered

        x_df = pd.DataFrame(
            outputs_namespace.OUTPUT_SET, dtype=str, columns=test_cols
        ).set_index("file_id")
        # print("x_df",x_df.head())
        # print("\ny_df",y_df.head())
        intersection = y_df.index.intersection(x_df.index)

        # Checking if the merge is okay
        if intersection.size == x_df.index.size:
            y_df = y_df.loc[intersection]
            x_df["TestResult"] = (x_df == y_df).all(axis=1).astype(int)
            logger.info(x_df.head())
            logger.info(
                f"\n\t Accuracy on the {test_file} Dataset: {round((x_df['TestResult'].sum() / x_df.shape[0]),6)}"
            )
        else:
            logger.error(
                "\nERROR: Insufficient Testing Data: Have you appended MultiMarked data yet?"
            )
            logger.error(
                f"Missing File-ids: {list(x_df.index.difference(intersection))}"
            )


def check_and_move(error_code, file_path, filepath2):
    # print("Dummy Move:  "+file_path, " --> ",filepath2)

    # TODO: fix file movement into error/multimarked/invalid etc again
    STATS.files_not_moved += 1
    return True

    if not os.path.exists(file_path):
        logger.warning(f"File already moved: {file_path}")
        return False
    if os.path.exists(filepath2):
        logger.error(f"ERROR {error_code}: Duplicate file at {filepath2}")
        return False

    logger.info(f"Moved: {file_path} --> {filepath2}")
    os.rename(file_path, filepath2)
    STATS.files_moved += 1
    return True


def report(status, streak, scheme, q_no, marked, ans, prev_marks, curr_marks, marks):
    logger.info(
        f"{q_no}\t {status}\t {str(streak)}\t [{scheme}] \t {str(prev_marks)} + {str(curr_marks)} = {str(marks)}\t {str(marked)}\t {str(ans)}"
    )


# TODO: Refactor into new process flow.
def preliminary_check():
    # filesCounter=0
    # mws, mbs = [],[]
    # # PRELIM_CHECKS for thresholding
    # if(config.PRELIM_CHECKS):
    #     # TODO: add more using unit testing
    #     TEMPLATE = TEMPLATES["H"]
    #     ALL_WHITE = 255 * np.ones((TEMPLATE.dimensions[1],TEMPLATE.dimensions[0]), dtype='uint8')
    #     response_dict, final_marked, multi_marked, multiroll = read_response(
    #         "H", ALL_WHITE, name="ALL_WHITE", save_dir=None, autoAlign=True
    #     )
    #     print("ALL_WHITE",response_dict)
    #     if(response_dict!={}):
    #         print("Preliminary Checks Failed.")
    #         exit(2)
    #     ALL_BLACK = np.zeros((TEMPLATE.dimensions[1],TEMPLATE.dimensions[0]), dtype='uint8')
    #     response_dict, final_marked, multi_marked, multiroll = read_response(
    #      "H", ALL_BLACK, name="ALL_BLACK", save_dir=None, autoAlign=True
    #     )
    #     print("ALL_BLACK",response_dict)
    #     show("Confirm : All bubbles are black",final_marked,1,1)
    pass
