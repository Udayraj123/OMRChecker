"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

import argparse
import os
from csv import QUOTE_NONNUMERIC
from pathlib import Path
from time import localtime, strftime, time

import cv2
import numpy as np
import pandas as pd

from src import constants

# TODO: use open_config_with_defaults after making a Config class.
from src.config import CONFIG_DEFAULTS as config
from src.logger import logger

# TODO: further break utils down and separate the imports
from src.utils.imgutils import (
    ImageUtils,
    MainOperations,
    draw_template_layout,
    setup_dirs,
)

# Note: dot-imported paths are relative to current directory
from .processors.manager import ProcessorManager
from .template import Template

# import matplotlib.pyplot as plt


# Load processors
PROCESSOR_MANAGER = ProcessorManager()
STATS = constants.Stats()

# TODO(beginner task) :-
# from colorama import init
# init()
# from colorama import Fore, Back, Style


def entry_point(root_dir, curr_dir, args):
    return process_dir(root_dir, curr_dir, args)


# TODO: make this function pure
def process_dir(root_dir, curr_dir, args, template=None):

    # Update local template (in current recursion stack)
    local_template_path = curr_dir.joinpath(constants.TEMPLATE_FILENAME)
    if os.path.exists(local_template_path):
        template = Template(local_template_path, PROCESSOR_MANAGER.processors)

    # Look for subdirectories for processing
    subdirs = [d for d in curr_dir.iterdir() if d.is_dir()]

    paths = constants.Paths(Path(args["output_dir"], curr_dir.relative_to(root_dir)))

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
            "\n------------------------------------------------------------------"
        )
        logger.info(f'Processing directory "{curr_dir}" with settings- ')
        logger.info("\tTotal images       : %d" % (len(omr_files)))
        logger.info(
            "\tCropping Enabled   : " + str("CropOnMarkers" in template.pre_processors)
        )
        logger.info("\tAuto Alignment     : " + str(args_local["autoAlign"]))
        logger.info("\tUsing Template     : " + str(template))
        # Print options
        for pp in template.pre_processors:
            logger.info(f"\tUsing preprocessor: {pp.__class__.__name__:13}")

        logger.info("")

        setup_dirs(paths)
        out = setup_output(paths, template)
        process_files(omr_files, template, args_local, out)

    elif not subdirs:
        # Each subdirectory should have images or should be non-leaf
        logger.info(
            f"Note: No valid images or sub-folders found in {curr_dir}.\
            Empty directories not allowed."
        )

    # recursively process subfolders
    for d in subdirs:
        process_dir(root_dir, d, args, template)


def check_and_move(error_code, file_path, filepath2):
    # print("Dummy Move:  "+file_path, " --> ",filepath2)
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


def process_omr(template, omr_resp):
    # Note: This is a reference function. It is not part of the OMR checker
    # So its implementation is completely subjective to user's requirements.
    csv_resp = {}

    # symbol for absent response
    unmarked_symbol = ""

    # print("omr_resp",omr_resp)

    # Multi-column/multi-row questions which need to be concatenated
    for q_no, resp_keys in template.concatenations.items():
        csv_resp[q_no] = "".join([omr_resp.get(k, unmarked_symbol) for k in resp_keys])

    # Single-column/single-row questions
    for q_no in template.singles:
        csv_resp[q_no] = omr_resp.get(q_no, unmarked_symbol)

    # Note: concatenations and singles together should be mutually exclusive
    # and should cover all questions in the template(exhaustive)
    # TODO: ^add a warning if omr_resp has unused keys remaining
    return csv_resp


def report(status, streak, scheme, q_no, marked, ans, prev_marks, curr_marks, marks):
    logger.info(
        "%s \t %s \t\t %s \t %s \t %s \t %s \t %s "
        % (
            q_no,
            status,
            str(streak),
            "[" + scheme + "] ",
            (str(prev_marks) + " + " + str(curr_marks) + " =" + str(marks)),
            str(marked),
            str(ans),
        )
    )


def setup_output(paths, template):
    ns = argparse.Namespace()
    logger.info("\nChecking Files...")

    # Include current output paths
    ns.paths = paths

    # custom sort: To use integer order in question names instead of
    # alphabetical - avoids q1, q10, q2 and orders them q1, q2, ..., q10
    ns.resp_cols = sorted(
        list(template.concatenations.keys()) + template.singles,
        key=lambda x: int(x[1:]) if ord(x[1]) in range(48, 58) else 0,
    )
    ns.empty_resp = [""] * len(ns.resp_cols)
    ns.sheetCols = ["file_id", "input_path", "output_path", "score"] + ns.resp_cols
    ns.OUTPUT_SET = []
    ns.files_obj = {}
    ns.filesMap = {
        # todo: use os.path.join(paths.results_dir, f"Results_{TIME_NOW_HRS}.csv") etc
        "Results": f"{paths.results_dir}Results_{TIME_NOW_HRS}.csv",
        "MultiMarked": f"{paths.manual_dir}MultiMarkedFiles.csv",
        "Errors": f"{paths.manual_dir}ErrorFiles.csv",
    }

    for file_key, file_name in ns.filesMap.items():
        if not os.path.exists(file_name):
            logger.info("Note: Created new file: %s" % (file_name))
            # moved handling of files to pandas csv writer
            ns.files_obj[file_key] = file_name
            # Create Header Columns
            pd.DataFrame([ns.sheetCols], dtype=str).to_csv(
                ns.files_obj[file_key],
                mode="a",
                quoting=QUOTE_NONNUMERIC,
                header=False,
                index=False,
            )
        else:
            logger.info("Present : appending to %s" % (file_name))
            ns.files_obj[file_key] = open(file_name, "a")

    return ns


# TODO: Refactor into new process flow.
def preliminary_check():
    pass
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


# TODO: take a look at 'out.paths'
def process_files(omr_files, template, args, out):
    start_time = int(time())
    files_counter = 0
    STATS.files_not_moved = 0

    for file_path in omr_files:
        files_counter += 1

        file_name = file_path.name
        args["current_file"] = file_path

        in_omr = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        logger.info(
            f"\n({files_counter}) Opening image: \t{file_path}\tResolution: {in_omr.shape}"
        )

        # TODO: Get rid of saveImgList
        for i in range(ImageUtils.save_image_level):
            ImageUtils.reset_save_img(i + 1)

        ImageUtils.append_save_img(1, in_omr)

        # resize to conform to template
        in_omr = ImageUtils.resize_util(
            in_omr,
            config.dimensions.processing_width,
            config.dimensions.processing_height,
        )

        # run pre_processors in sequence
        for pre_processor in template.pre_processors:
            in_omr = pre_processor.apply_filter(in_omr, args)

        if in_omr is None:
            # Error OMR case
            new_file_path = out.paths.errors_dir + file_name
            out.OUTPUT_SET.append([file_name] + out.empty_resp)
            if check_and_move(
                constants.ERROR_CODES.NO_MARKER_ERR, file_path, new_file_path
            ):
                err_line = [file_name, file_path, new_file_path, "NA"] + out.empty_resp
                pd.DataFrame(err_line, dtype=str).T.to_csv(
                    out.files_obj["Errors"],
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            continue

        if args["setLayout"]:
            template_layout = draw_template_layout(
                in_omr, template, shifted=False, border=2
            )
            MainOperations.show("Template Layout", template_layout, 1, 1)
            continue

        # uniquify
        file_id = str(file_name)
        save_dir = out.paths.save_marked_dir
        response_dict, final_marked, multi_marked, _ = MainOperations.read_response(
            template,
            image=in_omr,
            name=file_id,
            save_dir=save_dir,
            auto_align=args["autoAlign"],
        )

        # concatenate roll nos, set unmarked responses, etc
        resp = process_omr(template, response_dict)
        logger.info("\nRead Response: \t", resp, "\n")
        if config.outputs.show_image_level >= 2:
            MainOperations.show(
                "Final Marked Bubbles : " + file_id,
                ImageUtils.resize_util_h(
                    final_marked, int(config.dimensions.display_height * 1.3)
                ),
                1,
                1,
            )

        # This evaluates and returns the score attribute
        # TODO: Automatic scoring
        # score = evaluate(resp, explain_scoring=config.outputs.explain_scoring)
        score = 0

        resp_array = []
        for k in out.resp_cols:
            resp_array.append(resp[k])

        out.OUTPUT_SET.append([file_name] + resp_array)

        # TODO: Add roll number validation here
        if multi_marked == 0:
            STATS.files_not_moved += 1
            new_file_path = save_dir + file_id
            # Enter into Results sheet-
            results_line = [file_name, file_path, new_file_path, score] + resp_array
            # Write/Append to results_line file(opened in append mode)
            pd.DataFrame(results_line, dtype=str).T.to_csv(
                out.files_obj["Results"],
                mode="a",
                quoting=QUOTE_NONNUMERIC,
                header=False,
                index=False,
            )
            # Todo: Add score calculation from template.json
            # print(
            #     "[%d] Graded with score: %.2f" % (files_counter, score),
            #     "\t file_id: ",
            #     file_id,
            # )
            # print(files_counter,file_id,resp['Roll'],'score : ',score)
        else:
            # multi_marked file
            logger.info("[%d] multi_marked, moving File: %s" % (files_counter, file_id))
            new_file_path = out.paths.multi_marked_dir + file_name
            if check_and_move(
                constants.ERROR_CODES.MULTI_BUBBLE_WARN, file_path, new_file_path
            ):
                mm_line = [file_name, file_path, new_file_path, "NA"] + resp_array
                pd.DataFrame(mm_line, dtype=str).T.to_csv(
                    out.files_obj["MultiMarked"],
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            # else:
            #     TODO:  Add appropriate record handling here
            #     pass

    print_stats(start_time, files_counter)

    # flush after every 20 files for a live view
    # if(files_counter % 20 == 0 or files_counter == len(omr_files)):
    #     for file_key in out.filesMap.keys():
    #         out.files_obj[file_key].flush()


def print_stats(start_time, files_counter):
    time_checking = round(time() - start_time, 2) if files_counter else 1
    log = logger.info
    log("")
    log("Total file(s) moved        : %d " % (STATS.files_moved))
    log("Total file(s) not moved    : %d " % (STATS.files_not_moved))
    log("------------------------------")
    log(
        "Total file(s) processed    : %d (%s)"
        % (
            files_counter,
            "Sum Tallied!"
            if files_counter == (STATS.files_moved + STATS.files_not_moved)
            else "Not Tallying!",
        )
    )

    if config.outputs.show_image_level <= 0:
        log(
            "\nFinished Checking %d file(s) in %.1f seconds i.e. ~%.1f minute(s)."
            % (files_counter, time_checking, time_checking / 60)
        )
        log(
            "OMR Processing Rate  :\t ~ %.2f seconds/OMR"
            % (time_checking / files_counter)
        )
        log(
            "OMR Processing Speed :\t ~ %.2f OMRs/minute"
            % ((files_counter * 60) / time_checking)
        )
    else:
        log("\nTotal script time :", time_checking, "seconds")

    if config.outputs.show_image_level <= 1:
        # TODO: colorama this
        log(
            "\nTip: To see some awesome visuals, open config.json and increase 'show_image_level'"
        )

    # evaluate_correctness(out)

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


# Evaluate accuracy based on OMRDataset file generated through moderation
# portal on the same set of images
def evaluate_correctness(out):
    # TODO: test_file WOULD BE RELATIVE TO INPUT SUBDIRECTORY NOW-
    test_file = "inputs/OMRDataset.csv"
    if os.path.exists(test_file):
        logger.info("\nStarting evaluation for: " + test_file)

        test_cols = ["file_id"] + out.resp_cols
        y_df = (
            pd.read_csv(test_file, dtype=str)[test_cols]
            .replace(np.nan, "", regex=True)
            .set_index("file_id")
        )

        if np.any(y_df.index.duplicated):
            y_df_filtered = y_df.loc[~y_df.index.duplicated(keep="first")]
            logger.warning(
                "WARNING: Found duplicate File-ids in file %s. \
                Removed %d rows from testing data. Rows remaining: %d"
                % (
                    test_file,
                    y_df.shape[0] - y_df_filtered.shape[0],
                    y_df_filtered.shape[0],
                )
            )
            y_df = y_df_filtered

        x_df = pd.DataFrame(out.OUTPUT_SET, dtype=str, columns=test_cols).set_index(
            "file_id"
        )
        # print("x_df",x_df.head())
        # print("\ny_df",y_df.head())
        intersection = y_df.index.intersection(x_df.index)

        # Checking if the merge is okay
        if intersection.size == x_df.index.size:
            y_df = y_df.loc[intersection]
            x_df["TestResult"] = (x_df == y_df).all(axis=1).astype(int)
            logger.info(x_df.head())
            logger.info(
                "\n\t Accuracy on the %s Dataset: %.6f"
                % (test_file, (x_df["TestResult"].sum() / x_df.shape[0]))
            )
        else:
            logger.error(
                "\nERROR: Insufficient Testing Data: Have you appended MultiMarked data yet?"
            )
            logger.error(
                "Missing File-ids: ", list(x_df.index.difference(intersection))
            )


TIME_NOW_HRS = strftime("%I%p", localtime())
