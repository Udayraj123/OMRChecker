"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from csv import QUOTE_NONNUMERIC
from pathlib import Path
from time import time

import cv2
import pandas as pd
from rich.table import Table

from src.constants.common import (
    CONFIG_FILENAME,
    ERROR_CODES,
    EVALUATION_FILENAME,
    TEMPLATE_FILENAME,
)
from src.defaults import CONFIG_DEFAULTS
from src.evaluation import EvaluationConfig
from src.logger import console, logger
from src.template import Template
from src.utils.file import Paths, setup_dirs_for_paths, setup_outputs_for_template
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils, Stats
from src.utils.parallel import process_single_file_worker
from src.utils.parsing import open_config_with_defaults

# Load processors
STATS = Stats()


def entry_point(input_dir, args):
    if not os.path.exists(input_dir):
        raise Exception(f"Given input directory does not exist: '{input_dir}'")
    curr_dir = input_dir
    return process_dir(input_dir, curr_dir, args)


def print_config_summary(
    curr_dir,
    omr_files,
    template,
    tuning_config,
    local_config_path,
    evaluation_config,
    args,
):
    logger.info("")
    table = Table(title="Current Configurations", show_header=False, show_lines=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_row("Directory Path", f"{curr_dir}")
    table.add_row("Count of Images", f"{len(omr_files)}")
    table.add_row("Set Layout Mode ", "ON" if args["setLayout"] else "OFF")
    pre_processor_names = [pp.__class__.__name__ for pp in template.pre_processors]
    table.add_row(
        "Markers Detection",
        "ON" if "CropOnMarkers" in pre_processor_names else "OFF",
    )
    table.add_row("Auto Alignment", f"{tuning_config.alignment_params.auto_align}")
    table.add_row("Detected Template Path", f"{template}")
    if local_config_path:
        table.add_row("Detected Local Config", f"{local_config_path}")
    if evaluation_config:
        table.add_row("Detected Evaluation Config", f"{evaluation_config}")

    table.add_row(
        "Detected pre-processors",
        ", ".join(pre_processor_names),
    )
    console.print(table, justify="center")


def process_dir(
    root_dir,
    curr_dir,
    args,
    template=None,
    tuning_config=CONFIG_DEFAULTS,
    evaluation_config=None,
):
    # Update local tuning_config (in current recursion stack)
    local_config_path = curr_dir.joinpath(CONFIG_FILENAME)
    if os.path.exists(local_config_path):
        tuning_config = open_config_with_defaults(local_config_path)

    # Update local template (in current recursion stack)
    local_template_path = curr_dir.joinpath(TEMPLATE_FILENAME)
    local_template_exists = os.path.exists(local_template_path)
    if local_template_exists:
        template = Template(
            local_template_path,
            tuning_config,
        )
    # Look for subdirectories for processing
    subdirs = [d for d in curr_dir.iterdir() if d.is_dir()]

    output_dir = Path(args["output_dir"], curr_dir.relative_to(root_dir))
    paths = Paths(output_dir)

    # look for images in current dir to process
    exts = ("*.[pP][nN][gG]", "*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]")
    omr_files = sorted([f for ext in exts for f in curr_dir.glob(ext)])

    # Exclude images (take union over all pre_processors)
    excluded_files = []
    if template:
        for pp in template.pre_processors:
            excluded_files.extend(Path(p) for p in pp.exclude_files())

    local_evaluation_path = curr_dir.joinpath(EVALUATION_FILENAME)
    if not args["setLayout"] and os.path.exists(local_evaluation_path):
        if not local_template_exists:
            logger.warning(
                f"Found an evaluation file without a parent template file: {local_evaluation_path}"
            )
        evaluation_config = EvaluationConfig(
            curr_dir,
            local_evaluation_path,
            template,
            tuning_config,
        )

        excluded_files.extend(
            Path(exclude_file) for exclude_file in evaluation_config.get_exclude_files()
        )

    omr_files = [f for f in omr_files if f not in excluded_files]

    if omr_files:
        if not template:
            logger.error(
                f"Found images, but no template in the directory tree \
                of '{curr_dir}'. \nPlace {TEMPLATE_FILENAME} in the \
                appropriate directory."
            )
            raise Exception(
                f"No template file found in the directory tree of {curr_dir}"
            )

        setup_dirs_for_paths(paths)
        outputs_namespace = setup_outputs_for_template(paths, template)

        print_config_summary(
            curr_dir,
            omr_files,
            template,
            tuning_config,
            local_config_path,
            evaluation_config,
            args,
        )
        if args["setLayout"]:
            show_template_layouts(omr_files, template, tuning_config)
        else:
            process_files(
                omr_files,
                template,
                tuning_config,
                evaluation_config,
                outputs_namespace,
            )

    elif not subdirs:
        # Each subdirectory should have images or should be non-leaf
        logger.info(
            f"No valid images or sub-folders found in {curr_dir}.\
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
        )


def show_template_layouts(omr_files, template, tuning_config):
    for file_path in omr_files:
        file_name = file_path.name
        file_path = str(file_path)
        in_omr = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        in_omr = template.image_instance_ops.apply_preprocessors(
            file_path, in_omr, template
        )
        template_layout = template.image_instance_ops.draw_template_layout(
            in_omr, template, shifted=False, border=2
        )
        InteractionUtils.show(
            f"Template Layout: {file_name}", template_layout, 1, 1, config=tuning_config
        )


def process_files(
    omr_files,
    template,
    tuning_config,
    evaluation_config,
    outputs_namespace,
):
    start_time = int(time())
    files_counter = 0
    STATS.files_not_moved = 0

    save_dir = outputs_namespace.paths.save_marked_dir
    evaluation_dir = outputs_namespace.paths.evaluation_dir

    futures = []
    # If multiprocessing behaves badly, consider exposing max_workers configuraton.
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for file_path in omr_files:
            futures.append(
                executor.submit(
                    process_single_file_worker,
                    file_path,
                    template,
                    tuning_config,
                    evaluation_config,
                    save_dir,
                    evaluation_dir,
                )
            )

        for future in as_completed(futures):
            res = future.result()
            files_counter += 1
            file_name = res["file_name"]
            file_path = res["file_path"]

            logger.info("")
            if res["status"] == "error":
                new_file_path = outputs_namespace.paths.errors_dir.joinpath(file_name)
                outputs_namespace.OUTPUT_SET.append(
                    [file_name] + outputs_namespace.empty_resp
                )
                if check_and_move(res["error_code"], file_path, new_file_path):
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

            logger.info(
                f"({files_counter}) Opened image: \t'{file_path}'\tResolution: {res.get('resolution', 'N/A')}"
            )

            file_id = res["file_id"]

            if (
                evaluation_config is None
                or not evaluation_config.get_should_explain_scoring()
            ):
                logger.info(f"Read Response: \n{res['omr_response']}")

            if evaluation_config is not None:
                logger.info(
                    f"(/{files_counter}) Graded with score: {round(res['score'], 2)}\t for file: '{file_id}'"
                )
            else:
                logger.info(f"(/{files_counter}) Processed file: '{file_id}'")

            if res.get("final_marked") is not None:
                InteractionUtils.show(
                    f"Final Marked Bubbles : '{file_id}'",
                    ImageUtils.resize_util_h(
                        res["final_marked"],
                        int(tuning_config.dimensions.display_height * 1.3),
                    ),
                    1,
                    1,
                    config=tuning_config,
                )

            outputs_namespace.OUTPUT_SET.append([file_name] + res["resp_array"])

            if (
                res["multi_marked"] == 0
                or not tuning_config.outputs.filter_out_multimarked_files
            ):
                STATS.files_not_moved += 1
                new_file_path = save_dir.joinpath(file_id)
                # Enter into Results sheet-
                results_line = [
                    file_name,
                    file_path,
                    new_file_path,
                    res["score"],
                ] + res["resp_array"]
                pd.DataFrame(results_line, dtype=str).T.to_csv(
                    outputs_namespace.files_obj["Results"],
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            else:
                # multi_marked file
                logger.info(f"[{files_counter}] Found multi-marked file: '{file_id}'")
                new_file_path = outputs_namespace.paths.multi_marked_dir.joinpath(
                    file_name
                )
                if check_and_move(
                    ERROR_CODES.MULTI_BUBBLE_WARN, file_path, new_file_path
                ):
                    mm_line = [file_name, file_path, new_file_path, "NA"] + res[
                        "resp_array"
                    ]
                    pd.DataFrame(mm_line, dtype=str).T.to_csv(
                        outputs_namespace.files_obj["MultiMarked"],
                        mode="a",
                        quoting=QUOTE_NONNUMERIC,
                        header=False,
                        index=False,
                    )

    print_stats(start_time, files_counter, tuning_config)


def check_and_move(error_code, file_path, filepath2):
    # TODO: fix file movement into error/multimarked/invalid etc again
    STATS.files_not_moved += 1
    return True


def print_stats(start_time, files_counter, tuning_config):
    time_checking = max(1, round(time() - start_time, 2))
    log = logger.info
    log("")
    log(f"{'Total file(s) moved': <27}: {STATS.files_moved}")
    log(f"{'Total file(s) not moved': <27}: {STATS.files_not_moved}")
    log("--------------------------------")
    log(
        f"{'Total file(s) processed': <27}: {files_counter} ({'Sum Tallied!' if files_counter == (STATS.files_moved + STATS.files_not_moved) else 'Not Tallying!'})"
    )

    if tuning_config.outputs.show_image_level <= 0:
        log(
            f"\nFinished Checking {files_counter} file(s) in {round(time_checking, 1)} seconds i.e. ~{round(time_checking / 60, 1)} minute(s)."
        )
        log(
            f"{'OMR Processing Rate': <27}: \t ~ {round(time_checking / files_counter, 2)} seconds/OMR"
        )
        log(
            f"{'OMR Processing Speed': <27}: \t ~ {round((files_counter * 60) / time_checking, 2)} OMRs/minute"
        )
    else:
        log(f"\n{'Total script time': <27}: {time_checking} seconds")

    if tuning_config.outputs.show_image_level <= 1:
        log(
            "\nTip: To see some awesome visuals, open config.json and increase 'show_image_level'"
        )
