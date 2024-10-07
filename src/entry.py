import os
from csv import QUOTE_NONNUMERIC
from pathlib import Path, PurePosixPath
from time import time

import pandas as pd
from rich.table import Table
from rich_tools import table_to_df

from src.algorithm.evaluation.config import EvaluationConfig
from src.algorithm.evaluation.evaluation import evaluate_concatenated_response
from src.algorithm.template.alignment.template_alignment import apply_template_alignment
from src.algorithm.template.template import Template
from src.schemas.constants import DEFAULT_ANSWERS_SUMMARY_FORMAT_STRING
from src.schemas.defaults import CONFIG_DEFAULTS
from src.utils import constants
from src.utils.file import PathUtils
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils, Stats
from src.utils.logger import console, logger
from src.utils.parsing import open_config_with_defaults
from src.utils.template_drawing import TemplateDrawing

# Load processors
STATS = Stats()


def entry_point(input_dir, args):
    if not os.path.exists(input_dir):
        raise Exception(f"Given input directory does not exist: '{input_dir}'")
    curr_dir = input_dir
    return process_dir(input_dir, curr_dir, args)


# TODO: move into template.directory_handler?
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
    table.add_row("Debug Mode ", "ON" if args["debug"] else "OFF")
    table.add_row("Output Mode ", args["output_mode"])
    table.add_row("Set Layout Mode ", "ON" if args["setLayout"] else "OFF")
    table.add_row(
        "Markers Detection",
        "ON" if "CropOnMarkers" in template.get_pre_processor_names() else "OFF",
    )
    table.add_row("Detected Template Path", f"{template.path}")
    if local_config_path:
        table.add_row("Detected Local Config", f"{local_config_path}")
    if evaluation_config:
        table.add_row("Detected Evaluation Config", f"{evaluation_config}")

    table.add_row(
        "Detected pre-processors",
        f"{template.get_pre_processor_names()}",
    )
    table.add_row("Processing Image Shape", f"{template.get_processing_image_shape()}")

    console.print(table, justify="center")


# TODO: move into template.directory_handler?
def process_dir(
    root_dir,
    curr_dir,
    args,
    template=None,
    tuning_config=CONFIG_DEFAULTS,
    evaluation_config=None,
):
    # Update local tuning_config (in current recursion stack)
    local_config_path = curr_dir.joinpath(constants.CONFIG_FILENAME)
    if os.path.exists(local_config_path):
        tuning_config = open_config_with_defaults(local_config_path)

    # Update local template (in current recursion stack)
    local_template_path = curr_dir.joinpath(constants.TEMPLATE_FILENAME)
    local_template_exists = os.path.exists(local_template_path)
    if local_template_exists:
        template = Template(
            local_template_path,
            tuning_config,
        )
    # Look for subdirectories for processing
    subdirs = [d for d in curr_dir.iterdir() if d.is_dir()]

    output_mode = args["output_mode"]

    # look for images in current dir to process
    exts = ("*.[pP][nN][gG]", "*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]")
    omr_files = sorted([f for ext in exts for f in curr_dir.glob(ext)])

    # omr_files = PathUtils.filter_omr_files(omr_files)
    omr_files = [Path(PurePosixPath(omr_file).as_posix()) for omr_file in omr_files]

    # Exclude images (take union over all pre_processors)
    excluded_files = []
    if template:
        excluded_files.extend(template.get_exclude_files())

    local_evaluation_path = curr_dir.joinpath(constants.EVALUATION_FILENAME)
    # Note: if setLayout is passed, there's no need to load evaluation file
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
                of '{curr_dir}'. \nPlace {constants.TEMPLATE_FILENAME} in the \
                appropriate directory."
            )
            # TODO: restore support for --default-template flag
            raise Exception(
                f"No template file found in the directory tree of {curr_dir}"
            )

        output_dir = Path(args["output_dir"], curr_dir.relative_to(root_dir))

        # Reset all mutations to the template, and setup output directories
        template.reset_and_setup_for_directory(output_dir, output_mode)

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
            # TODO: pick these args from self/class instead of prop forwarding
            process_directory_files(
                omr_files,
                template,
                tuning_config,
                evaluation_config,
                output_mode,
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


# TODO: move into template.template_layout
def show_template_layouts(omr_files, template, tuning_config):
    for file_path in omr_files:
        file_name = file_path.name
        file_path = str(file_path)
        gray_image, colored_image = ImageUtils.read_image_util(file_path, tuning_config)

        (
            gray_image,
            colored_image,
            template,
        ) = template.apply_preprocessors(file_path, gray_image, colored_image)
        gray_layout, colored_layout = TemplateDrawing.draw_template_layout(
            gray_image,
            colored_image,
            template,
            tuning_config,
            shifted=False,
            border=2,
        )
        template_layout = (
            colored_layout
            if tuning_config.outputs.colored_outputs_enabled
            else gray_layout
        )
        # Size is template_dimensions
        InteractionUtils.show_for_roi(
            f"Template Layout: {file_name}",
            template_layout,
        )


# TODO: move into template.directory_handler
def process_directory_files(
    omr_files,
    template: Template,
    tuning_config,
    evaluation_config,
    output_mode,
):
    start_time = int(time())
    files_counter = 0
    # TODO: move STATS inside template.directory_handler
    STATS.files_not_moved = 0
    logger.set_log_levels(tuning_config.outputs.show_logs_by_type)
    for file_path in omr_files:
        files_counter += 1
        file_name = PathUtils.remove_non_utf_characters(file_path.name)
        file_id = str(file_name)

        gray_image, colored_image = ImageUtils.read_image_util(
            str(file_path), tuning_config
        )

        logger.info("")
        logger.info(
            f"({files_counter}) Opening image: \t'{file_path}'\tResolution: {gray_image.shape}"
        )

        # Start with blank saved images list
        template.save_image_ops.reset_all_save_img()

        template.save_image_ops.append_save_image(
            "Input Image", range(1, 7), gray_image, colored_image
        )

        # TODO: use try catch here and store paths to error files

        # Note: the returned template is a copy
        (
            gray_image,
            colored_image,
            template,
        ) = template.apply_preprocessors(file_path, gray_image, colored_image)

        # TODO[later]: template & evaluation as a "Processor"?
        # TODO: move apply_template_alignment into template class
        gray_image, colored_image, template = apply_template_alignment(
            gray_image, colored_image, template, tuning_config
        )

        if gray_image is None:
            # Error OMR case
            output_file_path = template.get_errors_dir().joinpath(file_name)

            template.append_empty_row(file_name)

            # TODO: move into template.directory_handler
            if check_and_move(
                constants.ERROR_CODES.NO_MARKER_ERR, file_path, output_file_path
            ):
                error_file_line = [
                    file_name,
                    file_path,
                    output_file_path,
                    "NA",
                ] + template.get_empty_response_array()

                pd.DataFrame(error_file_line, dtype=str).T.to_csv(
                    template.get_errors_file(),
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            continue

        concatenated_omr_response, raw_omr_response = template.read_omr_response(
            gray_image, colored_image, file_path
        )

        # TODO: add a try catch here?

        evaluation_config_for_response = (
            None
            if evaluation_config is None
            else evaluation_config.get_evaluation_config_for_response(
                concatenated_omr_response, file_path
            )
        )

        if (
            evaluation_config_for_response is None
            or not evaluation_config_for_response.get_should_explain_scoring()
        ):
            logger.info(f"Read Response: \n{concatenated_omr_response}")

        score, evaluation_meta = 0, None
        if evaluation_config_for_response is not None:
            score, evaluation_meta = evaluate_concatenated_response(
                concatenated_omr_response, evaluation_config_for_response
            )
            (
                default_answers_summary,
                *_,
            ) = evaluation_config_for_response.get_formatted_answers_summary(
                DEFAULT_ANSWERS_SUMMARY_FORMAT_STRING
            )
            logger.info(
                f"(/{files_counter}) Graded with score: {round(score, 2)}\t {default_answers_summary} \t file: '{file_id}'"
            )
            if evaluation_config_for_response.get_should_export_explanation_csv():
                explanation_table = evaluation_config_for_response.get_explanation_table()
                explanation_table = table_to_df(explanation_table)
                explanation_table.to_csv(
                    template.get_evaluations_dir().joinpath(file_name + ".csv"),
                    quoting=QUOTE_NONNUMERIC,
                    index=False,
                )

        else:
            logger.info(f"(/{files_counter}) Processed file: '{file_id}'")

        # TODO: move this logic inside the class
        save_marked_dir = template.get_save_marked_dir()

        # ----------

        # TODO: refactor this logic
        (
            is_multi_marked,
            field_number_to_scan_box_interpretation,
        ) = template.get_omr_metrics_for_file(str(file_path))

        # Save output image with bubble values and evaluation meta
        if output_mode != "moderation":
            # TODO: move TemplateDrawing functions inside the template class
            (
                final_marked,
                colored_final_marked,
            ) = TemplateDrawing.draw_template_layout(
                gray_image,
                colored_image,
                template,
                tuning_config,
                field_number_to_scan_box_interpretation,
                evaluation_meta=evaluation_meta,
                evaluation_config_for_response=evaluation_config_for_response,
            )
        else:
            # No drawing in moderation output mode
            final_marked, colored_final_marked = gray_image, colored_image

        # TODO(refactor): move to small function
        should_save_detections = (
            tuning_config.outputs.save_detections and save_marked_dir is not None
        )

        if should_save_detections:
            # TODO: migrate after support for is_multi_marked bucket based on identifier config
            # if multi_roll:
            #     save_marked_dir = save_marked_dir.joinpath("_MULTI_")
            ImageUtils.save_marked_image(save_marked_dir, file_id, final_marked)

            if (
                tuning_config.outputs.colored_outputs_enabled
                and save_marked_dir is not None
            ):
                # TODO: get dedicated path from top args
                colored_save_marked_dir = save_marked_dir.joinpath("colored")
                ImageUtils.save_marked_image(
                    colored_save_marked_dir, file_id, colored_final_marked
                )

        # Save output stack images
        template.save_image_ops.save_image_stacks(
            file_id,
            save_marked_dir,
            key=None,
            images_per_row=5 if tuning_config.outputs.show_image_level >= 5 else 4,
        )

        # TODO: move it inside template finalize_file_level_metrics()?
        # Save output metrics
        if tuning_config.outputs.save_image_metrics:
            template.export_omr_metrics_for_file(
                str(file_path),
                evaluation_meta,
                field_number_to_scan_box_interpretation,
            )

        # Save output CSV results
        output_omr_response = (
            raw_omr_response
            if output_mode == "moderation"
            else concatenated_omr_response
        )
        omr_response_array = template.append_output_omr_response(
            file_name, output_omr_response
        )

        posix_file_path = PathUtils.sep_based_posix_path(file_path)

        if (
            not is_multi_marked
            or not tuning_config.outputs.filter_out_multimarked_files
        ):
            STATS.files_not_moved += 1

            # Normalize path and convert to posix style
            output_file_path = PathUtils.sep_based_posix_path(
                save_marked_dir.joinpath(file_id)
            )
            # Enter into Results sheet-
            results_line = [
                file_name,
                posix_file_path,
                output_file_path,
                score,
            ] + omr_response_array

            # Write/Append to results_line file(opened in append mode)
            pd.DataFrame(results_line, dtype=str).T.to_csv(
                template.get_results_file(),
                mode="a",
                quoting=QUOTE_NONNUMERIC,
                header=False,
                index=False,
            )
        else:
            # is_multi_marked file
            logger.info(f"[{files_counter}] Found multi-marked file: '{file_id}'")
            output_file_path = PathUtils.sep_based_posix_path(
                template.get_multi_marked_dir().joinpath(file_name)
            )
            if check_and_move(
                constants.ERROR_CODES.MULTI_BUBBLE_WARN, file_path, output_file_path
            ):
                mm_line = [
                    file_name,
                    posix_file_path,
                    output_file_path,
                    "NA",
                ] + omr_response_array
                pd.DataFrame(mm_line, dtype=str).T.to_csv(
                    template.get_multi_marked_file(),
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            # else:
            #     TODO:  Add appropriate record handling here
            #     pass

    logger.reset_log_levels()

    # Calculate folder level stats here
    template.finalize_directory_metrics()
    # TODO: export directory level stats here

    print_stats(start_time, files_counter, tuning_config)


def check_and_move(error_code, file_path, filepath2):
    # TODO: use StatsByLabel class here
    # TODO: fix file movement into error/multimarked/invalid etc again
    STATS.files_not_moved += 1
    return True


# TODO: move into template.directory_handler
def print_stats(start_time, files_counter, tuning_config):
    time_checking = max(1, round(time() - start_time, 2))
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
            f"\nFinished Checking {files_counter} file(s) in {round(time_checking, 1)} seconds i.e. ~{round(time_checking / 60, 1)} minute(s)."
        )
        log(
            f"{'OMR Processing Rate':<27}:\t ~ {round(time_checking / files_counter,2)} seconds/OMR"
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
