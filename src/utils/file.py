import argparse
import json
import os
from collections import defaultdict
from csv import QUOTE_NONNUMERIC
from time import localtime, strftime

import pandas as pd

from src.utils.image import ImageUtils
from src.utils.logger import logger
from src.utils.math import MathUtils


def load_json(path, **rest):
    try:
        with open(path, "r") as f:
            loaded = json.load(f, **rest)
    except json.decoder.JSONDecodeError as error:
        logger.critical(error)
        raise Exception(f"Error when loading json file at: '{path}'")

    return loaded


class Paths:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.save_marked_dir = output_dir.joinpath("CheckedOMRs")
        self.image_metrics_dir = output_dir.joinpath("ImageMetrics")
        self.results_dir = output_dir.joinpath("Results")
        self.manual_dir = output_dir.joinpath("Manual")
        self.errors_dir = self.manual_dir.joinpath("ErrorFiles")
        self.multi_marked_dir = self.manual_dir.joinpath("MultiMarkedFiles")


def setup_dirs_for_paths(paths):
    logger.info("Checking Directories...")

    if not os.path.exists(paths.save_marked_dir):
        os.makedirs(paths.save_marked_dir)
    # Main output directories
    for save_output_dir in [
        paths.save_marked_dir.joinpath("colored"),
        paths.save_marked_dir.joinpath("stack"),
        paths.save_marked_dir.joinpath("stack", "colored"),
        paths.save_marked_dir.joinpath("_MULTI_"),
        paths.save_marked_dir.joinpath("_MULTI_", "colored"),
    ]:
        if not os.path.exists(save_output_dir):
            os.mkdir(save_output_dir)
            logger.info(f"Created : {save_output_dir}")

    # Image buckets
    for save_output_dir in [
        paths.manual_dir,
        paths.multi_marked_dir,
        paths.errors_dir,
    ]:
        if not os.path.exists(save_output_dir):
            logger.info(f"Created : {save_output_dir}")
            os.makedirs(save_output_dir)
            os.mkdir(save_output_dir.joinpath("colored"))

    # Non-image directories
    for save_output_dir in [
        paths.results_dir,
        paths.image_metrics_dir,
    ]:
        if not os.path.exists(save_output_dir):
            logger.info(f"Created : {save_output_dir}")
            os.makedirs(save_output_dir)


def setup_outputs_for_template(paths, template):
    # TODO: consider moving this into a class instance
    ns = argparse.Namespace()
    logger.info("Checking Files...")

    # Include current output paths
    ns.paths = paths

    ns.empty_resp = [""] * len(template.output_columns)
    ns.sheetCols = [
        "file_id",
        "input_path",
        "output_path",
        "score",
    ] + template.output_columns
    ns.OUTPUT_SET = []
    ns.files_obj = {}
    TIME_NOW_HRS = strftime("%I%p", localtime())
    ns.filesMap = {
        "Results": os.path.join(paths.results_dir, f"Results_{TIME_NOW_HRS}.csv"),
        "MultiMarked": os.path.join(paths.manual_dir, "MultiMarkedFiles.csv"),
        "Errors": os.path.join(paths.manual_dir, "ErrorFiles.csv"),
    }

    for file_key, file_name in ns.filesMap.items():
        if not os.path.exists(file_name):
            logger.info(f"Created new file: '{file_name}'")
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
            logger.info(f"Present : appending to '{file_name}'")
            ns.files_obj[file_key] = open(file_name, "a")

    return ns


class SaveImageOps:
    def __init__(self, tuning_config):
        self.gray_images = defaultdict(list)
        self.colored_images = defaultdict(list)
        self.tuning_config = tuning_config
        self.save_image_level = tuning_config.outputs.save_image_level

    def append_save_image(self, title, keys, gray_image=None, colored_image=None):
        assert type(title) == str
        if type(keys) == int:
            keys = [keys]
        gray_image_copy, colored_image_copy = None, None
        if gray_image is not None:
            gray_image_copy = gray_image.copy()
        if colored_image is not None:
            colored_image_copy = colored_image.copy()
        for key in keys:
            if int(key) > self.save_image_level:
                continue
            if gray_image_copy is not None:
                self.gray_images[key].append([title, gray_image_copy])
            if colored_image_copy is not None:
                self.colored_images[key].append([title, colored_image_copy])

    def save_image_stacks(self, filename, save_marked_dir, key=None, images_per_row=4):
        key = self.save_image_level if key is None else key
        if self.save_image_level >= int(key):
            name, _ext = os.path.splitext(filename)
            logger.info(
                f"Stack level: {key}", [title for title, _ in self.gray_images[key]]
            )

            if len(self.gray_images[key]) > 0:
                result = self.get_result_hstack(self.gray_images[key], images_per_row)
                stack_path = f"{save_marked_dir}/stack/{name}_{str(key)}_stack.jpg"
                logger.info(f"Saved stack image to: {stack_path}")
                ImageUtils.save_img(stack_path, result)
            else:
                logger.info(
                    f"Note: Nothing to save for gray image. Stack level: {self.save_image_level}"
                )

            if len(self.colored_images[key]) > 0:
                colored_result = self.get_result_hstack(
                    self.colored_images[key], images_per_row
                )
                colored_stack_path = (
                    f"{save_marked_dir}/stack/colored/{name}_{str(key)}_stack.jpg"
                )
                logger.info(f"Saved colored stack image to: {colored_stack_path}")

                ImageUtils.save_img(
                    colored_stack_path,
                    colored_result,
                )
            else:
                logger.info(
                    f"Note: Nothing to save for colored image. Stack level: {self.save_image_level}"
                )

    def get_result_hstack(self, titles_and_images, images_per_row):
        config = self.tuning_config
        _display_height, display_width = config.outputs.display_image_dimensions

        # TODO: attach title text as header to each stack image!
        images = [
            ImageUtils.resize_util(image, display_width)
            for _title, image in titles_and_images
        ]
        grid_images = MathUtils.chunks(images, images_per_row)
        result = ImageUtils.get_vstack_image_grid(grid_images)
        result = ImageUtils.resize_util(
            result,
            min(
                len(titles_and_images) * display_width // 3,
                int(display_width * 2.5),
            ),
        )
        return result

    def reset_all_save_img(self):
        # Max save image level is 6
        for i in range(7):
            self.gray_images[i + 1] = []
            self.colored_images[i + 1] = []
