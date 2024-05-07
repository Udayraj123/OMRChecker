import argparse
import json
import os
from collections import defaultdict
from csv import QUOTE_NONNUMERIC
from time import localtime, strftime

import numpy as np
import pandas as pd

from src.utils.image import ImageUtils
from src.utils.logger import logger


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
        paths.save_marked_dir.joinpath("stack"),
        paths.save_marked_dir.joinpath("colored"),
        paths.save_marked_dir.joinpath("_MULTI_"),
        paths.save_marked_dir.joinpath("_MULTI_", "stack"),
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
        self.save_img_list = defaultdict(list)
        self.tuning_config = tuning_config
        self.save_image_level = tuning_config.outputs.save_image_level

    def append_save_image(self, key, img):
        if img is None:
            return
        if self.save_image_level >= int(key):
            self.save_img_list[key].append(img.copy())

    # TODO: call this function appropriately to enable debug stacks again
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
