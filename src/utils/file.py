import argparse
import json
import os
from csv import QUOTE_NONNUMERIC
from time import localtime, strftime

import pandas as pd

from src.logger import logger


def load_json(path, **rest):
    with open(path, "r") as f:
        loaded = json.load(f, **rest)
    return loaded


def setup_outputs_for_template(paths, template):
    ns = argparse.Namespace()
    logger.info("Checking Files...")

    # Include current output paths
    ns.paths = paths

    # custom sort: To use integer order in question names instead of
    # alphabetical - avoids q1, q10, q2 and orders them q1, q2, ..., q10
    ns.resp_cols = sorted(
        list(template.concatenations.keys()) + template.singles,
        key=lambda x: int(x[1:]) if ord(x[1]) in range(48, 58) else 0,
    )
    # todo: consider using emptyVal for empty_resp
    ns.empty_resp = [""] * len(ns.resp_cols)
    ns.sheetCols = ["file_id", "input_path", "output_path", "score"] + ns.resp_cols
    ns.OUTPUT_SET = []
    ns.files_obj = {}
    TIME_NOW_HRS = strftime("%I%p", localtime())
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


class Paths:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.save_marked_dir = f"{self.output_dir}/CheckedOMRs/"
        self.results_dir = f"{self.output_dir}/Results/"
        self.manual_dir = f"{self.output_dir}/Manual/"
        self.errors_dir = f"{self.manual_dir}ErrorFiles/"
        self.multi_marked_dir = f"{self.manual_dir}MultiMarkedFiles/"


def setup_dirs_for_paths(paths):
    logger.info("Checking Directories...")
    for _dir in [paths.save_marked_dir]:
        if not os.path.exists(_dir):
            logger.info("Created : " + _dir)
            os.makedirs(_dir)
            os.mkdir(_dir + "/stack")
            os.mkdir(_dir + "/_MULTI_")
            os.mkdir(_dir + "/_MULTI_" + "/stack")
        # else:
        #     logger.info("Present : " + _dir)

    for _dir in [paths.manual_dir, paths.results_dir]:
        if not os.path.exists(_dir):
            logger.info("Created : " + _dir)
            os.makedirs(_dir)
        # else:
        #     logger.info("Present : " + _dir)

    for _dir in [paths.multi_marked_dir, paths.errors_dir]:
        if not os.path.exists(_dir):
            logger.info("Created : " + _dir)
            os.makedirs(_dir)
