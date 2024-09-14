import os
from csv import QUOTE_NONNUMERIC
from time import localtime, strftime

import pandas as pd

from src.utils.file import PathUtils
from src.utils.logger import logger


# Process files + directory
class DirectoryHandler:
    def __init__(self, template):
        self.template = template
        self.path_utils = None

    def reset_path_utils(self, output_dir, output_mode):
        # Override the paths utils
        self.path_utils = PathUtils(output_dir)
        self.path_utils.create_output_directories()
        self.setup_outputs_namespace(output_mode)

    def setup_outputs_namespace(self, output_mode):
        logger.info("Checking Files...")
        self.omr_response_columns = (
            list(self.template.all_parsed_labels)
            if output_mode == "moderation"
            else self.template.output_columns
        )

        # Include current output paths
        self.empty_response_array = [""] * len(self.omr_response_columns)

        self.sheet_columns = [
            "file_id",
            "input_path",
            "output_path",
            "score",
        ] + self.omr_response_columns

        self.output_files = {}
        TIME_NOW_HRS = strftime("%I%p", localtime())
        files_map = {
            "Results": os.path.join(
                self.path_utils.results_dir, f"Results_{TIME_NOW_HRS}.csv"
            ),
            "MultiMarked": os.path.join(
                self.path_utils.manual_dir, "MultiMarkedFiles.csv"
            ),
            "Errors": os.path.join(self.path_utils.manual_dir, "ErrorFiles.csv"),
        }

        for file_key, file_name in files_map.items():
            if not os.path.exists(file_name):
                logger.info(f"Created new file: '{file_name}'")
                # moved handling of files to pandas csv writer
                self.output_files[file_key] = file_name
                # Create Header Columns
                pd.DataFrame([self.sheet_columns], dtype=str).to_csv(
                    self.output_files[file_key],
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            else:
                logger.info(f"Present : appending to '{file_name}'")
                self.output_files[file_key] = open(file_name, "a")
