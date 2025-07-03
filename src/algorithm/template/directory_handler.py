from csv import QUOTE_NONNUMERIC
from pathlib import Path
from time import localtime, strftime

import pandas as pd

from src.utils.constants import OUTPUT_MODES
from src.utils.file import PathUtils
from src.utils.logger import logger


# Process files + directory
class TemplateDirectoryHandler:
    def __init__(self, template) -> None:
        self.template = template

    def reset_path_utils(self, output_dir, output_mode) -> None:
        if output_mode == OUTPUT_MODES.SET_LAYOUT:
            logger.info("Note: Skipped files creation in setLayout mode")
            return

        # Override the paths utils
        self.path_utils = PathUtils(output_dir)
        self.path_utils.create_output_directories()
        self.setup_outputs_namespace(output_mode)

    def setup_outputs_namespace(self, output_mode) -> None:
        logger.info("Checking Files...")
        self.omr_response_columns = (
            list(self.template.all_parsed_labels)
            if output_mode == OUTPUT_MODES.MODERATION
            else self.template.output_columns
        )

        # Include current output paths
        self.empty_response_array = [""] * len(self.omr_response_columns)

        self.sheet_columns = [
            "file_id",
            "input_path",
            "output_path",
            "score",
            *self.omr_response_columns,
        ]

        self.output_files = {}
        time_now_hrs = strftime("%I%p", localtime())
        files_map = {
            "Results": Path(self.path_utils.results_dir, f"Results_{time_now_hrs}.csv"),
            "MultiMarked": Path(self.path_utils.manual_dir, "MultiMarkedFiles.csv"),
            "Errors": Path(self.path_utils.manual_dir, "ErrorFiles.csv"),
        }

        for file_key, file_path in files_map.items():
            if not file_path.exists():
                logger.info(f"Created new file: '{file_path}'")
                # moved handling of files to pandas csv writer
                self.output_files[file_key] = file_path
                # Create Header Columns
                pd.DataFrame([self.sheet_columns], dtype=str).to_csv(
                    self.output_files[file_key],
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            else:
                logger.info(f"Present : appending to '{file_path}'")
                with Path.open(file_path, "a") as f:
                    self.output_files[file_key] = f

    def get_empty_response_array(self):
        return self.empty_response_array
